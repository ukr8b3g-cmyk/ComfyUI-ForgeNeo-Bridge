"""Chromium DOM/import smoke test, using real Bridge modules and a mocked Comfy host.
Does NOT certify the live Comfy frontend or any GPU/image output.
Requires developer-installed playwright and Chromium; never runs on node import.
"""
from pathlib import Path
import importlib.util
import json
import os
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
pkg = importlib.util.spec_from_file_location('fnb_browser', ROOT/'__init__.py', submodule_search_locations=[str(ROOT)])
module = importlib.util.module_from_spec(pkg); sys.modules[pkg.name] = module; pkg.loader.exec_module(module)
from fnb_browser.bridge.spec import schema, default_document, canonical, BridgeError
from fnb_browser.bridge.server import review_document
from fnb_browser.bridge.workflow import graph_plan
from fnb_browser.bridge.infotext import import_infotext

STUB = r'''
class Graph {
 constructor(){this.nodes=[];}
 add(n){n.id=this.nodes.length+1;this.nodes.push(n);}
 arrange(){}
 serialize(){return {nodes:this.nodes.map(n=>({type:n.type,widgets_values:n.widgets.map(w=>w.value)})),extra:{}};}
}
const defs={
 UNETLoader:[['unet_name','weight_dtype'],[]],CLIPLoader:[['clip_name','type','device'],[]],VAELoader:[['vae_name'],[]],
 ForgeCompatSpec:[['source_mode','profile','policy','payload','overrides_json','image_file'],[]],
 ForgeCompatText:[[],['spec','clip']],ForgeCompatNoise:[[],['spec','model','input_latent']],
 ForgeCompatSampler:[['trace_level'],['spec','model','conditioning','noise','latent_image']],
 VAEDecode:[[],['samples','vae']],SaveImage:[['filename_prefix'],['images']],
 CheckpointLoaderSimple:[['ckpt_name'],[]],LoraLoader:[['lora_name','strength_model','strength_clip'],['model','clip']]
};
globalThis.LiteGraph={createNode(type){const d=defs[type];if(!d)return null;return {type,widgets:d[0].map(name=>({name,value:'',options:{}})),inputs:d[1].map(name=>({name})),outputs:[{},{},{}],connect(){return {};}};}};
export const app={graph:new Graph(),nativeCalls:0,loads:[],extension:null,
 handleFile(){this.nativeCalls++;return 'native';},
 registerExtension(e){this.extension=e;e.setup();},
 async loadGraphData(d){this.loads.push(structuredClone(d));return true;},
 ui:{settings:{getSettingValue(){return false;}}}
};
globalThis.host=app;
'''

def dispatch(path, body=None):
    try:
        if path=='/forge_neo_bridge/schema':return {'schema':schema()}
        if path=='/forge_neo_bridge/catalog':return CATALOG
        if path.startswith('/forge_neo_bridge/defaults/'):return default_document(path.rsplit('/',1)[1])
        data=json.loads(body)
        if path.endswith('/infotext'):return import_infotext(data['text'],data['family']).document()
        spec=review_document(data['spec'],data.get('acknowledge',False))
        result={'spec':spec.document()}
        if path.endswith('/plan'):result['plan']=graph_plan(spec)
        return result
    except BridgeError as e:return {'error':{'message':str(e)}}


def main():
    global CATALOG
    from playwright.sync_api import sync_playwright
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        CATALOG={'checkpoints':[],'diffusion_models':['model.safetensors'],'text_encoders':['qwen.safetensors'],'vae':['vae.safetensors'],'loras':[]}
        for cat,files in CATALOG.items():
            (root/cat).mkdir()
            for file in files:(root/cat/file).write_text('browser harness only; not model weights')
        sys.modules['folder_paths']=SimpleNamespace(get_folder_paths=lambda cat:[str(root/cat)],get_full_path=lambda cat,name:str(root/cat/name))
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,executable_path=os.environ.get('CHROMIUM','/usr/bin/chromium'),args=['--no-sandbox'])
            page=browser.new_page(viewport={'width':1280,'height':960});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            # Pure DOM harness: no network or local HTTP server is required.
            page.expose_function('bridgeRequest',dispatch)
            page.set_content('<!doctype html><title>Bridge browser harness</title>')
            page.evaluate("""async ([stub,core,bridge])=>{
                const js=text=>URL.createObjectURL(new Blob([text],{type:'text/javascript'}));
                const appUrl=js(stub);
                const apiUrl=js(`export const api={fetchApi:async(path,opt={})=>{const data=await bridgeRequest(path,opt.body);return new Response(JSON.stringify(data),{status:data.error?400:200,headers:{'Content-Type':'application/json'}})}};`);
                const coreUrl=js(core);
                bridge=bridge.replace('../../scripts/app.js',appUrl).replace('../../scripts/api.js',apiUrl).replace('./bridge_core.js',coreUrl);
                await import(js(bridge));
            }""",[STUB,(ROOT/'web/bridge_core.js').read_text(),(ROOT/'web/bridge.js').read_text()])
            page.wait_for_function('host?.extension != null')
            assert page.evaluate('host.handleFile === host.handleFile')
            page.evaluate("host.extension.commands[0].function()")
            page.locator('dialog').wait_for()
            asset_selects=page.locator('fieldset select')
            asset_selects.nth(0).select_option(label='diffusion_models/model.safetensors')
            asset_selects.nth(1).select_option(label='text_encoders/qwen.safetensors')
            asset_selects.nth(2).select_option(label='vae/vae.safetensors')
            seed=page.locator('label').filter(has=page.locator('span',has_text='seed')).filter(has=page.locator('input[type=text]')).first.locator('input')
            seed.fill('18446744073709551615');seed.dispatch_event('change')
            page.locator('label').filter(has_text='推定値・省略された設定').locator('input').check()
            page.get_by_role('button',name='設定を検証',exact=True).click()
            page.wait_for_function("document.querySelector('dialog pre').textContent.includes('ready')")
            assert page.evaluate('host.loads.length')==0
            page.get_by_role('button',name='新しいワークフローを作成',exact=True).click()
            page.wait_for_function('host.loads.length === 1')
            saved=page.evaluate("host.loads[0].nodes.find(n=>n.type==='ForgeCompatSpec').widgets_values[3]")
            assert json.loads(saved)['effective']['noise']['seed']=='18446744073709551615'
            assert len(page.evaluate('host.loads[0].nodes'))==9
            page.evaluate("host.extension.commands[0].function()")
            page.locator('dialog').wait_for();page.get_by_role('button',name='キャンセル',exact=True).click()
            assert page.evaluate('host.loads.length')==1
            assert page.evaluate("host.handleFile(new File(['{}'],'test.json'))")=='native'
            page.evaluate('host.extension.settings[0].onChange(true)')
            page.evaluate("host.handleFile(new File(['{}'],'test.json'))")
            page.evaluate('host.extension.settings[0].onChange(false)')
            page.evaluate("host.handleFile(new File(['x'],'unchecked.png'))")
            assert page.evaluate('host.nativeCalls')==3
            assert errors==[],errors
            target=os.environ.get('BRIDGE_SCREENSHOT')
            if target:
                page.evaluate("host.extension.commands[0].function()")
                page.locator('dialog').wait_for();page.screenshot(path=target)
            browser.close()
        print(json.dumps({'status':'PASS','scope':'Chromium + real Bridge editor/backend helpers; mocked Comfy host','checks':['editor_open','explicit_assets','uint64_seed_roundtrip','validate_without_load','detached_9_node_plan','cancel_keeps_graph','drop_toggle_and_delegate','no_browser_errors']},indent=2))

if __name__=='__main__':main()
