"""Chromium image-to-workflow smoke test, using real Bridge modules and a mocked Comfy host.
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
from fnb_browser.bridge.reconstruction import reconstruct, reconstruction_plan

STUB = r'''
class Graph {
 constructor(){this.nodes=[];}
 add(n){n.id=this.nodes.length+1;this.nodes.push(n);}
 arrange(){}
 serialize(){return {nodes:this.nodes.map(n=>({type:n.type,widgets_values:n.widgets.map(w=>w.value)})),extra:{}};}
}
const defs={
 UNETLoader:[['unet_name','weight_dtype'],[]],CLIPLoader:[['clip_name','type','device'],[]],VAELoader:[['vae_name'],[]],
 ForgeNeoBridgeTextEncode:[['text'],['clip']],
 ForgeNeoBridgeSettings:[['family','mode','width','height','batch_size','rng','ensd','subseed','subseed_strength','emphasis','clip_skip','comma_padding_backtrack','seed_resize_width','seed_resize_height','shift','eta_ancestral','eta_ddim','s_churn','s_tmin','s_tmax','s_noise','sigma_min','sigma_max','rho','beta_alpha','beta_beta','discard_penultimate','sgm_noise_multiplier','skip_early_cfg','ngms','ngms_all_steps','img2img_step_mode','img2img_extra_noise','original_width','original_height','target_width','target_height','crop_x','crop_y','zero_empty_negative','source_json'],[]],
 ForgeNeoBridgeKSampler:[['seed','steps','cfg','sampler_name','scheduler','denoise'],['model','positive','negative','settings','input_latent']],
 VAEDecode:[[],['samples','vae']],SaveImage:[['filename_prefix'],['images']],
 CheckpointLoaderSimple:[['ckpt_name'],[]],LoraLoader:[['lora_name','strength_model','strength_clip'],['model','clip']],
 LoadImage:[['image'],[]],VAEEncode:[[],['pixels','vae']]
};
globalThis.LiteGraph={createNode(type){const d=defs[type];if(!d)return null;return {type,widgets:d[0].map(name=>({name,value:'',options:{}})),inputs:d[1].map(name=>({name})),outputs:[{},{},{}],connect(){return {};}};}};
export const app={graph:new Graph(),nativeCalls:0,loads:[],extension:null,
 handleFile(){this.nativeCalls++;return 'native';},
 registerExtension(e){this.extension=e;e.setup();},
 async loadGraphData(d){this.loads.push(structuredClone(d));return true;},
 ui:{settings:{getSettingValue(){return true;}}}
};
globalThis.host=app;
'''

def dispatch(path, body=None):
    if path != '/forge_neo_bridge/reconstruct':return {'error':{'message':'Unknown endpoint'}}
    text=('a teapot\nNegative prompt: blurry\nSteps: 10, Sampler: Euler, Schedule type: Beta, '
          'CFG scale: 1.5, Seed: 9007199254740993, Size: 1024x1344, '
          'Model: missing.safetensors, Module 1: qwen_3_06b.safetensors, Module 2: anima_vae.safetensors')
    doc,positive=reconstruct(text,CATALOG,probe=lambda *_:None)
    return {'kind':'forge','document':doc,'plan':reconstruction_plan(doc,positive,CATALOG)}


def main():
    global CATALOG
    from playwright.sync_api import sync_playwright
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        CATALOG={'checkpoints':[],'diffusion_models':[],'text_encoders':['qwen_3_06b.safetensors'],
                 'vae':['anima_vae.safetensors'],'loras':[]}
        for cat,files in CATALOG.items():
            (root/cat).mkdir()
            for file in files:(root/cat/file).write_text('browser harness only; not model weights')
        sys.modules['folder_paths']=SimpleNamespace(get_folder_paths=lambda cat:[str(root/cat)],get_full_path=lambda cat,name:str(root/cat/name))
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,executable_path=os.environ.get('CHROMIUM') or None,args=['--no-sandbox'])
            page=browser.new_page(viewport={'width':1280,'height':960});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            # Pure DOM harness: no network or local HTTP server is required.
            page.expose_function('bridgeRequest',dispatch)
            page.set_content('<!doctype html><title>Bridge browser harness</title>')
            page.evaluate("""async ([stub,core,bridge,accordion,seeds])=>{
                const js=text=>URL.createObjectURL(new Blob([text],{type:'text/javascript'}));
                const appUrl=js(stub);
                const apiUrl=js(`export const api={fetchApi:async(path,opt={})=>{const data=await bridgeRequest(path,opt.body);return new Response(JSON.stringify(data),{status:data.error?400:200,headers:{'Content-Type':'application/json'}})}};`);
                const coreUrl=js(core);
                const accordionUrl=js(accordion);
                bridge=bridge.replace('../../scripts/app.js',appUrl).replace('../../scripts/api.js',apiUrl).replace('./bridge_core.js',coreUrl).replace('./settings_accordion.js',accordionUrl).replace('./seed_controls.js',js(seeds));
                await import(js(bridge));
            }""",[STUB,(ROOT/'web/bridge_core.js').read_text(),(ROOT/'web/bridge.js').read_text(),(ROOT/'web/settings_accordion.js').read_text(),(ROOT/'web/seed_controls.js').read_text()])
            page.wait_for_function('host?.extension != null')
            assert page.evaluate('host.handleFile === host.handleFile')
            page.evaluate("host.extension.settings[0].onChange(true)")
            page.evaluate("async () => await host.handleFile(new File(['x'],'forge.png'))")
            page.wait_for_function('host.loads.length === 1')
            assert page.locator('dialog').count()==0
            seed=page.evaluate("host.loads[0].nodes.find(n=>n.type==='ForgeNeoBridgeKSampler').widgets_values[0]")
            assert seed=='9007199254740993'
            assert page.evaluate("host.loads[0].nodes.some(n=>n.type==='ForgeNeoBridgeTextEncode')")
            assert page.evaluate("host.loads[0].nodes.some(n=>n.type==='UNETLoader')")
            assert page.evaluate("host.handleFile(new File(['{}'],'test.json'))")=='native'
            page.evaluate('host.extension.settings[0].onChange(false)')
            page.evaluate("host.handleFile(new File(['x'],'unchecked.png'))")
            assert page.evaluate('host.nativeCalls')==2
            assert errors==[],errors
            target=os.environ.get('BRIDGE_SCREENSHOT')
            if target:
                page.screenshot(path=target)
            browser.close()
        print(json.dumps({'status':'PASS','scope':'Chromium + real Bridge drop/backend helpers; mocked Comfy host','checks':['direct_drop','no_dialog','uint64_seed_roundtrip','missing_model_graph','native_delegate','drop_toggle','no_browser_errors']},indent=2))

if __name__=='__main__':main()
