import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {buildDetached, captureCanvas, makeDropHandler, placeReconstructed, preserveMissingCombo, scalarValue} from '../web/bridge_core.js';

class Graph {
    constructor(){this.nodes=[];this._nodes=this.nodes;}
    add(node){node.id=this.nodes.length+1;node.pos=[0,0];node.size=[300,160];this.nodes.push(node);}
    serialize(){return {nodes:this.nodes.map(n=>({type:n.type,widgets_values:n.widgets.map(w=>w.value)})),extra:{}};}
}
const lite={createNode(type){return {type,inputs:[{name:'spec'}],outputs:[{}],widgets:[{name:'payload',value:'',options:{}}],connect(){return {};}};}};
const plan=()=>({schema_version:'1.0.0',config_hash:'a'.repeat(64),nodes:[{id:'1',type:'ForgeCompatSpec',title:'Spec',values:{payload:'18446744073709551615'}},{id:'2',type:'ForgeCompatText',values:{}}],edges:[['1',0,'2','spec']]});
test('build detached and retain exact seed text',()=>{
    const data=buildDetached(plan(),lite,Graph);
    assert.equal(data.nodes[0].widgets_values[0],'18446744073709551615');
    assert.equal(data.extra.forge_neo_bridge.qualification,'not_evaluated');
});
test('core nodes retain their ComfyUI title',()=>{
    let core;
    const creator={createNode(type){core={type,title:'Empty Latent Image',widgets:[],outputs:[{}],inputs:[]};return core;}};
    const p={schema_version:'1.0.0',nodes:[{id:'1',type:'EmptyLatentImage',values:{}}],edges:[]};
    buildDetached(p,creator,Graph);
    assert.equal(core.title,'Empty Latent Image');
});
test('native Comfy recipe keeps its qualification and core titles',()=>{
    const creator={createNode(type){return {type,title:type,widgets:[],outputs:[{}],inputs:[],connect(){return {};}};}};
    const p={schema_version:'1.0.0',qualification:'native_comfy_not_forge_parity',
        nodes:[{id:'1',type:'EmptyFlux2LatentImage',values:{}},{id:'2',type:'KSampler',values:{}},{id:'3',type:'ModelSamplingSD3',values:{}}],edges:[]};
    const data=buildDetached(p,creator,Graph);
    assert.deepEqual(data.nodes.map(node=>node.type),['EmptyFlux2LatentImage','KSampler','ModelSamplingSD3']);
    assert.equal(data.extra.forge_neo_bridge.qualification,'native_comfy_not_forge_parity');
});
test('imported native seeds remain fixed when queued',()=>{
    const creator={createNode(type){const name=type==='RandomNoise'?'noise_seed':'seed';return {
        type,widgets:[{name,value:0},{name:'control_after_generate',value:'randomize'}],
        outputs:[{}],inputs:[],connect(){return {};}};}};
    for(const [type,name,seed] of [['RandomNoise','noise_seed',1907370631],
                                   ['KSampler','seed',3129797896]]) {
        const p={schema_version:'1.0.0',nodes:[{id:'1',type,values:{[name]:seed}}],edges:[]};
        assert.deepEqual(buildDetached(p,creator,Graph).nodes[0].widgets_values,[seed,'fixed']);
    }
});
test('Hires core upscale nodes are accepted without retitling',()=>{
    const creator={createNode(type){return {type,title:type,widgets:[],outputs:[{}],inputs:[],connect(){return {};}};}};
    const p={schema_version:'1.0.0',nodes:['ImageScale','LatentUpscale'].map((type,i)=>({id:String(i),type,values:{}})),edges:[]};
    assert.deepEqual(buildDetached(p,creator,Graph).nodes.map(n=>n.type),['ImageScale','LatentUpscale']);
});
for(const kind of ['unknown','missing-widget','missing-input','duplicate-id','wrong-slot','missing-node']) {
    test(`transaction fails before graph load: ${kind}`,()=>{
        const p=plan();let creator=lite;
        if(kind==='unknown')p.nodes[0].type='RunArbitraryCode';
        if(kind==='missing-widget')p.nodes[0].values={unknown:2};
        if(kind==='missing-input')p.edges[0][3]='absent';
        if(kind==='duplicate-id')p.nodes[1].id='1';
        if(kind==='wrong-slot')p.edges[0][1]=9;
        if(kind==='missing-node')creator={createNode:()=>null};
        assert.throws(()=>buildDetached(p,creator,Graph));
    });
}
test('seed zero and uint64 stay strings',()=>{
    for(const value of ['0','9007199254740993','18446744073709551615'])assert.equal(scalarValue({value},''),value);
});
test('zero and false are not missing',()=>{
    assert.equal(scalarValue({value:'0'},1),0);
    assert.equal(scalarValue({checked:false},true),false);
    assert.throws(()=>scalarValue({value:'NaN'},1));
    assert.throws(()=>scalarValue({value:''},1));
});
for(const kind of ['native','other'])test(`drop delegates ${kind} exactly once`,async()=>{
    const owner={};let calls=0,opens=0;
    const fn=makeDropHandler(function(f,arg){calls++;assert.equal(this,owner);assert.equal(arg,7);return 23;},async()=>({kind}),()=>opens++,owner);
    assert.equal(await fn(new File(['x'],'test.webp'),7),23);assert.equal(calls,1);assert.equal(opens,0);
});
test('Forge drop passes the plan directly and never calls native load',async()=>{
    let calls=0,placed=0;const result={kind:'forge',plan:{schema_version:'1.0.0'}};
    const context={revision:4};
    const fn=makeDropHandler(()=>calls++,async()=>result,(r,f,c)=>{assert.equal(r,result);assert.equal(c,context);placed++;},{},()=>context);
    await fn(new File(['x'],'test.png'));assert.equal(placed,1);assert.equal(calls,0);
});
test('only designated missing model combo is preserved on this widget',()=>{
    const model={type:'UNETLoader',widgets:[{name:'unet_name',options:{values:['known.safetensors']}}]};
    const base=model.widgets[0].options.values;
    assert.equal(preserveMissingCombo(model,'unet_name','missing.safetensors'),true);
    assert.deepEqual(base,['known.safetensors']);
    assert.deepEqual(model.widgets[0].options.values,['known.safetensors','missing.safetensors']);
    const dtype={type:'UNETLoader',widgets:[{name:'weight_dtype',options:{values:['default']}}]};
    assert.equal(preserveMissingCombo(dtype,'weight_dtype','invalid'),false);
});
test('non-image files bypass inspection',async()=>{
    let inspected=0;const fn=makeDropHandler(()=>42,()=>inspected++,()=>{},{});
    assert.equal(await fn(new File(['x'],'workflow.json')),42);assert.equal(inspected,0);
});
test('failed inspection does not clear or load anything',async()=>{
    let calls=0;const fn=makeDropHandler(()=>calls++,async()=>{throw Error('Malformed EXIF');},()=>calls++,{});
    await assert.rejects(fn(new File(['x'],'bad.webp')),/Malformed EXIF/);assert.equal(calls,0);
});
test('direct placement loads once and leaves the original untouched until ready',async()=>{
    const graph=new Graph();const loaded=[];
    const app={graph,async loadGraphData(value){loaded.push(value);return true;}};
    const context=captureCanvas(app,1);
    await placeReconstructed({plan:{...plan(),name:'Forge'}},app,lite,context,()=>1);
    assert.equal(loaded.length,1);
    assert.equal(graph.nodes.length,0);
    assert.equal(loaded[0].nodes.length,2);
});
test('an edit or a newer drop cancels an older result without loading',async()=>{
    for(const changed of ['edit','newer']) {
        const graph=new Graph();let loads=0;
        const app={graph,async loadGraphData(){loads++;return true;}};
        const context=captureCanvas(app,1);
        if(changed==='edit')graph.nodes.push({type:'other',widgets:[]});
        await assert.rejects(placeReconstructed({plan:{...plan(),name:'Forge'}},app,lite,context,()=>changed==='newer'?2:1),/Workflow changed/);
        assert.equal(loads,0);
    }
});
test('a rejected graph load restores the prior graph snapshot',async()=>{
    const graph=new Graph();const loaded=[];
    const app={graph,async loadGraphData(value){loaded.push(value);return loaded.length>1;}};
    const context=captureCanvas(app,1);
    await assert.rejects(placeReconstructed({plan:{...plan(),name:'Forge'}},app,lite,context,()=>1),/rejected/);
    assert.equal(loaded.length,2);
    assert.deepEqual(loaded[1],context.before);
});
test('registered frontend handles a Forge image without opening a dialog',async()=>{
    const loaded=[],alerts=[];
    const app={graph:new Graph(),ui:{settings:{getSettingValue:()=>undefined}},
        handleFile(){throw Error('Unexpected native file load');},
        registerExtension(extension){this.extension=extension;extension.setup();},
        async loadGraphData(value){loaded.push(value);return true;}};
    globalThis.__forgeApp=app;
    globalThis.__forgeApi={fetchApi:async()=>({ok:true,json:async()=>({kind:'forge',plan:{
        schema_version:'1.0.0',name:'Forge',nodes:[{id:'1',type:'ForgeNeoBridgeSettings',values:{}}],edges:[]}})})};
    globalThis.LiteGraph=lite;
    globalThis.window={alert:value=>alerts.push(value)};
    const source=await readFile(new URL('../web/bridge.js',import.meta.url),'utf8');
    const appUrl='data:text/javascript,export const app=globalThis.__forgeApp';
    const apiUrl='data:text/javascript,export const api=globalThis.__forgeApi';
    const coreUrl=new URL('../web/bridge_core.js',import.meta.url).href;
    const script=source.replace('../../scripts/app.js',appUrl).replace('../../scripts/api.js',apiUrl)
        .replace('./bridge_core.js',coreUrl)
        .replace('./layout.js',new URL('../web/layout.js',import.meta.url).href)
        .replace('./settings_accordion.js',new URL('../web/settings_accordion.js',import.meta.url).href)
        .replace('./i18n.js',new URL('../web/i18n.js',import.meta.url).href)
        .replace('./help.js',new URL('../web/help.js',import.meta.url).href)
        .replace('./seed_controls.js',new URL('../web/seed_controls.js',import.meta.url).href);
    await import('data:text/javascript,'+encodeURIComponent(script));
    assert.equal(app.extension.settings[0].id,'ForgeNeo.Bridge.enable_canvas_drop');
    assert.equal(app.extension.settings[0].defaultValue,true);
    await app.handleFile(new File(['x'],'forge.png'));
    assert.equal(loaded.length,1);
    assert.equal(loaded[0].nodes[0].type,'ForgeNeoBridgeSettings');
    assert.deepEqual(alerts,[]);
    delete globalThis.__forgeApp;delete globalThis.__forgeApi;delete globalThis.window;
});
