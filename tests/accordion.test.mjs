import test from 'node:test';
import assert from 'node:assert/strict';
import {installSettingsAccordion, watchSettingsLatent} from '../web/settings_accordion.js';

const defaults = {family:'anima', mode:'txt2img', width:1024, height:1344, batch_size:1,
    emphasis:'Original', clip_skip:'2', comma_padding_backtrack:20, rng:'CPU', ensd:'31337',
    subseed:'9007199254740993', subseed_strength:0.25, seed_resize_width:0, seed_resize_height:0,
    eta_ancestral:1, eta_ddim:0, s_churn:0, s_tmin:0, s_tmax:0, s_noise:1,
    sigma_min:'auto', sigma_max:'auto', rho:'auto', beta_alpha:0.6, beta_beta:0.6,
    discard_penultimate:false, sgm_noise_multiplier:false, shift:'auto', skip_early_cfg:0,
    ngms:0, ngms_all_steps:false, img2img_step_mode:'forge_scaled', img2img_extra_noise:0,
    original_width:0, original_height:0, target_width:0, target_height:0, crop_x:0, crop_y:0,
    zero_empty_negative:false, source_json:'{"source":{"raw_infotext":"original metadata"}}',forge_compatibility:true,sampling_adjustments:false};

function createNode(properties = {}) {
    const node = {type:'ForgeNeoBridgeSettings', properties, size:[320,1000],
        widgets:Object.entries(defaults).map(([name,value]) => ({name,value, options:{},
            ...(name==='source_json' ? {inputEl:{style:{display:'block'},readOnly:false},computeSize:()=>[320,160]} : {})})),
        graph:{setDirtyCanvas(){},beforeChange(){},afterChange(){}},
        computeSize(){return [320,40+this.widgets.filter(w=>!w.hidden).length*24];},
        setSize(size){this.size=size;},
        addWidget(type,name,value,callback,options){const w={type,name,value,callback,options};this.widgets.push(w);return w;},
        serialize(){return structuredClone({properties:this.properties,widgets_values:this.widgets.map(w=>w.value),
            widgets_values_named:Object.fromEntries(this.widgets.map(w=>[w.name,w.value]))});},
        configure(data){this.properties=structuredClone(data.properties || {});this.widgets.forEach((w,i)=>{if(i<data.widgets_values.length)w.value=data.widgets_values[i];});},
    };
    node.app = {ui:{settings:{getSettingValue:()=>node.locale || 'ja'}}};
    installSettingsAccordion(node, node.app);
    return node;
}
const widget = (node,name) => node.widgets.find(w=>w.name===name);
const header = (node,title) => node.widgets.find(w=>w.type==='button' && w.name.includes(title));
function change(node,name,value){const w=widget(node,name);w.value=value;w.callback?.(value);}

test('advanced parent hides all detail headers, preserves child state and localizes help',()=>{
    const node=createNode();
    assert.equal(widget(node,'family').hidden,true);
    assert.equal(header(node,'テキスト処理').hidden,true);
    assert.equal(header(node,'読み込み情報').hidden,false);
    header(node,'詳細設定').callback();header(node,'テキスト処理').callback();
    assert.equal(widget(node,'emphasis').hidden,false);
    header(node,'詳細設定').callback();
    assert.equal(widget(node,'emphasis').hidden,true);
    const saved=node.serialize(),restored=createNode();restored.configure(saved);
    assert.equal(widget(restored,'emphasis').hidden,true);
    header(restored,'詳細設定').callback();assert.equal(widget(restored,'emphasis').hidden,false);
    restored.locale='en';restored.onDrawForeground();
    assert.match(header(restored,'Advanced settings').tooltip,/Collapsing keeps all values/);
    const legacy=createNode({forge_neo_bridge:{sections:{noise:true}}});
    assert.equal(widget(legacy,'rng').hidden,false);
});

test('initial closed sections keep Clip Skip and ENSD editable and preserve all execution values',()=>{
    const node=createNode();
    assert.equal(widget(node,'clip_skip').hidden,undefined);
    assert.equal(widget(node,'ensd').hidden,undefined);
    for(const name of ['rng','emphasis','eta_ancestral','sigma_min','source_json'])assert.equal(widget(node,name).hidden,true);
    assert.equal(header(node,'SDXL詳細').hidden,true);
    assert.equal(header(node,'img2img詳細').hidden,true);
    assert.deepEqual(node.serialize().widgets_values,Object.values(defaults));
    assert.equal(widget(node,'source_json').inputEl.style.display,'none');
    change(node,'ensd','0');
    assert.equal(node.serialize().widgets_values[9],'0');
});

test('Comfy constructor lifecycle identifies the node before LiteGraph sets type',()=>{
    const node={comfyClass:'ForgeNeoBridgeSettings',widgets:Object.entries(defaults).map(([name,value])=>({name,value})),properties:{},
        addWidget(type,name,value,callback,options){const w={type,name,value,callback,options};this.widgets.push(w);return w;}};
    installSettingsAccordion(node,{});
    assert.equal(typeof node.forgeRefreshSettings,'function');
    assert.equal(widget(node,'rng').hidden,true);
});

test('each accordion expands independently, shrinks, and saves only original widgets',()=>{
    const node=createNode();header(node,'詳細設定').callback();const size=node.size[1];
    header(node,'乱数・バリエーション').callback();
    assert.equal(widget(node,'rng').hidden,false);
    assert.equal(widget(node,'s_churn').hidden,true);
    assert(node.size[1]>size);
    change(node,'subseed_strength',0.5);
    const saved=node.serialize();
    assert.equal(saved.widgets_values.length,Object.keys(defaults).length);
    assert.equal(saved.properties.forge_neo_bridge.sections.noise,true);
    header(node,'乱数・バリエーション').callback();
    assert.equal(node.size[1],size);
    assert.equal(widget(node,'subseed_strength').value,0.5);
    assert.equal(widget(node,'subseed').value,'9007199254740993');
});

test('old positional workflows and new saved open states round-trip without reassigning values',()=>{
    const old={properties:{},widgets_values:Object.values({...defaults, family:'sdxl',mode:'img2img',ensd:'123'})};
    const node=createNode();node.configure(old);
    assert.equal(widget(node,'ensd').value,'123');
    header(node,'詳細設定').callback();
    assert.equal(header(node,'SDXL詳細').hidden,false);
    header(node,'SDXL詳細').callback();
    header(node,'img2img詳細').callback();
    change(node,'crop_x',64);
    const saved=node.serialize();const restored=createNode();restored.configure(saved);
    assert.equal(widget(restored,'crop_x').hidden,false);
    assert.equal(widget(restored,'img2img_extra_noise').hidden,false);
    assert.equal(widget(restored,'crop_x').value,64);
    assert.deepEqual(restored.serialize(),saved);
    change(restored,'family','anima');
    assert.equal(widget(restored,'crop_x').hidden,true);
    change(restored,'family','sdxl');
    assert.equal(widget(restored,'crop_x').hidden,false);
    assert.equal(widget(restored,'crop_x').value,64);
});

test('source information is readable on demand and warnings remain visible while closed',()=>{
    const node=createNode({forge_neo_bridge:{unresolved:[{code:'ASSET_MISSING',message:'Model file missing'}]}});
    const alert=header(node,'読み込み警告');assert.equal(alert.hidden,false);
    alert.callback();
    assert.equal(widget(node,'source_json').hidden,false);
    assert.equal(widget(node,'source_json').inputEl.style.display,'block');
    assert.equal(widget(node,'source_json').inputEl.readOnly,true);
    alert.callback();
    assert.equal(widget(node,'source_json').hidden,true);
    assert.equal(alert.hidden,false);
    assert.equal(widget(node,'source_json').value,defaults.source_json);
});

test('old generated warning titles are removed and their forced width is reset',()=>{
    const node=createNode();node.size=[690,330];
    node.configure({title:'ForgeNeo Bridge Settings [unsupported: Registered inventory item has no V1 executable adapter.]',
        properties:{forge_neo_bridge:{compact_width:true}},widgets_values:Object.values(defaults)});
    assert.equal(node.title,'ForgeNeo Bridge Settings');
    assert.equal(node.size[0],300);
});

test('connected standalone latent hides duplicate sizes; legacy disconnected graphs retain them',async()=>{
    const node=createNode();node.id=1;node.outputs=[{links:[10]}];
    const sampler={id:2,type:'ForgeNeoBridgeKSampler',inputs:[{name:'settings',link:10},{name:'input_latent',link:11}]};
    const graph={...node.graph,links:{10:{origin_id:1,target_id:2}},getNodeById(id){return id===1?node:sampler;}};
    node.graph=graph;sampler.graph=graph;watchSettingsLatent(sampler);
    sampler.onConnectionsChange();await Promise.resolve();
    assert.equal(widget(node,'width').hidden,true);
    sampler.inputs[1].link=null;sampler.onConnectionsChange();await Promise.resolve();
    assert.equal(widget(node,'width').hidden,false);
    assert.equal(widget(node,'width').value,1024);
});

test('mode remains ON for old graphs, OFF round-trips without losing inactive values; locale tracks Comfy',()=>{
    const node=createNode();
    node.configure({properties:{},widgets_values:Object.values(defaults).slice(0,-1)});
    assert.equal(widget(node,'forge_compatibility').value,true);
    assert.equal(widget(node,'sampling_adjustments').value,true);
    assert.equal(node.size[0],300);
    change(node,'forge_compatibility',false);
    assert.match(widget(node,'ensd').label,/適用なし/);
    const saved=node.serialize();
    const restored=createNode();restored.configure(saved);
    assert.equal(widget(restored,'forge_compatibility').value,false);
    assert.equal(widget(restored,'ensd').value,'31337');
    restored.locale='en';restored.onDrawForeground();
    assert.equal(widget(restored,'forge_compatibility').label,'Forge compatibility');
    assert.match(header(restored,'Noise / variation').name,/inactive/);
    change(restored,'forge_compatibility',true);
    assert.equal(widget(restored,'ensd').label,'ENSD');
});

test('Hires mode propagates through latent and image nodes, leaving unrelated settings alone',()=>{
    const base=createNode(),hires=createNode(),unrelated=createNode();base.id=1;hires.id=7;unrelated.id=8;
    const nodes=[base,{id:2,type:'ForgeNeoBridgeKSampler'},{id:3,type:'VAEDecode'},
        {id:4,type:'ImageScale'},{id:5,type:'VAEEncode'},{id:6,type:'ForgeNeoBridgeKSampler'},hires,unrelated];
    const pairs=[[1,2,'FORGE_SETTINGS'],[2,3,'LATENT'],[3,4,'IMAGE'],[4,5,'IMAGE'],[5,6,'LATENT'],[7,6,'FORGE_SETTINGS']];
    const graph={getNodeById:id=>nodes.find(n=>n.id===id),links:new Map(pairs.map(([origin_id,target_id,type],i)=>[i,{origin_id,target_id,type}]))};
    for(const n of nodes)n.graph=graph;
    change(base,'forge_compatibility',false);
    assert.equal(widget(hires,'forge_compatibility').value,false);
    assert.equal(widget(unrelated,'forge_compatibility').value,true);
    change(hires,'forge_compatibility',true);
    assert.equal(widget(base,'forge_compatibility').value,true);
    change(base,'sampling_adjustments',true);
    assert.equal(widget(hires,'sampling_adjustments').value,true);
    assert.equal(widget(unrelated,'sampling_adjustments').value,false);
});

test('new adjustment default stays OFF after save, old files migrate ON, help follows locale',()=>{
    const node=createNode();
    assert.equal(widget(node,'sampling_adjustments').value,false);
    const saved=node.serialize(),restored=createNode();restored.configure(saved);
    assert.equal(widget(restored,'sampling_adjustments').value,false);
    assert.equal(restored.size[0],300);
    assert.match(header(node,'サンプリング詳細').tooltip,/開閉/);
    node.locale='en';node.onDrawForeground();
    assert.match(header(node,'Sampling details').tooltip,/Expand or collapse/);
    node.configure({widgets_values:Object.values(defaults).slice(0,-2)});
    assert.equal(widget(node,'forge_compatibility').value,true);
    assert.equal(widget(node,'sampling_adjustments').value,true);
});
