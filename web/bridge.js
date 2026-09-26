import { app } from '../../scripts/app.js';
import {layoutMenu} from './layout.js';
import { api } from '../../scripts/api.js';
import { buildDetached, captureCanvas, makeDropHandler, placeReconstructed, preserveMissingCombo, scalarValue } from './bridge_core.js';

import { installSettingsAccordion, watchSettingsLatent } from './settings_accordion.js';
import { loadHelp, installHelp } from './help.js';
import { installSeedControls } from './seed_controls.js';
import { tr } from './i18n.js';

const BASE = '/forge_neo_bridge';
const SETTING = 'ForgeNeo.Bridge.enable_canvas_drop';
let activePanel = null, installedHandler = null, previousHandler = null;
let dropSerial = 0;

async function request(path, body, raw = false) {
    const response = await api.fetchApi(BASE + path, body === undefined ? {} : {
        method: 'POST', headers: raw ? {} : {'Content-Type':'application/json'}, body: raw ? body : JSON.stringify(body)
    });
    const value = await response.json();
    if (!response.ok || value.error) throw new Error(value.error?.message || response.statusText);
    return value;
}
function element(tag, text, parent) {
    const node = document.createElement(tag);
    if (text != null) node.textContent = text;
    parent?.appendChild(node); return node;
}
function button(parent, text, handler) {
    const node=element('button',text,parent);node.type='button';node.addEventListener('click',handler);return node;
}
function outputError(error) { console.error('[ForgeNeo Bridge]',error); window.alert(String(error.message || error)); }

async function editor(initial = null, sourceFile = null) {
    if (activePanel) { activePanel.showModal(); return; }
    const [catalog, definition] = await Promise.all([request('/catalog'),request('/schema')]);
    let doc = initial || await request('/defaults/anima');
    const dialog = element('dialog');activePanel=dialog;
    dialog.style.cssText='width:min(980px,94vw);max-height:92vh;overflow:auto;background:var(--comfy-menu-bg,#222);color:var(--input-text,#eee);border:1px solid #888;border-radius:8px;padding:18px';
    document.body.appendChild(dialog);
    dialog.addEventListener('close',()=>{activePanel=null;dialog.remove();});
    const title = element('h2','ComfyUI-ForgeNeo-Bridge',dialog);
    element('p','Forge Neo 710f1e25 • V1.0 • GPU parity: not evaluated',dialog);
    element('p',tr(app,'読み込んだ値・推定値・未対応項目を確認してから、新しいワークフローを作成します。生成は自動実行しません。','Review imported values, defaults and unsupported options before creating a workflow. Generation does not start automatically.'),dialog);
    const familyRow=element('div',null,dialog);
    element('label',tr(app,'モデル種別: ','Model family: '),familyRow);
    const family=element('select',null,familyRow);
    for(const key of ['anima','sd15','sdxl']) { const op=element('option',key,family);op.value=key; }
    family.value=doc.effective.family;
    family.addEventListener('change',async()=>{
        try {
            if(doc.source.raw_infotext) doc=await request('/infotext',{text:doc.source.raw_infotext,family:family.value});
            else doc=await request('/defaults/'+family.value);
            render();
        }catch(error){status.textContent=String(error);}
    });
    const toolbar=element('div',null,dialog);toolbar.style.margin='10px 0';
    const fileInput=element('input',null,toolbar);fileInput.type='file';fileInput.accept='.png,.webp,.jpg,.jpeg';
    fileInput.addEventListener('change',async()=>{
        try {
            const file=fileInput.files?.[0];if(!file)return;
            const result=await request('/inspect?family='+family.value,file,true);
            if(result.kind!=='forge')throw new Error(result.kind==='native'?tr(app,'Comfyのネイティブworkflowです。通常の画像読込みを使用してください。','This is a native Comfy workflow. Use the normal image importer.'):tr(app,'Forge infotextが見つかりません。','No Forge infotext found.'));
            doc=result.spec;sourceFile=file;render();
        }catch(error){status.textContent=String(error);}
    });
    button(toolbar,tr(app,'Infotextを貼り付け','Paste infotext'),async()=>{
        const value=window.prompt('Forge Neo / A1111 infotext');if(value===null)return;
        try{doc=await request('/infotext',{text:value,family:family.value});render();}catch(error){status.textContent=String(error);}
    });
    button(toolbar,tr(app,'Spec JSONを読込','Load Spec JSON'),()=>{
        const value=window.prompt('ForgeGenerationSpec JSON');if(value===null)return;
        try{const parsed=JSON.parse(value);if(parsed.schema_version!=='1.0.0')throw new Error('Unsupported spec');doc=parsed;family.value=doc.effective.family;render();}catch(error){status.textContent=String(error);}
    });
    const form=element('div',null,dialog),status=element('pre',null,dialog);
    status.style.cssText='white-space:pre-wrap;max-height:240px;overflow:auto;border:1px solid #666;padding:8px';
    const footer=element('div',null,dialog);
    const ackLabel=element('label',null,footer);const ack=element('input',null,ackLabel);ack.type='checkbox';
    element('span',tr(app,' 推定値・省略された設定を上記の値で使用することを確認しました',' Use the displayed defaults for inferred or omitted settings'),ackLabel);
    const controls=element('div',null,footer);controls.style.marginTop='12px';
    const run=async(action)=>{
        try {
            const value=await request(action,{spec:doc,acknowledge:ack.checked});doc=value.spec;status.textContent=JSON.stringify({status:doc.status,unresolved:doc.unresolved,unsupported:doc.unsupported,warnings:doc.warnings},null,2);
            if(action==='/plan') {
                const LiteGraph=globalThis.LiteGraph;
                const graphClass=(app.rootGraph || app.graph).constructor;
                if(!LiteGraph?.createNode || typeof app.loadGraphData!=='function')throw new Error('Unsupported frontend graph API. Export the Spec; the existing workflow was not touched.');
                const data=buildDetached(value.plan,LiteGraph,graphClass);
                const old=structuredClone((app.rootGraph || app.graph).serialize());
                try {
                    const result=await app.loadGraphData(data,true,true,value.plan.name+'.json');
                    if(result===false)throw new Error('Comfy rejected the workflow');
                    dialog.close();
                }catch(error){await app.loadGraphData(old,true,true,'ForgeNeo Bridge — restored previous workflow');throw error;}
            }
        }catch(error){status.textContent=String(error.message || error);}
    };
    button(controls,tr(app,'設定を検証','Validate settings'),()=>run('/validate'));
    button(controls,tr(app,'新しいワークフローを作成','Create workflow'),()=>run('/plan'));
    button(controls,tr(app,'Spec JSONを保存','Save Spec JSON'),()=>{
        const url=URL.createObjectURL(new Blob([JSON.stringify(doc,null,2)],{type:'application/json'}));
        const a=element('a');a.href=url;a.download='forge-generation-spec.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),0);
    });
    button(controls,tr(app,'キャンセル','Cancel'),()=>dialog.close());
    function asset(role,category,name,component=null,id=role) {
        return {asset_id:id,role,component,category,relative_name:name,sha256:null,short_hash:null,resolution:'resolved',binding_verification:'known_loader_chain'};
    }
    function assetUI(parent) {
        const section=element('fieldset',null,parent);element('legend',tr(app,'Model / TE / VAE — 正確なファイルを選択','Model / TE / VAE — select exact files'),section);
        element('p',tr(app,'Module番号や類似名だけでは自動選択しません。SD1.5/SDXLはCheckpoint内のCLIP/VAEを選択できます。','Assets are not selected by module number or similar names. SD1.5/SDXL can use CLIP/VAE from the checkpoint.'),section);
        for(const role of ['model','text_encoder','vae']) {
            const row=element('div',null,section);row.style.margin='6px 0';element('label',role+' ',row);
            const select=element('select',null,row);select.style.maxWidth='80%';element('option',tr(app,'選択してください','Select a file'),select).value='';
            const cats=role==='model'?['checkpoints','diffusion_models']:role==='text_encoder'?['checkpoints','text_encoders']:['checkpoints','vae'];
            for(const cat of cats)for(const name of catalog[cat]||[]) {const op=element('option',cat+'/'+name,select);op.value=JSON.stringify([cat,name]);}
            const current=doc.effective.assets.find(a=>a.role===role);if(current)select.value=JSON.stringify([current.category,current.relative_name]);
            select.addEventListener('change',()=>{doc.effective.assets=doc.effective.assets.filter(a=>a.role!==role);if(select.value){const [cat,name]=JSON.parse(select.value);doc.effective.assets.push(asset(role,cat,name));}});
        }
        if(doc.effective.family==='sdxl')element('p',tr(app,'CLIP L/G別ファイルはSpec JSONで2つのtext_encoder資産を指定できます。通常はCheckpointを選択します。','For separate CLIP L/G files, specify two text_encoder assets in Spec JSON. Usually select a checkpoint.'),section);
        const loras=element('div',null,section);
        const paintLoras=()=>{
            loras.replaceChildren();
            doc.effective.loras.forEach((l,index)=>{
                const a=doc.effective.assets.find(x=>x.asset_id===l.asset_id);
                const row=element('div',null,loras);element('span',(a?.relative_name||l.asset_id)+' ',row);
                for(const key of ['strength_model','strength_text_encoder']) {element('label',key+' ',row);const input=element('input',null,row);input.type='number';input.step='.05';input.value=l[key];input.style.width='72px';input.onchange=()=>{l[key]=Number(input.value);};}
                button(row,tr(app,'削除','Remove'),()=>{doc.effective.loras.splice(index,1);doc.effective.assets=doc.effective.assets.filter(x=>x.asset_id!==l.asset_id);doc.effective.loras.forEach((v,i)=>v.order=i);paintLoras();});
            });
        };
        paintLoras();
        const add=element('select',null,section);element('option',tr(app,'LoRAを追加','Add LoRA'),add).value='';
        for(const name of catalog.loras||[])element('option',name,add).value=name;
        add.onchange=()=>{
            if(!add.value)return;const id='lora_'+crypto.randomUUID();
            doc.effective.assets.push(asset('lora','loras',add.value,null,id));
            doc.effective.loras.push({asset_id:id,strength_model:1,strength_text_encoder:1,order:doc.effective.loras.length});add.value='';paintLoras();
        };
    }
    function fields(parent,value,path,rule={}) {
        if(rule.anyOf)rule=rule.anyOf.find(r=>r.type!=='null')||{};
        for(const [key,original] of Object.entries(value)) {
            if(['assets','loras','family','prediction_type'].includes(key)&&path==='')continue;
            const child=rule.properties?.[key]||{},pointer=path+'/'+key;
            if(original&&typeof original==='object'&&!Array.isArray(original)) {
                const details=element('details',null,parent);element('summary',key,details);if(['text','image','noise','sampling'].includes(key))details.open=true;
                fields(details,original,pointer,child);continue;
            }
            if(Array.isArray(original))continue;
            if(key==='sdxl'&&original===null)continue;
            const row=element('label',null,parent);row.style.cssText='display:flex;align-items:center;gap:12px;margin:5px 0';
            element('span',key,row).style.cssText='width:220px;flex-shrink:0';
            const nullable=child.anyOf?.some(r=>r.type==='null'),effectiveRule=child.anyOf?.find(r=>r.type!=='null')||child;
            let input;
            if(effectiveRule.enum){input=element('select',null,row);for(const v of effectiveRule.enum)element('option',String(v),input).value=String(v);}
            else if(typeof original==='boolean'){input=element('input',null,row);input.type='checkbox';input.checked=original;}
            else if(key.endsWith('_raw')){input=element('textarea',null,row);input.rows=4;input.style.width='100%';}
            else{input=element('input',null,row);input.type=(typeof original==='number'||effectiveRule.type==='number'||effectiveRule.type==='integer')?'number':'text';if(input.type==='number')input.step='any';input.style.flex='1';}
            if(typeof original!=='boolean')input.value=original===null?'':String(original);
            if(nullable)input.placeholder='model/default (null)';
            input.addEventListener('change',()=>{
                let result;
                try {
                    result=nullable&&input.value===''?null:scalarValue(input,original===null&&(effectiveRule.type==='number'||effectiveRule.type==='integer')?0:original);
                    value[key]=result;doc.requested[pointer]=result;
                    doc.provenance[pointer]={origin:'user_override',certainty:'explicit',acknowledged:true,raw_key:null,evidence:'Editor value'};
                }catch(error){status.textContent=String(error);}
            });
        }
    }
    function render() {
        form.replaceChildren();assetUI(form);
        fields(form,doc.effective,'',definition.schema.properties.effective.anyOf[0]);
        status.textContent=JSON.stringify({status:doc.status,raw_assets:doc.extensions.raw_assets,raw_loras:doc.extensions.raw_loras,unresolved:doc.unresolved,unsupported:doc.unsupported},null,2);
        ack.checked=false;
    }
    render();dialog.showModal();
}

let dropEnabled = false;
function updateDrop(enabled) {
    dropEnabled = enabled === true;
    if (dropEnabled && !installedHandler) {
        previousHandler = app.handleFile;
        const original = previousHandler;
        const importer = makeDropHandler(original, file => request('/reconstruct', file, true),
            (result,file,context) => placeReconstructed(result,app,globalThis.LiteGraph,context,()=>dropSerial),
            app, () => captureCanvas(app,++dropSerial));
        installedHandler = async function(file, ...args) {
            if (!dropEnabled) return original.call(app, file, ...args);
            try { return await importer(file, ...args); }
            catch (error) { outputError(error); }
        };
        app.handleFile = installedHandler;
    } else if (!dropEnabled && installedHandler && app.handleFile === installedHandler) {
        app.handleFile = previousHandler;
        installedHandler = null;
        previousHandler = null;
    }
    // If another extension wraps us, retain our disabled delegate. Do not break
    // its chain or install a second wrapper when this feature is enabled again.
}

app.registerExtension({
    name:'ForgeNeo.Bridge',
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.category === 'ForgeNeo Bridge') {
            try { nodeType.forgeHelp = await loadHelp(api); }
            catch (error) { console.warn(error); }
        }
    },
    settings:[{id:SETTING,name:tr(app,'ForgeNeo Bridge: 画像からワークフローを作成','ForgeNeo Bridge: canvas image to workflow'),type:'boolean',defaultValue:true,onChange:updateDrop}],
    setup(){updateDrop(app.extensionManager?.setting?.get?.(SETTING)??app.ui?.settings?.getSettingValue?.(SETTING)??true);},
    nodeCreated(node) {
        const assetInputs = {'CheckpointLoaderSimple':'ckpt_name','UNETLoader':'unet_name',
            'CLIPLoader':'clip_name','VAELoader':'vae_name','LoraLoader':'lora_name','LoraLoaderModelOnly':'lora_name'};
        const assetWidget = node.widgets?.find(w => w.name === assetInputs[node.type]);
        if (assetWidget) {
            const callback = assetWidget.callback;
            assetWidget.callback = function(value,...args) {
                if (node.properties?.forge_neo_bridge?.generated)
                    node.properties.forge_neo_bridge.current_name = value;
                return callback?.call(this,value,...args);
            };
        }
        const configure = node.configure;
        if (typeof configure === 'function') node.configure = function(info) {
            if (info?.properties?.forge_neo_bridge?.generated && Array.isArray(info.widgets_values)) {
                const serializedWidgets = (this.widgets || []).filter(widget => widget.serialize !== false);
                for (let i=0; i<Math.min(serializedWidgets.length,info.widgets_values.length); i++) {
                    const widget = serializedWidgets[i];
                    preserveMissingCombo(this,widget.name,info.widgets_values[i]);
                }
            }
            const result = configure.call(this,info);
            const current = this.widgets?.find(w => w.name === assetInputs[this.type]);
            if (current && this.properties?.forge_neo_bridge?.generated)
                this.properties.forge_neo_bridge.current_name = current.value;
            return result;
        };
        installSettingsAccordion(node, app);
        installSeedControls(node, app);
        if ((node.comfyClass || node.type) === 'ForgeNeoBridgeKSampler') {
            const control = node.widgets?.find(w => w.name === 'seed')?.linkedWidgets?.[0];
            if (control) Object.defineProperty(control,'tooltip',{get:()=>tr(app,
                'fixedは固定、incrementは+1、decrementは−1、randomizeはランダムにシードを変更します。変更のタイミングはComfyUIの設定に従います。',
                'Fixed keeps the seed; increment adds one, decrement subtracts one, and randomize chooses a random seed. Timing follows the ComfyUI widget-control setting.')});
        }
        watchSettingsLatent(node);
        installHelp(node,app,node.constructor.forgeHelp);
    },
    getCanvasMenuItems(canvas){return layoutMenu(app,LiteGraph,canvas);},
    getNodeMenuItems(node){return [...layoutMenu(app,LiteGraph,app.canvas,node),...(node.comfyClass==='ForgeCompatSpec'?[{content:tr(app,'ForgeNeo Bridge — 設定を編集','ForgeNeo Bridge — Edit Spec'),callback:()=>{
        try{const text=node.widgets.find(w=>w.name==='payload').value;editor(JSON.parse(text)).catch(outputError);}catch(error){outputError(error);}
    }}]:[])];}
});
