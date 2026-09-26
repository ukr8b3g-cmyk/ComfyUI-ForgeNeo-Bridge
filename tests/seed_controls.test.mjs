import test from 'node:test';
import assert from 'node:assert/strict';
import {installSeedControls, nextSeed, seedValue} from '../web/seed_controls.js';

function sampler(controlMode='after') {
    const values=['9007199254740993',30,7,'euler','automatic',1];
    const node={comfyClass:'ForgeNeoBridgeKSampler',properties:{},pos:[0,0],size:[320,300],
        widgets:['seed','steps','cfg','sampler_name','scheduler','denoise'].map((name,i)=>({name,value:values[i],options:{}})),
        graph:{incrementVersion(){},setDirtyCanvas(){}},
        addWidget(type,name,value,callback,options){const widget={type,name,value,callback,options};this.widgets.push(widget);return widget;},
        serialize(){const data={properties:structuredClone(this.properties),widgets_values:this.widgets.map(w=>w.serialize!==false?w.value:undefined)};this.onSerialize?.(data);return data;},
        configure(data){this.properties=structuredClone(data.properties||{});this.widgets.filter(w=>w.serialize!==false).forEach((w,i)=>w.value=data.widgets_values[i]);this.onConfigure?.(data);},
    };
    const helpers={addValueControlWidgets(node,seed,value){return [node.addWidget('combo','control_after_generate',value,()=>{},{values:['fixed','increment','decrement','randomize'],serialize:false})];}};
    const app={ui:{settings:{getSettingValue:()=>controlMode}}};
    installSeedControls(node,app,helpers);
    return {node,seed:node.widgets[0],control:node.widgets[1],values};
}

test('uint64 seeds step exactly and clamp at both boundaries',()=>{
    assert.equal(nextSeed('9007199254740993','increment'),'9007199254740994');
    assert.equal(nextSeed('18446744073709551615','increment'),'18446744073709551615');
    assert.equal(nextSeed('0','decrement'),'0');
    assert.equal(nextSeed('18446744073709551615','decrement'),'18446744073709551614');
    assert.equal(seedValue(' 000123 '),'123');
    assert.throws(()=>seedValue(9007199254740993));
    assert.throws(()=>seedValue('1.5'));
});

test('randomization retains every generated bit without Number rounding',()=>{
    assert.equal(nextSeed('0','randomize',words=>{words.set([0xffffffff,0xffffffff]);return words;}),'18446744073709551615');
    for(let i=0;i<8;i++){const value=nextSeed('0','randomize');assert.match(value,/^\d+$/);assert(BigInt(value)<2n**64n);}
});

test('numeric arrows and click entry retain string precision',()=>{
    const {node,seed}=sampler();
    assert.equal(seed.type,'number');assert.equal(seed._displayValue,'9007199254740993');
    seed.onClick({e:{canvasX:310},canvas:{}});
    assert.equal(seed.value,'9007199254740994');
    seed.onClick({e:{canvasX:20},canvas:{}});
    assert.equal(seed.value,'9007199254740993');
    seed.onClick({e:{canvasX:160},canvas:{prompt:(_label,_value,done)=>done('18446744073709551615')}});
    assert.equal(seed.value,'18446744073709551615');
    seed.onDrag({e:{canvasX:160,deltaX:-3},canvas:{}});
    assert.equal(seed.value,'18446744073709551612');
    assert.equal(node.widgets[1].name,'control_after_generate');
});

test('old positional values and new mode round-trip without shifting steps/CFG',()=>{
    const {node,control,values}=sampler();
    node.configure({widgets_values:values});
    assert.equal(control.value,'fixed');
    control.value='randomize';
    const saved=node.serialize();
    assert.deepEqual(saved.widgets_values,values);
    assert.equal(saved.properties.forge_neo_bridge.seed_control,'randomize');
    const restored=sampler();restored.node.configure(saved);
    assert.equal(restored.control.value,'randomize');
    assert.deepEqual(restored.node.serialize().widgets_values,values);
});

test('fixed/export do not advance; after-queue modes change only the next seed',()=>{
    const {node,seed,control}=sampler();
    control.beforeQueued();control.afterQueued();assert.equal(seed.value,'9007199254740993');
    control.value='increment';
    assert.equal(node.serialize().widgets_values[0],'9007199254740993');
    control.beforeQueued();assert.equal(seed.value,'9007199254740993');
    control.afterQueued();assert.equal(seed.value,'9007199254740994');
    control.value='decrement';control.afterQueued();assert.equal(seed.value,'9007199254740993');
    control.afterQueued({isPartialExecution:true});assert.equal(seed.value,'9007199254740993');
    node.inputs=[{widget:{name:'seed'},link:42}];
    control.afterQueued();assert.equal(seed.value,'9007199254740993');
});

test('Comfy before-generation preference preserves first seed and advances later queues',()=>{
    const {seed,control}=sampler('before');control.value='increment';
    control.beforeQueued();control.afterQueued();assert.equal(seed.value,'9007199254740993');
    control.beforeQueued();assert.equal(seed.value,'9007199254740994');
    control.afterQueued();assert.equal(seed.value,'9007199254740994');
});
