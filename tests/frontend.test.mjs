import test from 'node:test';
import assert from 'node:assert/strict';
import {buildDetached, makeDropHandler, scalarValue} from '../web/bridge_core.js';

class Graph {
    constructor(){this.nodes=[];this.arranged=false;}
    add(node){this.nodes.push(node);}
    arrange(){this.arranged=true;}
    serialize(){return {nodes:this.nodes.map(n=>({type:n.type,widgets_values:n.widgets.map(w=>w.value)})),extra:{}};}
}
const lite={createNode(type){return {type,inputs:[{name:'spec'}],outputs:[{}],widgets:[{name:'payload',value:'',options:{}}],connect(){return {};}};}};
const plan=()=>({schema_version:'1.0.0',config_hash:'a'.repeat(64),nodes:[{id:'1',type:'ForgeCompatSpec',title:'Spec',values:{payload:'18446744073709551615'}},{id:'2',type:'ForgeCompatText',values:{}}],edges:[['1',0,'2','spec']]});
test('build detached and retain exact seed text',()=>{
    const data=buildDetached(plan(),lite,Graph);
    assert.equal(data.nodes[0].widgets_values[0],'18446744073709551615');
    assert.equal(data.extra.forge_neo_bridge.qualification,'not_evaluated');
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
test('Forge drop opens review, never calls native load',async()=>{
    let calls=0,review=0;const spec={schema_version:'1.0.0'};
    const fn=makeDropHandler(()=>calls++,async()=>({kind:'forge',spec}),(s)=>{assert.equal(s,spec);review++;},{});
    await fn(new File(['x'],'test.png'));assert.equal(review,1);assert.equal(calls,0);
});
test('non-image files bypass inspection',async()=>{
    let inspected=0;const fn=makeDropHandler(()=>42,()=>inspected++,()=>{},{});
    assert.equal(await fn(new File(['x'],'workflow.json')),42);assert.equal(inspected,0);
});
test('failed inspection does not clear or load anything',async()=>{
    let calls=0;const fn=makeDropHandler(()=>calls++,async()=>{throw Error('Malformed EXIF');},()=>calls++,{});
    await assert.rejects(fn(new File(['x'],'bad.webp')),/Malformed EXIF/);assert.equal(calls,0);
});
