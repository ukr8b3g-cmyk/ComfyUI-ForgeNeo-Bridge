import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {installHelp} from '../web/help.js';
const catalog=Object.fromEntries(await Promise.all(['ja','en'].map(async lang=>
    [lang,{nodeDefs:JSON.parse(await readFile(new URL(`../locales/${lang}/nodeDefs.json`,import.meta.url),'utf8'))}])));
test('every Bridge help entry has Japanese and English text, and switches without changing values',()=>{
    let locale='en';const app={ui:{settings:{getSettingValue:()=>locale}}};
    assert.deepEqual(Object.keys(catalog.en.nodeDefs),Object.keys(catalog.ja.nodeDefs));
    for(const [name,definition] of Object.entries(catalog.en.nodeDefs)){
        const jp=catalog.ja.nodeDefs[name];
        assert(definition.description.length && jp.description.length);
        assert.deepEqual(Object.keys(definition.inputs),Object.keys(jp.inputs));
        const node={type:name,widgets:Object.keys(definition.inputs).map(name=>({name,value:'unchanged'}))};
        installHelp(node,app,catalog);
        for(const widget of node.widgets){
            locale='en';assert.equal(widget.tooltip,definition.inputs[widget.name].tooltip);
            locale='ja';assert.equal(widget.tooltip,jp.inputs[widget.name].tooltip);
            assert.notEqual(widget.tooltip,definition.inputs[widget.name].tooltip);
            assert.equal(widget.value,'unchanged');
        }
        for(const key in definition.outputs)assert(jp.outputs[key].tooltip.length && definition.outputs[key].tooltip.length);
    }
});
