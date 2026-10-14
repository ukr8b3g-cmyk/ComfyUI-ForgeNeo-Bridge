import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';

let instance = 0;
async function frontend(kind = 'forge') {
    const calls = {inspect:0, native:0, foreign:0, loads:0};
    const listeners = [];
    class Graph {
        constructor() { this._nodes = []; }
        add(node) { node.id=this._nodes.length+1; node.pos=[0,0]; node.size=[300,160]; this._nodes.push(node); }
        serialize() { return {nodes:this._nodes.map(n=>({type:n.type,widgets_values:[]})),extra:{}}; }
    }
    const app = {
        graph:new Graph(), ui:{settings:{getSettingValue:()=>undefined}},
        handleFile() { calls.native++; return 'native'; },
        registerExtension(extension) { this.extension=extension; extension.setup(); },
        async loadGraphData() { calls.loads++; return true; },
    };
    globalThis.__forgePriorityApp=app;
    globalThis.__forgePriorityApi={fetchApi:async()=>{
        calls.inspect++;
        return {ok:true,json:async()=>({kind,plan:{schema_version:'1.0.0',name:'Forge',
            nodes:[{id:'1',type:'ForgeNeoBridgeSettings',values:{}}],edges:[]}})};
    }};
    globalThis.LiteGraph={createNode:type=>({type,widgets:[],inputs:[],outputs:[{}]})};
    globalThis.window={alert:message=>assert.fail(message)};
    globalThis.document={addEventListener:(name,callback,capture)=>{
        assert.equal(name,'drop'); assert.equal(capture,true); listeners.push(callback);
    }};
    const id=++instance;
    let script=await readFile(new URL('../web/bridge.js',import.meta.url),'utf8');
    script=script.replace('../../scripts/app.js',`data:text/javascript,export const app=globalThis.__forgePriorityApp;#${id}`)
        .replace('../../scripts/api.js',`data:text/javascript,export const api=globalThis.__forgePriorityApi;#${id}`);
    for (const file of ['bridge_core','layout','settings_accordion','i18n','help','seed_controls'])
        script=script.replace(`./${file}.js`,new URL(`../web/${file}.js`,import.meta.url).href);
    await import('data:text/javascript,'+encodeURIComponent(script));
    assert.equal(listeners.length,1);
    const drop=()=>listeners[0]({preventDefault(){assert.fail('Drop event was consumed');},stopImmediatePropagation(){assert.fail('Drop event was consumed');}});
    const wrap=(consumeJpeg=false)=>{
        const previous=app.handleFile;
        app.handleFile=function(file,...args) {
            calls.foreign++;
            if (consumeJpeg && /\.jpe?g$/i.test(file.name)) return 'foreign JPEG';
            return previous.call(this,file,...args);
        };
    };
    return {app,calls,drop,wrap};
}

test.afterEach(()=>{
    delete globalThis.__forgePriorityApp; delete globalThis.__forgePriorityApi;
    delete globalThis.window; delete globalThis.document; delete globalThis.LiteGraph;
});

test('a later JPEG importer cannot consume a Forge image before Bridge',async()=>{
    const f=await frontend(); f.wrap(true); f.drop();
    await f.app.handleFile(new File(['x'],'forge.jpg',{type:'image/jpeg'}));
    assert.deepEqual(f.calls,{inspect:1,native:0,foreign:0,loads:1});
});

test('native Krea images and JSON still reach the existing handler exactly once',async()=>{
    const f=await frontend('native'); f.wrap();
    for (const name of ['krea.webp','krea.png','workflow.json']) {
        f.drop(); assert.equal(await f.app.handleFile(new File(['x'],name)),'native');
    }
    assert.deepEqual(f.calls,{inspect:2,native:3,foreign:3,loads:0});
});

test('repeated drops keep one active Bridge handler',async()=>{
    const f=await frontend(); f.wrap(true); f.drop(); const active=f.app.handleFile;
    for (let i=0;i<5;i++) {
        f.drop(); assert.equal(f.app.handleFile,active);
        await f.app.handleFile(new File(['x'],'forge.jpeg'));
    }
    assert.deepEqual(f.calls,{inspect:5,native:0,foreign:0,loads:5});
});

test('disabled Bridge preserves the foreign JPEG importer, then re-enables cleanly',async()=>{
    const f=await frontend(); f.wrap(true); const foreign=f.app.handleFile;
    f.app.extension.settings[0].onChange(false); f.drop();
    assert.equal(f.app.handleFile,foreign);
    assert.equal(await f.app.handleFile(new File(['x'],'forge.jpg')),'foreign JPEG');
    assert.equal(f.calls.inspect,0); assert.equal(f.calls.loads,0);
    f.app.extension.settings[0].onChange(true); f.drop();
    await f.app.handleFile(new File(['x'],'forge.jpg'));
    assert.deepEqual(f.calls,{inspect:1,native:0,foreign:1,loads:1});
});

test('retired Bridge wrappers delegate without recursion after more extensions and toggles',async()=>{
    const f=await frontend('native');
    for (let i=0;i<3;i++) {
        f.wrap(); f.drop();
        await f.app.handleFile(new File(['x'],'krea.webp'));
        f.app.extension.settings[0].onChange(false);
        f.app.extension.settings[0].onChange(true);
    }
    assert.equal(f.calls.inspect,3); assert.equal(f.calls.native,3); assert.equal(f.calls.loads,0);
});
