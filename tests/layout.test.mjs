import test from 'node:test';
import assert from 'node:assert/strict';
import {planLayout,arrangeLayout,selectLayoutNodes,layoutMenu} from '../web/layout.js';

function fixture() {
 const types=['CheckpointLoaderSimple','ForgeNeoBridgeSettings','ForgeNeoBridgeTextEncode','ForgeNeoBridgeTextEncode',
  'ForgeNeoBridgeKSampler','EmptyLatentImage','ForgeNeoBridgeSettings','VAEDecode','ImageScale','VAEEncode','ForgeNeoBridgeKSampler','VAEDecode','SaveImage'];
 const nodes=types.map((type,i)=>({type,id:i+1,pos:[100+i*30,130+i*70],size:[type.includes('TextEncode')?400:300,type.includes('Settings')?180:200],
  title:type,widgets_values:[i],properties:{forge_neo_bridge:{generated:true,layout_id:'a'}}}));
 const pairs=[[1,3,'CLIP'],[1,4,'CLIP'],[1,5,'MODEL'],[2,5,'FORGE_SETTINGS'],[3,5,'FORGE_CONDITIONING'],[4,5,'FORGE_CONDITIONING'],[6,5,'LATENT'],
  [5,8,'LATENT'],[8,9,'IMAGE'],[9,10,'IMAGE'],[10,11,'LATENT'],[7,11,'FORGE_SETTINGS'],[1,11,'MODEL'],[3,11,'FORGE_CONDITIONING'],[11,12,'LATENT'],[12,13,'IMAGE']];
 let next=0;
 const graph={_nodes:nodes,_groups:[],links:pairs.map(([origin_id,target_id,type])=>({origin_id,target_id,type})),
  add(g){g.id=++next;this._groups.push(g);},remove(g){this._groups.splice(this._groups.indexOf(g),1);},
  beforeChange(){this.before=(this.before||0)+1;},afterChange(){this.after=(this.after||0)+1;}};
 return graph;
}
const lite={LGraphGroup:class {constructor(title){this.title=title;}}};
const execution=g=>g._nodes.map(({id,type,title,size,widgets_values})=>({id,type,title,size,widgets_values}));

test('compact Hires layout keeps stages together, without node or group overlap',()=>{
 const g=fixture(),p=planLayout(g._nodes,g.links);
 assert.equal(p.groups.length,5);
 assert.deepEqual(p.groups.find(g=>g.stage===3).nodes.map(n=>n.id),[8,9,10,11,7]);
 assert.deepEqual(p.groups.find(g=>g.stage===2).nodes.map(n=>n.id),[5,2]);
 for(let i=0;i<p.groups.length-1;i++) assert(p.groups[i].pos[0]+p.groups[i].size[0]<p.groups[i+1].pos[0]);
 for(const a of g._nodes) for(const b of g._nodes) if(a.id<b.id){
  const [ax,ay]=p.positions.get(a),[bx,by]=p.positions.get(b);
  assert(ax+a.size[0]<=bx||bx+b.size[0]<=ax||ay+a.size[1]+30<=by||by+b.size[1]+30<=ay);
 }
});
test('saved node-array ordering does not reorder the decode/upscale/encode chain',()=>{
 const g=fixture(),a=planLayout(g._nodes,g.links),b=planLayout([...g._nodes].reverse(),g.links);
 for(const n of g._nodes)assert.deepEqual(a.positions.get(n),b.positions.get(n));
 assert(a.positions.get(g._nodes[7])[1]<a.positions.get(g._nodes[8])[1]);
 assert(a.positions.get(g._nodes[8])[1]<a.positions.get(g._nodes[9])[1]);
});
test('repeat arrangement is position-stable, replaces only its frames, and leaves inputs/sizes/links intact',()=>{
 const g=fixture();g._nodes.find(n=>n.type==='SaveImage').size=[640,900];
 const before=structuredClone(execution(g)),links=structuredClone(g.links);
 const userGroup={id:900,title:'User'};g._groups.push(userGroup);
 arrangeLayout(g,g._nodes,lite,{});const positions=g._nodes.map(n=>[...n.pos]);
 arrangeLayout(g,g._nodes,lite,{});
 assert.deepEqual(g._nodes.map(n=>n.pos),positions);assert.equal(g._groups.length,6);assert(g._groups.includes(userGroup));
 assert.deepEqual(execution(g),before);assert.deepEqual(g.links,links);assert.equal(g.before,2);assert.equal(g.after,2);
});
test('sampler reserves preview space and image outputs expand without shrinking user sizes',()=>{
 const g=fixture(),sampler=g._nodes[4],settings=g._nodes[1],output=g._nodes[12];
 const before=structuredClone(g.links);
 arrangeLayout(g,g._nodes,lite,{});
 assert.equal(settings.pos[1]-sampler.pos[1],560);
 assert.deepEqual(sampler.size,[300,200]);assert.deepEqual(output.size,[510,650]);
 assert.deepEqual(g.links,before);
 sampler.size[1]=750;output.size=[720,960];
 arrangeLayout(g,g._nodes,lite,{});
 assert.equal(settings.pos[1]-sampler.pos[1],810);assert.deepEqual(output.size,[720,960]);
 const group=g._groups.find(g=>g.title==='Base generation');
 assert.equal(sampler.pos[1]-group.pos[1],100);
 assert.equal(sampler.pos[0]-group.pos[0],30);
 assert.equal(group.pos[1]+group.size[1]-(settings.pos[1]+settings.size[1]),30);
});
test('collapsed nodes keep their size and do not reserve expanded preview space',()=>{
 const g=fixture(),sampler=g._nodes[4],settings=g._nodes[1],output=g._nodes[12];
 sampler.flags={collapsed:true};output.flags={collapsed:true};
 arrangeLayout(g,g._nodes,lite,{});
 assert.equal(settings.pos[1]-sampler.pos[1],60);assert.deepEqual(output.size,[300,200]);
});
test('target selection isolates imports and refuses ambiguous shared legacy workflows',()=>{
 const g=fixture(),extra={...g._nodes[0],id:99,pos:[9000,9000],properties:{forge_neo_bridge:{generated:true,layout_id:'b'}}};g._nodes.push(extra);
 assert.throws(()=>selectLayoutNodes(g),/select/);
 assert.equal(selectLayoutNodes(g,[],g._nodes[4]).length,13);
 assert.deepEqual(selectLayoutNodes(g,[g._nodes[0],g._nodes[1]]),g._nodes.slice(0,2));
 const subset=selectLayoutNodes(g,[],g._nodes[4]);arrangeLayout(g,subset,lite,{});assert.deepEqual(extra.pos,[9000,9000]);
 extra.type='ForgeNeoBridgeKSampler';delete extra.properties.forge_neo_bridge.layout_id;
 for(const n of g._nodes)delete n.properties.forge_neo_bridge.layout_id;
 g.links.push({origin_id:1,target_id:99,type:'MODEL'});
 assert.throws(()=>selectLayoutNodes(g,[],g._nodes[4]),/select/);
});
test('partially selected owned frame is retained',()=>{
 const g=fixture();arrangeLayout(g,g._nodes,lite,{});
 const before=g._groups[0];arrangeLayout(g,[g._nodes[0]],lite,{});
 assert(g._groups.includes(before));
});
test('clipboard ID remapping and saved ownership replace only copied frames',()=>{
 const g=fixture();arrangeLayout(g,g._nodes,lite,{});
 const originals=[...g._groups],originalNodes=[...g._nodes];
 const copies=structuredClone(originalNodes),frames=structuredClone(originals);
 for(const n of copies){n.id+=100;n.pos[0]+=6000;}
 for(const f of frames){f.pos[0]+=6000;g.add(f);}
 g._nodes.push(...copies);
 g.links.push(...g.links.map(l=>({...l,origin_id:l.origin_id+100,target_id:l.target_id+100})));
 const user={id:900,title:'User',pos:[5900,0],size:[10000,3000]};g._groups.push(user);
 arrangeLayout(g,copies,lite,{});
 assert.equal(g._groups.length,11);
 for(const f of originals)assert(g._groups.includes(f));
 for(const f of frames)assert(!g._groups.includes(f));
 assert(g._groups.includes(user));
 // Serialization preserves flags; IDs are not ownership evidence.
 g._groups=JSON.parse(JSON.stringify(g._groups));
 for(const f of g._groups)f.id+=1000;
 g._nodes=JSON.parse(JSON.stringify(g._nodes));
 arrangeLayout(g,g._nodes,lite,{});
 assert.equal(g._groups.length,6);assert(g._groups.some(f=>f.title==='User'));
});
test('legacy stale IDs and unrelated members cannot delete a frame',()=>{
 const g=fixture();
 const manual={id:11,title:'User notes',pos:[5000,5000],size:[300,200]};g._groups.push(manual);
 for(const n of g._nodes)n.properties.forge_neo_bridge.layout_groups=[11];
 arrangeLayout(g,g._nodes,lite,{});assert(g._groups.includes(manual));
 const owned=g._groups.find(f=>f.flags?.forge_neo_bridge);
 const foreign={id:99,type:'Note',pos:[owned.pos[0]+40,owned.pos[1]+110],size:[100,100]};g._nodes.push(foreign);
 arrangeLayout(g,g._nodes.filter(n=>n!==foreign),lite,{});
 assert(g._groups.includes(owned));assert(g._groups.includes(manual));
});
test('menus follow Comfy locale',()=>{
 const g=fixture(),app={graph:g,ui:{settings:{getSettingValue:()=> 'ja'}}};
 assert.equal(layoutMenu(app,lite,{graph:g})[0].submenu.options[0].content,'配置を整える');
 app.ui.settings.getSettingValue=()=> 'en';
 assert.equal(layoutMenu(app,lite,{graph:g},g._nodes[0])[0].submenu.options[0].content,'Arrange workflow');
});
