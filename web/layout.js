import {tr} from './i18n.js';

const SAMPLERS = new Set(['ForgeNeoBridgeKSampler','KSampler','KSamplerAdvanced','SamplerCustomAdvanced','ForgeCompatSampler']);
const IMAGE_OUTPUTS = new Set(['SaveImage','PreviewImage']);
export const BRIDGE_SAMPLER_WIDTH = 290;
// Leave room for the sampler's live preview and for a readable output image.
const SAMPLER_HEIGHT = 500, IMAGE_SIZE = [510,650];
const GROUP_PADDING = {top:100,side:30,bottom:30};
const info = n => n.properties?.forge_neo_bridge;
const linksOf = graph => graph.links instanceof Map ? [...graph.links.values()] : Object.values(graph.links || {});
const id = n => String(n.id);
const titles = [['モデル・サイズ','Model / size'],['プロンプト','Prompts'],['基本生成','Base generation'],['Hires','Hires'],['出力','Output']];
const colors = ['#35434b','#414052','#354b48','#434258','#41464c'];

function ancestors(node, links, nodes, stopAtSampler = false) {
    const found = new Set(), pending = [id(node)];
    while (pending.length) {
        const target = pending.pop();
        for (const link of links) {
            if (String(link.target_id) !== target) continue;
            const key = String(link.origin_id), parent = nodes.get(key);
            if (!parent || found.has(parent)) continue;
            found.add(parent);
            if (!stopAtSampler || !SAMPLERS.has(parent.type)) pending.push(key);
        }
    }
    found.delete(node);
    return found;
}

export function selectLayoutNodes(graph, selected = [], anchor = null) {
    const all = graph._nodes || [];
    selected = selected.filter(n => all.includes(n));
    if (selected.length > 1 && (!anchor || selected.includes(anchor))) {
        if (!selected.some(n => info(n)?.generated)) throw new Error('select');
        return selected;
    }
    const start = anchor || selected[0] || all.find(n => info(n)?.generated);
    if (!start || !info(start)?.generated) throw new Error('select');
    const candidates = all.filter(n => info(n)?.generated &&
        (!info(start).layout_id || info(n).layout_id === info(start).layout_id));
    const byId = new Map(candidates.map(n => [id(n),n]));
    const found = new Set([start]), pending = [start], links = linksOf(graph);
    while (pending.length) {
        const current = pending.pop();
        for (const link of links) {
            const other = String(link.origin_id) === id(current) ? link.target_id
                : String(link.target_id) === id(current) ? link.origin_id : null;
            const n = byId.get(String(other));
            if (n && !found.has(n)) {found.add(n);pending.push(n);}
        }
    }
    if (!anchor && !selected.length && all.some(n => info(n)?.generated && !found.has(n))) throw new Error('select');
    // Shared loaders can join independent legacy workflows: require explicit selection.
    const roots = [...found].filter(n => SAMPLERS.has(n.type) &&
        ![...ancestors(n,links,byId)].some(p => SAMPLERS.has(p.type)));
    if (roots.length > 1) throw new Error('select');
    return [...found];
}

/** Compute coordinates only; never change execution inputs or connection IDs. */
export function planLayout(nodes, links, origin = [100,130]) {
    const byId = new Map(nodes.map(n => [id(n),n]));
    const stages = new Map();
    const samplers = nodes.filter(n => SAMPLERS.has(n.type));
    const downstream = new Set();
    for (const n of nodes) {
        const upstream = ancestors(n,links,byId);
        if ([...upstream].some(p => SAMPLERS.has(p.type))) downstream.add(n);
        stages.set(n, /TextEncode|ForgeCompatText|ConditioningZeroOut/.test(n.type) ? 1
            : SAMPLERS.has(n.type) || /Scheduler|Guider|RandomNoise|KSamplerSelect|ForgeCompatNoise/.test(n.type) ? 2
            : /Save|Preview/.test(n.type) || n.type === 'VAEDecode' && downstream.has(n) ? 4 : 0);
    }
    for (const sampler of samplers) {
        const stage = downstream.has(sampler) ? 3 : 2;
        stages.set(sampler,stage);
        for (const link of links.filter(l => String(l.target_id) === id(sampler))) {
            const parent = byId.get(String(link.origin_id));
            if (!parent) continue;
            if (parent.type === 'ForgeNeoBridgeSettings') stages.set(parent,stage);
            if (stage === 3 && ['LATENT','IMAGE'].includes(link.type)) {
                for (const p of [parent,...ancestors(parent,links,byId,true)])
                    if (downstream.has(p) && !SAMPLERS.has(p.type)) stages.set(p,3);
            }
        }
    }
    const positions = new Map(), sizes = new Map(), groups = [];
    const depth = new Map(nodes.map(n=>[n,ancestors(n,links,byId).size]));
    let x = origin[0];
    const order = (a,b) => {
        const rank = n => n.type === 'ForgeNeoBridgeSettings' ? 2 : SAMPLERS.has(n.type) ? 1 : 0;
        return rank(a)-rank(b) || depth.get(a)-depth.get(b) || id(a).localeCompare(id(b),undefined,{numeric:true});
    };
    for (let stage=0;stage<5;stage++) {
        const members = nodes.filter(n => stages.get(n) === stage).sort(order);
        if (!members.length) continue;
        const columns = stage === 3 ? [members.filter(n => !SAMPLERS.has(n.type) && n.type !== 'ForgeNeoBridgeSettings'),
            members.filter(n => SAMPLERS.has(n.type) || n.type === 'ForgeNeoBridgeSettings')].filter(c=>c.length) : [members];
        let right=x, bottom=origin[1];
        for (const column of columns) {
            let y=origin[1],width=0;
            for (const n of column) {
                let w=n.size?.[0] || 300,h=n.size?.[1] || 160;
                if (n.type === 'ForgeNeoBridgeKSampler' && !n.flags?.collapsed && w < BRIDGE_SAMPLER_WIDTH) {
                    w=BRIDGE_SAMPLER_WIDTH;
                    sizes.set(n,[w,h]);
                }
                if (IMAGE_OUTPUTS.has(n.type) && !n.flags?.collapsed) {
                    w=Math.max(w,IMAGE_SIZE[0]);h=Math.max(h,IMAGE_SIZE[1]);
                    sizes.set(n,[w,h]);
                }
                h=n.flags?.collapsed ? 0 : SAMPLERS.has(n.type) ? Math.max(h,SAMPLER_HEIGHT) : h;
                positions.set(n,[right,y]);width=Math.max(width,w);
                y+=h+60;
            }
            bottom=Math.max(bottom,y-60);right+=width+40;
        }
        groups.push({stage,nodes:members,pos:[x-GROUP_PADDING.side,origin[1]-GROUP_PADDING.top],
            size:[right-40-x+GROUP_PADDING.side*2,Math.max(120,bottom-origin[1]+GROUP_PADDING.top+GROUP_PADDING.bottom)]});
        x=right-40+GROUP_PADDING.side*2+60;
    }
    return {positions,sizes,groups};
}

export function arrangeLayout(graph,nodes,LiteGraph,app,{initial=false}={}) {
    if (!nodes.length) return;
    const origin=initial ? [100,130] : [Math.min(...nodes.map(n=>n.pos[0])),Math.min(...nodes.map(n=>n.pos[1]))];
    const plan=planLayout(nodes,linksOf(graph),origin);
    // Persist group IDs on member nodes; standard LGraphGroup serializes its ID.
    const previous=new Set(nodes.flatMap(n=>info(n)?.layout_groups || []));
    const outside=graph._nodes?.filter(n=>!nodes.includes(n)) || [];
    const protectedGroups=new Set(outside.flatMap(n=>info(n)?.layout_groups || []));
    graph.beforeChange?.();
    try {
        for (const group of [...(graph._groups || [])])
            if (previous.has(group.id) && !protectedGroups.has(group.id)) graph.remove(group);
        for (const [node,pos] of plan.positions) node.pos=pos;
        for (const [node,size] of plan.sizes) {
            if (node.setSize) node.setSize(size);
            else node.size=size;
        }
        if (LiteGraph.LGraphGroup) for (const part of plan.groups) {
            const group=new LiteGraph.LGraphGroup(tr(app,...titles[part.stage]));
            group.color=colors[part.stage];group.font_size=20;
            group.pos=part.pos;group.size=part.size;graph.add(group);
            for (const n of part.nodes) {
                n.properties ??= {};n.properties.forge_neo_bridge ??= {};
                n.properties.forge_neo_bridge.layout_groups=[group.id];
            }
        }
    } finally {graph.afterChange?.();graph.setDirtyCanvas?.(true,true);}
}

export function layoutMenu(app,LiteGraph,canvas,anchor=null) {
    const graph=canvas?.graph || anchor?.graph || app.graph;
    if (!graph?._nodes?.some(n=>info(n)?.generated)) return [];
    return [{content:'ForgeNeo Bridge',submenu:{options:[{
        content:tr(app,'配置を整える','Arrange workflow'),callback:()=>{
            const selected=Object.values(canvas?.selected_nodes || {});
            let target;
            try {target=selectLayoutNodes(graph,selected,anchor);}
            catch {globalThis.alert(tr(app,'整理するBridgeノードを1つ、または対象ノード一式を選択してください。',
                'Select one Bridge node, or select the complete set of nodes to arrange.'));return;}
            arrangeLayout(graph,target,LiteGraph,app);
        }
    }]}}];
}
