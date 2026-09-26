/** Pure UI helpers: graph construction is detached; settings use named widgets/ports. */
import {arrangeLayout} from './layout.js';
export const MAX_IMAGE = 128 * 1024 * 1024;
export const ALLOWED = new Set(['CheckpointLoaderSimple','UNETLoader','CLIPLoader','DualCLIPLoader','VAELoader','LoraLoader','LoraLoaderModelOnly','EmptyLatentImage','EmptySD3LatentImage','EmptyFlux2LatentImage','CLIPTextEncode','TextEncodeQwenImageEdit','TextEncodeQwenImageEditPlus','ConditioningZeroOut','ModelSamplingAuraFlow','ModelSamplingSD3','CFGNorm','KSampler','FluxGuidance','Flux2Scheduler','RandomNoise','KSamplerSelect','BasicGuider','CFGGuider','SamplerCustomAdvanced','ForgeCompatSpec','ForgeCompatText','ForgeCompatNoise','ForgeCompatSampler','ForgeNeoBridgeTextEncode','ForgeNeoBridgeSettings','ForgeNeoBridgeKSampler','ForgeNeoBridgeScheduler','VAEDecode','VAEEncode','ImageScale','LatentUpscale','LoadImage','SaveImage']);
export const MISSING_COMBOS = new Set(['CheckpointLoaderSimple.ckpt_name','UNETLoader.unet_name','CLIPLoader.clip_name','DualCLIPLoader.clip_name1','DualCLIPLoader.clip_name2','VAELoader.vae_name','LoraLoader.lora_name','LoraLoaderModelOnly.lora_name','LoadImage.image','ForgeNeoBridgeKSampler.sampler_name','ForgeNeoBridgeKSampler.scheduler']);

export function preserveMissingCombo(node, name, value) {
    const widget = node.widgets?.find(w => w.name === name);
    if (!widget) return false;
    const values = widget.options?.values;
    if (!Array.isArray(values) || values.includes(value)) return true;
    if (!MISSING_COMBOS.has(`${node.type}.${name}`)) return false;
    widget.options = {...widget.options, values:[...values, value]};
    return true;
}

export function buildDetached(plan, LiteGraph, GraphClass, app) {
    if (!plan || plan.schema_version !== '1.0.0' || !Array.isArray(plan.nodes) || !Array.isArray(plan.edges)) throw new Error('Invalid graph plan');
    if (plan.nodes.length > 256 || plan.edges.length > 1024) throw new Error('Graph size limit exceeded');
    const graph = new GraphClass();
    const nodes = new Map();
    const layoutId = globalThis.crypto.randomUUID();
    for (const item of plan.nodes) {
        if (!ALLOWED.has(item.type) || nodes.has(item.id)) throw new Error('Unknown or duplicate node');
        const node = LiteGraph.createNode(item.type);
        if (!node) throw new Error(`Missing node: ${item.type}`);
        if (item.title != null) node.title = item.title;
        node.properties = {...node.properties, ...item.properties};
        node.properties.forge_neo_bridge ??= {};
        node.properties.forge_neo_bridge.generated = true;
        node.properties.forge_neo_bridge.layout_id = layoutId;
        graph.add(node); nodes.set(item.id, node);
        for (const [name, value] of Object.entries(item.values)) {
            const widget = node.widgets?.find(w => w.name === name);
            if (!widget) throw new Error(`Missing widget ${item.type}.${name}; frontend/backend versions do not match`);
            if (!preserveMissingCombo(node,name,value)) throw new Error(`Unavailable value: ${item.type}.${name}`);
            widget.value = value;
        }
        if ((item.type === 'KSampler' && 'seed' in item.values) ||
            (item.type === 'RandomNoise' && 'noise_seed' in item.values)) {
            const control = node.widgets?.find(w => w.name === 'control_after_generate');
            if (!control) throw new Error(`Missing widget ${item.type}.control_after_generate; frontend/backend versions do not match`);
            control.value = 'fixed';
        }
    }
    for (const [from, slot, to, input] of plan.edges) {
        const src = nodes.get(from), dst = nodes.get(to);
        const index = dst?.inputs?.findIndex(x => x.name === input);
        if (!src || !dst || index == null || index < 0 || !src.outputs?.[slot]) throw new Error('Invalid graph edge');
        if (src.connect(slot, dst, index) == null) throw new Error(`Cannot connect ${src.type} → ${dst.type}.${input}`);
    }
    for (const node of nodes.values()) node.forgeRefreshSettings?.();
    arrangeLayout(graph,[...nodes.values()],LiteGraph,app,{initial:true});
    const data = graph.serialize();
    data.extra ??= {};
    data.extra.forge_neo_bridge = {config_hash: plan.config_hash,
        qualification: plan.qualification || 'not_evaluated', source:plan.source, notes:plan.notes || []};
    return data;
}

export function captureCanvas(app, serial) {
    const graph = app.rootGraph || app.graph;
    const before = structuredClone(graph.serialize());
    return {graph,before,fingerprint:JSON.stringify(before),
        workflow:app.extensionManager?.workflowManager?.activeWorkflow?.id,serial};
}

export async function placeReconstructed(result, app, LiteGraph, context, currentSerial) {
    const {graph,before,fingerprint,workflow,serial} = context;
    if (!LiteGraph?.createNode || typeof app.loadGraphData !== 'function') throw new Error('Comfy graph API is unavailable');
    const data = buildDetached(result.plan, LiteGraph, graph.constructor,app);
    if (serial !== currentSerial() || graph !== (app.rootGraph || app.graph)
        || workflow !== app.extensionManager?.workflowManager?.activeWorkflow?.id
        || JSON.stringify(graph.serialize()) !== fingerprint) {
        throw new Error('Workflow changed while reading the image; drop again on the current canvas');
    }
    try {
        const loaded = await app.loadGraphData(data,true,true,result.plan.name+'.json');
        if (loaded === false) throw new Error('Comfy rejected the reconstructed workflow');
    } catch (error) {
        if (serial === currentSerial()) await app.loadGraphData(before,true,true,'ForgeNeo Bridge — previous workflow');
        throw error;
    }
}

export function makeDropHandler(original, inspect, open, owner, capture = () => undefined) {
    return async function(file, ...args) {
        if (!(file instanceof File) || !/\.(png|webp|jpe?g)$/i.test(file.name)) return original.call(owner, file, ...args);
        if (file.size > MAX_IMAGE) throw new Error('ForgeNeo Bridge: file exceeds 128 MiB');
        const context = capture();
        const result = await inspect(file);
        if (result.kind !== 'forge') return original.call(owner, file, ...args);
        await open(result, file, context);
    };
}

export function scalarValue(input, original) {
    if (typeof original === 'boolean') return input.checked;
    if (typeof original === 'number') {
        const number = Number(input.value);
        if (!Number.isFinite(number) || input.value.trim() === '') throw new Error('A finite number is required');
        return number;
    }
    // In particular, seed strings NEVER go through Number/parseInt.
    return input.value;
}
