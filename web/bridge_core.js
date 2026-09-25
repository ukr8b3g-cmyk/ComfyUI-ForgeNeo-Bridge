/** Pure UI helpers: graph construction is detached; settings use named widgets/ports. */
export const MAX_IMAGE = 128 * 1024 * 1024;
export const ALLOWED = new Set(['CheckpointLoaderSimple','UNETLoader','CLIPLoader','DualCLIPLoader','VAELoader','LoraLoader','LoraLoaderModelOnly','ForgeCompatSpec','ForgeCompatText','ForgeCompatNoise','ForgeCompatSampler','VAEDecode','VAEEncode','LoadImage','SaveImage']);

export function buildDetached(plan, LiteGraph, GraphClass) {
    if (!plan || plan.schema_version !== '1.0.0' || !Array.isArray(plan.nodes) || !Array.isArray(plan.edges)) throw new Error('Invalid graph plan');
    if (plan.nodes.length > 256 || plan.edges.length > 1024) throw new Error('Graph size limit exceeded');
    const graph = new GraphClass();
    const nodes = new Map();
    for (const item of plan.nodes) {
        if (!ALLOWED.has(item.type) || nodes.has(item.id)) throw new Error('Unknown or duplicate node');
        const node = LiteGraph.createNode(item.type);
        if (!node) throw new Error(`Missing node: ${item.type}`);
        node.title = item.title;
        graph.add(node); nodes.set(item.id, node);
        for (const [name, value] of Object.entries(item.values)) {
            const widget = node.widgets?.find(w => w.name === name);
            if (!widget) throw new Error(`Missing widget ${item.type}.${name}; frontend/backend versions do not match`);
            if (Array.isArray(widget.options?.values) && !widget.options.values.includes(value)) throw new Error(`Unavailable value: ${item.type}.${name}`);
            widget.value = value;
        }
    }
    for (const [from, slot, to, input] of plan.edges) {
        const src = nodes.get(from), dst = nodes.get(to);
        const index = dst?.inputs?.findIndex(x => x.name === input);
        if (!src || !dst || index == null || index < 0 || !src.outputs?.[slot]) throw new Error('Invalid graph edge');
        if (src.connect(slot, dst, index) == null) throw new Error(`Cannot connect ${src.type} → ${dst.type}.${input}`);
    }
    graph.arrange();
    const data = graph.serialize();
    data.extra ??= {};
    data.extra.forge_neo_bridge = {config_hash: plan.config_hash, qualification: 'not_evaluated'};
    return data;
}

export function makeDropHandler(original, inspect, open, owner) {
    return async function(file, ...args) {
        if (!(file instanceof File) || !/\.(png|webp|jpe?g)$/i.test(file.name)) return original.call(owner, file, ...args);
        if (file.size > MAX_IMAGE) throw new Error('ForgeNeo Bridge: file exceeds 128 MiB');
        const result = await inspect(file);
        if (result.kind !== 'forge') return original.call(owner, file, ...args);
        await open(result.spec, file);
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
