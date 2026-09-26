/** Standard numeric-widget UI, with exact uint64 strings at the backend boundary. */
const MAX_SEED = (1n << 64n) - 1n;
const MODES = ['fixed', 'increment', 'decrement', 'randomize'];

export function seedValue(value) {
    if (typeof value === 'number' && !Number.isSafeInteger(value))
        throw new Error('Seed must be entered as exact integer digits.');
    const text = String(value).trim();
    if (!/^-?\d+$/.test(text)) throw new Error('Seed must be an integer.');
    const seed = BigInt(text);
    return (seed < 0n ? 0n : seed > MAX_SEED ? MAX_SEED : seed).toString();
}

export function nextSeed(value, mode, random = crypto.getRandomValues.bind(crypto)) {
    const seed = BigInt(seedValue(value));
    if (mode === 'increment') return seedValue(seed + 1n);
    if (mode === 'decrement') return seedValue(seed - 1n);
    if (mode === 'randomize') {
        const words = random(new Uint32Array(2));
        return ((BigInt(words[0]) << 32n) | BigInt(words[1])).toString();
    }
    return seed.toString();
}

export function installSeedControls(node, app, widgets = globalThis.comfyAPI?.widgets) {
    if ((node.comfyClass || node.type) !== 'ForgeNeoBridgeKSampler') return;
    const old = node.widgets?.find(widget => widget.name === 'seed');
    if (!old || !widgets?.addValueControlWidgets) return;
    const index = node.widgets.indexOf(old);
    const seed = node.addWidget('number', 'seed', seedValue(old.value), old.callback,
        {...old.options, min:0, max:Number.MAX_SAFE_INTEGER, step:10, step2:1, precision:0, canvasOnly:true});
    node.widgets.splice(node.widgets.indexOf(seed), 1);
    node.widgets[index] = seed;
    seed.inputSpec = old.inputSpec;
    for (const input of node.inputs || []) if (input._widget === old) input._widget = seed;
    // Reuse the core number widget's theme, arrows and hit testing. Its arithmetic
    // and display must not coerce the stored decimal string through JS Number.
    Object.defineProperty(seed, '_displayValue', {get:() => seed.computedDisabled ? '' : String(seed.value)});
    seed.canIncrement = () => BigInt(seedValue(seed.value)) < MAX_SEED;
    seed.canDecrement = () => BigInt(seedValue(seed.value)) > 0n;
    seed.setValue = (value, context = {}) => {
        const next = seedValue(value), previous = seed.value;
        if (next === previous) return;
        const {e, canvas} = context;
        seed.value = next;
        seed.callback?.(next, canvas, node, canvas?.graph_mouse, e);
        node.onWidgetChanged?.('seed', next, previous, seed);
        node.graph?.incrementVersion?.();
        node.graph?.setDirtyCanvas?.(true, true);
    };
    seed.incrementValue = context => seed.setValue(nextSeed(seed.value, 'increment'), context);
    seed.decrementValue = context => seed.setValue(nextSeed(seed.value, 'decrement'), context);
    seed.onClick = context => {
        const {e, canvas} = context;
        const x = e.canvasX - node.pos[0], width = seed.width || node.size[0];
        if (x < 40) return seed.decrementValue(context);
        if (x > width - 40) return seed.incrementValue(context);
        canvas.prompt('Seed', String(seed.value), value => {
            try { seed.setValue(value, context); }
            catch (error) { window.alert(error.message); }
        }, e);
    };
    seed.onDrag = context => {
        const {e} = context;
        const x = e.canvasX - node.pos[0], width = seed.width || node.size[0];
        if ((x < 40 || x > width - 40) && x > -3 && x < width + 3) return;
        const delta = Math.trunc(e.deltaX || 0);
        if (delta) seed.setValue(BigInt(seedValue(seed.value)) + BigInt(delta), context);
    };
    // Core helper supplies the localized label and the standard four-option menu.
    const [control] = widgets.addValueControlWidgets(node, seed, 'fixed', {addFilterList:false});
    control.serialize = false;
    seed.linkedWidgets = [control];
    node.widgets.splice(node.widgets.indexOf(control), 1);
    node.widgets.splice(index + 1, 0, control);
    let queued = false;
    const runBefore = () => (app.extensionManager?.setting?.get?.('Comfy.WidgetControlMode')
        ?? app.ui?.settings?.getSettingValue?.('Comfy.WidgetControlMode')) === 'before';
    const advance = context => {
        if (context?.isPartialExecution || node.inputs?.some(input => input.widget?.name === 'seed' && input.link != null)) return;
        seed.setValue(nextSeed(seed.value, control.value));
    };
    control.beforeQueued = context => {
        if (runBefore() && queued) advance(context);
        queued = true;
    };
    control.afterQueued = context => { if (!runBefore()) advance(context); };
    // LiteGraph writes widgets_values by visual index, including holes for
    // nonserialized controls. Keep the original six-value file layout intact.
    const serialize = node.serialize;
    node.serialize = function (...args) {
        const displayed = this.widgets;
        this.widgets = displayed.filter(widget => widget !== control);
        try { return serialize.apply(this, args); }
        finally { this.widgets = displayed; }
    };
    const onSerialize = node.onSerialize;
    node.onSerialize = function (data) {
        const result = onSerialize?.call(this, data);
        data.properties = {...data.properties, forge_neo_bridge:{
            ...data.properties?.forge_neo_bridge, seed_control:control.value}};
        return result;
    };
    const onConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
        const result = onConfigure?.apply(this, args);
        const mode = this.properties?.forge_neo_bridge?.seed_control;
        control.value = MODES.includes(mode) ? mode : 'fixed';
        seed.value = seedValue(seed.value);
        queued = false;
        return result;
    };
    node.setSize?.(node.computeSize());
}
