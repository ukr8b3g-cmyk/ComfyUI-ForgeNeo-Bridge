/** Presentation only: keep the original widgets and their saved value order. */
import {language, tr, SETTINGS_LABELS} from './i18n.js';
const EN_GROUPS = {text:'Text processing',noise:'Noise / variation',sampling:'Sampling details',
    sigma:'Sigma / schedule',cfg:'CFG options',img2img:'Img2img details',sdxl:'SDXL details',source:'Source information'};
const GROUPS = [
    ['text', 'テキスト処理', ['emphasis', 'comma_padding_backtrack']],
    ['noise', '乱数・バリエーション', ['rng', 'subseed', 'subseed_strength', 'seed_resize_width', 'seed_resize_height']],
    ['sampling', 'サンプリング詳細', ['shift', 'eta_ancestral', 'eta_ddim', 's_churn', 's_tmin', 's_tmax', 's_noise']],
    ['sigma', 'Sigma・スケジュール詳細', ['sigma_min', 'sigma_max', 'rho', 'beta_alpha', 'beta_beta', 'discard_penultimate', 'sgm_noise_multiplier']],
    ['cfg', 'CFG補助', ['skip_early_cfg', 'ngms', 'ngms_all_steps']],
    ['img2img', 'img2img詳細', ['img2img_step_mode', 'img2img_extra_noise']],
    ['sdxl', 'SDXL詳細', ['original_width', 'original_height', 'target_width', 'target_height', 'crop_x', 'crop_y', 'zero_empty_negative']],
    ['source', '読み込み情報', ['source_json']],
];
const COMMON = ['forge_compatibility', 'sampling_adjustments', 'clip_skip', 'ensd'];
const ADVANCED = ['family', 'mode'];
const DIMENSIONS = ['width', 'height', 'batch_size'];

export function installSettingsAccordion(node, app) {
    if ((node.comfyClass || node.type) !== 'ForgeNeoBridgeSettings' || !node.addWidget) return;
    const original = [...node.widgets];
    const byName = new Map(original.map(widget => [widget.name, widget]));
    const presentation = new Map(original.map(widget => [widget, {
        computeSize:widget.computeSize,
        elements:[...new Set([widget.element, widget.inputEl].filter(Boolean))]
            .map(element => [element, element.style?.display || '']),
    }]));
    const value = name => byName.get(name)?.value;
    let compact = false, currentLanguage;
    const state = () => {
        node.properties ??= {};
        node.properties.forge_neo_bridge ??= {};
        return node.properties.forge_neo_bridge;
    };
    const sections = () => state().sections ??= {};
    const setVisible = (widget, visible) => {
        if (!widget) return;
        widget.hidden = !visible;
        const saved = presentation.get(widget);
        if (!saved) return;
        widget.computeSize = visible ? saved.computeSize : () => [0, -4];
        for (const [element, display] of saved.elements)
            if (element.style) element.style.display = visible ? display : 'none';
    };
    const details = node.addWidget('button', '詳細設定', null, () => {
        node.graph?.beforeChange?.();
        sections().advanced = !sections().advanced;
        refresh();
        node.graph?.afterChange?.();
    }, {serialize:false});
    details.serialize = false;
    Object.defineProperty(details,'tooltip',{get:()=>tr(app,
        'モデル種別・モードと詳細設定を開閉します。閉じても設定値は保持されます。',
        'Expand or collapse model family, mode and advanced settings. Collapsing keeps all values.')});
    const groups = GROUPS.map(([key, title, names]) => {
        const header = node.addWidget('button', title, null, () => {
            node.graph?.beforeChange?.();
            sections()[key] = !sections()[key];
            refresh();
            node.graph?.afterChange?.();
        }, {serialize:false});
        header.serialize = false;
        Object.defineProperty(header,'tooltip',{get:()=>tr(app,
            `${title}の詳細設定を開閉します。${['sampling','sigma','cfg','img2img'].includes(key) ? 'サンプリング差分補正ONで適用されます。' : ''}`,
            `Expand or collapse ${EN_GROUPS[key]}. ${['sampling','sigma','cfg','img2img'].includes(key) ? 'Applied when Sampling adjustments is ON.' : ''}`)});
        return {key, title, header, widgets:names.map(name => byName.get(name)).filter(Boolean)};
    });
    const warning = node.addWidget('button', '読み込み警告', null, () => {
        node.graph?.beforeChange?.();
        sections().source = !sections().source;
        refresh();
        node.graph?.afterChange?.();
    }, {serialize:false});
    warning.serialize = false;
    Object.defineProperty(warning,'tooltip',{get:()=>tr(app,'読み込み情報を開閉します。警告は元の情報に関するもので、編集後の実行状態とは異なる場合があります。',
        'Expand or collapse source information. Import warnings describe the original data and may differ from the edited run.')});
    const used = new Set([...COMMON, ...ADVANCED, ...DIMENSIONS, ...GROUPS.flatMap(group => group[2])]);
    const display = [
        ...COMMON.map(name => byName.get(name)).filter(Boolean), warning,
        ...DIMENSIONS.map(name => byName.get(name)).filter(Boolean),
        details, ...ADVANCED.map(name => byName.get(name)).filter(Boolean),
        ...groups.flatMap(group => [group.header, ...group.widgets]),
        ...original.filter(widget => !used.has(widget.name)),
    ];
    node.widgets = display;
    for (const [element] of presentation.get(byName.get('source_json'))?.elements || [])
        if ('readOnly' in element) element.readOnly = true;

    function refresh() {
        currentLanguage = language(app);
        const compatible = value('forge_compatibility') !== false;
        const adjusted = compatible && value('sampling_adjustments') !== false;
        for (const widget of original) {
            const labels = SETTINGS_LABELS[widget.name];
            if (labels) widget.label = tr(app,...labels);
            if (widget.name === 'ensd' || widget.name === 'sampling_adjustments')
                widget.label += (widget.name === 'ensd' ? adjusted : compatible) ? '' : tr(app,'（適用なし）',' (inactive)');
        }
        for (const name of ['forge_compatibility','sampling_adjustments']) {
            const toggle = byName.get(name);
            if (toggle) toggle.options = {...toggle.options,on:'ON',off:'OFF'};
        }
        const open = sections();
        // Preserve explicitly expanded details in workflows saved before this parent existed.
        open.advanced ??= groups.some(g => g.key !== 'source' && open[g.key]);
        details.name = `${open.advanced ? '▼' : '▶'} ${tr(app,'詳細設定','Advanced settings')}`;
        for (const name of ADVANCED) setVisible(byName.get(name),!!open.advanced);
        for (const group of groups) {
            const applicable = group.key === 'source' || !!open.advanced &&
                (group.key === 'sdxl' ? value('family') === 'sdxl'
                : group.key === 'img2img' ? value('mode') === 'img2img' : true);
            group.header.hidden = !applicable;
            group.header.computeSize = applicable ? undefined : () => [0, -4];
            const title = tr(app,group.title,EN_GROUPS[group.key]);
            const active = group.key === 'source' || (['sampling','sigma','cfg','img2img'].includes(group.key) ? adjusted : compatible);
            group.header.name = `${open[group.key] ? '▼' : '▶'} ${title}${active ? '' : tr(app,'（適用なし）',' (inactive)')}`;
            for (const widget of group.widgets) setVisible(widget, applicable && !!open[group.key]);
        }
        // Old hand-built graphs may still use the settings dimensions as fallback.
        const hasLatent = (node.outputs?.[0]?.links || []).some(id => {
            const link = node.graph?.links?.get?.(id) ?? node.graph?.links?.[id];
            const sampler = node.graph?.getNodeById?.(link?.target_id);
            return sampler?.type === 'ForgeNeoBridgeKSampler'
                && sampler.inputs?.some(input => input.name === 'input_latent' && input.link != null);
        });
        for (const name of DIMENSIONS) setVisible(byName.get(name), !hasLatent);
        const issues = [...(state().unsupported || []), ...(state().unresolved || [])]
            .filter(issue => issue.code !== 'INFERRED_SETTING');
        warning.hidden = !issues.length;
        warning.computeSize = issues.length ? undefined : () => [0, -4];
        warning.name = tr(app,`⚠ 読み込み警告 ${issues.length}件`,`⚠ Import warnings: ${issues.length}`);
        if (node.computeSize && node.setSize) {
            const size = node.computeSize();
            node.setSize([compact ? Math.max(node.size?.[0] || 300,300) : 300, size[1]]);
            compact = true;
            state().compact_width = true;
        }
        (node.graph || app.graph)?.setDirtyCanvas?.(true, true);
    }
    for (const widget of original) {
        const callback = widget.callback;
        widget.callback = function (...args) {
            const result = callback?.apply(this, args);
            if (['forge_compatibility','sampling_adjustments'].includes(widget.name)) syncHiresMode(node,widget.name,widget.value);
            refresh();
            return result;
        };
    }
    // File compatibility uses backend widget order, independent of visual grouping.
    for (const method of ['serialize', 'configure']) {
        const previous = node[method];
        if (!previous) continue;
        node[method] = function (...args) {
            const savedCompact = args[0]?.properties?.forge_neo_bridge?.compact_width === true;
            const legacyTitle = method === 'configure' && /^ForgeNeo Bridge Settings \[/.test(args[0]?.title || '');
            this.widgets = original;
            try { return previous.apply(this, args); }
            finally {
                this.widgets = display;
                if (method === 'configure') {
                    const index = original.findIndex(w => w.name === 'sampling_adjustments');
                    // Old files predate this switch and used the adjusted path.
                    if (index >= 0 && Array.isArray(args[0]?.widgets_values) && args[0].widgets_values.length <= index)
                        original[index].value = true;
                    if (legacyTitle) this.title = 'ForgeNeo Bridge Settings';
                    compact = savedCompact && !legacyTitle;
                    refresh(); queueMicrotask(refresh);
                }
            }
        };
    }
    const onConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function (...args) {
        const result = onConnectionsChange?.apply(this, args);
        queueMicrotask(refresh);
        return result;
    };
    node.forgeRefreshSettings = refresh;
    const draw = node.onDrawForeground;
    node.onDrawForeground = function (...args) {
        if (currentLanguage !== language(app)) refresh();
        return draw?.apply(this,args);
    };
    refresh();
}

/** Only synchronize passes connected through the actual Hires latent/image path. */
function syncHiresMode(settings, name, value) {
    const graph = settings.graph;
    if (!graph) return;
    const links = graph.links instanceof Map ? [...graph.links.values()] : Object.values(graph.links || {});
    const seen = new Set([settings.id]), pending = [settings.id];
    const path = new Set(['ForgeNeoBridgeSettings','ForgeNeoBridgeKSampler','VAEDecode','VAEEncode','ImageScale','LatentUpscale']);
    while (pending.length) {
        const id = pending.pop();
        for (const link of links) {
            if (!['FORGE_SETTINGS','LATENT','IMAGE'].includes(link.type)) continue;
            const other = link.origin_id === id ? link.target_id : link.target_id === id ? link.origin_id : null;
            if (other == null || seen.has(other)) continue;
            seen.add(other);
            const node = graph.getNodeById(other);
            if (!path.has(node?.type)) continue;
            pending.push(other);
            if (node.type === 'ForgeNeoBridgeSettings') {
                const toggle = node.widgets?.find(w => w.name === name);
                if (toggle && !node.inputs?.some(i => i.name === toggle.name && i.link != null)) {
                    toggle.value = value;
                    node.forgeRefreshSettings?.();
                }
            }
        }
    }
}

export function watchSettingsLatent(node) {
    if ((node.comfyClass || node.type) !== 'ForgeNeoBridgeKSampler') return;
    const previous = node.onConnectionsChange;
    node.onConnectionsChange = function (...args) {
        const result = previous?.apply(this, args);
        queueMicrotask(() => {
            const id = this.inputs?.find(input => input.name === 'settings')?.link;
            const link = this.graph?.links?.get?.(id) ?? this.graph?.links?.[id];
            this.graph?.getNodeById?.(link?.origin_id)?.forgeRefreshSettings?.();
        });
        return result;
    };
}
