import {language} from './i18n.js';

let translations;
/** Native widgets cache their creation-time tooltip. Read the current locale on hover. */
export async function loadHelp(api) {
    translations ??= api.fetchApi('/i18n').then(response => {
        if (!response.ok) throw new Error('ForgeNeo Bridge: could not load node help');
        return response.json();
    });
    return translations;
}

export function installHelp(node, app, catalog) {
    const name = node.comfyClass || node.type;
    if (!catalog?.en?.nodeDefs?.[name]) return;
    for (const widget of node.widgets || []) {
        if (!catalog.en.nodeDefs[name].inputs?.[widget.name]) continue;
        Object.defineProperty(widget,'tooltip',{configurable:true,get:()=>
            (catalog[language(app)] || catalog.en).nodeDefs[name].inputs[widget.name].tooltip});
    }
}
