"""Materialize one Forge run from the currently connected Comfy nodes."""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass
import torch

from .binding import GraphReader, model_family, verify_binding
from .spec import BridgeError, SAMPLERS, SCHEDULERS, default_document, finalize, parse_json, pointer_get, pointer_set, provenance, report


@dataclass(frozen=True, slots=True)
class ForgeTextInput:
    text: str
    clip: object
    node_id: str


@dataclass(frozen=True, slots=True)
class ForgeSettings:
    values: dict


def _asset(role, category, name, ident, component=None):
    return {'asset_id':ident,'role':role,'component':component,'category':category,'relative_name':name,
            'sha256':None,'short_hash':None,'resolution':'resolved','binding_verification':'known_loader_chain'}


def _resources(model_chain, clip_chain):
    roots_m = [x for x in model_chain if x['category'] != 'loras']
    roots_c = [x for x in clip_chain if x['category'] != 'loras']
    if len(roots_m) != 1 or not 1 <= len(roots_c) <= 2:
        raise BridgeError('ASSET_BINDING_MISMATCH','Expected one model and one or two text encoders')
    assets = [_asset('model',roots_m[0]['category'],roots_m[0]['name'],'model')]
    assets += [_asset('text_encoder',x['category'],x['name'],f'text_encoder_{i}',x.get('component')) for i,x in enumerate(roots_c)]
    model_loras = [x for x in model_chain if x['category'] == 'loras']
    clip_loras = [x for x in clip_chain if x['category'] == 'loras']
    if [x['name'] for x in model_loras if x['strength']] != [x['name'] for x in clip_loras if x['strength']]:
        # A zero-strength branch is valid; compare the complete loader order below.
        if [x['name'] for x in model_loras] != [x['name'] for x in clip_loras]:
            raise BridgeError('ASSET_BINDING_MISMATCH','MODEL and CLIP use different LoRA order')
    if [x['name'] for x in model_loras] != [x['name'] for x in clip_loras]:
        raise BridgeError('ASSET_BINDING_MISMATCH','MODEL and CLIP use different LoRA loaders')
    loras = []
    for i,(m,c) in enumerate(zip(model_loras,clip_loras)):
        ident = f'lora_{i}'
        assets.append(_asset('lora','loras',m['name'],ident))
        loras.append({'asset_id':ident,'strength_model':m['strength'],'strength_text_encoder':c['strength'],'order':i})
    return assets,loras


def _optional_float(value, fallback):
    if value in ('auto','',None):return fallback
    try:return float(value)
    except (ValueError,TypeError,OverflowError) as exc:raise BridgeError('INVALID_SPEC',f'Expected a number or auto: {value}') from exc


def _clip_skip(value, fallback):
    if value in ('auto','',None):return fallback
    if not re.fullmatch(r'[1-9][0-9]*',str(value)):
        raise BridgeError('INVALID_SPEC','Clip Skip must be auto or a positive integer')
    return int(value)


def _image_from_latent(latent):
    samples = latent.get('samples') if isinstance(latent,dict) else None
    if not isinstance(samples,torch.Tensor) or samples.is_nested or samples.ndim not in (4,5) or (samples.ndim == 5 and samples.shape[2] != 1):
        raise BridgeError('LATENT_MISMATCH','Expected a single-image latent batch')
    ratio = latent.get('downscale_ratio_spacial',8)
    if type(ratio) is not int or ratio <= 0:
        raise BridgeError('LATENT_MISMATCH','Invalid latent spatial scale')
    return {'width':int(samples.shape[-1])*ratio,'height':int(samples.shape[-2])*ratio,
            'batch_size':int(samples.shape[0])}


def build_execution_spec(settings, sampler_values, positive, negative, model, reader, node_id, input_latent=None):
    if not isinstance(settings, ForgeSettings) or not isinstance(positive, ForgeTextInput) or not isinstance(negative, ForgeTextInput):
        raise BridgeError('INVALID_SPEC','Expected ForgeNeo Bridge Settings and Text Encode nodes')
    family = model_family(model)
    values = settings.values
    if values['family'] not in ('auto', family):raise BridgeError('MODEL_FAMILY_MISMATCH',f"Selected {values['family']}, loaded {family}")
    sampler_values = dict(sampler_values, sampler_name={'uni_pc':'unipc'}.get(sampler_values['sampler_name'],sampler_values['sampler_name']))
    scheduler = {'ddim_uniform':'ddim'}.get(sampler_values['scheduler'],sampler_values['scheduler'])
    if sampler_values['sampler_name'] not in SAMPLERS:
        raise BridgeError('UNSUPPORTED_COMBINATION',f"Forge compatibility does not support sampler {sampler_values['sampler_name']}. Choose a supported sampler or turn Forge compatibility OFF.")
    if scheduler not in SCHEDULERS:
        raise BridgeError('UNSUPPORTED_COMBINATION',f"Forge compatibility does not support scheduler {scheduler}. Choose a supported scheduler or turn Forge compatibility OFF.")
    if re.search(r'<(?:lora|lyco):[^>]+>',positive.text+' '+negative.text,re.IGNORECASE):
        raise BridgeError('UNSUPPORTED_COMBINATION','Add LoRA with a connected LoraLoader; prompt tags cannot load a model implicitly')
    source = parse_json(values['source_json'])
    if not isinstance(source, dict):raise BridgeError('INVALID_SPEC','Invalid source data')
    doc = default_document(family, policy='exploratory')
    cfg = doc['effective']
    doc['execution_scope'] = 'sampling'
    doc['provenance'] = {f'/{key}':provenance('profile_default',False) for key in cfg}
    if source.get('schema_version') == '1.0.0':
        original = source['effective']
        for path, raw_value in source['requested'].items():
            if path.startswith(('/assets','/loras','/family','/prediction_type','/mode','/sdxl')):continue
            try:pointer_set(cfg,path,pointer_get(original,path))
            except BridgeError:continue
            doc['requested'][path] = raw_value
            doc['provenance'][path] = source['provenance'].get(path,provenance('metadata_explicit'))
        doc['source'] = copy.deepcopy(source['source'])
        doc['extensions'] = copy.deepcopy(source['extensions'])
        doc['unresolved'] = [x for x in source['unresolved'] if x['code'] not in ('ASSET_BINDING_REQUIRED','ASSET_MISSING','FAMILY_UNRESOLVED','INFERRED_SETTING')]
        doc['unsupported'] = [x for x in source['unsupported'] if x['path'] not in ('/sampling/sampler','/sampling/scheduler','/text/positive_raw')]
    for key, text_input in (('positive',positive),('negative',negative)):
        linked_id,slot = reader.input(node_id,key)
        if str(linked_id) != text_input.node_id or slot != 0 or reader.get(text_input.node_id)['class_type'] != 'ForgeNeoBridgeTextEncode':
            raise BridgeError('UNVERIFIED_PATCH_CHAIN',f'{key} must come from the connected ForgeNeo Bridge Text Encode')
    settings_id, settings_slot = reader.input(node_id,'settings')
    if settings_slot != 0 or reader.get(settings_id)['class_type'] != 'ForgeNeoBridgeSettings':
        raise BridgeError('UNVERIFIED_PATCH_CHAIN','Settings must come from ForgeNeo Bridge Settings')
    model_chain = reader.trace(reader.input(node_id,'model'),'model')
    p_chain = reader.trace(reader.input(positive.node_id,'clip'),'text_encoder')
    n_chain = reader.trace(reader.input(negative.node_id,'clip'),'text_encoder')
    if p_chain != n_chain or positive.clip is not negative.clip:
        raise BridgeError('ASSET_BINDING_MISMATCH','Positive and negative must use the same CLIP/LoRA chain')
    cfg['assets'],cfg['loras'] = _resources(model_chain,p_chain)
    cfg['mode'] = values['mode']
    cfg['image'].update(width=values['width'],height=values['height'],batch_size=values['batch_size'])
    if input_latent is not None:cfg['image'].update(_image_from_latent(input_latent))
    cfg['text'].update(positive_raw=positive.text,negative_raw=negative.text,emphasis=values['emphasis'],
                       clip_skip=_clip_skip(values['clip_skip'],cfg['text']['clip_skip']),
                       comma_padding_backtrack=values['comma_padding_backtrack'])
    cfg['noise'].update(seed=sampler_values['seed'],source=values['rng'],ensd=values['ensd'],
                        subseed=values['subseed'],subseed_strength=values['subseed_strength'],
                        seed_resize_from={'width':values['seed_resize_width'],'height':values['seed_resize_height']})
    cfg['sampling'].update(sampler=sampler_values['sampler_name'],scheduler=scheduler,
                           adjustments=values.get('sampling_adjustments',True),
                           steps=sampler_values['steps'],cfg=sampler_values['cfg'],denoise=sampler_values['denoise'],
                           shift=_optional_float(values['shift'],cfg['sampling']['shift']),
                           eta_ancestral=values['eta_ancestral'],eta_ddim=values['eta_ddim'],
                           s_churn=values['s_churn'],s_tmin=values['s_tmin'],s_tmax=values['s_tmax'],
                           s_noise=values['s_noise'],
                           sigma_min=_optional_float(values['sigma_min'],None),
                           sigma_max=_optional_float(values['sigma_max'],None),
                           rho=_optional_float(values['rho'],None),
                           beta_alpha=values['beta_alpha'],beta_beta=values['beta_beta'],
                           discard_penultimate_requested=values['discard_penultimate'],
                           sgm_noise_multiplier_requested=values['sgm_noise_multiplier'],
                           img2img_step_mode=values['img2img_step_mode'],
                           img2img_extra_noise=values['img2img_extra_noise'])
    cfg['guidance'].update(skip_early_cfg=values['skip_early_cfg'],ngms=values['ngms'],
                           ngms_all_steps=values['ngms_all_steps'])
    if family == 'sdxl':
        for key in ('original_width','target_width'):
            cfg['sdxl'][key] = values[key] or cfg['image']['width']
        for key in ('original_height','target_height'):
            cfg['sdxl'][key] = values[key] or cfg['image']['height']
        cfg['sdxl'].update(crop_left=values['crop_x'],crop_top=values['crop_y'],
                           zero_empty_negative=values['zero_empty_negative'])
    for path in ('/assets','/loras','/mode','/image','/text','/noise','/sampling','/guidance'):
        doc['provenance'][path] = provenance('user_override')
        doc['requested'][path] = copy.deepcopy(pointer_get(cfg,path))
    # Source-requested values may have been edited in their widgets since drop.
    # Preserve the original only in source_json; report this run's current values.
    for path in tuple(doc['requested']):
        doc['requested'][path] = copy.deepcopy(pointer_get(cfg,path))
        doc['provenance'][path] = provenance('user_override')
    return finalize(doc)


def sample_workflow(model, positive, negative, settings, sampler_values, input_latent=None,
                    prompt=None, unique_id=None, dynprompt=None):
    if isinstance(settings, ForgeSettings) and settings.values.get('forge_compatibility', True) is False:
        from .comfy_mode import sample_comfy
        return sample_comfy(model,positive,negative,settings,sampler_values,input_latent)
    from .noise import generate_noise
    from .runtime import execute
    from .text import encode_bundle
    import latent_preview

    reader = GraphReader(prompt,dynprompt)
    spec = build_execution_spec(settings,sampler_values,positive,negative,model,reader,unique_id,input_latent)
    cfg = spec.require_executable()
    model_binding = verify_binding(cfg,reader,unique_id,'model','model')
    text_binding = verify_binding(cfg,reader,positive.node_id,'clip','text_encoder')
    conditioning = encode_bundle(spec,positive.clip,text_binding)
    latent,noise = generate_noise(spec,model,model_binding,input_latent)
    callback = latent_preview.prepare_callback(model,cfg['sampling']['steps'])
    result,_,facts,_ = execute(spec,model,conditioning,noise,latent,model_binding,'summary',callback)
    return {'ui':{'text':[report('ForgeNeoBridgeKSampler',spec,**facts)]},'result':(result,)}
