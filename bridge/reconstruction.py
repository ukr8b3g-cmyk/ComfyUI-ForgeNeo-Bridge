"""Turn Forge infotext into an editable graph without loading model weights."""
from __future__ import annotations

import math
import re
import struct

from .infotext import import_infotext, parse_loras
from .spec import BridgeError, canonical, finalize

NATIVE_FAMILIES = frozenset(('ernie', 'zimage', 'qwen', 'qwen21', 'krea2',
                             'flux1', 'flux2_klein', 'flux2_dev'))
UNSUPPORTED_FAMILIES = frozenset(('flux2', 'lumina2', 'pid', 'wan'))


def _family_label(value):
    label = re.sub(r'[^a-z0-9]+', '', str(value).casefold())
    if 'ernie' in label:return 'ernie'
    if 'zimage' in label or label.startswith('zit'):return 'zimage'
    if 'qwenimage21' in label:return 'qwen21'
    if 'qwenimage' in label:return 'qwen'
    if 'krea2' in label or label.startswith('k2turbo'):return 'krea2'
    if 'klein' in label:return 'flux2_klein'
    if 'flux2' in label:return 'flux2_dev' if 'dev' in label else 'flux2'
    if 'flux1' in label or label.startswith('fluxdev') or label.startswith('fluxschnell'):return 'flux1'
    if 'anima' in label:return 'anima'
    if 'sdxl' in label:return 'sdxl'
    if 'sd15' in label or 'stable diffusion 1.5' in str(value).casefold():return 'sd15'
    if 'lumina' in label:return 'lumina2'
    if label.startswith('pid'):return 'pid'
    if label.startswith('wan'):return 'wan'
    return None


def _name(value):
    return str(value or '').replace('\\', '/').split('/')[-1]


def resolve_asset(raw, categories, inventory, roots=None):
    name = str(raw or '').replace('\\', '/')
    basename = _name(name)
    if roots is None:
        try:
            import folder_paths
            roots = {category:folder_paths.get_folder_paths(category) for category in categories}
        except ImportError:roots = {}
    for category in categories:
        for root in roots.get(category, ()):
            prefix = str(root).replace('\\','/').rstrip('/') + '/'
            if not name.casefold().startswith(prefix.casefold()):continue
            relative = name[len(prefix):]
            found = [candidate for candidate in inventory.get(category,())
                     if candidate.replace('\\','/').casefold() == relative.casefold()]
            if len(found) == 1:
                return {'category':category,'name':found[0],'resolution':'resolved','raw_name':raw}
    for match in (
        lambda candidate: candidate.replace('\\', '/').casefold() == name.casefold(),
        lambda candidate: _name(candidate).casefold() == basename.casefold(),
        lambda candidate: _name(candidate).rsplit('.', 1)[0].casefold() == basename.casefold(),
    ):
        found = [(category, candidate) for category in categories for candidate in inventory.get(category, ()) if match(candidate)]
        if found:
            if len(found) == 1:
                category, candidate = found[0]
                return {'category': category, 'name': candidate, 'resolution': 'resolved', 'raw_name': raw}
            return {'category': None, 'name': basename, 'resolution': 'ambiguous', 'raw_name': raw}
    return {'category': None, 'name': basename, 'resolution': 'missing', 'raw_name': raw}


def probe_family(category, name):
    """Read only a bounded safetensors header; never deserialize checkpoint weights."""
    if not name.lower().endswith('.safetensors'):return None
    try:
        import comfy.utils
        import folder_paths
        path = folder_paths.get_full_path(category,name)
        if path is None:return None
        raw = comfy.utils.safetensors_header(path,max_size=16 * 1024 * 1024)
        if raw is None:return None
        import json
        header = json.loads(raw)
        keys = tuple(header)
        def has(suffix):return any(key.endswith(suffix) for key in keys)
        if has('layers.0.mlp.linear_fc2.weight') and has('adaLN_modulation.1.weight'):
            return 'ernie'
        if has('txtfusion.projector.weight') and has('first.weight'):
            return 'krea2'
        if all(has(key) for key in ('txt_in.text_norm.weight', 'modulation.1.weight',
                                    'transformer_blocks.0.attn.norm_q.weight', 'img_in.weight', 'proj_out.weight')):
            return 'qwen21'
        if has('txt_norm.weight') and has('img_in.weight'):
            return 'qwen'
        if has('double_stream_modulation_img.lin.weight') and (
                has('double_blocks.0.img_attn.norm.key_norm.scale') or
                has('double_blocks.0.img_attn.norm.key_norm.weight')):
            return 'flux2'
        if has('double_blocks.0.img_attn.norm.key_norm.weight') and has('img_in.weight'):
            return 'flux1'
        if has('cap_embedder.1.weight') and has('noise_refiner.0.attention.k_norm.weight'):
            key = next((k for k in keys if k.endswith('cap_embedder.1.weight')), None)
            if key and header[key].get('shape', [0])[0] == 3840:return 'zimage'
        if any(k.endswith('llm_adapter.blocks.0.cross_attn.q_proj.weight') for k in keys) and any(k.endswith('blocks.0.mlp.layer1.weight') for k in keys):
            return 'anima'
        for key in keys:
            if key.endswith('add_embedding.linear_1.weight') and header[key].get('shape',[0,0])[-1] == 2816:
                return 'sdxl'
        if any(k.endswith('model.diffusion_model.input_blocks.0.0.weight') for k in keys) and any(k.endswith('cond_stage_model.transformer.text_model.embeddings.token_embedding.weight') for k in keys):
            return 'sd15'
    except (OSError,ValueError,TypeError,KeyError,ImportError,IndexError,struct.error):
        return None
    return None


def infer_family(fields, assets):
    for field in fields:
        if field['key'].casefold() in ('model type', 'architecture', 'model architecture'):
            family = _family_label(field['value'])
            if family:return family
    model = next((a['name'] for a in assets if a['key'].casefold() == 'model'), '')
    modules = ' '.join(a['name'] for a in assets if a['key'].casefold().startswith('module ')).casefold()
    family = _family_label(model)
    if family:return family
    if any(x in modules for x in ('qwen_3_06b','qwen3_06b')):return 'anima'
    return None


def _default_module(family, role, model_name, inventory):
    klein_9b = '9b' in model_name.casefold()
    defaults = {
        'ernie':('ministral-3-3b', 'flux2-vae'),
        'zimage':('qwen_3_4b', 'ae'),
        'qwen':('qwen_2.5_vl_7b_fp8_scaled', 'qwen_image_vae'),
        'qwen21':('qwen3vl_8b_int8_convrot', 'qwen_image_2.1_vae_bf16'),
        'krea2':('qwen3vl_4b_fp8_scaled', 'qwen_image_vae'),
        'flux1':('clip_l', 'ae'),
        'flux2_dev':('mistral_3_small_flux2_bf16', 'flux2-vae'),
        'flux2_klein':('qwen_3_8b_fp8mixed' if klein_9b else 'qwen_3_4b', 'flux2-vae'),
    }
    if family not in defaults:return None
    category = 'vae' if role == 'vae' else 'text_encoders'
    name = defaults[family][1 if role == 'vae' else 0]
    if family == 'flux2_klein' and klein_9b and role == 'vae':
        small_decoder = resolve_asset('full_encoder_small_decoder', ('vae',), inventory)
        if small_decoder['resolution'] == 'resolved':
            small_decoder['category'] = 'vae'
            return small_decoder
    asset = resolve_asset(name, (category,), inventory)
    asset['category'] = category
    return asset


def _component_shape(asset, suffix):
    if asset['resolution'] != 'resolved' or not asset['name'].lower().endswith('.safetensors'):
        return None
    try:
        import comfy.utils
        import folder_paths
        path = folder_paths.get_full_path(asset['category'], asset['name'])
        if path is None:return None
        import json
        header = json.loads(comfy.utils.safetensors_header(path, max_size=16 * 1024 * 1024))
        return next((value['shape'] for key, value in header.items() if key.endswith(suffix)), None)
    except (OSError, ValueError, TypeError, KeyError, ImportError, StopIteration, struct.error):
        return None


def _klein_component_matches(asset, role, model_name):
    if not asset:return False
    name = _name(asset['name']).casefold()
    if role == 'clip':
        expected_width = 4096 if '9b' in model_name.casefold() else 2560
        shape = _component_shape(asset, 'model.embed_tokens.weight')
        if shape:return shape[-1] == expected_width
        return bool(re.search(r'qwen[_-]?3[_-]?(8b|4b)', name) and
                    (('8b' in name) == (expected_width == 4096)))
    shape = _component_shape(asset, 'decoder.conv_in.weight')
    if shape:return len(shape) > 1 and shape[1] == 32
    return 'flux2' in name or name.startswith('full_encoder_small_decoder')


def reconstruct(text, inventory, probe=probe_family, *, image_size=None):
    p = text.rfind('\nSteps:')
    if p < 0:raise BridgeError('NOT_FORGE_METADATA', 'Missing generation parameters')
    from .infotext import split_fields
    fields = split_fields(text[p + 1:])
    raw_assets = [{'key':f['key'],'name':f['value']} for f in fields
                  if f['key'].casefold() in ('model', 'vae') or re.fullmatch(r'module \d+', f['key'].casefold())]
    family = infer_family(fields, raw_assets)
    model_raw = next((a['name'] for a in raw_assets if a['key'].casefold() == 'model'), '')
    model = resolve_asset(model_raw, ('diffusion_models', 'checkpoints'), inventory)
    if model['resolution'] == 'ambiguous' and family in NATIVE_FAMILIES | {'anima','sd15','sdxl'}:
        preferred = 'checkpoints' if family in ('sd15','sdxl') else 'diffusion_models'
        candidate = resolve_asset(model_raw, (preferred,), inventory)
        if candidate['resolution'] == 'resolved':model = candidate
    probed = probe(model['category'],model['name']) if model['resolution'] == 'resolved' else None
    if probed:
        family = family if probed == 'flux2' and family in ('flux2_klein','flux2_dev') else probed
    if family in UNSUPPORTED_FAMILIES:
        raise BridgeError('FAMILY_RECIPE_UNAVAILABLE', f'{family} has no verified image workflow recipe')
    native = family in NATIVE_FAMILIES
    if native or family == 'anima':
        if model['category'] != 'diffusion_models':
            model['category'] = 'diffusion_models'
            model['resolution'] = 'missing'
    elif model['category'] is None:
        model['category'] = 'checkpoints'
    doc = import_infotext(text, family=family if family in ('anima','sd15','sdxl') else 'sd15',
                          policy='exploratory', tolerant=True).document()
    cfg = doc['effective']
    vae_raw = next((a['name'] for a in raw_assets if a['key'].casefold() == 'vae'), '')
    vae = resolve_asset(vae_raw, ('vae',), inventory) if vae_raw else None
    clip = None;clip_secondary = None;unassigned_modules = []
    for item in raw_assets:
        if not item['key'].casefold().startswith('module '):continue
        candidate = resolve_asset(item['name'], ('text_encoders', 'vae'), inventory)
        label = _name(item['name']).casefold()
        if candidate['category'] == 'vae' or (candidate['category'] is None and 'vae' in label):
            if vae is None:vae = candidate;vae['category'] = 'vae'
            elif _name(vae['raw_name']).casefold() != _name(item['name']).casefold():unassigned_modules.append(item)
        elif candidate['category'] == 'text_encoders' or (candidate['category'] is None and any(x in label for x in (
                'qwen', 'ministral', 'mistral', 't5xxl', 'text_encoder', 'clip'))):
            if clip is None:clip = candidate;clip['category'] = 'text_encoders'
            elif native and family == 'flux1' and clip_secondary is None:
                clip_secondary = candidate;clip_secondary['category'] = 'text_encoders'
            elif _name(clip['raw_name']).casefold() != _name(item['name']).casefold():unassigned_modules.append(item)
        else:unassigned_modules.append(item)
    if family == 'flux1' and clip is not None and 't5' in _name(clip['name']).casefold():
        clip, clip_secondary = clip_secondary, clip
    corrections = []
    if family == 'flux2_klein':
        for role in ('clip', 'vae'):
            original = clip if role == 'clip' else vae
            if original and not _klein_component_matches(original, role, model['name']):
                replacement = _default_module(family, role, model['name'], inventory)
                replacement['raw_name'] = original['raw_name']
                corrections.append({'role':role, 'source':original['raw_name'],
                                    'selected':replacement['name']})
                if role == 'clip':clip = replacement
                else:vae = replacement
    if clip is None and native:clip = _default_module(family,'clip',model['name'],inventory)
    if vae is None and native:vae = _default_module(family,'vae',model['name'],inventory)
    if family == 'flux1' and clip_secondary is None:
        clip_secondary = resolve_asset('t5xxl_fp16', ('text_encoders',), inventory)
        for name in ('t5xxl_fp16', 't5xxl_fp8_e4m3fn', 't5xxl_fp8_e4m3fn_scaled', 't5xxl'):
            candidate = resolve_asset(name, ('text_encoders',), inventory)
            if candidate['resolution'] == 'resolved':
                clip_secondary = candidate
                break
        clip_secondary['category'] = 'text_encoders'
    if clip is None and model['category'] != 'checkpoints':clip = resolve_asset('', ('text_encoders',), inventory);clip['category'] = 'text_encoders'
    if vae is None and model['category'] != 'checkpoints':vae = resolve_asset('', ('vae',), inventory);vae['category'] = 'vae'
    if family is None:doc['unresolved'].append({'code':'FAMILY_UNRESOLVED','path':'/family','message':'Model family will be checked when a model is loaded'})
    # The shipped example specs contain demonstration prompts and seed values. Never
    # mistake those for metadata when a field was absent from the dropped image.
    defaults = {'/noise/seed':'0','/sampling/scheduler':'automatic'} if not native else {}
    if family is None:defaults.update({'/image/width':512,'/image/height':512,'/sampling/steps':20,'/sampling/cfg':7.0})
    from .spec import pointer_set
    for path,value in defaults.items():
        if path not in doc['requested']:pointer_set(cfg,path,value)
    if native:
        from .native_reconstruction import apply_defaults
        apply_defaults(cfg,doc['requested'],family,model['name'])
    # Header evidence can correct a filename-based guess. Rebuild with the right
    # profile while retaining observed settings and provenance.
    if family in ('anima','sd15','sdxl') and cfg['family'] != family:
        from .spec import default_document, pointer_get
        replacement = default_document(family,'exploratory')
        for path in doc['requested']:
            try:pointer_set(replacement['effective'],path,pointer_get(cfg,path))
            except BridgeError:pass
        if family == 'sdxl':
            for key in ('original_width','target_width'):
                replacement['effective']['sdxl'][key] = replacement['effective']['image']['width']
            for key in ('original_height','target_height'):
                replacement['effective']['sdxl'][key] = replacement['effective']['image']['height']
        replacement.update({k:doc[k] for k in ('source','requested','provenance','unresolved','unsupported','extensions')})
        doc = replacement;cfg = doc['effective']
    doc = finalize(doc).document();cfg = doc['effective']
    for item in (model, clip, clip_secondary, vae):
        if item and item['resolution'] != 'resolved':
            doc['unresolved'].append({'code':'ASSET_MISSING','path':'/assets','message':f"{item['category']}: {item['raw_name'] or 'not specified'} ({item['resolution']})"})
    for item in unassigned_modules:
        doc['unresolved'].append({'code':'UNKNOWN_PARAMETER','path':'/extensions',
                                  'message':f"Unassigned {item['key']}: {item['name']}"})
    for item in corrections:
        doc['unresolved'].append({'code':'MODULE_MISMATCH_CORRECTED','path':'/assets',
                                  'message':f"Flux.2 Klein {item['role']}: {item['source']} -> {item['selected']}"})
    try:cleaned, loras = parse_loras(cfg['text']['positive_raw'])
    except BridgeError:cleaned, loras = cfg['text']['positive_raw'], []
    doc['extensions']['reconstruction'] = {'family':family, 'family_evidence':'safetensors_header' if probed else 'metadata_or_name' if family else 'unknown',
                                             'model':model, 'clip':clip, 'clip_secondary':clip_secondary,
                                             'vae':vae, 'loras':loras,
                                             'unassigned_modules':unassigned_modules,
                                             'component_corrections':corrections}
    if image_size is not None:
        if len(image_size) != 2 or any(type(v) is not int or v <= 0 for v in image_size):
            raise BridgeError('INVALID_METADATA', 'Invalid source image dimensions')
        doc['extensions']['reconstruction']['source_image_size'] = list(image_size)
    if native:
        doc['extensions']['reconstruction']['native_config'] = {
            'family':family, **{key:cfg[key] for key in ('mode','image','noise','sampling','text')}}
        # The old GenerationSpec only represents SD1.5/SDXL/Anima. Do not return
        # a misleading SD1.5 effective configuration for an ERNIE/Flux/etc graph.
        doc['effective'] = None
        doc['config_hash'] = None
        doc['status'] = 'needs_review'
        doc['unresolved'] = [item for item in doc['unresolved']
                             if item['code'] not in ('ASSET_BINDING_REQUIRED','INFERRED_SETTING')]
        binding = doc['extensions']['reconstruction']
        binding['notes'] = []
        if '/text/clip_skip' in doc['requested']:
            binding['notes'].append('Recorded Clip Skip is not applied: Forge uses no layer skip for this model conditioning.')
        if cfg['mode'] == 'img2img':
            binding['notes'].append('The original input/reference image is not embedded in this generated image. Select it in LoadImage.')
        if family == 'flux1':
            raw_guidance = next((f['value'] for f in fields if f['key'].casefold() == 'distilled cfg scale'), '3.5')
            try:binding['distilled_cfg'] = float(raw_guidance)
            except ValueError as exc:raise BridgeError('INVALID_METADATA', 'Invalid Distilled CFG Scale') from exc
            if not math.isfinite(binding['distilled_cfg']):
                raise BridgeError('INVALID_METADATA', 'Distilled CFG Scale must be finite')
            doc['unresolved'] = [x for x in doc['unresolved'] if x['message'].casefold() != 'review unhandled parameter distilled cfg scale']
        from .native_reconstruction import CORE_SAMPLERS, CORE_SCHEDULERS, core_sampling
        sampler, scheduler = core_sampling(cfg['sampling'])
        if sampler in CORE_SAMPLERS and (scheduler in CORE_SCHEDULERS or family in ('flux2_klein','flux2_dev')):
            doc['unsupported'] = [item for item in doc['unsupported'] if not (
                item['code'] == 'UNSUPPORTED_COMBINATION' and item['path'] == '/sampling' and
                item['message'] == 'Registered inventory item has no V1 executable adapter.')]
    return doc, cleaned


def reconstruction_plan(doc, positive, inventory):
    cfg = doc['effective']; binding = doc['extensions']['reconstruction']
    if binding['family'] in NATIVE_FAMILIES:
        from .native_reconstruction import native_plan
        return native_plan(doc,positive,inventory)
    from .hires import hires_recipe, pass_document
    original = doc
    hires = hires_recipe(doc)
    if hires:
        doc = pass_document(doc,hires)
        cfg = doc['effective']
    nodes, edges = [], []
    def add(kind, values=None, title=None, properties=None):
        ident = str(len(nodes) + 1)
        node = {'id':ident,'type':kind,'values':values or {},'properties':properties or {}}
        if title is not None:node['title'] = title
        nodes.append(node)
        return ident
    def connect(source, slot, target, key):edges.append([source,slot,target,key])
    def resource(a):return {'forge_neo_bridge':{'raw_name':a['raw_name'],'current_name':a['name'],
                                                'resolution':a['resolution']}}
    model_asset = binding['model']
    if model_asset['category'] == 'checkpoints':
        checkpoint = add('CheckpointLoaderSimple',{'ckpt_name':model_asset['name']},properties=resource(model_asset))
        model, clip, vae = (checkpoint,0), (checkpoint,1), (checkpoint,2)
    else:
        model = (add('UNETLoader',{'unet_name':model_asset['name'],'weight_dtype':'default'},
                     properties=resource(model_asset)),0)
        clip = vae = None
    if binding['clip']:
        a = binding['clip'];clip = (add('CLIPLoader',{'clip_name':a['name'],'type':'stable_diffusion','device':'default'},
                                    properties=resource(a)),0)
    if binding['vae']:
        a = binding['vae'];vae = (add('VAELoader',{'vae_name':a['name']},properties=resource(a)),0)
    if clip is None:
        a = resolve_asset('', ('text_encoders',), inventory);a['category'] = 'text_encoders'
        clip = (add('CLIPLoader',{'clip_name':'','type':'stable_diffusion','device':'default'},
                    properties=resource(a)),0)
    if vae is None:
        a = resolve_asset('', ('vae',), inventory);a['category'] = 'vae'
        vae = (add('VAELoader',{'vae_name':''},properties=resource(a)),0)
    for entry in binding['loras']:
        a = resolve_asset(entry['name'], ('loras',), inventory)
        node = add('LoraLoader',{'lora_name':a['name'],'strength_model':entry['strength_model'],'strength_clip':entry['strength_text_encoder']},
                   properties=resource(a))
        connect(*model,node,'model');connect(*clip,node,'clip')
        model,clip = (node,0),(node,1)
    noise,sampling,text = cfg['noise'],cfg['sampling'],cfg['text']
    values = {'family':binding['family'] or 'auto','mode':cfg['mode'] if cfg['mode'] in ('txt2img','img2img') else 'txt2img',
              'width':cfg['image']['width'],'height':cfg['image']['height'],'batch_size':cfg['image']['batch_size'],
              'rng':noise['source'],'ensd':noise['ensd'],'subseed':noise['subseed'],'subseed_strength':noise['subseed_strength'],
              'emphasis':text['emphasis'],
              'clip_skip':str(text['clip_skip']) if '/text/clip_skip' in doc['requested'] and text['clip_skip'] is not None else 'auto',
              'comma_padding_backtrack':text['comma_padding_backtrack'],
              'seed_resize_width':noise['seed_resize_from']['width'],'seed_resize_height':noise['seed_resize_from']['height'],
              'shift':str(sampling['shift']) if '/sampling/shift' in doc['requested'] and sampling['shift'] is not None else 'auto',
              'eta_ancestral':sampling['eta_ancestral'],'eta_ddim':sampling['eta_ddim'],
              's_churn':sampling['s_churn'],'s_tmin':sampling['s_tmin'],'s_tmax':sampling['s_tmax'],
              's_noise':sampling['s_noise'],
              'sigma_min':str(sampling['sigma_min']) if sampling['sigma_min'] is not None else 'auto',
              'sigma_max':str(sampling['sigma_max']) if sampling['sigma_max'] is not None else 'auto',
              'rho':str(sampling['rho']) if sampling['rho'] is not None else 'auto',
              'beta_alpha':sampling['beta_alpha'],'beta_beta':sampling['beta_beta'],
              'discard_penultimate':sampling['discard_penultimate_requested'],
              'sgm_noise_multiplier':sampling['sgm_noise_multiplier_requested'],
              'skip_early_cfg':cfg['guidance']['skip_early_cfg'],'ngms':cfg['guidance']['ngms'],
              'ngms_all_steps':cfg['guidance']['ngms_all_steps'],
              'img2img_step_mode':sampling['img2img_step_mode'],
              'img2img_extra_noise':sampling['img2img_extra_noise'],
              'original_width':(cfg['sdxl'] or {}).get('original_width',0),
              'original_height':(cfg['sdxl'] or {}).get('original_height',0),
              'target_width':(cfg['sdxl'] or {}).get('target_width',0),
              'target_height':(cfg['sdxl'] or {}).get('target_height',0),
              'crop_x':(cfg['sdxl'] or {}).get('crop_left',0),
              'crop_y':(cfg['sdxl'] or {}).get('crop_top',0),
              'zero_empty_negative':(cfg['sdxl'] or {}).get('zero_empty_negative',False),
              'source_json':canonical(doc),'sampling_adjustments':True}
    settings = add('ForgeNeoBridgeSettings',values,'ForgeNeo Bridge Settings',
                   {'forge_neo_bridge':{'unsupported':doc['unsupported'],'unresolved':doc['unresolved']}})
    pos = add('ForgeNeoBridgeTextEncode',{'text':positive},'ForgeNeo Bridge Positive')
    neg = add('ForgeNeoBridgeTextEncode',{'text':text['negative_raw']},'ForgeNeo Bridge Negative')
    sampler_name = doc['extensions'].get('unhandled_sampler',sampling['sampler'])
    scheduler = doc['extensions'].get('unhandled_scheduler',sampling['scheduler'])
    sampler = add('ForgeNeoBridgeKSampler',{'seed':noise['seed'],'steps':sampling['steps'],'cfg':sampling['cfg'],
                 'sampler_name':sampler_name,'scheduler':scheduler,'denoise':sampling['denoise']},'ForgeNeo Bridge KSampler')
    connect(*clip,pos,'clip');connect(*clip,neg,'clip')
    connect(*model,sampler,'model');connect(pos,0,sampler,'positive');connect(neg,0,sampler,'negative');connect(settings,0,sampler,'settings')
    if values['mode'] == 'img2img':
        source = add('LoadImage',{'image':''})
        encode = add('VAEEncode')
        connect(source,0,encode,'pixels');connect(*vae,encode,'vae');connect(encode,0,sampler,'input_latent')
    else:
        empty = add('EmptyLatentImage',{'width':cfg['image']['width'],'height':cfg['image']['height'],
                                        'batch_size':cfg['image']['batch_size']})
        connect(empty,0,sampler,'input_latent')
    if hires:
        second_doc = pass_document(original,hires,second=True)
        second_values = dict(values,mode='img2img',width=hires['width'],height=hires['height'],
                             img2img_step_mode='exact_steps',source_json=canonical(second_doc))
        if hires['sampling']['shift'] is not None:
            second_values['shift'] = str(hires['sampling']['shift'])
        # Zero means the connected latent determines SDXL original/target size.
        for key in ('original_width','original_height','target_width','target_height'):
            second_values[key] = 0
        second_settings = add('ForgeNeoBridgeSettings',second_values,'ForgeNeo Bridge Hires Settings',
                              {'forge_neo_bridge':{'unsupported':second_doc['unsupported'],
                                                  'unresolved':second_doc['unresolved']}})
        if hires['upscale_type'] == 'ImageScale':
            first_decode = add('VAEDecode')
            connect(sampler,0,first_decode,'samples');connect(*vae,first_decode,'vae')
            upscale = add('ImageScale',{'upscale_method':hires['upscale_method'],
                                       'width':hires['width'],'height':hires['height'],'crop':'disabled'})
            connect(first_decode,0,upscale,'image')
            latent = add('VAEEncode')
            connect(upscale,0,latent,'pixels');connect(*vae,latent,'vae')
        else:
            latent = add('LatentUpscale',{'upscale_method':hires['upscale_method'],
                                          'width':hires['width'],'height':hires['height'],'crop':'disabled'})
            connect(sampler,0,latent,'samples')
        for key, prompt_text, current in (('positive',hires['positive'],positive),
                                          ('negative',hires['negative'],text['negative_raw'])):
            if prompt_text != current:
                encoded = add('ForgeNeoBridgeTextEncode',{'text':prompt_text},'ForgeNeo Bridge Hires '+key.title())
                connect(*clip,encoded,'clip')
                if key == 'positive':pos = encoded
                else:neg = encoded
        hr = hires['sampling']
        sampler = add('ForgeNeoBridgeKSampler',{'seed':noise['seed'],'steps':hr['steps'],'cfg':hr['cfg'],
                      'sampler_name':hr['sampler'],'scheduler':hr['scheduler'],'denoise':hr['denoise']},
                      'ForgeNeo Bridge Hires KSampler')
        connect(*model,sampler,'model');connect(pos,0,sampler,'positive');connect(neg,0,sampler,'negative')
        connect(second_settings,0,sampler,'settings');connect(latent,0,sampler,'input_latent')
    decode = add('VAEDecode');save = add('SaveImage',{'filename_prefix':'ForgeNeo-Bridge'})
    connect(sampler,0,decode,'samples');connect(*vae,decode,'vae');connect(decode,0,save,'images')
    return {'schema_version':'1.0.0','name':'ForgeNeo Bridge '+(binding['family'] or 'unresolved'),
            'nodes':nodes,'edges':edges,'qualification':'not_evaluated'}
