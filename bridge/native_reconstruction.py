"""ComfyUI-core workflow recipes for image families outside the Bridge sampler.

These graphs follow the installed ComfyUI blueprints. They are editable native
workflows, not a claim that ComfyUI and Forge Neo produce identical pixels.
"""
from __future__ import annotations

from .spec import BridgeError


CORE_SAMPLERS = frozenset((
    'euler', 'euler_cfg_pp', 'euler_ancestral', 'euler_ancestral_cfg_pp',
    'heun', 'dpm_2', 'lms', 'dpmpp_2s_ancestral', 'dpmpp_sde', 'dpmpp_2m',
    'dpmpp_2m_sde', 'dpmpp_3m_sde', 'lcm', 'res_multistep', 'er_sde',
    'ddim', 'uni_pc', 'uni_pc_bh2', 'dpmpp_2m_cfg_pp',
))
CORE_SCHEDULERS = frozenset(('simple', 'sgm_uniform', 'karras', 'exponential',
                             'ddim_uniform', 'beta', 'normal', 'linear_quadratic', 'kl_optimal'))


def defaults(family, model_name):
    name = model_name.casefold()
    turbo = 'turbo' in name or 'schnell' in name
    base = 'base' in name
    if family == 'ernie':return (8, 1.0, 'euler') if turbo else (20, 4.0, 'euler')
    if family == 'zimage':return (8, 1.0, 'res_multistep') if turbo else (25, 4.0, 'euler')
    if family == 'krea2':return 8, 1.0, 'euler'
    if family == 'qwen21':return 25, 1.0, 'euler'
    if family == 'flux2_klein':return (20, 5.0, 'euler') if base else (4, 1.0, 'euler')
    if family == 'flux2_dev':return 20, 4.0, 'euler'
    if family == 'flux1':return (4 if 'schnell' in name else 20), 1.0, 'euler'
    return 20, 4.0, 'euler'


def apply_defaults(cfg, requested, family, model_name):
    steps, guidance, sampler = defaults(family, model_name)
    for path, value in (('/noise/seed', '0'), ('/image/width', 1024),
                        ('/image/height', 1024), ('/sampling/steps', steps),
                        ('/sampling/cfg', guidance), ('/sampling/sampler', sampler),
                        ('/sampling/scheduler', 'simple')):
        if path not in requested:
            group, field = path.strip('/').split('/')
            cfg[group][field] = value


def core_sampling(sampling):
    sampler = {'unipc':'uni_pc'}.get(sampling['sampler'], sampling['sampler'])
    scheduler = {'ddim':'ddim_uniform', 'uniform':'normal'}.get(sampling['scheduler'], sampling['scheduler'])
    if scheduler == 'automatic':
        scheduler = {'dpm_2':'karras', 'dpmpp_2m':'karras', 'dpmpp_2m_cfg_pp':'karras', 'dpmpp_2m_sde':'exponential',
                     'dpmpp_3m_sde':'exponential', 'dpmpp_sde':'karras',
                     'dpmpp_2s_ancestral':'karras'}.get(sampler, 'normal')
    return sampler, scheduler


def native_plan(doc, positive, inventory):
    binding = doc['extensions']['reconstruction']
    family, cfg = binding['family'], binding['native_config']
    sampling, noise, image, prompts = (cfg[k] for k in ('sampling', 'noise', 'image', 'text'))
    if doc['unsupported']:
        issue = doc['unsupported'][0]
        raise BridgeError('NATIVE_EFFECT_UNSUPPORTED', issue['message'])
    if cfg['mode'] not in ('txt2img', 'img2img'):
        raise BridgeError('NATIVE_MODE_UNSUPPORTED', f'{family}: {cfg["mode"]} requires a separate workflow')
    seed = int(noise['seed'])
    if not 0 <= seed <= 9007199254740991:
        raise BridgeError('NATIVE_SEED_UNSUPPORTED', 'Core workflow cannot preserve a seed above JavaScript safe integer')
    sampler_name, scheduler = core_sampling(sampling)
    if sampler_name not in CORE_SAMPLERS:
        raise BridgeError('NATIVE_SAMPLER_UNSUPPORTED', f'{family}: {sampler_name} is unavailable in the native recipe')
    if family not in ('flux2_klein', 'flux2_dev') and scheduler not in CORE_SCHEDULERS:
        raise BridgeError('NATIVE_SCHEDULER_UNSUPPORTED', f'{family}: {scheduler} is unavailable in the native recipe')
    if family == 'flux2_dev' and prompts['negative_raw'].strip():
        raise BridgeError('NATIVE_NEGATIVE_UNSUPPORTED', 'Flux.2 Dev blueprint has no negative conditioning input')
    if family == 'qwen21' and sampling['shift'] is not None and '/sampling/shift' in doc['requested'] and abs(sampling['shift'] - 0.69) > 1e-6:
        raise BridgeError('NATIVE_SHIFT_UNSUPPORTED', 'Qwen-Image 2.1 recipe uses its built-in shift of 0.69')
    if family in ('flux2_klein', 'flux2_dev') and cfg['mode'] == 'img2img':
        raise BridgeError('NATIVE_MODE_UNSUPPORTED', 'Flux.2 image editing needs its own conditioning recipe')

    nodes, edges = [], []
    def add(kind, values=None, asset=None):
        ident = str(len(nodes) + 1)
        node = {'id':ident, 'type':kind, 'values':values or {}, 'properties':{}}
        if asset:
            node['properties']['forge_neo_bridge'] = {
                'raw_name':asset['raw_name'], 'current_name':asset['name'],
                'resolution':asset['resolution']}
        nodes.append(node)
        return ident
    def connect(source, slot, target, key):edges.append([source, slot, target, key])

    model_asset = binding['model']
    qwen_edit = family == 'qwen' and 'edit' in model_asset['name'].casefold()
    edit_conditioning = qwen_edit and cfg['mode'] == 'img2img'
    model = add('UNETLoader', {'unet_name':model_asset['name'], 'weight_dtype':'default'}, model_asset)
    encoder = binding['clip']
    if family == 'flux1':
        second = binding['clip_secondary']
        clip = add('DualCLIPLoader', {'clip_name1':encoder['name'], 'clip_name2':second['name'],
                                      'type':'flux', 'device':'default'}, encoder)
    else:
        clip_type = {'ernie':'flux2', 'zimage':'lumina2', 'qwen':'qwen_image',
                     'qwen21':'qwen_image', 'krea2':'krea2',
                     'flux2_klein':'flux2', 'flux2_dev':'flux2'}[family]
        clip = add('CLIPLoader', {'clip_name':encoder['name'], 'type':clip_type,
                                  'device':'default'}, encoder)
    vae_asset = binding['vae']
    vae = add('VAELoader', {'vae_name':vae_asset['name']}, vae_asset)
    model_slot = clip_slot = 0
    for item in binding['loras']:
        from .reconstruction import resolve_asset
        asset = resolve_asset(item['name'], ('loras',), inventory)
        if qwen_edit:
            loader = add('LoraLoaderModelOnly', {'lora_name':asset['name'],
                                                'strength_model':item['strength_model']}, asset)
            connect(model, model_slot, loader, 'model')
            model, model_slot = loader, 0
            continue
        loader = add('LoraLoader', {'lora_name':asset['name'],
                                    'strength_model':item['strength_model'],
                                    'strength_clip':item['strength_text_encoder']}, asset)
        connect(model, model_slot, loader, 'model');connect(clip, clip_slot, loader, 'clip')
        model = clip = loader
        model_slot, clip_slot = 0, 1

    if family in ('zimage', 'ernie'):
        shift = sampling['shift'] if '/sampling/shift' in doc['requested'] and sampling['shift'] is not None else 3.0
        # ERNIE consumes timesteps scaled by 1000; Z-Image consumes 0..1.
        patch = add('ModelSamplingSD3' if family == 'ernie' else 'ModelSamplingAuraFlow', {'shift':shift})
        connect(model, model_slot, patch, 'model');model, model_slot = patch, 0
    if edit_conditioning:
        patch = add('CFGNorm', {'strength':1.0})
        connect(model, model_slot, patch, 'model');model, model_slot = patch, 0

    # Forge uses its ordinary text template when no reference image was supplied.
    edit_plus = any(version in model_asset['name'] for version in ('2509', '2511'))
    text_node = ('TextEncodeQwenImageEditPlus' if edit_plus else 'TextEncodeQwenImageEdit') if edit_conditioning else 'CLIPTextEncode'
    text_key = 'prompt' if edit_conditioning else 'text'
    pos = add(text_node, {text_key:positive})
    connect(clip, clip_slot, pos, 'clip')
    if edit_conditioning:connect(vae, 0, pos, 'vae')
    if family != 'flux2_dev':
        if edit_conditioning:
            neg = add(text_node, {text_key:prompts['negative_raw']})
            connect(clip, clip_slot, neg, 'clip');connect(vae, 0, neg, 'vae')
        elif not prompts['negative_raw'].strip() and sampling['cfg'] == 1.0:
            neg = add('ConditioningZeroOut')
            connect(pos, 0, neg, 'conditioning')
        else:
            neg = add('CLIPTextEncode', {'text':prompts['negative_raw']})
            connect(clip, clip_slot, neg, 'clip')

    if family == 'flux1' and 'schnell' not in model_asset['name'].casefold():
        guidance = add('FluxGuidance', {'guidance':binding['distilled_cfg']})
        connect(pos, 0, guidance, 'conditioning');pos = guidance
        negative_guidance = add('FluxGuidance', {'guidance':binding['distilled_cfg']})
        connect(neg, 0, negative_guidance, 'conditioning');neg = negative_guidance

    latent_type = ('EmptyFlux2LatentImage' if family in ('ernie','flux2_klein','flux2_dev')
                   else 'EmptyLatentImage' if family in ('krea2','qwen21') else 'EmptySD3LatentImage')
    if cfg['mode'] == 'img2img':
        source = add('LoadImage', {'image':''})
        latent = add('VAEEncode')
        connect(source, 0, latent, 'pixels');connect(vae, 0, latent, 'vae')
        if edit_conditioning:
            connect(source, 0, pos, 'image1' if edit_plus else 'image')
            connect(source, 0, neg, 'image1' if edit_plus else 'image')
    else:
        latent = add(latent_type, {'width':image['width'],'height':image['height'],
                                    'batch_size':image['batch_size']})

    if family in ('flux2_klein', 'flux2_dev'):
        if family == 'flux2_dev':
            guidance = add('FluxGuidance', {'guidance':sampling['cfg']})
            connect(pos, 0, guidance, 'conditioning')
            guider = add('BasicGuider')
            connect(model, model_slot, guider, 'model')
            connect(guidance, 0, guider, 'conditioning')
        else:
            guider = add('CFGGuider', {'cfg':sampling['cfg']})
            connect(model, model_slot, guider, 'model')
            connect(pos, 0, guider, 'positive');connect(neg, 0, guider, 'negative')
        noise_node = add('RandomNoise', {'noise_seed':seed})
        sampler_node = add('KSamplerSelect', {'sampler_name':sampler_name})
        sigmas = add('Flux2Scheduler', {'steps':sampling['steps'],
                                        'width':image['width'], 'height':image['height']})
        sampler = add('SamplerCustomAdvanced')
        for src, key in ((noise_node,'noise'),(guider,'guider'),(sampler_node,'sampler'),
                         (sigmas,'sigmas'),(latent,'latent_image')):
            connect(src, 0, sampler, key)
    elif sampling['discard_penultimate_requested']:
        sigmas = add('ForgeNeoBridgeScheduler', {'scheduler':scheduler, 'steps':sampling['steps'],
                     'denoise':sampling['denoise']})
        connect(model, model_slot, sigmas, 'model')
        noise_node = add('RandomNoise', {'noise_seed':seed})
        sampler_node = add('KSamplerSelect', {'sampler_name':sampler_name})
        guider = add('CFGGuider', {'cfg':sampling['cfg']})
        connect(model, model_slot, guider, 'model')
        connect(pos, 0, guider, 'positive');connect(neg, 0, guider, 'negative')
        sampler = add('SamplerCustomAdvanced')
        for src, key in ((noise_node,'noise'),(guider,'guider'),(sampler_node,'sampler'),
                         (sigmas,'sigmas'),(latent,'latent_image')):
            connect(src, 0, sampler, key)
    else:
        sampler = add('KSampler', {'seed':seed, 'steps':sampling['steps'],
                                   'cfg':sampling['cfg'], 'sampler_name':sampler_name,
                                   'scheduler':scheduler, 'denoise':sampling['denoise']})
        connect(model, model_slot, sampler, 'model')
        connect(pos, 0, sampler, 'positive');connect(neg, 0, sampler, 'negative')
        connect(latent, 0, sampler, 'latent_image')
    decode = add('VAEDecode')
    save = add('SaveImage', {'filename_prefix':'ForgeNeo-Bridge'})
    connect(sampler, 0, decode, 'samples');connect(vae, 0, decode, 'vae')
    connect(decode, 0, save, 'images')
    return {'schema_version':'1.0.0', 'name':'ForgeNeo Bridge '+('qwen_edit' if qwen_edit else family),
            'nodes':nodes, 'edges':edges, 'qualification':'native_comfy_not_forge_parity',
            'source':doc['source'], 'notes':binding.get('notes', [])}
