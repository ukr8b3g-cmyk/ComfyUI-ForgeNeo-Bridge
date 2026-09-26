"""Known-loader-chain verification and runtime identity checks, scoped to input edges."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from .spec import BridgeError, canonical, safe_relative


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def tensor_hash(tensor):
    import torch
    if not isinstance(tensor, torch.Tensor) or tensor.is_nested:
        raise BridgeError('LATENT_MISMATCH', 'Expected a dense tensor')
    t = tensor.detach().contiguous().cpu()
    raw = t.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(canonical({'dtype': str(t.dtype), 'shape': list(t.shape)}).encode() + raw).hexdigest()


def category_path(category, name):
    import folder_paths
    name = safe_relative(name)
    if category not in ('checkpoints', 'diffusion_models', 'text_encoders', 'vae', 'loras', 'embeddings', 'input'):
        raise BridgeError('INVALID_PATH', 'Unsupported asset category')
    if category == 'input':
        roots = [Path(folder_paths.get_input_directory()).resolve()]
        candidate = roots[0] / name
    else:
        roots = [Path(p).absolute() for p in folder_paths.get_folder_paths(category)]
        found = folder_paths.get_full_path(category, name)
        if found is None:
            raise BridgeError('ASSET_MISSING', f'{category}/{name}')
        candidate = Path(found).absolute()
        # Comfy's registered model folders deliberately support file/directory
        # symlinks (including Stability Matrix's shared model store). Verify the
        # registered entry before resolving it, then stat/hash its actual target.
        if not any(candidate == root / name for root in roots):
            raise BridgeError('INVALID_PATH', 'Asset is not an entry in its registered category')
    candidate = candidate.resolve()
    if category == 'input' and not any(candidate.is_relative_to(root) for root in roots):
        raise BridgeError('INVALID_PATH', 'Asset escapes its registered category root')
    if not candidate.is_file():
        raise BridgeError('ASSET_MISSING', f'{category}/{name}')
    return candidate


def file_identity(asset):
    path = category_path(asset['category'], asset['relative_name'])
    stat = path.stat()
    result = {'asset_id': asset['asset_id'], 'category': asset['category'], 'relative_name': asset['relative_name'],
              'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': None}
    if asset.get('sha256'):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(block)
        result['sha256'] = h.hexdigest()
        if result['sha256'] != asset['sha256']:
            raise BridgeError('ASSET_HASH_MISMATCH', f"Hash mismatch: {asset['asset_id']}")
    return result


class GraphReader:
    def __init__(self, prompt=None, dynprompt=None):
        self.prompt, self.dynamic = prompt or {}, dynprompt
    def get(self, node_id):
        node = self.dynamic.get_node(str(node_id)) if self.dynamic is not None else self.prompt.get(str(node_id))
        if not node or not isinstance(node.get('inputs'), dict):
            raise BridgeError('UNVERIFIED_PATCH_CHAIN', 'Graph node is missing')
        return node
    def input(self, node_id, key):
        value = self.get(node_id)['inputs'].get(key)
        if not isinstance(value, list) or len(value) != 2 or type(value[1]) is not int:
            raise BridgeError('UNVERIFIED_PATCH_CHAIN', f'{key} must be a known loader connection')
        return value
    def literal(self, inputs, key):
        value = inputs.get(key)
        if isinstance(value, (list, dict)) or value is None:
            raise BridgeError('UNVERIFIED_PATCH_CHAIN', f'Dynamic loader parameter {key} needs a separate adapter')
        return value
    def trace(self, link, role, seen=None):
        seen = set() if seen is None else seen
        node_id, slot = str(link[0]), link[1]
        if node_id in seen or len(seen) > 64:
            raise BridgeError('UNVERIFIED_PATCH_CHAIN', 'Loader chain cycle/limit')
        seen.add(node_id)
        node = self.get(node_id); kind = node['class_type']; ins = node['inputs']
        if kind == 'CheckpointLoaderSimple':
            expected = {'model': 0, 'text_encoder': 1, 'vae': 2}[role]
            if slot != expected: raise BridgeError('UNVERIFIED_PATCH_CHAIN', 'Wrong checkpoint output slot')
            return [{'category': 'checkpoints', 'name': safe_relative(self.literal(ins, 'ckpt_name')), 'component': role}]
        loaders = {'UNETLoader': ('model', 'diffusion_models', 'unet_name', 0),
                   'CLIPLoader': ('text_encoder', 'text_encoders', 'clip_name', 0),
                   'VAELoader': ('vae', 'vae', 'vae_name', 0)}
        if kind in loaders:
            expected, category, key, expected_slot = loaders[kind]
            if role != expected or slot != expected_slot: raise BridgeError('UNVERIFIED_PATCH_CHAIN', 'Loader role/slot mismatch')
            out = {'category': category, 'name': safe_relative(self.literal(ins, key)), 'component': role}
            for opt in ('weight_dtype', 'type', 'device'):
                if opt in ins: out[opt] = self.literal(ins, opt)
            return [out]
        if kind == 'DualCLIPLoader' and role == 'text_encoder' and slot == 0:
            return [{'category': 'text_encoders', 'name': safe_relative(self.literal(ins, k)), 'component': k,
                     'type': self.literal(ins, 'type')} for k in ('clip_name1', 'clip_name2')]
        if kind in ('LoraLoader', 'LoraLoaderModelOnly'):
            if role == 'model' and slot == 0: key, strength = 'model', 'strength_model'
            elif role == 'text_encoder' and kind == 'LoraLoader' and slot == 1: key, strength = 'clip', 'strength_clip'
            else: raise BridgeError('UNVERIFIED_PATCH_CHAIN', 'LoRA output role mismatch')
            chain = self.trace(self.input(node_id, key), role, seen)
            chain.append({'category': 'loras', 'name': safe_relative(self.literal(ins, 'lora_name')),
                          'strength': float(self.literal(ins, strength)), 'component': role, 'node_id': node_id})
            return chain
        raise BridgeError('UNVERIFIED_PATCH_CHAIN', f'Unsupported loader/patch node: {kind}')


def verify_binding(cfg, reader, node_id, input_key, role):
    chain = reader.trace(reader.input(node_id, input_key), role)
    assets = {a['asset_id']: a for a in cfg['assets']}
    roots = [a for a in cfg['assets'] if a['role'] == role]
    expected_roots = [(a['category'], a['relative_name']) for a in roots]
    actual_roots = [(x['category'], x['name']) for x in chain if x['category'] != 'loras']
    if sorted(expected_roots) != sorted(actual_roots):
        raise BridgeError('ASSET_BINDING_MISMATCH', f'{role} loaders differ from Spec')
    weight_key = 'strength_model' if role == 'model' else 'strength_text_encoder'
    expected_loras = [(assets[l['asset_id']]['relative_name'], l[weight_key]) for l in cfg['loras'] if l[weight_key] != 0]
    actual_loras = [(x['name'], x['strength']) for x in chain if x['category'] == 'loras' and x['strength'] != 0]
    if actual_loras != expected_loras:
        raise BridgeError('ASSET_BINDING_MISMATCH', f'{role} LoRA order/strength differs from Spec (double application is forbidden)')
    identities = [file_identity(a) for a in roots]
    identities += [file_identity(assets[l['asset_id']]) for l in cfg['loras'] if l[weight_key] != 0]
    return {'chain': chain, 'assets': identities, 'hash': digest({'chain': chain, 'assets': identities})}


def model_family(model):
    import comfy.model_base as mb
    obj = model.model
    if isinstance(obj, getattr(mb, 'Anima', ())): return 'anima'
    if type(obj) is mb.SDXL: return 'sdxl'
    if type(obj) is mb.BaseModel:
        # SD1 base model, not SD2 or arbitrary BaseModel subclasses.
        config = getattr(obj, 'model_config', None)
        if type(config).__name__ == 'SD15': return 'sd15'
    raise BridgeError('MODEL_FAMILY_MISMATCH', 'V1 only accepts Anima / SD1.5 / SDXL Base, not video or derivative families')


def validate_model(model, cfg):
    import comfy.model_sampling as ms
    family = model_family(model)
    if family != cfg['family']: raise BridgeError('MODEL_FAMILY_MISMATCH', f'{family} != {cfg["family"]}')
    predictor = model.get_model_object('model_sampling')
    if family == 'anima': ok = isinstance(predictor, ms.CONST)
    else: ok = isinstance(predictor, ms.EPS) and not isinstance(predictor, (ms.V_PREDICTION, ms.CONST))
    if not ok: raise BridgeError('UNSUPPORTED_COMBINATION', 'Actual prediction type differs from the approved profile')
    attachments = getattr(model, 'attachments', {}) or {}
    # Core LoraLoader passes safetensors header metadata through this attachment.
    # It does not patch inference behavior; the loader chain and LoRA file were
    # already verified against cfg before this function is called.
    core_lora_metadata = bool(cfg['loras']) and isinstance(attachments, dict) and set(attachments) == {'lora_metadata'}
    if getattr(model, 'object_patches', {}) or (attachments and not core_lora_metadata):
        raise BridgeError('UNVERIFIED_PATCH_CHAIN', 'Object patches/attachments require a separate adapter')
    def contains_hooks(value):
        if callable(value): return True
        if isinstance(value, dict): return any(contains_hooks(v) for v in value.values())
        if isinstance(value, (list, tuple)): return any(contains_hooks(v) for v in value)
        return False
    if contains_hooks(getattr(model, 'model_options', {})):
        raise BridgeError('UNVERIFIED_PATCH_CHAIN', 'Custom model hooks are outside the strict V1 execution contract')
    return family


def latent_shape(model, cfg):
    fmt = model.get_model_object('latent_format')
    channels = int(fmt.latent_channels)
    ratio = getattr(fmt, 'spacial_downscale_ratio', 8)
    ratio = int(ratio) if not isinstance(ratio, (list,tuple)) else int(ratio[-1])
    image = cfg['image']; dims = int(getattr(fmt, 'latent_dimensions', 2))
    expected_c, expected_d = (16, 3) if cfg['family'] == 'anima' else (4, 2)
    if channels != expected_c or dims != expected_d or image['width'] % ratio or image['height'] % ratio:
        raise BridgeError('MODEL_FAMILY_MISMATCH', 'Actual latent format is outside the profile')
    return (image['batch_size'], channels, *((1,) if dims == 3 else ()), image['height']//ratio, image['width']//ratio)


def check_latent(latent, shape):
    import torch
    if not isinstance(latent, dict) or 'samples' not in latent:
        raise BridgeError('LATENT_MISMATCH', 'Missing latent samples')
    if latent.get('noise_mask') is not None or latent.get('batch_index') is not None:
        raise BridgeError('UNSUPPORTED_COMBINATION', 'Inpaint masks / latent batch indices are outside V1')
    t = latent['samples']
    if not isinstance(t, torch.Tensor) or t.is_nested or tuple(t.shape) != tuple(shape):
        raise BridgeError('LATENT_MISMATCH', 'Nested/video/wrong-shaped latent')
    if not t.is_floating_point() or not torch.isfinite(t).all():
        raise BridgeError('LATENT_MISMATCH', 'Latent must contain finite floating-point samples')
    return t
