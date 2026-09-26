"""Split Hires.fix into ordinary sampling passes, retaining the source infotext.

Forge sample_hr_pass uses explicit (exact) steps, resets ImageRNG at the enlarged
size and recalculates SDXL conditioning. Only handled metadata blockers are removed.
"""
from __future__ import annotations
import copy
import math
import re
from .infotext import SAMPLER_NAMES, SCHEDULER_NAMES, parse_loras
from .spec import BridgeError, finalize

FIELDS = frozenset(('hires upscale', 'hires resize', 'hires upscaler', 'hires steps',
                    'hires cfg scale', 'hires sampler', 'hires schedule type',
                    'hires prompt', 'hires negative prompt', 'hires checkpoint', 'hires shift'))
LATENT_MODES = {'latent':'bilinear', 'latent (bicubic)':'bicubic',
                'latent (nearest-exact)':'nearest-exact'}


def hires_recipe(doc):
    cfg = doc['effective']
    if cfg['mode'] != 'hires_fix':
        return None
    family = doc['extensions']['reconstruction']['family']
    def unsupported(reason):
        raise BridgeError('HIRES_UNSUPPORTED', f'Hires.fix ({family}): {reason}')
    if family not in ('sd15', 'sdxl', 'anima'):
        unsupported('no two-pass workflow recipe for this model family')
    fields, handled = {}, []
    for item in doc['source']['raw_fields']:
        key, value = item['key'].casefold(), item['value']
        if not key.startswith('hires'):continue
        if key in fields and fields[key] != value:
            raise BridgeError('INVALID_METADATA', f'Conflicting {item["key"]}')
        fields[key] = value
        if re.fullmatch(r'hires module \d+', key) or key == 'hires checkpoint':
            if value.casefold() not in ('use same choices', 'use same checkpoint'):
                unsupported(f'{item["key"]}: switching model/encoder/VAE is not supported ({value})')
        elif key not in FIELDS:unsupported(f'unhandled setting {item["key"]}: {value}')
        if key == 'hires shift' and family != 'anima':
            unsupported('Hires Shift is supported only for Anima in this recipe')
        handled.append(f'{item["key"]} requires a separate adapter')
    method = fields.get('hires upscaler', '').casefold()
    if method != 'lanczos' and method not in LATENT_MODES:
        unsupported(f'unsupported or missing Hires upscaler: {method}')
    if '/sampling/denoise' not in doc['requested']:
        unsupported('Denoising strength is missing')

    def number(key, default, convert=float):
        try:result = convert(fields.get(key, default))
        except (ValueError, TypeError, OverflowError) as exc:
            raise BridgeError('INVALID_METADATA', f'Invalid {key}') from exc
        if not math.isfinite(result):raise BridgeError('INVALID_METADATA', f'Non-finite {key}')
        return result

    width, height = cfg['image']['width'], cfg['image']['height']
    size_source = 'hires_resize'
    scaled_size = None
    if 'hires resize' in fields:
        match = re.fullmatch(r'(\d+)[x×](\d+)', fields['hires resize'])
        if not match:raise BridgeError('INVALID_METADATA', 'Invalid Hires resize')
        target_w, target_h = map(int, match.groups())
        if target_w == 0 and target_h > 0:target_w = target_h * width / height
        if target_h == 0 and target_w > 0:target_h = target_w * height / width
    elif 'hires upscale' in fields:
        scale = number('hires upscale', 2)
        if scale <= 0:raise BridgeError('INVALID_METADATA', 'Hires upscale must be positive')
        target_w, target_h = width * scale, height * scale
        scaled_size = [target_w, target_h]
        source_size = doc['extensions']['reconstruction'].get('source_image_size')
        if source_size:
            target_w, target_h = source_size
            size_source = 'source_image'
        else:size_source = 'scale_estimate'
    else:unsupported('Hires resize or Hires upscale is missing')
    # Forge's configurable resolution rounding is not recorded in infotext.
    if any(not math.isfinite(v) or v < 64 or v > 16384 or v % 8 for v in (target_w, target_h)):
        raise BridgeError('HIRES_SIZE_UNSUPPORTED', 'Hires target must be a multiple of 8 between 64 and 16384; record an explicit Hires resize')
    steps = number('hires steps', 0, int) or cfg['sampling']['steps']
    scale_cfg = number('hires cfg scale', cfg['sampling']['cfg'])
    if not 1 <= steps <= 10000 or scale_cfg < 0 or not 0 < cfg['sampling']['denoise'] <= 1:
        raise BridgeError('INVALID_METADATA', 'Hires steps/CFG/denoise is outside the supported range')
    sampling = dict(cfg['sampling'], steps=steps, cfg=scale_cfg, img2img_step_mode='exact_steps')
    if 'hires shift' in fields:
        sampling['shift'] = number('hires shift', cfg['sampling']['shift'])
        if sampling['shift'] <= 0:raise BridgeError('INVALID_METADATA', 'Hires Shift must be positive')
    for field, dest, names in (('hires sampler', 'sampler', SAMPLER_NAMES),
                               ('hires schedule type', 'scheduler', SCHEDULER_NAMES)):
        if field not in fields or fields[field].casefold() == 'use same sampler':continue
        value = {k.casefold():v for k, v in names.items()}.get(fields[field].casefold())
        if value is None:unsupported(f'unsupported {field}: {fields[field]}')
        sampling[dest] = value
    positive = fields.get('hires prompt') or cfg['text']['positive_raw']
    negative = fields.get('hires negative prompt') or cfg['text']['negative_raw']
    try:
        positive, loras = parse_loras(positive)
        _, negative_loras = parse_loras(negative)
    except BridgeError as exc:unsupported(str(exc))
    if loras != doc['extensions']['reconstruction']['loras'] or negative_loras:
        unsupported('changing LoRAs in the Hires pass is not supported')
    return {'width':int(target_w), 'height':int(target_h), 'sampling':sampling,
            'upscale_type':'ImageScale' if method == 'lanczos' else 'LatentUpscale',
            'upscale_method':'lanczos' if method == 'lanczos' else LATENT_MODES[method],
            'positive':positive, 'negative':negative, 'handled':handled,
            'size_source':size_source, 'scale_estimate':scaled_size}


def pass_document(doc, recipe, second=False):
    result = copy.deepcopy(doc)
    cfg = result['effective']
    cfg['mode'] = 'img2img' if second else 'txt2img'
    if second:
        cfg['sampling'] = copy.deepcopy(recipe['sampling'])
        cfg['image'].update(width=recipe['width'], height=recipe['height'])
        cfg['text'].update(positive_raw=recipe['positive'], negative_raw=recipe['negative'])
        if cfg['sdxl']:
            for key in ('original_width', 'target_width'):cfg['sdxl'][key] = recipe['width']
            for key in ('original_height', 'target_height'):cfg['sdxl'][key] = recipe['height']
    else:cfg['sampling']['denoise'] = 1.0
    result['unsupported'] = [item for item in result['unsupported'] if not (
        (item['path'] == '/extensions' and item['message'] in recipe['handled']) or
        (item['path'] == '/mode' and item['message'] == 'V1 supports txt2img and unmasked img2img only.'))]
    result['extensions']['hires_pass'] = 'second' if second else 'base'
    result['extensions']['hires_target'] = {'width':recipe['width'], 'height':recipe['height'],
        'source':recipe['size_source'], 'scale_estimate':recipe['scale_estimate']}
    target = [recipe['width'], recipe['height']]
    if recipe['size_source'] == 'scale_estimate':
        message = f'Hires target {target[0]}x{target[1]} is estimated from scale; Forge resolution rounding is unknown.'
    elif recipe['size_source'] == 'source_image' and recipe['scale_estimate'] != target:
        message = f'Hires target uses source image dimensions {target[0]}x{target[1]}, differing from the scale estimate {recipe["scale_estimate"]}. A resized source image changes this target.'
    else:message = None
    if message:
        result['unresolved'].append({'code':'HIRES_TARGET_SIZE', 'path':'/image', 'message':message})
    return finalize(result).document()
