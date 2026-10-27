"""MOSS override metadata and run-local sigma schedules; no Core registrations."""
import math

from .infotext import parse_bool
from .spec import BridgeError

MODES = ('MOSS Shift Scheduler', 'MOSS Curve Scheduler', 'MOSS Sigma Scheduler')
STEP_MODES = ('comfy', 'forge_scaled', 'exact_steps')
METADATA_FIELDS = frozenset(('moss scheduler', 'moss scheduler mode', 'moss scheduler version',
                             'moss schedule steps', 'moss schedule intervals'))


def schedule_config(fields, cfg):
    values = {}
    for field in fields:
        key, value = field['key'].casefold(), field['value']
        if key not in METADATA_FIELDS:continue
        if key in values and values[key] != value:
            raise BridgeError('INVALID_METADATA', f'Conflicting {field["key"]}')
        values[key] = value
    if not parse_bool(values.get('moss scheduler', 'False')):
        return {'enabled': False}
    mode = values.get('moss scheduler mode')
    if mode not in MODES:
        raise BridgeError('MOSS_SCHEDULE_UNSUPPORTED',
                          'MOSS Scheduler is ON but its mode is missing or unknown. '
                          'Select Shift, Curve or Sigma in Forge Neo and export the generation information again.')
    recorded = {}
    for key in ('moss schedule steps', 'moss schedule intervals'):
        if key not in values:continue
        try:
            number = int(values[key])
            if number < 1:raise ValueError()
        except ValueError as exc:
            raise BridgeError('INVALID_METADATA', f'Invalid {key}') from exc
        recorded[key] = number
    step_mode = 'comfy'
    steps, denoise = cfg['sampling']['steps'], cfg['sampling']['denoise']
    total = recorded.get('moss schedule steps')
    if cfg['mode'] == 'img2img' and denoise > 0:
        if total == steps:
            step_mode = 'forge_scaled'
        elif total == int(steps / min(denoise, 0.999)):
            step_mode = 'exact_steps'
        else:
            raise BridgeError('MOSS_SCHEDULE_UNRESOLVED',
                              'MOSS img2img needs MOSS Schedule steps matching the recorded Steps and denoise. '
                              'Export the original generation information again.')
    elif total is not None and total != steps:
        raise BridgeError('MOSS_SCHEDULE_UNRESOLVED', 'MOSS Schedule steps does not match the recorded Steps.')
    return {'enabled': True, 'mode': mode, 'step_mode': step_mode,
            'source_version': values.get('moss scheduler version'), 'recorded': recorded}


def build_sigmas(model_sampling, mode, steps, denoise, step_mode='comfy'):
    from ..vendor.moss_scheduler import MODE_HANDLERS, get_sigmas
    import torch
    if mode not in MODES or step_mode not in STEP_MODES:
        raise BridgeError('MOSS_SCHEDULE_UNSUPPORTED', 'Choose a supported MOSS mode and step mode.')
    if type(steps) is not int or steps < 1 or not math.isfinite(denoise) or not 0 <= denoise <= 1:
        raise BridgeError('INVALID_SPEC', 'MOSS Scheduler needs positive integer steps and denoise between 0 and 1.')
    if denoise <= 0:return torch.FloatTensor([])
    if step_mode == 'comfy':
        result = get_sigmas(model_sampling, mode, steps, denoise)
    else:
        total = int(steps / min(denoise, 0.999)) if step_mode == 'exact_steps' else steps
        cut = total - steps if step_mode == 'exact_steps' else total - int(min(denoise, 0.999) * total) - 1
        result = MODE_HANDLERS[mode](model_sampling, total)[cut:].cpu()
    if (result.numel() < 2 or not torch.isfinite(result).all() or result[-1] != 0
            or not (result[:-1] > 0).all() or not (result[:-1] >= result[1:]).all()):
        raise BridgeError('MOSS_SCHEDULE_INVALID', 'MOSS Scheduler has no valid sampling interval. '
                          'Check the model sigma table, reduce steps or increase denoise.')
    return result
