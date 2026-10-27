"""Boundary errors must precede graph construction or invalid sampling math."""
from types import SimpleNamespace

import pytest
import torch

from fnb.bridge import schedules
from fnb.bridge.reconstruction import reconstruct, reconstruction_plan
from fnb.bridge.spec import BridgeError, default_document


@pytest.mark.parametrize('steps', [1, 2, 3])
@pytest.mark.parametrize('discard', [False, True])
@pytest.mark.parametrize('sampler', ['euler', 'dpm_2'])
@pytest.mark.parametrize('mode', ['txt2img', 'img2img'])
def test_bong_boundary_uses_effective_schedule_length(steps, discard, sampler, mode):
    cfg = default_document('sd15')['effective']
    cfg['mode'] = mode
    cfg['sampling'].update(steps=steps, scheduler='bong_tangent', sampler=sampler,
                           discard_penultimate_requested=discard, denoise=1,
                           img2img_step_mode='forge_scaled')
    predictor = SimpleNamespace(sigmas=torch.tensor([.01, 14.6]))
    if steps == 1 and not discard and sampler == 'euler':
        with pytest.raises(BridgeError) as error:
            schedules.generate_sigmas(cfg, predictor)
        assert error.value.code == 'UNSUPPORTED_COMBINATION'
        assert error.value.path == '/sampling/steps'
        assert 'Bong Tangent' in str(error.value)
    else:
        sigmas, report = schedules.generate_sigmas(cfg, predictor)
        assert torch.isfinite(sigmas).all()
        assert torch.all(sigmas[:-1] > sigmas[1:])
        assert sigmas[-1] == 0
        assert len(sigmas) == report['launch_steps'] + 1


def native(family, extra=''):
    model, clip, vae = {
        'zimage': ('z_image_turbo_bf16', 'qwen_3_4b', 'ae'),
        'moss': ('moss_image_sft_bf16', 'qwen3.5_9b_int8_convrot', 'flux2-vae'),
        'flux2_dev': ('flux2-dev', 'mistral_3_small_flux2_bf16', 'flux2-vae'),
        'flux2_klein': ('flux-2-klein-4b', 'qwen_3_4b', 'flux2-vae'),
    }[family]
    inventory = {category: [name + '.safetensors'] for category, name in
                 [('diffusion_models', model), ('text_encoders', clip), ('vae', vae)]}
    text = (f'portrait\nSteps: 8, Sampler: Euler, CFG scale: 1, Seed: 42, '
            f'Size: 1024x1024, Model: {model}, Module 1: {clip}, Module 2: {vae}' + extra)
    doc, positive = reconstruct(text, inventory, probe=lambda *_: None)
    return doc, positive, inventory


@pytest.mark.parametrize('family', ['zimage', 'moss', 'flux2_dev', 'flux2_klein'])
@pytest.mark.parametrize('strength', [0, .8])
def test_native_variation_is_inert_or_explicitly_rejected(family, strength):
    doc, positive, inventory = native(family, f', Variation seed: 123, Variation seed strength: {strength}')
    if strength:
        with pytest.raises(BridgeError) as error:
            reconstruction_plan(doc, positive, inventory)
        assert error.value.code == 'NATIVE_VARIATION_UNSUPPORTED'
        assert error.value.path == '/noise/subseed_strength'
        assert '0.8' in str(error.value) and '123' in str(error.value)
    else:
        plan = reconstruction_plan(doc, positive, inventory)
        plain, text, inv = native(family)
        reference = reconstruction_plan(plain, text, inv)
        assert plan['nodes'] == reference['nodes']
        assert plan['edges'] == reference['edges']


@pytest.mark.parametrize('family', ['flux2_dev', 'flux2_klein'])
@pytest.mark.parametrize('scheduler', [None, 'Simple', 'Automatic', 'Karras', 'not-a-scheduler'])
def test_flux2_never_silently_substitutes_explicit_nondefault_schedule(family, scheduler):
    doc, positive, inventory = native(family, '' if scheduler is None else ', Schedule type: '+scheduler)
    if scheduler == 'not-a-scheduler':
        with pytest.raises(BridgeError, match='Unknown scheduler'):
            reconstruction_plan(doc, positive, inventory)
    elif scheduler == 'Karras':
        with pytest.raises(BridgeError, match='Flux2Scheduler') as error:
            reconstruction_plan(doc, positive, inventory)
        assert error.value.code == 'NATIVE_SCHEDULER_UNSUPPORTED'
        assert error.value.path == '/sampling/scheduler'
    else:
        plan = reconstruction_plan(doc, positive, inventory)
        assert any(n['type'] == 'Flux2Scheduler' for n in plan['nodes'])
