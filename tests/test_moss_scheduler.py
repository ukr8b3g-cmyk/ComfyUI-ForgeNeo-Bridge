"""CPU schedules and node registration, independent of optional MOSS extensions."""
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from fnb.bridge.moss_scheduler import MODES, build_sigmas
from fnb.bridge.spec import BridgeError
from fnb.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from fnb.vendor.moss_scheduler import MODE_HANDLERS


@pytest.fixture
def sampling():
    t = torch.arange(1, 1001) / 1000
    return NS(sigmas=5 * t / (1 + 4 * t))


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('step_mode,steps,denoise,total,cut', [
    ('comfy',8,1.,8,0), ('comfy',8,.5,16,-9),
    ('forge_scaled',8,1.,8,0), ('forge_scaled',8,.5,8,3),
    ('exact_steps',8,1.,8,0), ('exact_steps',8,.5,16,8),
])
def test_sigmas_match_the_selected_moss_curve_and_forge_tail(sampling, mode, step_mode, steps, denoise, total, cut):
    before = sampling.sigmas.clone()
    expected = MODE_HANDLERS[mode](sampling,total)[cut:]
    result = build_sigmas(sampling,mode,steps,denoise,step_mode)
    assert torch.equal(result,expected)
    assert result.dtype == torch.float32 and result.device.type == 'cpu'
    assert result[-1] == 0 and torch.isfinite(result).all()
    assert torch.equal(before,sampling.sigmas)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('step_mode', ['comfy','forge_scaled','exact_steps'])
def test_zero_denoise_produces_an_empty_schedule(sampling, mode, step_mode):
    assert build_sigmas(sampling,mode,8,0.,step_mode).numel() == 0


@pytest.mark.parametrize('installed', [False, True])
def test_own_node_works_without_external_moss_and_does_not_touch_registries(sampling, monkeypatch, installed):
    names = ['normal'] + (list(MODES) if installed else [])
    handlers = {name: object() for name in names}
    registry = NS(SCHEDULER_NAMES=names, SCHEDULER_HANDLERS=handlers)
    monkeypatch.setitem(sys.modules,'comfy.samplers',registry)
    node_class = NODE_CLASS_MAPPINGS['ForgeNeoBridgeMOSSScheduler']
    assert NODE_DISPLAY_NAME_MAPPINGS['ForgeNeoBridgeMOSSScheduler'] == 'ForgeNeo Bridge MOSS Scheduler'
    assert node_class.INPUT_TYPES()['required']['mode'][0] == list(MODES)
    assert node_class.RETURN_TYPES == ('SIGMAS',)
    assert node_class.FUNCTION == 'schedule'
    model = NS(get_model_object=lambda key:sampling)
    for mode in MODES:
        result, = node_class().schedule(model,mode,8,1.)
        assert torch.equal(result,MODE_HANDLERS[mode](sampling,8))
    assert registry.SCHEDULER_NAMES is names and registry.SCHEDULER_HANDLERS is handlers
    assert list(handlers) == names


@pytest.mark.parametrize('table', [torch.full((1000,),float('nan')), torch.full((1000,),-1.)])
def test_invalid_generated_schedule_stops_with_a_reason(table):
    with pytest.raises(BridgeError,match='MOSS Scheduler has no valid sampling interval'):
        build_sigmas(NS(sigmas=table),'MOSS Shift Scheduler',8,1.)
