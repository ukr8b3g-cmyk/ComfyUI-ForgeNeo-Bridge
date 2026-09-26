import json
from pathlib import Path
import sys
import types
import pytest
from fnb.nodes import ForgeNeoBridgeSettings, NODE_CLASS_MAPPINGS
from fnb.bridge.spec import finalize,BridgeError
from fnb.bridge.schedules import step_plan,core_sigmas
from fnb.bridge.runtime import unused_parameters

def test_new_default_and_legacy_spec(document):
    assert ForgeNeoBridgeSettings.INPUT_TYPES()['optional']['sampling_adjustments'][1]['default'] is False
    old=finalize(document)
    assert old.require_executable()['sampling'].get('adjustments',True) is True
    document['effective']['sampling']['adjustments']=False
    new=finalize(document)
    assert new.config_hash!=old.config_hash
    assert new.require_executable()['sampling']['adjustments'] is False

def test_unadjusted_steps_zero_denoise_and_retained_values(document):
    cfg=document['effective'];cfg['mode']='img2img'
    cfg['sampling'].update(adjustments=False,steps=20,denoise=.5,img2img_step_mode='forge_scaled',eta_ancestral=.67)
    assert step_plan(cfg)==(40,20,20)
    assert '/sampling/eta_ancestral' in unused_parameters(cfg)
    assert cfg['sampling']['eta_ancestral']==.67
    cfg['sampling'].update(denoise=0,img2img_step_mode='exact_steps')
    assert step_plan(finalize(document).require_executable())==(0,0,None)

def test_forge_only_schedule_does_not_silently_fall_back(document,monkeypatch):
    core=types.ModuleType('comfy');core.samplers=types.ModuleType('comfy.samplers')
    core.samplers.SAMPLER_NAMES=['euler'];core.samplers.SCHEDULER_NAMES=['normal']
    monkeypatch.setitem(sys.modules,'comfy',core);monkeypatch.setitem(sys.modules,'comfy.samplers',core.samplers)
    cfg=document['effective'];cfg['sampling'].update(adjustments=False,sampler='euler',scheduler='bong_tangent')
    with pytest.raises(BridgeError,match='Enable Sampling adjustments'):core_sigmas(cfg,object())

def test_native_help_covers_all_visible_fields_and_ports(monkeypatch):
    core=types.ModuleType('comfy');core.samplers=types.ModuleType('comfy.samplers')
    core.samplers.SAMPLER_NAMES=['euler'];core.samplers.SCHEDULER_NAMES=['normal']
    monkeypatch.setitem(sys.modules,'comfy',core);monkeypatch.setitem(sys.modules,'comfy.samplers',core.samplers)
    for lang in ('en','ja'):
        doc=json.loads((Path(__file__).resolve().parents[1]/'locales'/lang/'nodeDefs.json').read_text(encoding='utf-8'))
        for name,cls in NODE_CLASS_MAPPINGS.items():
            assert set(doc[name]['inputs'])=={key for group,values in cls.INPUT_TYPES().items() if group!='hidden' for key in values}
            assert len(doc[name]['outputs'])==len(cls.RETURN_TYPES)
