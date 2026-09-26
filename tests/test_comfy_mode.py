"""Routing tests; no fake-model image quality or GPU claims."""
import copy
import sys
from types import SimpleNamespace as NS
import pytest
from fnb.bridge.workflow_adapter import ForgeSettings, ForgeTextInput, sample_workflow
from fnb.bridge.spec import BridgeError
from fnb.nodes import ForgeNeoBridgeSettings, ForgeNeoBridgeKSampler


@pytest.fixture
def core(monkeypatch):
    calls=[]
    class ClipLayer:
        def set_last_layer(self,clip,layer):
            calls.append(('clip',clip,layer));return (('clone',clip,layer),)
    class Text:
        def encode(self,clip,text):
            calls.append(('text',clip,text));return (('conditioning',clip,text),)
    class Sampler:
        def sample(self,*args,**kwargs):
            calls.append(('sample',args,kwargs));return ({'samples':'core-output'},)
    class Empty:
        def generate(self,*args):
            calls.append(('empty',args));return ({'samples':'empty'},)
    monkeypatch.setitem(sys.modules,'nodes',NS(CLIPSetLastLayer=ClipLayer,CLIPTextEncode=Text,KSampler=Sampler,EmptyLatentImage=Empty))
    samplers=NS(SAMPLER_NAMES=['euler','uni_pc'],SCHEDULER_NAMES=['normal','simple','ddim_uniform'])
    monkeypatch.setitem(sys.modules,'comfy.samplers',samplers)
    return calls


def inputs():
    values={name:opts[1].get('default') if len(opts)>1 else opts[0][0]
            for name,opts in ForgeNeoBridgeSettings.INPUT_TYPES()['required'].items()}
    # Deliberately invalid Forge metadata/options: OFF must not run the Forge validator.
    values.update(forge_compatibility=False,ensd='31337',source_json='not JSON',clip_skip='2')
    return ForgeSettings(values),dict(seed='18446744073709551615',steps=23,cfg=4.5,
                                     sampler_name='uni_pc',scheduler='automatic',denoise=0.5)


def run(settings,sampler,latent=None):
    return sample_workflow('patched model',ForgeTextInput('edited positive','clip P','1'),
                           ForgeTextInput('edited negative','clip N','2'),settings,sampler,latent)


def test_off_calls_core_with_current_values_exact_seed_and_original_latent(core):
    settings,sampler=inputs();before=copy.deepcopy(settings.values)
    latent={'samples':'connected','noise_mask':'mask','batch_index':[1]}
    result=run(settings,sampler,latent)
    assert result['result']==({'samples':'core-output'},)
    assert core[-1]==('sample',('patched model',18446744073709551615,23,4.5,'uni_pc','normal',
        ('conditioning',('clone','clip P',-2),'edited positive'),
        ('conditioning',('clone','clip N',-2),'edited negative'),latent),{'denoise':0.5})
    assert core[-1][1][-1] is latent
    assert settings.values==before


def test_auto_clip_and_empty_latent_use_core_defaults(core):
    settings,sampler=inputs();settings.values['clip_skip']='auto'
    run(settings,sampler)
    assert core[0]==('empty',(512,512,1))
    assert not any(call[0]=='clip' for call in core)


def test_off_accepts_imported_unipc_alias_without_changing_saved_values(core):
    settings,sampler=inputs();sampler['sampler_name']='unipc'
    run(settings,sampler)
    assert core[-1][1][4]=='uni_pc'
    assert sampler['sampler_name']=='unipc'


def test_off_img2img_requires_image_latent(core):
    settings,sampler=inputs();settings.values['mode']='img2img'
    with pytest.raises(BridgeError,match='Connect the encoded image'):run(settings,sampler)


def test_unsupported_core_schedule_is_explicit_error(core):
    settings,sampler=inputs();sampler['scheduler']='bong_tangent'
    with pytest.raises(BridgeError,match='COMFY_UNSUPPORTED_SAMPLING'):run(settings,sampler)
    assert not core


def test_old_settings_default_to_forge_path(core,monkeypatch):
    import fnb.bridge.workflow_adapter as adapter
    settings,sampler=inputs();settings.values.pop('forge_compatibility')
    def sentinel(*args):raise RuntimeError('Forge path reached')
    monkeypatch.setattr(adapter,'build_execution_spec',sentinel)
    monkeypatch.setitem(sys.modules,'latent_preview',NS())
    with pytest.raises(RuntimeError,match='Forge path reached'):run(settings,sampler)
    assert not core


def test_core_choices_are_available_and_old_settings_input_order_is_unchanged(core):
    fields=ForgeNeoBridgeSettings.INPUT_TYPES()
    assert list(fields['required'])[-1]=='source_json'
    assert fields['optional']['forge_compatibility'][1]['default'] is True
    sampler=ForgeNeoBridgeKSampler.INPUT_TYPES()['required']
    assert 'uni_pc' in sampler['sampler_name'][0]
    assert 'ddim_uniform' in sampler['scheduler'][0]
