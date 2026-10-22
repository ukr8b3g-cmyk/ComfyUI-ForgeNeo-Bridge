import pytest

from fnb.bridge.reconstruction import reconstruct, reconstruction_plan
from fnb.bridge.spec import BridgeError


TEXT = ('portrait <lora:lora_latest:1>\nSteps: 8, Sampler: ER SDE, Schedule type: Simple, '
        'CFG scale: 1, Shift: 5, Seed: 3178412173, Size: 880x1184, Model: moss_image_sft_bf16, '
        'Module 1: qwen3.5_9b_int8_convrot, Module 2: full_encoder_small_decoder, RNG: CPU, '
        'Skip Layer Guidance: True, SLG Single Layers: 22, SLG Start Percent: 0.0, '
        'SLG End Percent: 0.5, Discard penultimate sigma: True, Version: neo-2.31')
INVENTORY = {'diffusion_models':['MOSS-Image/moss_image_sft_bf16.safetensors'],
             'text_encoders':['qwen3.5_9b_int8_convrot.safetensors'],
             'vae':['full_encoder_small_decoder.safetensors'],
             'loras':['moss_image/lora_latest.safetensors']}


def plan_for(text=TEXT, inventory=INVENTORY):
    doc,positive=reconstruct(text,inventory,probe=lambda *_:None)
    return doc,reconstruction_plan(doc,positive,inventory)


def test_moss_distilled_recipe_preserves_all_explicit_settings():
    doc,plan=plan_for()
    by_type={n['type']:n for n in plan['nodes']}
    binding=doc['extensions']['reconstruction']
    assert binding['family']=='moss' and doc['unsupported']==[]
    assert binding['clip']['resolution']==binding['vae']['resolution']=='resolved'
    assert not any('SLG' in x['message'] or 'Skip Layer Guidance' in x['message'] for x in doc['unresolved'])
    assert by_type['CLIPLoader']['values']['type']=='moss_image'
    assert by_type['LoraLoader']['values']=={'lora_name':'moss_image/lora_latest.safetensors',
                                            'strength_model':1.,'strength_clip':1.}
    assert by_type['EmptyFlux2LatentImage']['values']=={'width':880,'height':1184,'batch_size':1}
    assert by_type['SkipLayerGuidanceDiTSimple']['values']=={
        'double_layers':'','single_layers':'22','start_percent':0.,'end_percent':.5}
    assert by_type['RandomNoise']['values']['noise_seed']==3178412173
    assert by_type['KSamplerSelect']['values']['sampler_name']=='er_sde'
    assert by_type['CFGGuider']['values']['cfg']==1.
    assert by_type['ForgeNeoBridgeScheduler']['values']=={'scheduler':'simple','steps':8,'denoise':1.}
    assert [by_type['SkipLayerGuidanceDiTSimple']['id'],0,by_type['CFGGuider']['id'],'model'] in plan['edges']
    assert not {'ModelSamplingSD3','ModelSamplingAuraFlow','ForgeNeoBridgeKSampler','ConditioningZeroOut'} & by_type.keys()
    assert len([n for n in plan['nodes'] if n['type']=='CLIPTextEncode'])==2
    assert any('CFG 1' in note for note in plan['notes'])


def test_missing_moss_components_still_build_graph_with_original_names():
    doc,plan=plan_for(inventory={})
    binding=doc['extensions']['reconstruction']
    assert binding['family']=='moss'
    assert binding['vae']['name']=='full_encoder_small_decoder'
    assert binding['vae']['resolution']=='missing'
    assert binding['unassigned_modules']==[]
    assert any(n['type']=='UNETLoader' and n['values']['unet_name']=='moss_image_sft_bf16' for n in plan['nodes'])


@pytest.mark.parametrize('name',['moss_image_base_bf16','moss_image_sft_int8_convrot'])
def test_moss_base_and_sft_names_use_native_recipe(name):
    doc,_=plan_for(TEXT.replace('moss_image_sft_bf16',name))
    assert doc['extensions']['reconstruction']['family']=='moss'


def test_nondefault_moss_shift_uses_unit_timestep_multiplier():
    _,plan=plan_for(TEXT.replace('Shift: 5','Shift: 9'))
    assert next(n for n in plan['nodes'] if n['type']=='ModelSamplingAuraFlow')['values']['shift']==9
    assert not any(n['type']=='ModelSamplingSD3' for n in plan['nodes'])


def test_slg_disabled_and_plain_sampling_stays_editable_at_cfg1():
    _,plan=plan_for(TEXT.replace('Skip Layer Guidance: True','Skip Layer Guidance: False')
                      .replace('Discard penultimate sigma: True','Discard penultimate sigma: False'))
    assert not any(n['type'] in ('SkipLayerGuidanceDiTSimple','SamplerCustomAdvanced') for n in plan['nodes'])
    sampler=next(n for n in plan['nodes'] if n['type']=='KSampler')
    assert sampler['values']['seed']==3178412173 and sampler['values']['steps']==8
    assert len([n for n in plan['nodes'] if n['type']=='CLIPTextEncode'])==2


@pytest.mark.parametrize('value',['nan','inf','not-a-number'])
def test_invalid_slg_stops_before_graph_creation(value):
    with pytest.raises(BridgeError,match='Invalid MOSS SLG'):
        plan_for(TEXT.replace('SLG End Percent: 0.5','SLG End Percent: '+value))


def test_moss_omitted_components_and_sampling_use_documented_defaults():
    _,plan=plan_for('portrait\nSteps: 50, Model: moss_image_base')
    assert next(n for n in plan['nodes'] if n['type']=='CLIPLoader')['values']['clip_name']=='qwen3.5_9b_int8_convrot.safetensors'
    sampler=next(n for n in plan['nodes'] if n['type']=='KSampler')
    assert sampler['values']['cfg']==3. and sampler['values']['sampler_name']=='euler'
    assert sampler['values']['scheduler']=='sgm_uniform'
    assert next(n for n in plan['nodes'] if n['type']=='VAELoader')['values']['vae_name']=='flux2-vae'
    assert not any(n['type']=='SkipLayerGuidanceDiTSimple' for n in plan['nodes'])


def test_moss_hires_is_not_silently_omitted():
    with pytest.raises(BridgeError):plan_for(TEXT+', Hires upscale: 2')
