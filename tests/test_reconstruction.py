"""The drop path is deliberately independent of a loaded model or GPU."""
import json
import torch
import pytest

from fnb.bridge.binding import GraphReader
from fnb.bridge.reconstruction import reconstruct, reconstruction_plan, resolve_asset
from fnb.bridge.spec import BridgeError, default_document, finalize
from fnb.bridge.workflow_adapter import ForgeSettings, ForgeTextInput, build_execution_spec
from fnb.nodes import ForgeNeoBridgeSettings


TEXT = ('portrait <lora:ink:0.4:0.7>\nNegative prompt: blur\nSteps: 27, Sampler: Euler, '
        'Schedule type: Karras, CFG scale: 6.5, Seed: 9007199254740993, Size: 768x1024, '
        'Model: missing.safetensors, Module 1: qwen_3_06b.safetensors, Module 2: anima_vae.safetensors')


def test_exact_asset_resolution_and_missing():
    inventory={'checkpoints':['sub/a.safetensors','other/a.safetensors','unique.safetensors']}
    assert resolve_asset('unique',('checkpoints',),inventory)['name']=='unique.safetensors'
    assert resolve_asset('a.safetensors',('checkpoints',),inventory)['resolution']=='ambiguous'
    assert resolve_asset('absent',('checkpoints',),inventory)['resolution']=='missing'
    assert resolve_asset(r'C:\models\sub\unique.safetensors',('checkpoints',),inventory)['name']=='unique.safetensors'
    assert resolve_asset(r'sub\a.safetensors',('checkpoints',),
                         {'checkpoints':[r'sub\a.safetensors',r'other\a.safetensors']})['name']==r'sub\a.safetensors'
    assert resolve_asset(r'D:\models\sub\a.safetensors',('checkpoints',),inventory,
                         roots={'checkpoints':[r'D:\models']})['name']=='sub/a.safetensors'
    assert resolve_asset('anima-turbo-lora-v0.2',('loras',),
                         {'loras':[r'anima\anima-turbo-lora-v0.2.safetensors']}) == {
        'category':'loras','name':r'anima\anima-turbo-lora-v0.2.safetensors',
        'resolution':'resolved','raw_name':'anima-turbo-lora-v0.2'}


def test_missing_assets_still_create_complete_editable_graph():
    doc,positive=reconstruct(TEXT,{},probe=lambda *_:None)
    plan=reconstruction_plan(doc,positive,{})
    assert positive=='portrait '
    assert doc['extensions']['reconstruction']['model']['resolution']=='missing'
    assert any(x['code']=='ASSET_MISSING' for x in doc['unresolved'])
    kinds=[n['type'] for n in plan['nodes']]
    assert 'ForgeNeoBridgeTextEncode' in kinds and 'ForgeNeoBridgeSettings' in kinds
    sampler=next(n for n in plan['nodes'] if n['type']=='ForgeNeoBridgeKSampler')
    settings=next(n for n in plan['nodes'] if n['type']=='ForgeNeoBridgeSettings')
    assert sampler['values']['seed']=='9007199254740993'
    assert sampler['values']['steps']==27 and sampler['values']['cfg']==6.5
    assert settings['values']['width']==768 and settings['values']['height']==1024
    empty=next(n for n in plan['nodes'] if n['type']=='EmptyLatentImage')
    assert empty['values']=={'width':768,'height':1024,'batch_size':1}
    assert [empty['id'],0,sampler['id'],'input_latent'] in plan['edges']
    assert all('title' not in n for n in plan['nodes'] if n['type'] in
               ('CheckpointLoaderSimple','CLIPLoader','VAELoader','LoraLoader','EmptyLatentImage','VAEDecode','SaveImage'))
    assert set(settings['values'])==set(ForgeNeoBridgeSettings.INPUT_TYPES()['required'])|{'sampling_adjustments'}
    assert settings['values']['sampling_adjustments'] is False
    assert json.loads(settings['values']['source_json'])['source']['raw_infotext']==TEXT
    assert any(n['type']=='LoraLoader' and n['values']['strength_model']==0.7 for n in plan['nodes'])
    assert len(plan['edges'])>=8
    assert settings['title']=='ForgeNeo Bridge Settings'


@pytest.mark.parametrize('sampler',['LCM','LMS','DPM++ SDE','DPM++ 3M SDE','Res Multistep',
                                  'DDIM','UniPC','Euler CFG++','Euler a CFG++','DPM++ 2M CFG++'])
def test_shared_samplers_reconstruct_hires_and_current_spec(monkeypatch,sampler):
    import fnb.bridge.workflow_adapter as adapter
    monkeypatch.setattr(adapter,'model_family',lambda model:'sd15')
    text=(f'portrait\nSteps: 8, Sampler: {sampler}, Schedule type: Karras, CFG scale: 2, Seed: 1181113836, '
          'Size: 512x768, Model: luc, Clip skip: 2, RNG: CPU, Hires upscale: 1.5, Hires upscaler: Lanczos, '
          'Hires steps: 12, Hires CFG Scale: 4.5, Denoising strength: 0.45, Discard penultimate sigma: True')
    inventory={'checkpoints':['luc.safetensors']}
    doc,positive=reconstruct(text,inventory,probe=lambda *_:'sd15')
    plan=reconstruction_plan(doc,positive,inventory)
    prompt={n['id']:{'class_type':n['type'],'inputs':dict(n['values'])} for n in plan['nodes']}
    for source,slot,target,key in plan['edges']:prompt[target]['inputs'][key]=[source,slot]
    samplers=[n for n in plan['nodes'] if n['type']=='ForgeNeoBridgeKSampler']
    assert len(samplers)==2
    clip=object()
    for i,node in enumerate(samplers):
        inputs=prompt[node['id']]['inputs']
        settings=prompt[inputs['settings'][0]]['inputs']
        pos_id,neg_id=inputs['positive'][0],inputs['negative'][0]
        current=dict(node['values'])
        if i:current['scheduler']='ddim_uniform'
        if sampler=='UniPC' and i:current['sampler_name']='uni_pc'
        spec=build_execution_spec(ForgeSettings(settings),current,
            ForgeTextInput('edited',clip,pos_id),ForgeTextInput('',clip,neg_id),object(),
            GraphReader(prompt),node['id'],{'samples':torch.zeros((1,4,144 if i else 96,96 if i else 64))})
        cfg=spec.require_executable()
        assert cfg['sampling']['steps']==(12 if i else 8)
        assert cfg['sampling']['scheduler']==('ddim' if i else 'karras')
        assert cfg['sampling']['discard_penultimate_requested'] is True


def test_current_loader_and_lora_values_drive_new_execution_spec(monkeypatch):
    text=TEXT.replace('missing.safetensors','original.safetensors')
    inventory={'checkpoints':['original.safetensors'],'text_encoders':['qwen_3_06b.safetensors'],
               'vae':['anima_vae.safetensors'],'loras':['ink.safetensors']}
    doc,positive=reconstruct(text,inventory,probe=lambda *_:'sd15')
    plan=reconstruction_plan(doc,positive,inventory)
    prompt={n['id']:{'class_type':n['type'],'inputs':dict(n['values'])} for n in plan['nodes']}
    for source,slot,target,key in plan['edges']:
        prompt[target]['inputs'][key]=[source,slot]
    loader=next(n for n in plan['nodes'] if n['type']=='CheckpointLoaderSimple')
    lora=next(n for n in plan['nodes'] if n['type']=='LoraLoader')
    sampler=next(n for n in plan['nodes'] if n['type']=='ForgeNeoBridgeKSampler')
    pos=next(n for n in plan['nodes'] if n.get('title')=='ForgeNeo Bridge Positive')
    neg=next(n for n in plan['nodes'] if n.get('title')=='ForgeNeo Bridge Negative')
    settings=next(n for n in plan['nodes'] if n['type']=='ForgeNeoBridgeSettings')
    prompt[loader['id']]['inputs']['ckpt_name']='edited.safetensors'
    prompt[lora['id']]['inputs']['strength_model']=0.85
    prompt[lora['id']]['inputs']['strength_clip']=0.25
    clip=object()
    import fnb.bridge.workflow_adapter as adapter
    monkeypatch.setattr(adapter,'model_family',lambda model:'sd15')
    spec=build_execution_spec(ForgeSettings(settings['values']),sampler['values'],
                              ForgeTextInput(positive+' edited',clip,pos['id']),
                              ForgeTextInput('edited negative',clip,neg['id']),object(),
                              GraphReader(prompt),sampler['id'],
                              {'samples':torch.zeros((2,4,80,64)),'downscale_ratio_spacial':8})
    cfg=spec.require_executable()
    assert cfg['image']=={'width':512,'height':640,'batch_size':2}
    assert cfg['assets'][0]['relative_name']=='edited.safetensors'
    assert cfg['loras'][0]['strength_model']==0.85
    assert cfg['loras'][0]['strength_text_encoder']==0.25
    assert cfg['text']['positive_raw']=='portrait  edited'
    assert cfg['text']['negative_raw']=='edited negative'
    assert spec.document()['requested']['/text/positive_raw']=='portrait  edited'


def test_unconnected_legacy_sampler_keeps_settings_size(monkeypatch):
    doc,positive=reconstruct(TEXT,{},probe=lambda *_:'sd15')
    plan=reconstruction_plan(doc,positive,{})
    prompt={n['id']:{'class_type':n['type'],'inputs':dict(n['values'])} for n in plan['nodes']}
    for source,slot,target,key in plan['edges']:
        if key != 'input_latent':prompt[target]['inputs'][key]=[source,slot]
    sampler=next(n for n in plan['nodes'] if n['type']=='ForgeNeoBridgeKSampler')
    settings=next(n for n in plan['nodes'] if n['type']=='ForgeNeoBridgeSettings')
    pos=next(n for n in plan['nodes'] if n.get('title')=='ForgeNeo Bridge Positive')
    neg=next(n for n in plan['nodes'] if n.get('title')=='ForgeNeo Bridge Negative')
    import fnb.bridge.workflow_adapter as adapter
    monkeypatch.setattr(adapter,'model_family',lambda model:'anima')
    clip=object()
    spec=build_execution_spec(ForgeSettings(settings['values']),sampler['values'],
                              ForgeTextInput(positive,clip,pos['id']),
                              ForgeTextInput('blur',clip,neg['id']),object(),
                              GraphReader(prompt),sampler['id'])
    assert spec.require_executable()['image']=={'width':768,'height':1024,'batch_size':1}


def test_img2img_uses_encoded_latent_without_empty_latent():
    doc,positive=reconstruct(TEXT,{},probe=lambda *_:'sd15')
    doc['effective']['mode']='img2img'
    plan=reconstruction_plan(doc,positive,{})
    sampler=next(n for n in plan['nodes'] if n['type']=='ForgeNeoBridgeKSampler')
    encode=next(n for n in plan['nodes'] if n['type']=='VAEEncode')
    assert not any(n['type']=='EmptyLatentImage' for n in plan['nodes'])
    assert [encode['id'],0,sampler['id'],'input_latent'] in plan['edges']
    assert all('title' not in n for n in plan['nodes'] if n['type'] in ('LoadImage','VAEEncode'))


def test_unsupported_effect_keeps_visible_blocker():
    text=TEXT+', ADetailer model: face'
    doc,positive=reconstruct(text,{},probe=lambda *_:None)
    plan=reconstruction_plan(doc,positive,{})
    settings=next(n for n in plan['nodes'] if n['type']=='ForgeNeoBridgeSettings')
    assert any(x['path']=='/extensions' for x in doc['unsupported'])
    assert settings['title']=='ForgeNeo Bridge Settings'
    assert settings['properties']['forge_neo_bridge']['unsupported']==doc['unsupported']


def test_incomplete_hires_is_rejected_before_creating_graph():
    doc,positive=reconstruct(TEXT+', Hires upscale: 2',{},probe=lambda *_:None)
    with pytest.raises(BridgeError,match='Hires upscaler'):
        reconstruction_plan(doc,positive,{})


def test_unknown_sampler_is_kept_for_editing():
    text=TEXT.replace('Sampler: Euler','Sampler: NotYetImplemented')
    doc,positive=reconstruct(text,{},probe=lambda *_:None)
    plan=reconstruction_plan(doc,positive,{})
    sampler=next(n for n in plan['nodes'] if n['type']=='ForgeNeoBridgeKSampler')
    assert sampler['values']['sampler_name']=='NotYetImplemented'
    assert positive=='portrait '


def test_unknown_family_uses_bridge_defaults_not_example_seed():
    doc,positive=reconstruct('plain\nSteps: 20, Model: unknown.safetensors',{},probe=lambda *_:None)
    plan=reconstruction_plan(doc,positive,{})
    settings=next(n for n in plan['nodes'] if n['type']=='ForgeNeoBridgeSettings')
    sampler=next(n for n in plan['nodes'] if n['type']=='ForgeNeoBridgeKSampler')
    assert settings['values']['family']=='auto'
    assert (settings['values']['width'],settings['values']['height'])==(512,512)
    assert next(n for n in plan['nodes'] if n['type']=='EmptyLatentImage')['values']['width']==512
    assert settings['values']['clip_skip']=='auto'
    assert sampler['values']['seed']=='0' and sampler['values']['cfg']==7.0
    assert positive=='plain'


def test_header_family_correction_keeps_observed_size_for_sdxl():
    text='scene\nSteps: 18, Size: 768x1024, Model: anima_name.safetensors'
    doc,positive=reconstruct(text,{'checkpoints':['anima_name.safetensors']},probe=lambda *_:'sdxl')
    plan=reconstruction_plan(doc,positive,{})
    settings=next(n for n in plan['nodes'] if n['type']=='ForgeNeoBridgeSettings')
    assert settings['values']['family']=='sdxl'
    assert settings['values']['original_width']==768
    assert settings['values']['target_height']==1024
    assert settings['values']['shift']=='auto'


def test_unassigned_module_is_visible_without_guessing_its_role():
    doc,positive=reconstruct('scene\nSteps: 10, Model: unknown.safetensors, Module 1: unexplained.safetensors',
                             {},probe=lambda *_:None)
    assert doc['extensions']['reconstruction']['unassigned_modules'][0]['name']=='unexplained.safetensors'
    assert any('Unassigned Module 1' in x['message'] for x in doc['unresolved'])
    settings=next(n for n in reconstruction_plan(doc,positive,{})['nodes'] if n['type']=='ForgeNeoBridgeSettings')
    assert settings['title']=='ForgeNeo Bridge Settings'
    assert any('Unassigned Module 1' in x['message'] for x in settings['properties']['forge_neo_bridge']['unresolved'])


def test_sampling_scope_does_not_require_vae_but_old_scope_does():
    doc=default_document('sd15','exploratory')
    doc['effective']['assets']=[{'asset_id':role,'role':role,'component':None,'category':category,
                                 'relative_name':'test.safetensors','sha256':None,'short_hash':None,
                                 'resolution':'resolved','binding_verification':'known_loader_chain'}
                                for role,category in [('model','checkpoints'),('text_encoder','checkpoints')]]
    old=finalize(doc).document()
    assert any(x['code']=='ASSET_BINDING_REQUIRED' for x in old['unresolved'])
    doc['execution_scope']='sampling'
    new=finalize(doc).document()
    assert not any(x['code']=='ASSET_BINDING_REQUIRED' for x in new['unresolved'])
    assert new['config_hash']!=old['config_hash']


def _native_inventory(model, encoder, vae):
    return {'diffusion_models':[model], 'text_encoders':[encoder], 'vae':[vae]}


def _native_text(model, encoder, vae, *, steps=8, cfg=1, sampler='Euler', negative=''):
    return (f'portrait\nNegative prompt: {negative}\nSteps: {steps}, Sampler: {sampler}, '
            f'Schedule type: Simple, CFG scale: {cfg}, Seed: 3241723114, Size: 848x1264, '
            f'Model: {model}, Module 1: {encoder}, Module 2: {vae}')


def test_ernie_turbo_reconstructs_core_recipe_and_keeps_missing_model_visible():
    model='ernieImageTurboQT_fp8V2'
    inventory=_native_inventory('ERNIE-Image/ernie-image-turbo-QT-v2-mxfp8.safetensors',
                                 'ministral-3-3b.safetensors','flux2-vae.safetensors')
    text=_native_text(model,'ministral-3-3b','flux2-vae')
    doc,positive=reconstruct(text,inventory,probe=lambda *_:None)
    plan=reconstruction_plan(doc,positive,inventory)
    binding=doc['extensions']['reconstruction']
    assert binding['family']=='ernie'
    assert binding['model']['category']=='diffusion_models'
    assert binding['model']['resolution']=='missing'
    assert doc['effective'] is None and doc['config_hash'] is None
    assert any(issue['code']=='ASSET_MISSING' for issue in doc['unresolved'])
    assert 'CheckpointLoaderSimple' not in [node['type'] for node in plan['nodes']]
    assert not any(node['type'].startswith('ForgeNeoBridge') for node in plan['nodes'])
    clip=next(node for node in plan['nodes'] if node['type']=='CLIPLoader')
    assert clip['values']['type']=='flux2'
    latent=next(node for node in plan['nodes'] if node['type']=='EmptyFlux2LatentImage')
    sampler=next(node for node in plan['nodes'] if node['type']=='KSampler')
    assert latent['values']=={'width':848,'height':1264,'batch_size':1}
    assert [latent['id'],0,sampler['id'],'latent_image'] in plan['edges']
    assert sampler['values']['seed']==3241723114
    assert (sampler['values']['steps'],sampler['values']['cfg'])==(8,1)
    shift=next(node for node in plan['nodes'] if node['type']=='ModelSamplingSD3')
    assert shift['values']=={'shift':3.0}
    assert [shift['id'],0,sampler['id'],'model'] in plan['edges']
    assert 'ModelSamplingAuraFlow' not in [node['type'] for node in plan['nodes']]
    assert plan['qualification']=='native_comfy_not_forge_parity'


def test_modern_model_prefers_diffusion_category_when_basename_exists_twice():
    inventory={'diffusion_models':['ernie-image-turbo.safetensors'],
               'checkpoints':['ernie-image-turbo.safetensors'],
               'text_encoders':['ministral-3-3b.safetensors'], 'vae':['flux2-vae.safetensors']}
    text=_native_text('ernie-image-turbo.safetensors','ministral-3-3b','flux2-vae')
    doc,positive=reconstruct(text,inventory,probe=lambda *_:None)
    model=doc['extensions']['reconstruction']['model']
    assert model['resolution']=='resolved' and model['category']=='diffusion_models'
    assert 'CheckpointLoaderSimple' not in [n['type'] for n in reconstruction_plan(doc,positive,inventory)['nodes']]


@pytest.mark.parametrize('family,model,encoder,vae,clip_type,latent,patch',[
    ('zimage','z_image_turbo_bf16.safetensors','qwen_3_4b.safetensors','ae.safetensors','lumina2','EmptySD3LatentImage',True),
    ('qwen','qwen_image_2512.safetensors','qwen_2.5_vl_7b_fp8_scaled.safetensors','qwen_image_vae.safetensors','qwen_image','EmptySD3LatentImage',False),
    ('qwen21','qwen_image_2.1_int8_convrot.safetensors','qwen3vl_8b_int8_convrot.safetensors','qwen_image_2.1_vae_bf16.safetensors','qwen_image','EmptyLatentImage',False),
    ('krea2','krea2_turbo_bf16.safetensors','qwen3vl_4b_fp8_scaled.safetensors','qwen_image_vae.safetensors','krea2','EmptyLatentImage',False),
])
def test_native_family_loaders_and_latents(family,model,encoder,vae,clip_type,latent,patch):
    inventory=_native_inventory(model,encoder,vae)
    text=_native_text(model,encoder.removesuffix('.safetensors'),vae.removesuffix('.safetensors'))
    doc,positive=reconstruct(text,inventory,probe=lambda *_:None)
    plan=reconstruction_plan(doc,positive,inventory)
    assert doc['extensions']['reconstruction']['family']==family
    assert next(node for node in plan['nodes'] if node['type']=='CLIPLoader')['values']['type']==clip_type
    assert any(node['type']==latent for node in plan['nodes'])
    assert ('ModelSamplingAuraFlow' in [node['type'] for node in plan['nodes']]) == patch


def test_flux2_klein_uses_native_scheduler_and_qwen_encoder():
    model='FLUX2/flux-2-klein-9b.safetensors'
    encoder='qwen_3_8b_fp8mixed.safetensors'
    inventory=_native_inventory(model,encoder,'flux2-vae.safetensors')
    doc,positive=reconstruct(_native_text(model,encoder,'flux2-vae',steps=4),inventory,probe=lambda *_:'flux2')
    plan=reconstruction_plan(doc,positive,inventory)
    kinds={node['type'] for node in plan['nodes']}
    assert doc['extensions']['reconstruction']['family']=='flux2_klein'
    assert {'Flux2Scheduler','CFGGuider','RandomNoise','KSamplerSelect','SamplerCustomAdvanced',
            'EmptyFlux2LatentImage'} <= kinds
    assert next(node for node in plan['nodes'] if node['type']=='CLIPLoader')['values']['type']=='flux2'
    assert 'CheckpointLoaderSimple' not in kinds


def test_flux2_klein_9b_corrects_stale_forge_modules():
    inventory={'diffusion_models':['FLUX2/flux-2-klein-9b.safetensors'],
               'text_encoders':['qwen_3_4b.safetensors','qwen_3_8b_fp8mixed.safetensors'],
               'vae':['ae.safetensors','flux2-vae.safetensors','full_encoder_small_decoder.safetensors']}
    text=_native_text('flux-2-klein-9b','qwen_3_4b','ae',steps=20)
    doc,positive=reconstruct(text,inventory,probe=lambda *_:'flux2')
    plan=reconstruction_plan(doc,positive,inventory)
    clip=next(node for node in plan['nodes'] if node['type']=='CLIPLoader')
    vae=next(node for node in plan['nodes'] if node['type']=='VAELoader')
    assert clip['values']['clip_name']=='qwen_3_8b_fp8mixed.safetensors'
    assert vae['values']['vae_name']=='full_encoder_small_decoder.safetensors'
    assert clip['properties']['forge_neo_bridge']['raw_name']=='qwen_3_4b'
    assert vae['properties']['forge_neo_bridge']['raw_name']=='ae'
    assert len(doc['extensions']['reconstruction']['component_corrections'])==2
    assert sum(issue['code']=='MODULE_MISMATCH_CORRECTED' for issue in doc['unresolved'])==2


def test_flux2_klein_missing_compatible_encoder_never_uses_4b_for_9b():
    inventory={'diffusion_models':['flux-2-klein-9b.safetensors'],
               'text_encoders':['qwen_3_4b.safetensors'], 'vae':['flux2-vae.safetensors']}
    doc,positive=reconstruct(_native_text('flux-2-klein-9b','qwen_3_4b','flux2-vae'),
                             inventory,probe=lambda *_:'flux2')
    clip=doc['extensions']['reconstruction']['clip']
    assert clip['name']=='qwen_3_8b_fp8mixed' and clip['resolution']=='missing'
    assert any(issue['code']=='ASSET_MISSING' for issue in doc['unresolved'])


def test_flux2_klein_4b_corrects_stale_modules_without_choosing_9b_assets():
    inventory={'diffusion_models':['flux-2-klein-4b.safetensors'],
               'text_encoders':['qwen_3_4b.safetensors','qwen_3_8b_fp8mixed.safetensors'],
               'vae':['ae.safetensors','flux2-vae.safetensors']}
    doc,positive=reconstruct(_native_text('flux-2-klein-4b','qwen_3_8b_fp8mixed','ae'),
                             inventory,probe=lambda *_:'flux2')
    plan=reconstruction_plan(doc,positive,inventory)
    assert next(n for n in plan['nodes'] if n['type']=='CLIPLoader')['values']['clip_name']=='qwen_3_4b.safetensors'
    assert next(n for n in plan['nodes'] if n['type']=='VAELoader')['values']['vae_name']=='flux2-vae.safetensors'


def test_zimage_keeps_forge_shift_and_native_modules():
    inventory=_native_inventory('z_image_turbo_bf16.safetensors',
                                 'qwen_3_4b.safetensors','ae.safetensors')
    text=_native_text('z_image_turbo_bf16','qwen_3_4b','ae',steps=9)
    text=text.replace('Schedule type: Simple','Schedule type: Beta').replace('Seed: 3241723114',
                     'Shift: 9, Seed: 3129797896')
    doc,positive=reconstruct(text,inventory,probe=lambda *_:'zimage')
    plan=reconstruction_plan(doc,positive,inventory)
    assert doc['extensions']['reconstruction']['component_corrections']==[]
    assert next(n for n in plan['nodes'] if n['type']=='ModelSamplingAuraFlow')['values']['shift']==9
    sampler=next(n for n in plan['nodes'] if n['type']=='KSampler')
    assert sampler['values']['seed']==3129797896
    assert (sampler['values']['steps'],sampler['values']['scheduler'])==(9,'beta')


def test_qwen_edit_2511_lcm_lightning_builds_its_core_conditioning():
    inventory={'diffusion_models':['qwen/qwen_image_edit_2511_int8_convrot.safetensors'],
               'text_encoders':['qwen_2.5_vl_7b_fp8_scaled.safetensors'],
               'vae':['qwen_image_vae.safetensors'],
               'loras':['Qwen/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors']}
    text=_native_text('qwen_image_edit_2511_int8_convrot','qwen_2.5_vl_7b_fp8_scaled',
                      'qwen_image_vae',steps=8,sampler='LCM')
    text=text.replace('portrait',
        'portrait <lora:Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16:1>',1)
    text=text.replace('Schedule type: Simple','Schedule type: Normal')
    text+=', Shift: 9'  # Qwen's Forge engine does not consume the UI Shift setting.
    doc,positive=reconstruct(text,inventory,probe=lambda *_:'qwen')
    plan=reconstruction_plan(doc,positive,inventory)
    kinds={n['type'] for n in plan['nodes']}
    assert doc['unsupported']==[]
    assert plan['name']=='ForgeNeo Bridge qwen_edit'
    assert {'LoraLoaderModelOnly','CLIPTextEncode','EmptySD3LatentImage','KSampler'} <= kinds
    assert not {'TextEncodeQwenImageEditPlus','CFGNorm','ModelSamplingAuraFlow','LoadImage'} & kinds
    assert next(n for n in plan['nodes'] if n['type']=='CLIPTextEncode')['values']['text']=='portrait '
    sampler=next(n for n in plan['nodes'] if n['type']=='KSampler')
    assert (sampler['values']['steps'],sampler['values']['sampler_name'],
            sampler['values']['scheduler'])==(8,'lcm','normal')


def test_qwen_edit_img2img_connects_source_to_edit_conditioning():
    inventory=_native_inventory('qwen_image_edit_2511_int8_convrot.safetensors',
                                 'qwen_2.5_vl_7b_fp8_scaled.safetensors','qwen_image_vae.safetensors')
    text=_native_text('qwen_image_edit_2511_int8_convrot','qwen_2.5_vl_7b_fp8_scaled',
                      'qwen_image_vae')+', Denoising strength: 0.8'
    doc,positive=reconstruct(text,inventory,probe=lambda *_:'qwen')
    plan=reconstruction_plan(doc,positive,inventory)
    source=next(n for n in plan['nodes'] if n['type']=='LoadImage')
    pos=next(n for n in plan['nodes'] if n['type']=='TextEncodeQwenImageEditPlus')
    assert [source['id'],0,pos['id'],'image1'] in plan['edges']


def test_flux1_uses_dual_encoder_even_when_module_order_is_reversed():
    model='flux1-dev.safetensors'
    inventory={'diffusion_models':[model], 'text_encoders':['clip_l.safetensors','t5xxl_fp16.safetensors'],
               'vae':['ae.safetensors']}
    text=_native_text(model,'t5xxl_fp16','ae')+', Module 3: clip_l'
    doc,positive=reconstruct(text,inventory,probe=lambda *_:None)
    plan=reconstruction_plan(doc,positive,inventory)
    loader=next(node for node in plan['nodes'] if node['type']=='DualCLIPLoader')
    assert loader['values']['clip_name1']=='clip_l.safetensors'
    assert loader['values']['clip_name2']=='t5xxl_fp16.safetensors'
    assert loader['values']['type']=='flux'


def test_native_recipe_refuses_to_drop_hires_silently():
    inventory=_native_inventory('ernie-image-turbo.safetensors',
                                 'ministral-3-3b.safetensors','flux2-vae.safetensors')
    base=_native_text('ernie-image-turbo','ministral-3-3b','flux2-vae')
    for suffix,code in ((', Hires upscale: 2','NATIVE_EFFECT_UNSUPPORTED'),):
        doc,positive=reconstruct(base+suffix,inventory,probe=lambda *_:None)
        with pytest.raises(BridgeError) as caught:
            reconstruction_plan(doc,positive,inventory)
        assert caught.value.code==code


@pytest.mark.parametrize('name',('wan2.2','lumina2','pid_model'))
def test_unverified_families_stop_before_a_wrong_checkpoint_recipe(name):
    with pytest.raises(BridgeError,match='no verified image workflow recipe'):
        reconstruct(f'portrait\nSteps: 20, Model: {name}',{},probe=lambda *_:None)

@pytest.mark.parametrize('model,encoder,vae',[
    ('z_image_turbo_bf16','qwen3_4b_f32-q8_0','ae'),
    ('ernie-image-turbo','ministral-3-3b','flux2-vae'),
    ('qwen_image_edit_2511','qwen_2.5_vl_7b_fp8_scaled','qwen_image_vae'),
    ('flux1-dev','clip_l','ae'),
])
def test_recorded_noop_clip_skip_never_blocks_missing_asset_workflow(model,encoder,vae):
    text=_native_text(model,encoder,vae)+', Clip skip: 2'
    doc,positive=reconstruct(text,{},probe=lambda *_:None)
    plan=reconstruction_plan(doc,positive,{})
    assert not any(n['type']=='CLIPSetLastLayer' for n in plan['nodes'])
    assert plan['source']['raw_infotext']==text
    assert any('Clip Skip' in n for n in plan['notes'])
    assert next(n for n in plan['nodes'] if n['type']=='UNETLoader')['values']['unet_name']==model


@pytest.mark.parametrize('version,node,port',[
    ('','TextEncodeQwenImageEdit','image'),
    ('_2509','TextEncodeQwenImageEditPlus','image1'),
    ('_2511','TextEncodeQwenImageEditPlus','image1'),
])
def test_qwen_edit_versions_use_reference_conditioning_only_with_source(version,node,port):
    model='qwen_image_edit'+version
    text=_native_text(model,'qwen_2.5_vl_7b_fp8_scaled','qwen_image_vae',cfg=4,negative='blur')+', Denoising strength: 0.8'
    doc,pos=reconstruct(text,{},probe=lambda *_:None)
    plan=reconstruction_plan(doc,pos,{})
    source=next(n for n in plan['nodes'] if n['type']=='LoadImage')
    encode=next(n for n in plan['nodes'] if n['type']==node)
    assert [source['id'],0,encode['id'],port] in plan['edges']
    negative=next(n for n in plan['nodes'] if n['type']==node and n['values']['prompt']=='blur')
    assert [source['id'],0,negative['id'],port] in plan['edges']
    assert next(e[:2] for e in plan['edges'] if e[2:]==[encode['id'],'vae'])==next(e[:2] for e in plan['edges'] if e[2:]==[negative['id'],'vae'])
    assert any('original input' in n for n in plan['notes'])


def test_flux_distilled_guidance_and_schnell_defaults():
    for variant in ('dev','schnell'):
        text=f'photo\nModel: flux1-{variant}, Distilled CFG Scale: 2.25, Clip skip: 2'
        # Metadata requires the standard Steps header; omit other fields for profile defaults.
        text=text.replace('\nModel:', '\nSteps: '+('4' if variant=='schnell' else '20')+', Model:')
        doc,pos=reconstruct(text,{'text_encoders':['t5xxl_fp8_e4m3fn.safetensors','clip_l.safetensors']},probe=lambda *_:None)
        plan=reconstruction_plan(doc,pos,{})
        assert doc['extensions']['reconstruction']['clip_secondary']['name']=='t5xxl_fp8_e4m3fn.safetensors'
        guidance=[n for n in plan['nodes'] if n['type']=='FluxGuidance']
        assert len(guidance)==(2 if variant=='dev' else 0)
        assert all(n['values']['guidance']==2.25 for n in guidance)
        assert next(n for n in plan['nodes'] if n['type']=='KSampler')['values']['cfg']==1


@pytest.mark.parametrize('sampler,schedule,expected',[
    ('Euler','Automatic',('euler','normal')),('UniPC','DDIM',('uni_pc','ddim_uniform')),
])
def test_native_core_sampling_aliases(sampler,schedule,expected):
    text=_native_text('z_image_turbo','qwen_3_4b','ae',sampler=sampler).replace('Schedule type: Simple',f'Schedule type: {schedule}')
    doc,pos=reconstruct(text,{},probe=lambda *_:None)
    plan=reconstruction_plan(doc,pos,{})
    n=next(n for n in plan['nodes'] if n['type']=='KSampler')
    assert (n['values']['sampler_name'],n['values']['scheduler'])==expected


def test_qwen_lcm_discard_preserves_recorded_eight_steps_and_seed():
    text=_native_text('qwen_image_edit_2511','qwen_2.5_vl_7b_fp8_scaled','qwen_image_vae',sampler='LCM')+', Discard penultimate sigma: True'
    doc,pos=reconstruct(text,{},probe=lambda *_:None)
    plan=reconstruction_plan(doc,pos,{})
    assert next(n for n in plan['nodes'] if n['type']=='ForgeNeoBridgeScheduler')['values']['steps']==8
    assert next(n for n in plan['nodes'] if n['type']=='RandomNoise')['values']['noise_seed']==3241723114
    assert any(n['type']=='SamplerCustomAdvanced' for n in plan['nodes'])


@pytest.mark.parametrize('denoise,expected_steps,expected',[(1.0,5,[5,4,3,2,0]),(0.5,9,[5,4,3,2,0])])
def test_discard_scheduler_keeps_terminal_zero_and_step_count(monkeypatch,denoise,expected_steps,expected):
    import sys,types
    from fnb.nodes import ForgeNeoBridgeScheduler
    comfy=types.ModuleType('comfy');samplers=types.ModuleType('comfy.samplers')
    def calculate(model,scheduler,steps):
        assert steps==expected_steps
        return torch.arange(steps,-1,-1,dtype=torch.float32)
    samplers.calculate_sigmas=calculate;comfy.samplers=samplers
    monkeypatch.setitem(sys.modules,'comfy',comfy);monkeypatch.setitem(sys.modules,'comfy.samplers',samplers)
    model=types.SimpleNamespace(get_model_object=lambda _:None)
    sigmas,=ForgeNeoBridgeScheduler().schedule(model,'normal',4,denoise)
    assert sigmas.tolist()==expected
