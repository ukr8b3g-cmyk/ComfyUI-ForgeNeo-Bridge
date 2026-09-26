"""Regression for the supplied SDXL Hires.fix image and execution contracts."""
import json
import pytest
import torch
from fnb.bridge.binding import GraphReader
from fnb.bridge.reconstruction import reconstruct, reconstruction_plan
from fnb.bridge.schedules import step_plan
from fnb.bridge.spec import BridgeError
from fnb.bridge.workflow_adapter import ForgeSettings, ForgeTextInput, build_execution_spec

TEXT = ('portrait\nNegative prompt: blur\nSteps: 30, Sampler: Euler a, Schedule type: Automatic, '
        'CFG scale: 7, Seed: 424011486, Size: 1024x1344, Model: waiIllustriousSDXL_v170, '
        'Denoising strength: 0.5, ENSD: 31337, RNG: CPU, Hires Module 1: Use same choices, '
        'Hires upscale: 2, Discard penultimate sigma: True, Eta: 0.67, '
        'Hires upscaler: Lanczos, Hires steps: 20, Hires CFG Scale: 4.5, Version: neo-2.29.1')
INVENTORY = {'checkpoints':['waiIllustriousSDXL_v170.safetensors']}


def plan_for(text=TEXT):
    doc, positive = reconstruct(text, INVENTORY, probe=lambda *_:'sdxl')
    return reconstruction_plan(doc, positive, INVENTORY)


def test_two_pass_graph_and_exact_execution(monkeypatch):
    plan = plan_for()
    nodes = {n['id']:n for n in plan['nodes']}
    prompt = {key:{'class_type':n['type'], 'inputs':dict(n['values'])} for key,n in nodes.items()}
    for source,slot,target,key in plan['edges']:prompt[target]['inputs'][key] = [source,slot]
    samplers = [n for n in nodes.values() if n['type'] == 'ForgeNeoBridgeKSampler']
    assert len(samplers) == 2
    assert len([n for n in nodes.values() if n['type'] == 'CheckpointLoaderSimple']) == 1
    upscale = next(n for n in nodes.values() if n['type'] == 'ImageScale')
    assert upscale['values'] == {'upscale_method':'lanczos','width':2048,'height':2688,'crop':'disabled'}
    assert all('title' not in n for n in nodes.values() if n['type'] in ('ImageScale','VAEDecode','VAEEncode','EmptyLatentImage'))
    reader, clip = GraphReader(prompt), object()
    monkeypatch.setattr('fnb.bridge.workflow_adapter.model_family', lambda _:'sdxl')
    for index, (sampler, size, expected) in enumerate(zip(samplers, [(1024,1344),(2048,2688)], [(30,7,1),(20,4.5,.5)])):
        inputs = prompt[sampler['id']]['inputs']
        settings = nodes[inputs['settings'][0]]['values']
        pos = nodes[inputs['positive'][0]]
        neg = nodes[inputs['negative'][0]]
        latent_id = inputs['input_latent'][0]
        assert nodes[latent_id]['type'] == ('EmptyLatentImage' if index == 0 else 'VAEEncode')
        if index:
            assert prompt[latent_id]['inputs']['pixels'] == [upscale['id'],0]
            first_decode = prompt[upscale['id']]['inputs']['image'][0]
            assert prompt[first_decode]['inputs']['samples'] == [samplers[0]['id'],0]
        spec = build_execution_spec(ForgeSettings(settings),sampler['values'],
            ForgeTextInput(pos['values']['text'],clip,pos['id']),ForgeTextInput(neg['values']['text'],clip,neg['id']),
            object(),reader,sampler['id'],{'samples':torch.zeros(1,4,size[1]//8,size[0]//8)})
        cfg = spec.require_executable()
        assert tuple(cfg['sampling'][k] for k in ('steps','cfg','denoise')) == expected
        assert (cfg['sdxl']['original_width'],cfg['sdxl']['target_height']) == size
        assert cfg['noise']['seed'] == '424011486' and cfg['noise']['ensd'] == '31337'
        assert cfg['text']['clip_skip'] == 2  # SDXL profile default when metadata is absent.
        assert cfg['sampling']['eta_ancestral'] == .67
        assert cfg['sampling']['discard_penultimate_requested'] is True
        assert step_plan(cfg) == ((30,30,None) if index == 0 else (40,20,20))
        assert json.loads(settings['source_json'])['source']['raw_infotext'] == TEXT


@pytest.mark.parametrize('suffix', [', ADetailer model: face'])
def test_unsupported_operations_are_not_cleared(suffix):
    plan = plan_for(TEXT + suffix)
    settings = [n for n in plan['nodes'] if n['type'] == 'ForgeNeoBridgeSettings']
    assert settings and all(json.loads(n['values']['source_json'])['unsupported'] for n in settings)


@pytest.mark.parametrize('suffix', [', Hires Module 2: other_vae', ', Hires checkpoint: other',
                                    ', Hires mystery: True'])
def test_unhandled_hires_does_not_build_a_misleading_single_pass(suffix):
    with pytest.raises(BridgeError, match='HIRES_UNSUPPORTED'):
        plan_for(TEXT + suffix)


ANIMA = TEXT.replace('waiIllustriousSDXL_v170', 'waiANIMA_v10Base10').replace(
    'Hires upscale: 2', 'Hires upscale: 1.5').replace('Hires CFG Scale: 4.5', 'Hires CFG Scale: 4') + ', Shift: 3, Hires Shift: 5'


def anima_plan(text=ANIMA, image_size=(1536, 2048)):
    doc, positive = reconstruct(text, {}, probe=lambda *_:'anima', image_size=image_size)
    return reconstruction_plan(doc, positive, {})


def test_anima_hires_image_size_shift_and_two_passes():
    plan = anima_plan()
    samplers = [n for n in plan['nodes'] if n['type']=='ForgeNeoBridgeKSampler']
    settings = [n for n in plan['nodes'] if n['type']=='ForgeNeoBridgeSettings']
    assert len(samplers) == len(settings) == 2
    assert [(n['values']['steps'], n['values']['cfg'], n['values']['denoise']) for n in samplers] == [(30,7,1),(20,4,.5)]
    assert [n['values']['shift'] for n in settings] == ['3.0','5.0']
    assert all(n['values']['sampling_adjustments'] is True for n in settings)
    up = next(n for n in plan['nodes'] if n['type']=='ImageScale')
    assert (up['values']['width'],up['values']['height']) == (1536,2048)
    assert sum(n['type']=='VAEEncode' for n in plan['nodes']) == 1
    for n in settings:
        doc = json.loads(n['values']['source_json'])
        assert not doc['unsupported']
        assert doc['extensions']['hires_target']['source'] == 'source_image'
        assert any(x['code']=='HIRES_TARGET_SIZE' for x in doc['unresolved'])
    assert json.loads(settings[1]['values']['source_json'])['effective']['sampling']['shift'] == 5


@pytest.mark.parametrize('size', [None, (1536,2048)])
def test_explicit_hires_resize_has_priority(size):
    plan = anima_plan(ANIMA+', Hires resize: 1600x2112', image_size=size)
    up = next(n for n in plan['nodes'] if n['type']=='ImageScale')
    assert (up['values']['width'],up['values']['height']) == (1600,2112)


def test_text_only_hires_marks_resolution_as_estimated():
    plan = anima_plan(image_size=None)
    doc = json.loads(next(n for n in plan['nodes'] if n['type']=='ForgeNeoBridgeSettings')['values']['source_json'])
    assert doc['extensions']['hires_target'] == {'width':1536, 'height':2016, 'source':'scale_estimate', 'scale_estimate':[1536,2016]}
    assert any('estimated' in x['message'] for x in doc['unresolved'])


@pytest.mark.parametrize('value', ['0', '-1', 'NaN', 'Infinity'])
def test_invalid_hires_shift_rejected(value):
    with pytest.raises(BridgeError, match='INVALID_METADATA'):
        anima_plan(ANIMA.replace('Hires Shift: 5', 'Hires Shift: '+value))


@pytest.mark.parametrize('method', ['Latent', 'Latent (bicubic)', 'Latent (nearest-exact)'])
def test_latent_upscaling_avoids_decode_encode_roundtrip(method):
    plan = plan_for(TEXT.replace('Lanczos', method))
    assert sum(n['type'] == 'LatentUpscale' for n in plan['nodes']) == 1
    assert sum(n['type'] == 'VAEDecode' for n in plan['nodes']) == 1
    assert not any(n['type'] == 'VAEEncode' for n in plan['nodes'])


def test_second_pass_overrides_and_fallback_steps():
    plan = plan_for(TEXT.replace('Hires steps: 20','Hires steps: 0').replace('Hires upscale: 2','Hires resize: 1536x2016') +
                    ', Hires sampler: DPM++ 2M, Hires schedule type: Karras, Hires prompt: another scene')
    sampler = [n for n in plan['nodes'] if n['type'] == 'ForgeNeoBridgeKSampler'][1]
    assert sampler['values']['steps'] == 30
    assert sampler['values']['sampler_name'] == 'dpmpp_2m'
    assert sampler['values']['scheduler'] == 'karras'
    positive = next(n for n in plan['nodes'] if n.get('title') == 'ForgeNeo Bridge Hires Positive')
    assert positive['values']['text'] == 'another scene'
    assert [positive['id'],0,sampler['id'],'positive'] in plan['edges']


def test_sd15_has_two_passes_and_explicit_clip_skip_is_preserved():
    doc,positive = reconstruct(TEXT+', Clip Skip: 3',INVENTORY,probe=lambda *_:'sd15')
    plan = reconstruction_plan(doc,positive,INVENTORY)
    settings = [n['values'] for n in plan['nodes'] if n['type']=='ForgeNeoBridgeSettings']
    assert len(settings)==2 and all(n['clip_skip']=='3' and n['family']=='sd15' for n in settings)


@pytest.mark.parametrize('old,new', [('Hires upscale: 2','Hires upscale: NaN'),
                                    ('Hires steps: 20','Hires steps: -2'),
                                    ('Hires CFG Scale: 4.5','Hires CFG Scale: Infinity'),
                                    ('Denoising strength: 0.5','Denoising strength: 0')])
def test_invalid_hires_is_not_silently_used(old, new):
    with pytest.raises(BridgeError):plan_for(TEXT.replace(old,new))
