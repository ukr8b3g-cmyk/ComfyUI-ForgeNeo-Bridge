"""Regression cases from the 2026-09 audit; no pretrained weights required."""
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from fnb.bridge import text, workflow_adapter as adapter
from fnb.bridge.binding import GraphReader
from fnb.bridge.classic import apply_emphasis
from fnb.bridge.noise import make_rng
from fnb.bridge.reconstruction import reconstruct, reconstruction_plan
from fnb.bridge.spec import BridgeError, MAX_SEED, default_document, finalize


@pytest.fixture
def encoder(monkeypatch):
    calls=[]
    class Encoder:
        def __init__(self,*args):pass
        def encode(self,line,device):
            calls.append(line)
            return torch.ones(1,3,4),{'pooled_output':torch.ones(1,4)},{'tokens':line}
    monkeypatch.setattr(text,'ensure_clip_family',lambda *args:None)
    monkeypatch.setattr(text,'prepare_clip',lambda clip:('cpu',NS()))
    monkeypatch.setattr(text,'clip_identity',lambda clip:())
    monkeypatch.setattr(text,'ClassicEncoder',Encoder)
    monkeypatch.setattr(text,'AnimaEncoder',Encoder)
    return calls


@pytest.mark.parametrize('family',['sd15','sdxl','anima'])
@pytest.mark.parametrize('sampler',['euler','euler_cfg_pp','euler_ancestral_cfg_pp','dpmpp_2m_cfg_pp'])
@pytest.mark.parametrize('adjustments',[False,True])
@pytest.mark.parametrize('cfg',[1.,7.])
@pytest.mark.parametrize('negative',['','blur'])
def test_encode_preserves_cfgpp_negative(encoder,family,sampler,adjustments,cfg,negative):
    d=default_document(family,'exploratory');c=d['effective']
    c['sampling'].update(cfg=cfg,sampler=sampler,adjustments=adjustments)
    c['text'].update(positive_raw='portrait',negative_raw=negative)
    bundle=text.encode_bundle(finalize(d),object(),{})
    assert bool(bundle.negative)==(cfg!=1 or sampler!='euler')
    assert negative in encoder if bundle.negative else encoder==['portrait']


@pytest.mark.parametrize('adjustments',[False,True])
def test_public_workflow_keeps_negative_at_cfg1(monkeypatch,encoder,adjustments):
    import fnb.bridge.runtime as runtime
    import fnb.bridge.noise as noise
    d,p=reconstruct('portrait\nSteps: 4, Sampler: Euler CFG++, CFG scale: 1, Seed: 42, Size: 64x64, Model: sd15',
                    {'checkpoints':['sd15.safetensors']},probe=lambda *_:'sd15')
    plan=reconstruction_plan(d,p,{})
    prompt={n['id']:{'class_type':n['type'],'inputs':dict(n['values'])} for n in plan['nodes']}
    for src,slot,dst,key in plan['edges']:prompt[dst]['inputs'][key]=[src,slot]
    sampler=next(n for n in plan['nodes'] if n['type']=='ForgeNeoBridgeKSampler')
    settings=next(n for n in plan['nodes'] if n['type']=='ForgeNeoBridgeSettings')['values']
    settings['sampling_adjustments']=adjustments
    txt=[n for n in plan['nodes'] if n['type']=='ForgeNeoBridgeTextEncode'];clip=object()
    monkeypatch.setattr(adapter,'model_family',lambda model:'sd15')
    monkeypatch.setattr(adapter,'verify_binding',lambda *args:{'hash':'test'})
    monkeypatch.setattr(noise,'generate_noise',lambda *args:({'samples':torch.zeros(1,4,8,8)},None))
    monkeypatch.setitem(sys.modules,'latent_preview',NS(prepare_callback=lambda *args:None))
    def execute(spec,model,bundle,noise,latent,binding,level,callback):
        assert spec.config['sampling']['cfg']==1
        assert bundle.negative and '' in encoder
        return latent,None,{},{}
    monkeypatch.setattr(runtime,'execute',execute)
    result=adapter.sample_workflow(object(),adapter.ForgeTextInput('portrait',clip,txt[0]['id']),
        adapter.ForgeTextInput('',clip,txt[1]['id']),adapter.ForgeSettings(settings),sampler['values'],
        {'samples':torch.zeros(1,4,8,8)},prompt=prompt,unique_id=sampler['id'])
    assert result['result'][0]['samples'].shape==(1,4,8,8)


def lora(name,strength,node):
    return {'name':name,'strength':strength,'category':'loras','node_id':node}


@pytest.mark.parametrize('m,c,expected',[
    ([lora('a',.7,'a')],[],[('a',.7,0)]),
    ([lora('a',.7,'a')],[lora('a',0,'a')],[('a',.7,0)]),
    ([lora('a',1,'a'),lora('b',1,'b')],[lora('b',.5,'b')],[('a',1,0),('b',1,.5)]),
    ([lora('a',.2,'only'),lora('a',.7,'both')],[lora('a',.3,'both')],[('a',.2,0),('a',.7,.3)]),
    ([lora('a',.7,'both')],[lora('a',.1,'only'),lora('a',.3,'both')],[('a',0,.1),('a',.7,.3)]),
    ([lora('a',.2,'model-only'),lora('b',.7,'both')],
     [lora('b',.3,'both'),lora('a',.1,'clip-only')],
     [('a',.2,0),('b',.7,.3),('a',0,.1)]),
    ([],[lora('a',.3,'a')],[('a',0,.3)]),
])
def test_lora_independent_branches_and_duplicate_events(m,c,expected):
    root={'name':'model','category':'checkpoints'}
    assets,loras=adapter._resources([root,*m],[root,*c]);names={a['asset_id']:a['relative_name'] for a in assets}
    assert [(names[x['asset_id']],x['strength_model'],x['strength_text_encoder']) for x in loras]==expected
    for branch,key in ((m,'strength_model'),(c,'strength_text_encoder')):
        assert [(names[x['asset_id']],x[key]) for x in loras if x[key]]==[(x['name'],x['strength']) for x in branch if x['strength']]


def test_lora_conflicting_common_order_rejected():
    root={'name':'model','category':'checkpoints'}
    with pytest.raises(BridgeError,match='different LoRA order'):
        adapter._resources([root,lora('a',1,'a'),lora('b',1,'b')],
                           [root,lora('b',1,'other-b'),lora('a',1,'other-a')])


def test_lora_shared_node_separates_independent_same_name_events():
    prompt={'1':{'class_type':'CheckpointLoaderSimple','inputs':{'ckpt_name':'base.safetensors'}},
            '2':{'class_type':'LoraLoaderModelOnly','inputs':{'model':['1',0],'lora_name':'same.safetensors','strength_model':.2}},
            '3':{'class_type':'LoraLoader','inputs':{'model':['2',0],'clip':['1',1],'lora_name':'shared.safetensors',
                                                   'strength_model':.7,'strength_clip':.3}},
            '4':{'class_type':'LoraLoader','inputs':{'model':['3',0],'clip':['3',1],'lora_name':'same.safetensors',
                                                   'strength_model':0.,'strength_clip':.1}}}
    reader=GraphReader(prompt)
    assets,loras=adapter._resources(reader.trace(['3',0],'model'),reader.trace(['4',1],'text_encoder'))
    names={a['asset_id']:a['relative_name'] for a in assets}
    assert [(names[x['asset_id']],x['strength_model'],x['strength_text_encoder']) for x in loras]==[
        ('same.safetensors',.2,0),('shared.safetensors',.7,.3),('same.safetensors',0,.1)]


def test_model_only_graph_provenance():
    prompt={'1':{'class_type':'UNETLoader','inputs':{'unet_name':'base.safetensors'}},
            '2':{'class_type':'LoraLoaderModelOnly','inputs':{'model':['1',0],'lora_name':'a.safetensors','strength_model':.7}},
            '3':{'class_type':'LoraLoaderModelOnly','inputs':{'model':['2',0],'lora_name':'a.safetensors','strength_model':.3}}}
    chain=GraphReader(prompt).trace(['3',0],'model')
    assert [x['node_id'] for x in chain if x['category']=='loras']==['2','3']


@pytest.mark.parametrize('source',['CPU','NV'])
@pytest.mark.parametrize('adjustments',[False,True])
def test_active_seed_boundaries_before_rng_creation(source,adjustments):
    d=default_document('sd15','exploratory');c=d['effective'];c['image']['batch_size']=2
    c['sampling']['adjustments']=adjustments
    c['noise'].update(source=source,seed='42',subseed=str(MAX_SEED),subseed_strength=0.,ensd='0')
    finalize(d)  # Unused subseed offset must not block generation.
    rng=make_rng(c,(4,4,4),'cpu');assert torch.isfinite(rng.next()).all() and torch.isfinite(rng.next()).all()
    c['noise']['subseed_strength']=.5
    with pytest.raises(BridgeError,match='subseed.*uint64'):finalize(d)
    with pytest.raises(BridgeError,match='subseed.*uint64'):make_rng(c,(4,4,4),'cpu')
    c['noise'].update(subseed='0',seed=str(MAX_SEED))
    finalize(d)  # Variation keeps the base seed constant across the batch.
    rng=make_rng(c,(4,4,4),'cpu');assert torch.isfinite(rng.next()).all() and torch.isfinite(rng.next()).all()
    c['noise']['subseed_strength']=0
    with pytest.raises(BridgeError,match='seed.*uint64'):finalize(d)
    with pytest.raises(BridgeError,match='seed.*uint64'):make_rng(c,(4,4,4),'cpu')


@pytest.mark.parametrize('source',['CPU','NV'])
def test_off_ensd_retained_but_not_applied(source):
    d=default_document('sd15','exploratory');c=d['effective'];c['image']['batch_size']=1
    c['sampling']['adjustments']=False
    c['noise'].update(source=source,seed=str(MAX_SEED),ensd='31337',subseed_strength=0.)
    assert finalize(d).config['noise']['ensd']=='31337'
    rng=make_rng(c,(4,4,4),'cpu');assert rng.cfg.eta_noise_seed_delta==0
    assert torch.isfinite(rng.next()).all() and torch.isfinite(rng.next()).all()
    c['sampling']['adjustments']=True
    with pytest.raises(BridgeError,match='ensd.*uint64'):finalize(d)
    with pytest.raises(BridgeError,match='ensd.*uint64'):make_rng(c,(4,4,4),'cpu')


@pytest.mark.parametrize('shift',[0,3.5,None])
@pytest.mark.parametrize('family',['zimage','ernie'])
def test_zero_shift_import_preserves_request(family,shift):
    value='' if shift is None else f', Shift: {shift}'
    model='z_image_turbo_bf16' if family=='zimage' else 'ernie_image_turbo'
    raw=f'portrait\nSteps: 9, Sampler: Euler, Schedule type: Simple, CFG scale: 1, Seed: 3515071732, Size: 1024x1536, Model: {model}, Clip skip: 2{value}'
    d,p=reconstruct(raw,{},probe=lambda *_:None);plan=reconstruction_plan(d,p,{})
    assert d['source']['raw_infotext']==raw
    if shift is not None:assert d['requested']['/sampling/shift']==shift
    patch=next(n for n in plan['nodes'] if n['type'] in ('ModelSamplingAuraFlow','ModelSamplingSD3'))
    assert patch['values']['shift']==(3. if shift in (0,None) else shift)


def test_negative_shift_still_rejected():
    with pytest.raises(BridgeError):
        reconstruct('portrait\nSteps: 9, Model: z_image_turbo_bf16, Shift: -1',{},probe=lambda *_:None)


def test_zero_mean_emphasis_boundary_is_explicit(encoder,monkeypatch):
    out=apply_emphasis('Original',torch.zeros(1,3,4),torch.ones(1,3))
    assert not torch.isfinite(out).all()
    monkeypatch.setattr(text.ClassicEncoder,'encode',lambda *args:(out,{}, {'tokens':[]}))
    with pytest.raises(BridgeError,match='NONFINITE_CONDITIONING'):
        text.encode_bundle(finalize(default_document('sd15','exploratory')),object(),{})
