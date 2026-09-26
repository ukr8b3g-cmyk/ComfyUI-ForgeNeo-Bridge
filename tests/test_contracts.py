"""Backend edge/CFG/lifecycle regression tests; fake models do not certify GPU parity."""
import copy
import json
import sys
import types
from types import SimpleNamespace as NS
import pytest
import torch
from fnb.bridge.spec import BridgeError, finalize, default_document
from fnb.bridge.binding import GraphReader, verify_binding, category_path, check_latent, validate_model
from fnb.bridge.workflow import graph_plan
from fnb.bridge.runtime import Denoiser, pick

@pytest.fixture
def assets(tmp_path,monkeypatch,document):
    roots={k:tmp_path/k for k in ('checkpoints','diffusion_models','text_encoders','vae','loras','input')}
    for p in roots.values():p.mkdir()
    for a in document['effective']['assets']:(roots[a['category']]/a['relative_name']).write_bytes(b'fixture')
    fp=NS(get_input_directory=lambda:str(roots['input']),get_folder_paths=lambda k:[str(roots[k])],get_full_path=lambda k,n:str(roots[k]/n) if (roots[k]/n).exists() else None)
    monkeypatch.setitem(sys.modules,'folder_paths',fp)
    return roots

def graph():
    return {'1':{'class_type':'UNETLoader','inputs':{'unet_name':'test.safetensors','weight_dtype':'default'}},
            '2':{'class_type':'ForgeCompatNoise','inputs':{'model':['1',0]}}}

def test_binding_actual_edge_and_file(assets,document):
    b=verify_binding(document['effective'],GraphReader(graph()),'2','model','model')
    assert b['assets'][0]['size']==7 and len(b['hash'])==64
    (assets['diffusion_models']/'test.safetensors').write_bytes(b'changed')
    assert verify_binding(document['effective'],GraphReader(graph()),'2','model','model')['hash']!=b['hash']

def test_binding_not_manifest_flag(assets,document):
    g=graph();g['1']['inputs']['unet_name']='wrong.safetensors'
    with pytest.raises(BridgeError,match='ASSET_BINDING_MISMATCH'):verify_binding(document['effective'],GraphReader(g),'2','model','model')

@pytest.mark.parametrize('mutation',['cycle','custom','slot','dynamic'])
def test_graph_rejects_unverified_edges(assets,document,mutation):
    g=graph()
    if mutation=='cycle':g['1']={'class_type':'LoraLoader','inputs':{'model':['1',0],'lora_name':'x','strength_model':1}}
    elif mutation=='custom':g['1']['class_type']='ArbitraryModelPatch'
    elif mutation=='slot':g['2']['inputs']['model'][1]=1
    else:g['1']['inputs']['unet_name']=['other',0]
    with pytest.raises(BridgeError,match='UNVERIFIED'):verify_binding(document['effective'],GraphReader(g),'2','model','model')

def test_duplicate_lora_rejected(assets,document):
    a={'asset_id':'l','role':'lora','component':None,'category':'loras','relative_name':'l.safetensors','sha256':None,'short_hash':None,'resolution':'resolved','binding_verification':'known_loader_chain'}
    cfg=document['effective'];cfg['assets'].append(a);cfg['loras']=[dict(asset_id='l',order=0,strength_model=1.,strength_text_encoder=0.)]
    (assets['loras']/a['relative_name']).write_bytes(b'lora')
    g=graph();g['3']={'class_type':'LoraLoaderModelOnly','inputs':{'model':['1',0],'lora_name':a['relative_name'],'strength_model':1.}}
    g['2']['inputs']['model']=['3',0]
    verify_binding(cfg,GraphReader(g),'2','model','model')
    g['4']=copy.deepcopy(g['3']);g['4']['inputs']['model']=['3',0];g['2']['inputs']['model']=['4',0]
    with pytest.raises(BridgeError,match='double application'):verify_binding(cfg,GraphReader(g),'2','model','model')


def test_core_lora_metadata_attachment_is_not_a_model_patch(monkeypatch):
    package=types.ModuleType('comfy');package.__path__=[]
    base=types.ModuleType('comfy.model_base');sampling=types.ModuleType('comfy.model_sampling')
    class Anima:pass
    class CONST:pass
    class EPS:pass
    class V_PREDICTION:pass
    base.Anima=Anima;base.SDXL=type('SDXL',(),{});base.BaseModel=type('BaseModel',(),{})
    sampling.CONST=CONST;sampling.EPS=EPS;sampling.V_PREDICTION=V_PREDICTION
    package.model_base=base;package.model_sampling=sampling
    monkeypatch.setitem(sys.modules,'comfy',package)
    monkeypatch.setitem(sys.modules,'comfy.model_base',base)
    monkeypatch.setitem(sys.modules,'comfy.model_sampling',sampling)
    cfg=default_document('anima')['effective']
    cfg['loras']=[{'asset_id':'lora_0','strength_model':1.0,'strength_text_encoder':1.0,'order':0}]
    model=NS(model=Anima(),get_model_object=lambda key:CONST(),object_patches={},
             attachments={'lora_metadata':{'format':'pt'}},model_options={})
    assert validate_model(model,cfg)=='anima'
    cfg['loras']=[]
    with pytest.raises(BridgeError,match='UNVERIFIED_PATCH_CHAIN'):validate_model(model,cfg)
    cfg['loras']=[{'asset_id':'lora_0','strength_model':1.0,'strength_text_encoder':1.0,'order':0}]
    model.attachments['custom_adapter']=object()
    with pytest.raises(BridgeError,match='UNVERIFIED_PATCH_CHAIN'):validate_model(model,cfg)
    del model.attachments['custom_adapter']
    model.object_patches={'model_sampling':object()}
    with pytest.raises(BridgeError,match='UNVERIFIED_PATCH_CHAIN'):validate_model(model,cfg)

def test_file_symlink_escape(assets,tmp_path):
    outside=tmp_path/'private';outside.write_text('x')
    try:(assets['input']/'escape').symlink_to(outside)
    except OSError:pytest.skip('Symlink privilege not available')
    with pytest.raises(BridgeError,match='INVALID_PATH'):category_path('input','escape')

def test_registered_model_symlink_uses_actual_target(assets,tmp_path):
    outside=tmp_path/'shared.safetensors';outside.write_bytes(b'model')
    try:(assets['checkpoints']/'linked.safetensors').symlink_to(outside)
    except OSError:pytest.skip('Symlink privilege not available')
    assert category_path('checkpoints','linked.safetensors')==outside.resolve()

def test_model_resolver_cannot_return_unregistered_path(assets,tmp_path,monkeypatch):
    outside=tmp_path/'private.safetensors';outside.write_bytes(b'model')
    monkeypatch.setattr(sys.modules['folder_paths'],'get_full_path',lambda *_:str(outside))
    with pytest.raises(BridgeError,match='INVALID_PATH'):category_path('checkpoints','linked.safetensors')

@pytest.mark.parametrize('family',['anima','sd15','sdxl'])
def test_graph_plan_no_hidden_lora(document,family):
    d=default_document(family);d['effective']['assets']=document['effective']['assets']
    s=finalize(d);p=graph_plan(s)
    assert len([n for n in p['nodes'] if n['type'].startswith('ForgeCompat')])==4
    assert not any(n['type']=='LoraLoader' for n in p['nodes'])
    json.dumps(p,allow_nan=False)
    assert p['config_hash']==s.config_hash

@pytest.mark.parametrize('bad',['mask','shape','nested','nan'])
def test_latent_guard(bad):
    x=torch.zeros(1,16,1,8,8);latent={'samples':x}
    if bad=='mask':latent['noise_mask']=torch.ones(1,8,8)
    elif bad=='shape':latent['samples']=torch.zeros(1,16,2,8,8)
    elif bad=='nested':latent['samples']=torch.nested.nested_tensor([torch.zeros(2,2),torch.zeros(3,2)])
    else:x[...,0,0]=float('nan')
    with pytest.raises(BridgeError):check_latent(latent,(1,16,1,8,8))

@pytest.fixture
def comfy_stub(monkeypatch):
    package=types.ModuleType('comfy');package.__path__=[]
    sampling=types.ModuleType('comfy.samplers');mm=types.ModuleType('comfy.model_management')
    mm.throw_exception_if_processing_interrupted=lambda:None
    calls=[]
    def calc(model,conds,x,sigma,options):
        calls.append(copy.deepcopy(conds))
        def val(items):
            if items is None:return torch.zeros_like(x)
            total=sum(c.get('strength',1) for c in items)
            return torch.ones_like(x)*(sum(c['value']*c.get('strength',1) for c in items)/total)
        return val(conds[0]),val(conds[1])
    sampling.calc_cond_batch=calc;package.samplers=sampling;package.model_management=mm
    monkeypatch.setitem(sys.modules,'comfy',package)
    monkeypatch.setitem(sys.modules,'comfy.samplers',sampling)
    monkeypatch.setitem(sys.modules,'comfy.model_management',mm)
    return calls,mm

@pytest.mark.parametrize('skip,ngms,all_steps',[(0,0,False),(.5,0,False),(0,.8,False),(0,.8,True)])
def test_cfg_schedule_boundaries(document,comfy_stub,skip,ngms,all_steps):
    calls,mm=comfy_stub;cfg=document['effective'];cfg['sampling']['cfg']=2.
    cfg['guidance'].update(skip_early_cfg=skip,ngms=ngms,ngms_all_steps=all_steps)
    bundle=NS(positive=((1.5,((2,'p'),(4,'p2'))),(.5,((4,'q'),))),negative=((None,((4,'n'),)),))
    guider=NS(conds={'p':[{'value':3}],'p2':[{'value':5}],'q':[{'value':7}],'n':[{'value':1}]},inner_model=object())
    trace=[];d=Denoiser(guider,object(),cfg,bundle,{},4,trace)
    for i in range(4):
        sigma=.9-i*.2
        result=d(torch.ones(1,4,2,2),torch.tensor([sigma]))
        skipped=0<i/4<=skip or ((i%2 or all_steps) and 0<sigma<ngms)
        pos=3 if i<=2 else 5
        p=(pos*1.5+7*.5)/2
        expected=p*2 if skipped else 1+(p-1)*2*2
        assert result[0,0,0,0].item()==pytest.approx(expected)
        assert (calls[-1][1] is None)==bool(skipped)
        assert trace[-1]['cfg']==(1 if skipped else 2)
    assert guider.conds['p']==[{'value':3}]

def test_cancel_does_not_advance_call(document,comfy_stub):
    _,mm=comfy_stub;mm.throw_exception_if_processing_interrupted=lambda:(_ for _ in ()).throw(InterruptedError())
    d=Denoiser(NS(),object(),document['effective'],NS(),{},10,[])
    with pytest.raises(InterruptedError):d(torch.zeros(1),torch.ones(1))
    assert d.call==0

def test_brownian_close_releases_state():
    pytest.importorskip('torchsde')
    from fnb.bridge.brownian import make_brownian
    from fnb.bridge.rng import RNGConfig
    stream=make_brownian(torch.zeros(1,4,4,4),torch.tensor([1.,.5,.2,0]),[42],RNGConfig('CPU','cpu'))
    stream(torch.tensor(1.),torch.tensor(.5))
    root=stream.intervals[0];namespace=stream.namespace
    stream.close();stream.close()
    assert stream.intervals==[] and namespace=={}
    assert root._w_h is None and root._top is None and root._parent is None
    with pytest.raises(BridgeError,match='RNG_CLOSED'):stream(torch.tensor(1.),torch.tensor(.5))

def test_review_ack_type(document):
    from fnb.bridge.server import review_document
    with pytest.raises(BridgeError,match='boolean'):review_document(document,'false')
