"""Numerical oracle tests use the pinned Forge source, not Neo-Sampler's claims.

Analytic denoisers isolate sampling math. These are CPU unit tests, NOT trained
model/image parity tests. Set FORGE_REFERENCE to an unpacked 710f1e25 tree.
"""
from __future__ import annotations
import ast,copy,math,os,pathlib,types
from functools import partial
import numpy as np
import pytest
import torch
from scipy import stats,integrate
from fnb.bridge.spec import default_document
from fnb.bridge.rng import ImageRNG,RNGConfig
from fnb.bridge.schedules import Predictor,generate_sigmas,SAMPLER_OPTIONS
from fnb.bridge.runtime import loop
from fnb.vendor.forge import rng_philox

ROOT=pathlib.Path(os.environ.get('FORGE_REFERENCE','/mnt/data/bridge_upstream/forge'))
pytestmark=pytest.mark.skipif(not (ROOT/'modules/rng.py').exists(),reason='Pinned Forge oracle tree not supplied')


def definitions(path,namespace,names=None):
    tree=ast.parse((ROOT/path).read_text(encoding='utf-8'))
    selected=[]
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and (names is None or node.name in names):selected.append(node)
        elif isinstance(node,(ast.Assign,ast.AnnAssign)) and names is None:selected.append(node)
    tree=ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[]))
    exec(compile(tree,str(path),'exec'),namespace)
    return namespace


def predictor(family):
    ns=definitions('backend/modules/k_prediction.py',{'torch':torch,'np':np,'math':math})
    if family=='anima':return ns['PredictionDiscreteFlow'](types.SimpleNamespace(sampling_settings={'shift':3.0}))
    return ns['Prediction'](1.0,'epsilon','linear',.00085,.012,1000)


def reference_rng(source,shape,seeds,ensd,subseeds=None,strength=0,resize=(0,0)):
    opts=types.SimpleNamespace(randn_source=source,eta_noise_seed_delta=ensd)
    ns={'torch':torch,'devices':types.SimpleNamespace(device=torch.device('cpu'),cpu=torch.device('cpu')),'rng_philox':rng_philox,'shared':types.SimpleNamespace(opts=opts,device=torch.device('cpu'))}
    definitions('modules/rng.py',ns)
    return ns['ImageRNG'](shape,seeds,subseeds,strength,*resize)


@pytest.mark.parametrize('source',['CPU','NV'])
@pytest.mark.parametrize('shape',[(4,8,8),(16,1,8,8)])
@pytest.mark.parametrize('ensd',[0,31337])
@pytest.mark.parametrize('strength',[0.0,0.4])
def test_rng_oracle(source,shape,ensd,strength):
    seeds=[42,43];sub=[123,124]
    ours=ImageRNG(RNGConfig(source,'cpu',ensd),shape,seeds,sub,strength)
    with torch.random.fork_rng(devices=[]):
        ref=reference_rng(source,shape,seeds,ensd,sub,strength)
        for _ in range(4):assert torch.equal(ref.next(),ours.next())

@pytest.mark.parametrize('resize',[(32,32),(128,128),(32,128)])
def test_resize_oracle(resize):
    shape=(4,8,8)
    ours=ImageRNG(RNGConfig('CPU','cpu'),shape,[42],[9],.2,*resize)
    with torch.random.fork_rng(devices=[]):
        ref=reference_rng('CPU',shape,[42],0,[9],.2,resize)
        for _ in range(3):assert torch.equal(ref.next(),ours.next())


class TorchProxy:
    def __init__(self,rng):self.rng=rng
    def __getattr__(self,key):return (lambda x:self.rng.next()) if key=='randn_like' else getattr(torch,key)


def source_sampling(rng):
    names={'_is_const','append_zero','to_d','get_ancestral_step','default_noise_sampler','sigma_to_half_log_snr','half_log_snr_to_sigma','offset_first_sigma_for_snr',
      'sample_euler','sample_euler_ancestral','sample_euler_ancestral_RF','sample_er_sde','sample_dpmpp_2m','sample_dpmpp_2m_sde','sample_dpm_2','sample_heun',
      'get_sigmas_karras','get_sigmas_exponential','get_sigmas_polyexponential'}
    ns={'torch':TorchProxy(rng),'math':math,'partial':partial,'trange':lambda n,**kw:range(n),'utils':types.SimpleNamespace(append_dims=lambda x,n:x[(...,)+(None,)*(n-x.ndim)])}
    definitions('modules_forge/packages/k_diffusion/sampling.py',ns,names)
    # sd_schedulers overrides this in Forge at import; emulate only in oracle namespace.
    ns['to_d']=lambda x,s,d:(x-d)/s
    return ns


def reference_sigmas(cfg,pred,functions):
    s=cfg['sampling'];op=SAMPLER_OPTIONS[s['sampler']]
    n=s['steps']
    if cfg['mode']=='img2img' and s['img2img_step_mode']=='exact_steps':n=int(n/min(s['denoise'],.999))
    discard=s['discard_penultimate_requested'] or op.get('discard_next_to_last_sigma',False)
    count=n+int(discard)
    opts=types.SimpleNamespace(beta_dist_alpha=s['beta_alpha'],beta_dist_beta=s['beta_beta'])
    ns={'torch':torch,'np':np,'stats':stats,'exp':math.exp,'pi':math.pi,'atan':math.atan,'shared':types.SimpleNamespace(opts=opts,sd_model=types.SimpleNamespace(is_sdxl=cfg['family']=='sdxl'))}
    definitions('modules/sd_schedulers.py',ns,{'normal_scheduler','simple_scheduler','uniform','sgm_uniform','beta_scheduler','_loglinear_interp','get_align_your_steps_sigmas','linear_quadratic','kl_optimal','ddim_scheduler','get_bong_tangent_sigmas','bong_tangent_scheduler'})
    linker=types.SimpleNamespace(sigmas=pred.sigmas,sigma_to_t=pred.timestep,t_to_sigma=pred.sigma,get_sigmas=lambda n:torch.cat((pred.sigma(torch.linspace(len(pred.sigmas)-1,0,n)),torch.zeros(1))))
    name=s['scheduler'];name=op.get('scheduler','normal' if cfg['family']=='anima' else 'uniform') if name=='automatic' else name
    lo=float(pred.sigmas[0]) if s['sigma_min'] is None else s['sigma_min'];hi=float(pred.sigmas[-1]) if s['sigma_max'] is None else s['sigma_max']
    if name in ('simple','beta','normal','sgm_uniform','ddim','uniform'):
        fn=ns[{'simple':'simple_scheduler','beta':'beta_scheduler','normal':'normal_scheduler','sgm_uniform':'sgm_uniform','ddim':'ddim_scheduler','uniform':'uniform'}[name]]
        sigmas=fn(count,lo,hi,linker,'cpu')
    elif name in ('karras','exponential','polyexponential'):
        kwargs={} if s['rho'] is None or name=='exponential' else {'rho':s['rho']}
        sigmas=functions['get_sigmas_'+name](count,lo,hi,device='cpu',**kwargs)
    else:
        fn=ns[{'align_your_steps':'get_align_your_steps_sigmas','linear_quadratic':'linear_quadratic','kl_optimal':'kl_optimal','bong_tangent':'bong_tangent_scheduler'}[name]]
        sigmas=fn(count,lo,hi,'cpu')
    if discard:sigmas=torch.cat((sigmas[:-2],sigmas[-1:]))
    full=sigmas.clone()
    if cfg['mode']=='img2img':
        t=s['steps']-1 if s['img2img_step_mode']=='exact_steps' else int(min(s['denoise'],.999)*n)
        sigmas=sigmas[n-t-1:]
    return sigmas,full


class AnalyticDenoiser:
    def __init__(self,pred):self.inner_model=types.SimpleNamespace(predictor=pred);self.calls=[]
    def __call__(self,x,sigma,**kwargs):
        # Finite nonlinear deterministic function, independent of real trained weights.
        self.calls.append(float(sigma[0]));return x*.125+torch.sin(sigma).reshape((-1,)+(1,)*(x.ndim-1))*.075

PAIRS=[('euler','simple'),('euler','beta'),('er_sde','beta'),('euler_ancestral','simple'),('dpmpp_2m','karras'),('dpm_2','karras'),('heun','simple')]
@pytest.mark.parametrize('family',['anima','sd15','sdxl'])
@pytest.mark.parametrize('pair',PAIRS)
@pytest.mark.parametrize('mode',['txt2img','scaled','exact'])
@pytest.mark.parametrize('discard',[False,True])
def test_sampling_oracle(family,pair,mode,discard):
    cfg=default_document(family)['effective'];s=cfg['sampling'];s.update(sampler=pair[0],scheduler=pair[1],steps=6,discard_penultimate_requested=discard)
    if mode!='txt2img':cfg['mode']='img2img';s.update(denoise=.6,img2img_step_mode='exact_steps' if mode=='exact' else 'forge_scaled',img2img_extra_noise=.1)
    shape=(16,1,4,4) if family=='anima' else (4,4,4)
    pred=Predictor(predictor(family),cfg['prediction_type'])
    rng=ImageRNG(RNGConfig('CPU','cpu',31337),shape,[42,43]);initial=rng.next()
    with torch.random.fork_rng(devices=[]):
        ref_rng=reference_rng('CPU',shape,[42,43],31337);ref_initial=ref_rng.next();assert torch.equal(initial,ref_initial)
        funcs=source_sampling(ref_rng)
        sigmas,info=generate_sigmas(cfg,pred);ref_sigmas,full=reference_sigmas(cfg,pred,funcs)
        assert torch.equal(sigmas,ref_sigmas)
        lat=initial*.07;model=AnalyticDenoiser(pred);ref_model=AnalyticDenoiser(pred)
        ours=loop(cfg,pred,model,initial,lat,sigmas,rng,torch.tensor(info['full_sigmas']))
        # Independent pinned Forge AbstractPrediction scaling, not bridge loop's helper.
        p0=pred.ms
        start=p0.noise_scaling(ref_sigmas[0],ref_initial,torch.zeros_like(lat) if cfg['mode']=='txt2img' else lat,max_denoise=s['sgm_noise_multiplier_requested'] if cfg['mode']=='txt2img' else False)
        if cfg['mode']=='img2img':start+=ref_initial*s['img2img_extra_noise']
        kwargs={}
        if pair[0] in ('euler','heun','dpm_2'):kwargs=dict(s_churn=s['s_churn'],s_tmin=s['s_tmin'],s_tmax=s['s_tmax'] or float('inf'),s_noise=s['s_noise'])
        if pair[0]=='euler_ancestral':kwargs=dict(eta=s['eta_ancestral'],s_noise=s['s_noise'])
        ref=funcs['sample_'+pair[0]](ref_model,start,ref_sigmas,disable=True,**kwargs)
        assert torch.equal(ours,ref),float((ours-ref).abs().max())
        assert model.calls==ref_model.calls

@pytest.mark.parametrize('family',['anima','sd15','sdxl'])
@pytest.mark.parametrize('scheduler',['automatic','karras','exponential','polyexponential','normal','simple','uniform','sgm_uniform','linear_quadratic','kl_optimal','ddim','align_your_steps','beta','bong_tangent'])
def test_scheduler_oracle(family,scheduler):
    cfg=default_document(family)['effective'];cfg['sampling'].update(scheduler=scheduler,steps=10)
    pred=Predictor(predictor(family),cfg['prediction_type']);rng=ImageRNG(RNGConfig('CPU','cpu'),(4,4,4),[1]);fn=source_sampling(rng)
    expected,_=reference_sigmas(cfg,pred,fn)
    if not torch.all(expected[:-1] > expected[1:]):
        # Fixed Forge Uniform/flow can include a duplicate terminal zero.
        # Do not repair the schedule silently or run a divide-by-zero loop.
        from fnb.bridge.spec import BridgeError
        with pytest.raises(BridgeError,match='UNSUPPORTED_COMBINATION'):generate_sigmas(cfg,pred)
    else:
        actual,_=generate_sigmas(cfg,pred)
        assert torch.equal(actual,expected)

@pytest.mark.parametrize('family',['anima','sd15','sdxl'])
def test_sgm_scaling(family):
    p=predictor(family);ours=Predictor(p,'flow' if family=='anima' else 'epsilon')
    x=torch.ones((1,4,4,4));s=p.sigmas[-1]
    for flag in (False,True):assert torch.equal(ours.noise_scaling(s,x,torch.zeros_like(x),flag),p.noise_scaling(s,x,torch.zeros_like(x),flag))

@pytest.mark.parametrize('source',['CPU','NV'])
@pytest.mark.parametrize('family',['anima','sd15','sdxl'])
def test_brownian_and_sde(source,family):
    torchsde=pytest.importorskip('torchsde')
    from torchsde._brownian import brownian_interval
    from fnb.bridge.brownian import make_brownian
    from fnb.bridge.rng import randn_local
    cfg=default_document(family)['effective'];cfg['sampling'].update(sampler='dpmpp_2m_sde',scheduler='exponential',steps=6)
    shape=(16,1,4,4) if family=='anima' else (4,4,4)
    ours_rng=ImageRNG(RNGConfig(source,'cpu'),shape,[42,43]);x=ours_rng.next()
    pred=Predictor(predictor(family),cfg['prediction_type']);sigmas,info=generate_sigmas(cfg,pred)
    with torch.random.fork_rng(devices=[]):
        ref_rng=reference_rng(source,shape,[42,43],0);ref_x=ref_rng.next();funcs=source_sampling(ref_rng)
        before=brownian_interval._randn;global_state=torch.random.get_rng_state().clone()
        ours=make_brownian(x,sigmas,(42,43),ours_rng.cfg)
        assert brownian_interval._randn is before
        t0,t1=sigmas[sigmas>0].min(),sigmas.max()
        try:
            # Oracle reproduces Forge's own globally patched torchsde in a bounded
            # test context. Production only uses the private namespace adapter.
            brownian_interval._randn=lambda size,dtype,device,seed:randn_local(ours_rng.cfg,seed,size).to(device=device,dtype=dtype)
            trees=[torchsde.BrownianTree(t0,torch.zeros_like(x[0]),t1,entropy=s) for s in (42,43)]
            def ref_noise(a,b):
                lo,hi,sign=(a,b,1) if a<b else (b,a,-1)
                return torch.stack([tree(lo,hi) for tree in trees])*sign/(b-a).abs().sqrt()
            for a,b in [(sigmas[0],sigmas[1]),(sigmas[1],sigmas[2]),(sigmas[0],sigmas[2])]:assert torch.equal(ours(a,b),ref_noise(a,b))
            model=AnalyticDenoiser(pred);ref_model=AnalyticDenoiser(pred)
            actual=loop(cfg,pred,model,x,torch.zeros_like(x),sigmas,ours_rng,sigmas)
            start=pred.ms.noise_scaling(sigmas[0],ref_x,torch.zeros_like(x),False)
            # New reference tree: the sequence of calls matters to the Brownian cache.
            trees=[torchsde.BrownianTree(t0,torch.zeros_like(x[0]),t1,entropy=s) for s in (42,43)]
            expected=funcs['sample_dpmpp_2m_sde'](ref_model,start,sigmas,disable=True,noise_sampler=ref_noise,eta=1.,s_noise=1.)
            assert torch.equal(actual,expected)
        finally:
            brownian_interval._randn=before
            if hasattr(ours,'close'):ours.close()
        assert torch.equal(global_state,torch.random.get_rng_state())
