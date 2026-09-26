"""Core-backed solver routing, private noise, and report boundaries."""
import sys
import types
from types import SimpleNamespace as NS
import pytest
import torch

from fnb.bridge.runtime import Denoiser, loop, sampler_function, sampler_implementation, unused_parameters
from fnb.bridge.spec import BridgeError, COMFY_SHARED_SAMPLERS, default_document


def install_core(monkeypatch, fn, names=('ddim','uni_pc','euler_cfg_pp','euler_ancestral_cfg_pp','dpmpp_2m_cfg_pp')):
    package=types.ModuleType('comfy');package.__path__=[]
    calls=[]
    def select(name):
        calls.append(name);return NS(sampler_function=fn)
    core=NS(SAMPLER_NAMES=names,sampler_object=select)
    package.samplers=core
    monkeypatch.setitem(sys.modules,'comfy',package)
    monkeypatch.setitem(sys.modules,'comfy.samplers',core)
    return calls


@pytest.mark.parametrize('name',COMFY_SHARED_SAMPLERS)
def test_resolve_exact_core_selection_and_report_limit(monkeypatch,name):
    def fn(*args,**kwargs):pass
    calls=install_core(monkeypatch,fn)
    assert sampler_function(name) is fn
    assert calls==[{'unipc':'uni_pc'}.get(name,name)]
    assert sampler_implementation(name)['forge_parity'] is False
    cfg=default_document('sd15')['effective'];cfg['sampling']['sampler']=name
    assert '/sampling/eta_ddim' in unused_parameters(cfg)


def test_missing_core_sampler_errors_without_substituting_another(monkeypatch):
    calls=install_core(monkeypatch,lambda:None,names=('euler',))
    with pytest.raises(BridgeError,match='no uni_pc'):sampler_function('unipc')
    assert calls==[]


def test_core_solver_uses_private_continued_noise_and_cannot_mutate_sigmas(monkeypatch):
    noises=[]
    def fn(model,x,sigmas,extra_args,callback,disable,eta=1.,s_noise=1.,noise_sampler=None):
        assert extra_args=={'seed':42}
        assert eta==.7 and s_noise==.8
        noises.extend([noise_sampler(None,None),noise_sampler(None,None)])
        sigmas[-1]=.001
        return x+noises[0]+noises[1]
    install_core(monkeypatch,fn)
    cfg=default_document('sd15')['effective']
    cfg['sampling'].update(sampler='euler_ancestral_cfg_pp',eta_ancestral=.7,s_noise=.8)
    seq=iter([torch.full((1,4,2,2),2.),torch.full((1,4,2,2),3.)])
    rng=NS(seeds=(42,),next=lambda:next(seq))
    initial=torch.ones(1,4,2,2);latent=torch.zeros_like(initial)
    sigmas=torch.tensor([1.,.5,0.]);before=sigmas.clone()
    predictor=NS(noise_scaling=lambda sigma,noise,image,max_denoise:noise*sigma+image)
    output=loop(cfg,predictor,None,initial,latent,sigmas,rng,sigmas)
    assert torch.equal(sigmas,before)
    assert torch.equal(output,torch.full_like(initial,6.))


@pytest.mark.parametrize('sampler',['euler_cfg_pp','euler_ancestral_cfg_pp','dpmpp_2m_cfg_pp'])
@pytest.mark.parametrize('mode',['cfg1','skip_early','ngms'])
def test_cfgpp_gets_real_negative_prediction_even_when_cfg_is_one(monkeypatch,sampler,mode):
    install_core(monkeypatch,lambda:None)
    package=sys.modules['comfy'];core=package.samplers
    mm=NS(throw_exception_if_processing_interrupted=lambda:None)
    package.model_management=mm
    monkeypatch.setitem(sys.modules,'comfy.model_management',mm)
    x=torch.zeros(1,4,2,2)
    def calc(model,conditions,x,sigma,opts):
        assert conditions[1] is not None
        return torch.full_like(x,2.),torch.full_like(x,1.)
    core.calc_cond_batch=calc
    cfg=default_document('sd15')['effective'];cfg['sampling'].update(sampler=sampler,cfg=1. if mode=='cfg1' else 7.)
    cfg['guidance'].update(skip_early_cfg=.5 if mode=='skip_early' else 0,ngms=2. if mode=='ngms' else 0)
    guider=NS(model_patcher=object(),inner_model=object(),conds={'p':[{}],'n':[{}]})
    bundle=NS(positive=((1.,((4,'p'),)),),negative=((None,((4,'n'),)),))
    seen=[]
    def capture(args):
        seen.append(args['uncond_denoised'])
        return args['denoised']
    denoiser=Denoiser(guider,None,cfg,bundle,{},4,None)
    denoiser.call=1
    out=denoiser(x,torch.ones(1),model_options={
        'sampler_post_cfg_function':[capture]})
    assert torch.equal(out,torch.full_like(x,2.))
    assert len(seen)==1 and torch.equal(seen[0],torch.ones_like(x))
    assert guider.conds=={'p':[{}],'n':[{}]}
