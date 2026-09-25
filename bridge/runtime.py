"""Forge sampling loop through Comfy's supported per-run sampler interface.

No module/function replacement, no implicit LoRA loading and no OOM fallback.
"""
from __future__ import annotations
import copy
import inspect
import math
from types import SimpleNamespace
import torch
from .binding import check_latent, digest, tensor_hash, validate_model
from .noise import model_identity
from .schedules import Predictor, Linker, generate_sigmas, SAMPLER_OPTIONS
from .spec import BridgeError, canonical, parse_json
from .text import require_dtype, dtype_name
from ..vendor.forge import sampling


def pick(schedule,call):
    index=0
    for i,(end,_) in enumerate(schedule):
        index=i
        if call<=end:break
    return schedule[index][1]


def merge_options(base,extra):
    out=base.copy()
    for key,value in extra.items():
        if isinstance(value,dict) and isinstance(out.get(key),dict):out[key]=merge_options(out[key],value)
        elif isinstance(value,list) and isinstance(out.get(key),list):out[key]=out[key]+value
        else:out[key]=value
    return out


class Denoiser:
    def __init__(self,guider,predictor,config,bundle,options,total_calls,trace):
        self.guider,self.inner_model,self.config,self.bundle=guider,Linker(predictor),config,bundle
        self.options,self.total_calls,self.trace=options,total_calls,trace
        self.call=0
    def conditions(self,parts):
        result=[]
        for weight,schedule in parts:
            key=pick(schedule,self.call)
            for entry in self.guider.conds[key]:
                entry=entry.copy()
                if weight is not None:entry['strength']=weight
                result.append(entry)
        return result
    def __call__(self,x,sigma,**extra):
        import comfy.model_management as mm
        import comfy.samplers
        mm.throw_exception_if_processing_interrupted()
        cfg=self.config['sampling']['cfg'];g=self.config['guidance']
        if 0<self.call/self.total_calls<=g['skip_early_cfg'] or ((self.call%2 or g['ngms_all_steps']) and 0<float(sigma[0])<g['ngms']):cfg=1.0
        cond=self.conditions(self.bundle.positive)
        uncond=self.conditions(self.bundle.negative) if self.bundle.negative else None
        opts=merge_options(self.options,extra.get('model_options',{}))
        if math.isclose(cfg,1.0) and not opts.get('disable_cfg1_optimization',False):uncond=None
        # Standard V1 paths have no third-party CFG callbacks; validation rejects them.
        pos,neg=comfy.samplers.calc_cond_batch(self.guider.inner_model,[cond,uncond],x,sigma,opts)
        strength=sum(c.get('strength',1) for c in cond)
        out=neg+(pos-neg)*cfg*strength if not math.isclose(strength,1.0) else neg+(pos-neg)*cfg
        if not torch.isfinite(out).all():raise BridgeError('NONFINITE_DENOISED','Denoiser produced NaN/Inf')
        if self.trace is not None:self.trace.append({'call':self.call,'sigma':float(sigma[0]),'cfg':cfg,'output_hash':tensor_hash(out)})
        self.call+=1
        return out


PARAMETERS = {
 'euler':('s_churn','s_tmin','s_tmax','s_noise'), 'heun':('s_churn','s_tmin','s_tmax','s_noise'),
 'dpm_2':('s_churn','s_tmin','s_tmax','s_noise'),
 'euler_ancestral':('s_noise',), 'dpmpp_2m_sde':('s_noise',),
}



def unused_parameters(cfg):
    s=cfg['sampling'];name=s['sampler']
    accepted=set(PARAMETERS.get(name,()))
    unused=['/sampling/'+k for k in ('s_churn','s_tmin','s_tmax','s_noise') if k not in accepted]
    unused.append('/sampling/eta_ddim')
    if 'eta' not in inspect.signature(getattr(sampling,'sample_'+name)).parameters:unused.append('/sampling/eta_ancestral')
    if cfg['family']=='anima':unused+=['/text/clip_skip','/sampling/sgm_noise_multiplier_requested']
    elif s['shift'] is not None:unused.append('/sampling/shift')
    if cfg['mode']=='txt2img':unused+=['/sampling/denoise','/sampling/img2img_extra_noise']
    return unused

def loop(config,predictor,denoiser,initial,latent,sigmas,rng,full_sigmas,callback=None,disable=True):
    """Pure numeric loop entry; also used by pinned-reference CPU tests."""
    s=config['sampling'];name=s['sampler'];fn=getattr(sampling,'sample_'+name)
    extra={'_bridge_rng':rng};kwargs={k:s[k] for k in PARAMETERS.get(name,())}
    if 's_tmax' in kwargs:kwargs['s_tmax']=kwargs['s_tmax'] or float('inf')
    if 'eta' in inspect.signature(fn).parameters:kwargs['eta']=s['eta_ancestral']
    brownian=None
    if SAMPLER_OPTIONS[name].get('brownian_noise'):
        from .brownian import make_brownian
        brownian=make_brownian(initial,full_sigmas,tuple(rng.seeds),rng.cfg)
        kwargs['noise_sampler']=brownian
    try:
        if config['mode']=='txt2img':
            x=predictor.noise_scaling(sigmas[0],initial,torch.zeros_like(initial),s['sgm_noise_multiplier_requested'])
        else:
            x=predictor.noise_scaling(sigmas[0],initial,latent.to(initial),False)
            if s['img2img_extra_noise']>0:x=x+initial*s['img2img_extra_noise']
        return fn(denoiser,x,sigmas,extra_args=extra,callback=callback,disable=disable,**kwargs)
    finally:
        if brownian is not None and hasattr(brownian,'close'):brownian.close()


class BridgeSampler:
    def __init__(self,config,predictor,bundle,noise_bundle,schedule_info,trace_level):
        self.config,self.predictor,self.bundle,self.noise_bundle=config,predictor,bundle,noise_bundle
        self.schedule_info,self.trace_level=schedule_info,trace_level
        self.calls=[];self.step_trace=[];self.tensor_trace={};self.conditioning_trace={}
    def sample(self,model_wrap,sigmas,extra_args,callback,noise,latent_image=None,denoise_mask=None,disable_pbar=False):
        if denoise_mask is not None:raise BridgeError('UNSUPPORTED_COMBINATION','Masks are outside the V1 profile')
        rng=self.noise_bundle.replay(self.config,noise.device)
        # Hash the actual post-adapter conditions, separately from Qwen outputs.
        if self.trace_level!='none':
            for key,entries in model_wrap.conds.items():
                for i,entry in enumerate(entries):
                    for name,value in entry.get('model_conds',{}).items():
                        data=getattr(value,'cond',None)
                        if isinstance(data,torch.Tensor):
                            self.conditioning_trace[f'{key}.{i}.{name}']={'hash':tensor_hash(data),'dtype':dtype_name(data.dtype),'shape':list(data.shape)}
        options=extra_args.get('model_options',{}).copy()
        denoiser=Denoiser(model_wrap,self.predictor,self.config,self.bundle,options,self.schedule_info['total_denoiser_calls'],self.calls if self.trace_level!='none' else None)
        def on_step(data):
            if callback:callback(data['i'],data['denoised'],data['x'],len(sigmas)-1)
            if self.trace_level!='none':self.step_trace.append({'step':data['i'],'sigma':float(data['sigma']),'latent_hash':tensor_hash(data['x'])})
            if self.trace_level=='tensors':self.tensor_trace[f'step_{data["i"]:04d}']=data['x'].detach().cpu().clone()
        full=torch.tensor(self.schedule_info['full_sigmas'],device=noise.device,dtype=torch.float32)
        out=loop(self.config,self.predictor,denoiser,noise,latent_image,sigmas,rng,full,on_step,disable_pbar)
        if self.trace_level=='tensors':
            self.tensor_trace.update(initial_noise=noise.detach().cpu().clone(),final_sampling_latent=out.detach().cpu().clone(),sigmas=sigmas.detach().cpu().clone())
        return out


def prepare_model(model,cfg):
    validate_model(model,cfg)
    require_dtype(cfg['numeric']['model_dtype'],model.model_dtype(),'Model')
    if cfg['numeric']['attention_backend']!='as_loaded':
        raise BridgeError('DTYPE_NOT_APPLIED','V1 does not globally switch attention backends')
    if cfg['guidance']['batching_policy']!='as_loaded':
        raise BridgeError('UNSUPPORTED_COMBINATION','Reference batch execution must be configured by its own qualified adapter')
    if cfg['numeric']['vae_dtype']!='as_loaded':
        raise BridgeError('DTYPE_NOT_APPLIED','VAE precision is controlled by the external VAE loader; use as_loaded here')
    if cfg['family']=='anima':
        dtype=model.model.get_dtype_inference()
        require_dtype(cfg['numeric']['anima_adapter_dtype'],dtype,'Anima adapter input')
    clone=model.clone()
    # Core clone is shallow for the model; only its object-patch mapping may change.
    current=model.get_model_object('model_sampling')
    shift=cfg['sampling']['shift']
    if shift is not None and cfg['family']=='anima' and getattr(current,'shift',None)!=shift:
        private=copy.deepcopy(current)
        params={'shift':shift}
        if hasattr(private,'multiplier'):params['multiplier']=private.multiplier
        private.set_parameters(**params)
        clone.add_object_patch('model_sampling',private)
        current=private
    return clone,Predictor(current,cfg['prediction_type'])


@torch.inference_mode()
def execute(spec,model,bundle,noise_bundle,latent,binding,trace_level='summary',callback=None):
    import comfy.samplers
    import comfy.model_management as mm
    cfg=spec.require_executable()
    if bundle.config_hash!=spec.config_hash or noise_bundle.config_hash!=spec.config_hash:
        raise BridgeError('SPEC_MISMATCH','Conditioning/Noise was built with a different Spec')
    if bundle.family!=cfg['family'] or noise_bundle.family!=cfg['family']:
        raise BridgeError('MODEL_FAMILY_MISMATCH','Bundle family differs from Spec')
    if noise_bundle.model_identity!=model_identity(model) or parse_json(noise_bundle.binding_json)['hash']!=binding['hash']:
        raise BridgeError('ASSET_BINDING_MISMATCH','MODEL or its loader/LoRA chain changed after noise creation')
    samples=check_latent(latent,noise_bundle.shape)
    if tensor_hash(samples)!=noise_bundle.latent_hash or tensor_hash(noise_bundle.initial)!=noise_bundle.initial_hash:
        raise BridgeError('LATENT_MISMATCH','Latent or cached initial noise was modified')
    clone,predictor=prepare_model(model,cfg)
    sigmas,info=generate_sigmas(cfg,predictor)
    sampler=BridgeSampler(cfg,predictor,bundle,noise_bundle,info,trace_level)
    guider=comfy.samplers.CFGGuider(clone)
    guider.inner_set_conds(bundle.cloned_entries());guider.set_cfg(cfg['sampling']['cfg'])
    try:
        output=guider.sample(noise_bundle.initial.clone(),samples.clone(),sampler,sigmas,denoise_mask=None,callback=callback,
                             disable_pbar=not getattr(__import__('comfy.utils',fromlist=['PROGRESS_BAR_ENABLED']),'PROGRESS_BAR_ENABLED',True),seed=noise_bundle.seeds[0])
        if not torch.isfinite(output).all():raise BridgeError('NONFINITE_LATENT','Sampler produced NaN/Inf')
        result=latent.copy();result['samples']=output.to(mm.intermediate_device())
        if trace_level=='tensors':sampler.tensor_trace['final_latent']=output.detach().cpu().clone()
        facts={'schedule':info,'calls':sampler.calls,'steps':sampler.step_trace,'post_adapter_conditions':sampler.conditioning_trace,
               'final_latent_hash':tensor_hash(output),'model_binding':binding,'text_binding':parse_json(bundle.binding_json),
               'numeric':{'sampling_dtype':dtype_name(output.dtype),'model_dtype':dtype_name(model.model_dtype()),'text':parse_json(bundle.trace_json)},
               'environment':{'torch':torch.__version__,'cuda':torch.version.cuda,'device':str(model.load_device)},
               'applied':{'sampler':cfg['sampling']['sampler'],'scheduler':info['resolved_scheduler'],'rng':cfg['noise']['source'],'steps_requested':cfg['sampling']['steps'],'intervals_executed':len(sigmas)-1},
               'not_applied':unused_parameters(cfg)}
        facts['run_hash']=digest(facts)
        return result,sigmas,facts,sampler.tensor_trace
    finally:
        # Core's successful path clears these; clean the exception path too.
        if hasattr(guider,'loaded_models') and hasattr(guider,'conds'):
            import comfy.sampler_helpers
            comfy.sampler_helpers.cleanup_models(guider.conds,guider.loaded_models)
        for key in ('loaded_models','conds','inner_model'):
            if hasattr(guider,key):delattr(guider,key)
        guider.original_conds.clear()
