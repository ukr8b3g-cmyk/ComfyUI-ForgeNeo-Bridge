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
from .schedules import Predictor, Linker, generate_sigmas, core_sigmas, core_names, SAMPLER_OPTIONS
from .spec import BridgeError, COMFY_SHARED_SAMPLERS, canonical, parse_json
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
        if config['sampling']['sampler'] in COMFY_SHARED_SAMPLERS:
            # Public core samplers inspect this object to distinguish EPS/flow.
            self.inner_model.model_patcher = guider.model_patcher
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
        if self.config['sampling'].get('adjustments',True) and (0<self.call/self.total_calls<=g['skip_early_cfg'] or ((self.call%2 or g['ngms_all_steps']) and 0<float(sigma[0])<g['ngms'])):cfg=1.0
        cond=self.conditions(self.bundle.positive)
        uncond=self.conditions(self.bundle.negative) if self.bundle.negative else None
        opts=merge_options(self.options,extra.get('model_options',{}))
        if math.isclose(cfg,1.0) and not opts.get('disable_cfg1_optimization',False):uncond=None
        # Standard V1 paths have no third-party CFG callbacks; validation rejects them.
        pos,neg=comfy.samplers.calc_cond_batch(self.guider.inner_model,[cond,uncond],x,sigma,opts)
        strength=sum(c.get('strength',1) for c in cond)
        out=neg+(pos-neg)*cfg*strength if not math.isclose(strength,1.0) else neg+(pos-neg)*cfg
        # CFG++ solvers install a per-call callback to retrieve the negative
        # prediction. Keep it local; never patch core functions or shared options.
        if not self.config['sampling'].get('adjustments',True) or self.config['sampling']['sampler'] in COMFY_SHARED_SAMPLERS:
            for fn in opts.get('sampler_post_cfg_function',()):
                out=fn({'denoised':out,'cond':cond,'uncond':uncond,'cond_scale':cfg,
                        'model':self.guider.inner_model,'uncond_denoised':neg,'cond_denoised':pos,
                        'sigma':sigma,'model_options':opts,'input':x})
        if not torch.isfinite(out).all():raise BridgeError('NONFINITE_DENOISED','Denoiser produced NaN/Inf')
        if self.trace is not None:self.trace.append({'call':self.call,'sigma':float(sigma[0]),'cfg':cfg,'output_hash':tensor_hash(out)})
        self.call+=1
        return out


PARAMETERS = {
 'euler':('s_churn','s_tmin','s_tmax','s_noise'), 'heun':('s_churn','s_tmin','s_tmax','s_noise'),
 'dpm_2':('s_churn','s_tmin','s_tmax','s_noise'),
 'euler_ancestral':('s_noise',), 'dpmpp_2m_sde':('s_noise',),
 'dpmpp_sde':('s_noise',), 'dpmpp_3m_sde':('s_noise',),
 'euler_ancestral_cfg_pp':('s_noise',),
}


def sampler_function(name):
    if name not in COMFY_SHARED_SAMPLERS:
        return getattr(sampling,'sample_'+name)
    import comfy.samplers as core
    core_name = {'unipc':'uni_pc'}.get(name,name)
    if core_name not in core.SAMPLER_NAMES:
        raise BridgeError('COMFY_UNSUPPORTED_SAMPLING',f'This ComfyUI version has no {core_name} sampler')
    return core.sampler_object(core_name).sampler_function


def sampler_implementation(name,adjustments=True):
    if not adjustments:
        return {'backend':'comfy','name':{'unipc':'uni_pc'}.get(name,name),'sampling_adjustments':False,
                'note':'Core sampler and schedule; Forge text and initial noise are retained.'}
    if name in COMFY_SHARED_SAMPLERS:
        return {'backend':'comfy_shared','name':{'unipc':'uni_pc'}.get(name,name),
                'forge_parity':False,
                'note':'Comfy solver with Forge text, initial noise and schedule. See README compatibility table.'}
    return {'backend':'forge_reference','name':name}



def unused_parameters(cfg):
    s=cfg['sampling'];name=s['sampler']
    if not s.get('adjustments',True):
        return ['/sampling/'+key for key in (
            'eta_ancestral','eta_ddim','s_churn','s_tmin','s_tmax','s_noise','sigma_min','sigma_max',
            'rho','beta_alpha','beta_beta','discard_penultimate_requested','sgm_noise_multiplier_requested',
            'shift','img2img_step_mode','img2img_extra_noise','flow_match_options')]+[
                '/noise/ensd','/guidance/skip_early_cfg','/guidance/ngms','/guidance/ngms_all_steps']+(
                    ['/text/clip_skip'] if cfg['family']=='anima' else [])
    accepted=set(PARAMETERS.get(name,()))
    unused=['/sampling/'+k for k in ('s_churn','s_tmin','s_tmax','s_noise') if k not in accepted]
    unused.append('/sampling/eta_ddim')
    if name in COMFY_SHARED_SAMPLERS and name != 'euler_ancestral_cfg_pp':unused.append('/noise/ensd')
    if 'eta' not in inspect.signature(sampler_function(name)).parameters:unused.append('/sampling/eta_ancestral')
    if cfg['family']=='anima':unused+=['/text/clip_skip','/sampling/sgm_noise_multiplier_requested']
    elif s['shift'] is not None:unused.append('/sampling/shift')
    if cfg['mode']=='txt2img':unused+=['/sampling/denoise','/sampling/img2img_extra_noise']
    return unused

def loop(config,predictor,denoiser,initial,latent,sigmas,rng,full_sigmas,callback=None,disable=True):
    """Pure numeric loop entry; also used by pinned-reference CPU tests."""
    s=config['sampling'];name=s['sampler'];fn=sampler_function(name)
    extra={'_bridge_rng':rng};kwargs={k:s[k] for k in PARAMETERS.get(name,())}
    if name in COMFY_SHARED_SAMPLERS:
        extra={'seed':int(rng.seeds[0])}
        if 'noise_sampler' in inspect.signature(fn).parameters:
            kwargs['noise_sampler']=lambda *_:rng.next()
        # UniPC adjusts its terminal sigma in-place in some core versions.
        sigmas=sigmas.clone()
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
            if self.trace_level!='none':self.step_trace.append({'step':data['i'],'sigma':float(data.get('sigma',sigmas[min(data['i'],len(sigmas)-1)])),'latent_hash':tensor_hash(data['x'])})
            if self.trace_level=='tensors':self.tensor_trace[f'step_{data["i"]:04d}']=data['x'].detach().cpu().clone()
        if self.config['sampling'].get('adjustments',True):
            rng=self.noise_bundle.replay(self.config,noise.device)
            full=torch.tensor(self.schedule_info['full_sigmas'],device=noise.device,dtype=torch.float32)
            out=loop(self.config,self.predictor,denoiser,noise,latent_image,sigmas,rng,full,on_step,disable_pbar)
        else:
            import comfy.samplers as core
            selected=core.sampler_object(core_names(self.config['sampling'])[0])
            # There is no mask on this path. DDIM's unused random inpaint noise
            # would otherwise reset the process-wide RNG in core KSAMPLER.
            selected=core.KSAMPLER(selected.sampler_function,selected.extra_options,inpaint_options={})
            adapter=CoreDenoiser(model_wrap,denoiser)
            def core_callback(i,denoised,x,total):
                on_step({'i':i,'denoised':denoised,'x':x})
            out=selected.sample(adapter,sigmas.clone(),dict(extra_args),core_callback,noise,
                                latent_image=latent_image,denoise_mask=None,disable_pbar=disable_pbar)
        if self.trace_level=='tensors':
            self.tensor_trace.update(initial_noise=noise.detach().cpu().clone(),final_sampling_latent=out.detach().cpu().clone(),sigmas=sigmas.detach().cpu().clone())
        return out


class CoreDenoiser:
    """Core sampler model interface with Bridge's scheduled text conditioning."""
    def __init__(self,guider,denoiser):
        self.inner_model=guider.inner_model
        self.model_patcher=guider.model_patcher
        self.cfg=guider.cfg
        self.denoiser=denoiser
    def __call__(self,x,sigma,**kwargs):
        return self.denoiser(x,sigma,**kwargs)


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
    if cfg['sampling'].get('adjustments',True) and shift is not None and cfg['family']=='anima' and getattr(current,'shift',None)!=shift:
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
    adjustments=cfg['sampling'].get('adjustments',True)
    sigmas,info=generate_sigmas(cfg,predictor) if adjustments else core_sigmas(cfg,clone)
    sampler=BridgeSampler(cfg,predictor,bundle,noise_bundle,info,trace_level)
    guider=comfy.samplers.CFGGuider(clone)
    guider.inner_set_conds(bundle.cloned_entries());guider.set_cfg(cfg['sampling']['cfg'])
    try:
        output=guider.sample(noise_bundle.initial.clone(),samples.clone(),sampler,sigmas,denoise_mask=None,callback=callback,
                             disable_pbar=not getattr(__import__('comfy.utils',fromlist=['PROGRESS_BAR_ENABLED']),'PROGRESS_BAR_ENABLED',True),seed=noise_bundle.seeds[0])
        if not torch.isfinite(output).all():raise BridgeError('NONFINITE_LATENT','Sampler produced NaN/Inf')
        result=latent.copy();result['samples']=output.to(mm.intermediate_device())
        if trace_level=='tensors':sampler.tensor_trace['final_latent']=output.detach().cpu().clone()
        facts={'schedule':info,'sampler_implementation':sampler_implementation(cfg['sampling']['sampler'],adjustments),
               'calls':sampler.calls,'steps':sampler.step_trace,'post_adapter_conditions':sampler.conditioning_trace,
               'final_latent_hash':tensor_hash(output),'model_binding':binding,'text_binding':parse_json(bundle.binding_json),
               'numeric':{'sampling_dtype':dtype_name(output.dtype),'model_dtype':dtype_name(model.model_dtype()),'text':parse_json(bundle.trace_json)},
               'environment':{'torch':torch.__version__,'cuda':torch.version.cuda,'device':str(model.load_device)},
               'applied':{'sampler':cfg['sampling']['sampler'],'scheduler':info['resolved_scheduler'],'rng':cfg['noise']['source'],'steps_requested':cfg['sampling']['steps'],'intervals_executed':max(0,len(sigmas)-1),'sampling_adjustments':adjustments},
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
