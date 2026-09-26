"""Replayable private noise. Cached bundles contain snapshots, never live RNGs."""
from __future__ import annotations
import copy
from dataclasses import dataclass
import torch
from .binding import check_latent, digest, latent_shape, tensor_hash, validate_model
from .rng import ImageRNG, RNGConfig
from .spec import BridgeError, canonical, noise_seed_values


def model_identity(model):
    return (id(model.model),str(getattr(model,'patches_uuid','')),digest({'patch_keys':sorted(getattr(model,'patches',{}))}))


def seeds_for(config):
    return noise_seed_values(config)[:2]


def make_rng(config,shape,device):
    n=config['noise'];seeds,subseeds,ensd=noise_seed_values(config)
    if n['source']=='GPU' and torch.device(device).type!='cuda':
        raise BridgeError('RNG_DEVICE_UNAVAILABLE','GPU RNG requires a CUDA device; no CPU fallback was substituted')
    rng=ImageRNG(RNGConfig(n['source'],device,ensd),shape,seeds,subseeds,n['subseed_strength'],n['seed_resize_from']['height'],n['seed_resize_from']['width'])
    return rng


def freeze_state(rng):
    state=rng.get_state()
    states=[]
    for g in state['generators']:
        states.append((g[0],g[1].cpu().numpy().tobytes()) if g[0]=='torch' else tuple(g))
    return state['is_first'],tuple(states)


def restore_state(rng,state):
    states=[]
    for g in state[1]:
        states.append((g[0],torch.frombuffer(bytearray(g[1]),dtype=torch.uint8).clone()) if g[0]=='torch' else g)
    rng.set_state({'is_first':state[0],'generators':states})
    return rng


@dataclass(frozen=True,slots=True)
class NoiseBundle:
    config_hash:str
    family:str
    shape:tuple
    model_identity:tuple
    binding_json:str
    initial:torch.Tensor
    initial_hash:str
    latent_hash:str
    state:tuple
    seeds:tuple
    device:str

    def replay(self,config,device):
        # Same-device generator state is required; do not reinterpret CUDA bytes as CPU state.
        if config['noise']['source']=='GPU' and str(device)!=self.device:
            raise BridgeError('RNG_DEVICE_UNAVAILABLE','GPU RNG device changed after the noise snapshot')
        return restore_state(make_rng(config,self.shape[1:],device),self.state)


def generate_noise(spec,model,binding,input_latent=None):
    cfg=spec.require_executable();validate_model(model,cfg);shape=latent_shape(model,cfg)
    if cfg['mode']=='img2img':
        samples=check_latent(input_latent,shape)
        latent=copy.copy(input_latent);latent['samples']=samples.clone()
    else:
        if input_latent is None:
            latent={'samples':torch.zeros(shape,dtype=torch.float32,device='cpu')}
        else:
            import comfy.sample
            latent=copy.copy(input_latent)
            latent['samples']=comfy.sample.fix_empty_latent_channels(model,input_latent['samples'],
                input_latent.get('downscale_ratio_spacial'),input_latent.get('downscale_ratio_temporal'))
            samples=check_latent(latent,shape)
            if torch.count_nonzero(samples):raise BridgeError('LATENT_MISMATCH','txt2img requires an empty latent')
            latent['samples']=samples.clone()
    device=model.load_device
    rng=make_rng(cfg,shape[1:],device)
    initial=rng.next().detach().cpu().float()
    bundle=NoiseBundle(spec.config_hash,cfg['family'],tuple(shape),model_identity(model),canonical(binding),initial,
                       tensor_hash(initial),tensor_hash(latent['samples']),freeze_state(rng),seeds_for(cfg)[0],str(device))
    return latent,bundle
