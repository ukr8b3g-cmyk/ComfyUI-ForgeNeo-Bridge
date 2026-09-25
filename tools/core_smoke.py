"""Run actual pinned Comfy CFGGuider/ModelPatcher on CPU with analytic tiny models.
This is an API/lifecycle test, not pretrained-model, VAE, or GPU/image parity.
Usage: python tools/core_smoke.py /path/to/pinned/ComfyUI
"""
from pathlib import Path
import copy
import gc
import importlib.util
import json
import sys
from types import MappingProxyType

root=Path(__file__).resolve().parents[1]
core=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(core))
from comfy.cli_args import args
args.cpu=True
import torch
import comfy.model_base as mb
import comfy.supported_models as supported
import comfy.model_patcher as mp
import comfy.model_management as mm
import comfy.samplers
import comfy.sample

module_spec=importlib.util.spec_from_file_location('fnb_smoke',root/'__init__.py',submodule_search_locations=[str(root)])
bridge=importlib.util.module_from_spec(module_spec);sys.modules[module_spec.name]=bridge;module_spec.loader.exec_module(bridge)
from fnb_smoke.bridge.spec import default_document,finalize,canonical
from fnb_smoke.bridge.noise import generate_noise
from fnb_smoke.bridge.text import ConditioningBundle
from fnb_smoke.bridge.runtime import execute

class AnalyticDiffusion(torch.nn.Module):
    def __init__(self):
        super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(1));self.interrupt=False
    @property
    def dtype(self):return self.anchor.dtype
    def forward(self,x,t,context=None,**kwargs):
        if self.interrupt:raise InterruptedError('Injected test interruption')
        mean=context.mean(dim=tuple(range(1,context.ndim))) if context is not None else torch.zeros(x.shape[0])
        return torch.tanh(x)*.03+mean.reshape((-1,)+(1,)*(x.ndim-1))*.001
    def preprocess_text_embeds(self,text,ids,t5xxl_weights=None):
        # Synthetic adapter only: exercise Anima extra_conds before denoising.
        out=text.mean(dim=1,keepdim=True).expand(-1,ids.shape[1],-1).clone()
        if t5xxl_weights is not None:out=out*t5xxl_weights
        return torch.nn.functional.pad(out,(0,0,0,max(512-out.shape[1],0)))

functions=(comfy.sample.prepare_noise,comfy.samplers.calc_cond_batch,comfy.samplers.KSampler.sample)
results=[]
for family,klass,cfgclass in [('anima',mb.Anima,supported.Anima),('sd15',mb.BaseModel,supported.SD15),('sdxl',mb.SDXL,supported.SDXL)]:
    config=cfgclass({'disable_unet_model_creation':True,'in_channels':16 if family=='anima' else 4})
    config.set_inference_dtype(torch.float32,None)
    model=klass(config,device=torch.device('cpu'))
    model.diffusion_model=AnalyticDiffusion()
    patcher=mp.ModelPatcher(model,torch.device('cpu'),torch.device('cpu'))
    for sampler,scheduler in [('euler','simple'),('er_sde','beta'),('dpmpp_2m_sde','exponential')]:
        d=default_document(family)
        c=d['effective'];c['image'].update(width=64,height=64,batch_size=1);c['sampling'].update(sampler=sampler,scheduler=scheduler,steps=4,cfg=2.)
        c['assets']=[dict(asset_id=role,role=role,component=None,category=cat,relative_name='synthetic.safetensors',sha256=None,short_hash=None,resolution='resolved',binding_verification='known_loader_chain') for role,cat in [('model','diffusion_models'),('text_encoder','text_encoders'),('vae','vae')]]
        spec=finalize(d);binding={'hash':'smoke-test-only','chain':[],'assets':[]}
        extra={}
        if family=='anima':extra={'t5xxl_ids':torch.tensor([5,8,1],dtype=torch.int),'t5xxl_weights':torch.tensor([1.,2.,1.])}
        if family=='sdxl':
            extra={'pooled_output':torch.ones(1,1280),'width':64,'height':64,'target_width':64,'target_height':64,'crop_w':0,'crop_h':0}
        entries=MappingProxyType({'p':(torch.ones(1,3,8),extra),'n':(torch.zeros(1,3,8),copy.deepcopy(extra))})
        cond=ConditioningBundle(spec.config_hash,family,canonical(binding),(),((1.,((4,'p'),)),),((None,((4,'n'),)),),entries,'{}')
        before=torch.random.get_rng_state().clone();options=copy.deepcopy(patcher.model_options);objects=patcher.object_patches.copy()
        latent,noise=generate_noise(spec,patcher,binding)
        a,_,facts,tensors=execute(spec,patcher,cond,noise,latent,binding,'tensors')
        b,_,_,_=execute(spec,patcher,cond,noise,latent,binding,'summary')
        assert torch.equal(a['samples'],b['samples'])
        assert torch.equal(before,torch.random.get_rng_state())
        assert patcher.model_options==options and patcher.object_patches==objects
        assert (comfy.sample.prepare_noise,comfy.samplers.calc_cond_batch,comfy.samplers.KSampler.sample)==functions
        assert tensors['final_latent'].shape==a['samples'].shape
        if sampler=='euler':
            model.diffusion_model.interrupt=True
            try:execute(spec,patcher,cond,noise,latent,binding)
            except InterruptedError:pass
            else:raise AssertionError('Interruption did not propagate')
            finally:model.diffusion_model.interrupt=False
            after,_,_,_=execute(spec,patcher,cond,noise,latent,binding)
            assert torch.equal(after['samples'],a['samples'])
        results.append({'family':family,'sampler':sampler,'status':'PASS','shape':list(a['samples'].shape),'hash':facts['final_latent_hash']})
    mm.unload_all_models()
    del patcher, model
    gc.collect()
# Real Comfy transformer classes, small synthetic weights/tokenizers. This checks
# call signatures, layer selection, pooled outputs and masks, not model quality.
from types import SimpleNamespace as NS
import re
import comfy.clip_model
import comfy.ops
from comfy.text_encoders.llama import Qwen3_06B
from fnb_smoke.bridge.text import AnimaEncoder, ClassicEncoder

class SyntheticTokenizer:
    def get_vocab(self):return {',</w>':2}
    def __call__(self, texts, **kwargs):
        return {'input_ids':[[sum(map(ord,word))%100+3 for word in re.findall(r'[^\s]+',text)] for text in texts]}

def fill_weights(model):
    generator=torch.Generator(device='cpu').manual_seed(19)
    with torch.no_grad():
        for name,p in model.named_parameters():
            if 'norm' in name and name.endswith('weight'):p.fill_(1)
            elif name.endswith('bias'):p.zero_()
            else:p.copy_(torch.randn(p.shape,generator=generator)*.01)
    return model.eval()

text_results=[]
for family in ('anima','sd15','sdxl'):
    cfg=default_document(family)['effective']
    tokenizer=SyntheticTokenizer()
    if family=='anima':
        transformer=fill_weights(Qwen3_06B(dict(hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1),torch.float32,'cpu',comfy.ops.manual_cast))
        clip=NS(cond_stage_model=NS(qwen3_06b=NS(transformer=transformer)),tokenizer=NS(qwen3_06b=NS(tokenizer=tokenizer),t5xxl=NS(tokenizer=tokenizer)))
        encoder=AnimaEncoder(clip,cfg)
    else:
        options=json.loads((core/'comfy/sd1_clip_config.json').read_text())
        options.update(hidden_size=16,intermediate_size=32,num_hidden_layers=4,num_attention_heads=2,projection_dim=16)
        def sdclip(pad):
            return NS(special_tokens={'start':49406,'end':49407,'pad':pad},transformer=fill_weights(comfy.clip_model.CLIPTextModel(options,torch.float32,'cpu',comfy.ops.manual_cast)))
        csm=NS(clip_l=sdclip(49407));tok=NS(clip_l=NS(tokenizer=tokenizer))
        if family=='sdxl':csm.clip_g=sdclip(0);tok.clip_g=NS(tokenizer=tokenizer)
        clip=NS(cond_stage_model=csm,tokenizer=tok);encoder=ClassicEncoder(clip,cfg)
    before=torch.random.get_rng_state().clone()
    with torch.inference_mode():
        out,extra,trace=encoder.encode('red (teapot:1.2), cup','cpu')
        again,_,_=encoder.encode('red (teapot:1.2), cup','cpu')
    assert torch.isfinite(out).all() and torch.equal(out,again)
    assert torch.equal(before,torch.random.get_rng_state())
    if family=='sdxl':assert extra['pooled_output'].shape==(1,16)
    text_results.append({'family':family,'status':'PASS','shape':list(out.shape)})
    del encoder,clip
    gc.collect()
print(json.dumps({'test':'actual_comfy_cpu_analytic_smoke','pretrained_weights':False,'gpu':False,'results':results,'text_api_results':text_results},indent=2))
