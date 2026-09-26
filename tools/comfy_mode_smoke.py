"""Actual Comfy CPU sampler comparison with synthetic models, not image parity.

Usage: python tools/comfy_mode_smoke.py /path/to/ComfyUI
"""
from pathlib import Path
import copy
import importlib.util
import json
import sys

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(sys.argv[1]).resolve()))
from comfy.cli_args import args
args.cpu=True
import torch
import nodes as core
import comfy.model_base as mb
import comfy.supported_models as supported
import comfy.model_patcher as mp
import comfy.model_management as mm

spec=importlib.util.spec_from_file_location('bridge_qa',root/'__init__.py',submodule_search_locations=[str(root)])
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
from bridge_qa.bridge.workflow_adapter import ForgeSettings,ForgeTextInput,sample_workflow


class AnalyticDiffusion(torch.nn.Module):
    def __init__(self):
        super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(1))
    @property
    def dtype(self):return self.anchor.dtype
    def forward(self,x,t,context=None,**kwargs):
        mean=context.mean(dim=tuple(range(1,context.ndim)))
        return torch.tanh(x)*.03+mean.reshape((-1,)+(1,)*(x.ndim-1))*.001
    def preprocess_text_embeds(self,text,ids,t5xxl_weights=None):
        out=text.mean(dim=1,keepdim=True).expand(-1,ids.shape[1],-1).clone()
        return torch.nn.functional.pad(out,(0,0,0,max(512-out.shape[1],0)))


class SyntheticClip:
    def __init__(self,family):self.family=family;self.layer=None
    def clone(self):return copy.copy(self)
    def clip_layer(self,layer):self.layer=layer
    def tokenize(self,text):return text
    def encode_from_tokens_scheduled(self,text):
        extra={}
        if self.family=='sdxl':extra={'pooled_output':torch.ones(1,1280)}
        if self.family=='anima':extra={'t5xxl_ids':torch.tensor([5,8,1]),'t5xxl_weights':torch.ones(3)}
        return [[torch.full((1,3,8),1. if text=='positive' else 0.),extra]]


results=[]
for family,klass,config_cls in [('sd15',mb.BaseModel,supported.SD15),('sdxl',mb.SDXL,supported.SDXL),('anima',mb.Anima,supported.Anima)]:
    config=config_cls({'disable_unet_model_creation':True,'in_channels':16 if family=='anima' else 4})
    config.set_inference_dtype(torch.float32,None)
    model=klass(config,device=torch.device('cpu'));model.diffusion_model=AnalyticDiffusion()
    patcher=mp.ModelPatcher(model,torch.device('cpu'),torch.device('cpu'))
    clip=SyntheticClip(family)
    settings=ForgeSettings(dict(forge_compatibility=False,clip_skip='2',mode='txt2img',ensd='31337'))
    latent={'samples':torch.zeros((1,16 if family=='anima' else 4,8,8))}
    for denoise in (1.,.5):
        params=dict(seed='9007199254740993',steps=4,cfg=2.,sampler_name='euler',scheduler='normal',denoise=denoise)
        output=sample_workflow(patcher,ForgeTextInput('positive',clip,'1'),ForgeTextInput('negative',clip,'2'),settings,params,latent)['result'][0]
        core_clip,=core.CLIPSetLastLayer().set_last_layer(clip,-2)
        positive,=core.CLIPTextEncode().encode(core_clip,'positive')
        negative,=core.CLIPTextEncode().encode(core_clip,'negative')
        expected,=core.KSampler().sample(patcher,int(params['seed']),4,2.,'euler','normal',positive,negative,latent,denoise=denoise)
        assert torch.equal(output['samples'],expected['samples'])
        assert torch.isfinite(output['samples']).all()
        assert clip.layer is None
        results.append(dict(family=family,denoise=denoise,exact_core_match=True,shape=list(output['samples'].shape)))
    mm.unload_all_models()
print(json.dumps(dict(scope='actual core CPU; synthetic model and conditioning; no GPU or pretrained-image parity',results=results),indent=2))
