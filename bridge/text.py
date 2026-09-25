"""Forge text planning/encoding without changing the shared CLIP wrapper or transformer."""
from __future__ import annotations
import contextlib
import copy
import re
from dataclasses import dataclass
from types import MappingProxyType
import torch
from .binding import digest, tensor_hash
from .infotext import parse_loras
from .spec import BridgeError, canonical
from .schedules import SAMPLER_OPTIONS, step_plan
from ..vendor.forge.parsing import parse_prompt_attention

DTYPES = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}


def dtype_name(dtype):
    return next((k for k,v in DTYPES.items() if v==dtype), str(dtype))


def require_dtype(requested, actual, label):
    if requested not in (None, 'as_loaded') and DTYPES.get(requested) != actual:
        raise BridgeError('DTYPE_NOT_APPLIED', f'{label}: requested {requested}, actual {actual}; load the required precision explicitly')


def ensure_clip_family(clip, family):
    csm, tok = clip.cond_stage_model, clip.tokenizer
    if family=='anima': ok=hasattr(csm,'qwen3_06b') and hasattr(tok,'qwen3_06b') and hasattr(tok,'t5xxl')
    elif family=='sdxl': ok=hasattr(csm,'clip_l') and hasattr(csm,'clip_g') and hasattr(tok,'clip_l') and hasattr(tok,'clip_g')
    else: ok=hasattr(csm,'clip_l') and not hasattr(csm,'clip_g') and hasattr(tok,'clip_l')
    if not ok: raise BridgeError('MODEL_FAMILY_MISMATCH','Text encoder is not the expected model family')
    if getattr(clip,'layer_idx',None) is not None or getattr(clip,'apply_hooks_to_conds',None):
        raise BridgeError('UNVERIFIED_PATCH_CHAIN','Layer/hook overrides must be represented in Spec')


def prepare_clip(clip):
    import comfy.model_management as mm
    # Core owns weight loading/offload. Never reset_clip_options / set_clip_options.
    clip.load_model()
    return clip.patcher.load_device, mm


def plan_prompts(cfg):
    clean, tags = parse_loras(cfg['text']['positive_raw'])
    expected = cfg['loras']
    if tags:
        assets={a['asset_id']:a for a in cfg['assets']}
        actual=[(x['strength_model'],x['strength_text_encoder']) for x in tags]
        ref=[(x['strength_model'],x['strength_text_encoder']) for x in expected]
        if actual != ref:
            raise BridgeError('ASSET_BINDING_MISMATCH','Prompt LoRA tags must match the explicit loader list, including order')
        for tag, l in zip(tags, expected):
            name=assets[l['asset_id']]['relative_name']
            if name is None or name.replace('\\','/').split('/')[-1].rsplit('.',1)[0] != tag['name'].replace('\\','/').split('/')[-1]:
                raise BridgeError('ASSET_BINDING_MISMATCH','Prompt LoRA name requires an explicit unambiguous binding')
    if re.search(r'<\w+:',cfg['text']['negative_raw']):
        # Forge does not activate negative-prompt LoRAs. Preserve literal text there.
        pass
    _,calls,_=step_plan(cfg)
    calls *= 2 if SAMPLER_OPTIONS[cfg['sampling']['sampler']].get('second_order') else 1
    try:
        from ..vendor.forge import prompt_parser
    except ImportError as exc:
        raise BridgeError('DEPENDENCY_MISSING','Bundled prompt parser is unavailable; reinstall the complete Bridge folder (no automatic installation was attempted)') from exc
    indices, flat, _ = prompt_parser.get_multicond_prompt_list([clean])
    schedules = prompt_parser.get_learned_conditioning_prompt_schedules(flat,calls)
    positives=[{'weight':weight,'schedule':schedules[index]} for index,weight in indices[0]]
    negatives=[{'weight':None,'schedule':prompt_parser.get_learned_conditioning_prompt_schedules([cfg['text']['negative_raw']],calls)[0]}]
    total=sum(len(p['schedule']) for p in positives+negatives)
    if total>4096: raise BridgeError('IMPORT_LIMIT_EXCEEDED','Prompt plan exceeds 4096 schedule entries')
    return positives,negatives,calls


class AnimaEncoder:
    def __init__(self,clip,cfg):
        self.clip,self.cfg=clip,cfg
        self.qwen=clip.tokenizer.qwen3_06b.tokenizer
        self.t5=clip.tokenizer.t5xxl.tokenizer
    def tokens(self,line):
        parsed=parse_prompt_attention(line,self.cfg['text']['emphasis'])
        texts=[t for t,w in parsed]
        q=self.qwen(texts,truncation=False,add_special_tokens=False)['input_ids']
        t=self.t5(texts,truncation=False,add_special_tokens=False)['input_ids']
        qids=[x for row in q for x in row] or [151643]
        tids=[x for row in t for x in row]+[1]
        weights=[w for row,(_,w) in zip(t,parsed) for x in row]+[1.0]
        mask=[];eos=False
        for token in qids:
            mask.append(0 if eos else 1)
            if token==151643:eos=True
        return {'qwen_ids':qids,'t5_ids':tids,'t5_weights':weights,'qwen_mask':mask}
    def encode(self,line,device):
        data=self.tokens(line)
        transformer=self.clip.cond_stage_model.qwen3_06b.transformer
        ids=torch.tensor([data['qwen_ids']],device=device,dtype=torch.long)
        # Forge anima_engine.process_embeds supplies embedding output; no fallback-layer trick.
        embeds=transformer.get_input_embeddings()(ids)
        requested=self.cfg['numeric']['text_encoder_dtype']
        if requested!='as_loaded':embeds=embeds.to(DTYPES[requested])
        mask=torch.tensor([data['qwen_mask']],device=device,dtype=torch.long)
        out=transformer(None,embeds=embeds,attention_mask=mask,num_tokens=[sum(data['qwen_mask'])],embeds_info=[],dtype=embeds.dtype)
        require_dtype(requested,out[0].dtype,'Anima text encoder output')
        extra={'t5xxl_ids':torch.tensor(data['t5_ids'],dtype=torch.int),
               't5xxl_weights':torch.tensor(data['t5_weights'],dtype=torch.float32)}
        z=out[0].float().cpu()
        return z,extra,{'tokens':data,'embedding_dtype':dtype_name(embeds.dtype),'output_dtype':dtype_name(out[0].dtype)}


class ClassicEncoder:
    def __init__(self,clip,cfg):
        from .classic import ClassicEngine
        self.cfg=cfg;self.engines={}
        sdxl=cfg['family']=='sdxl'
        for name in (('clip_l','clip_g') if sdxl else ('clip_l',)):
            self.engines[name]=ClassicEngine(getattr(clip.cond_stage_model,name),getattr(clip.tokenizer,name).tokenizer,
                cfg['text']['emphasis'],clip_skip=cfg['text']['clip_skip'],minimal_clip_skip=2 if sdxl else 1,
                text_projection=name=='clip_g',return_pooled=name=='clip_g',final_layer_norm=not sdxl,
                comma_padding_backtrack=cfg['text']['comma_padding_backtrack'])
    def encode(self,line,device):
        z_l=self.engines['clip_l'](line,device)
        trace={'clip_l':self.engines['clip_l'].last_token_trace}
        pooled=None;extra={}
        if self.cfg['family']=='sdxl':
            z_g,pooled=self.engines['clip_g'](line,device)
            n=max(z_l.shape[1],z_g.shape[1])
            z_l=torch.nn.functional.pad(z_l,(0,0,0,n-z_l.shape[1]))
            z_g=torch.nn.functional.pad(z_g,(0,0,0,n-z_g.shape[1]))
            z=torch.cat((z_l,z_g),dim=2)
            trace['clip_g']=self.engines['clip_g'].last_token_trace
            d=self.cfg['sdxl']
            extra.update(width=d['original_width'],height=d['original_height'],crop_w=d['crop_left'],crop_h=d['crop_top'],target_width=d['target_width'],target_height=d['target_height'])
        else:z=z_l
        # The reference CLIP arithmetic uses fp32 activations. Other requests must not be silently ignored.
        require_dtype(self.cfg['numeric']['text_encoder_dtype'],z.dtype,'CLIP activation output')
        extra['pooled_output']=None if pooled is None else pooled.float().cpu()
        return z.float().cpu(),extra,{'tokens':trace,'output_dtype':dtype_name(z.dtype)}


@dataclass(frozen=True,slots=True)
class ConditioningBundle:
    config_hash:str
    family:str
    binding_json:str
    clip_identity:tuple
    positive:tuple
    negative:tuple
    entries:object
    trace_json:str

    def cloned_entries(self):
        return {k:[[t.clone(),{name:v.clone() if isinstance(v,torch.Tensor) else copy.deepcopy(v) for name,v in meta.items()}]] for k,(t,meta) in self.entries.items()}


def clip_identity(clip):
    return (id(clip.cond_stage_model),str(getattr(clip.patcher,'patches_uuid','')),digest({'keys':sorted(getattr(clip.patcher,'patches',{}))}))


@torch.inference_mode()
def encode_bundle(spec,clip,binding):
    cfg=spec.require_executable();ensure_clip_family(clip,cfg['family'])
    positive,negative,calls=plan_prompts(cfg)
    device,mm=prepare_clip(clip)
    encoder=AnimaEncoder(clip,cfg) if cfg['family']=='anima' else ClassicEncoder(clip,cfg)
    entries={};trace={};lookup={}
    def expand(parts,is_negative):
        result=[]
        for part in parts:
            plan=[]
            for end,text in part['schedule']:
                key=(text,is_negative)
                if key not in lookup:
                    cid='condition_'+str(len(entries));lookup[key]=cid
                    z,extra,info=encoder.encode(text,device)
                    if cfg['family']=='sdxl' and is_negative and text=='' and cfg['sdxl']['zero_empty_negative']:
                        z=torch.zeros_like(z);extra['pooled_output']=torch.zeros_like(extra['pooled_output'])
                    if not torch.isfinite(z).all():raise BridgeError('NONFINITE_CONDITIONING','Encoder produced NaN/Inf')
                    entries[cid]=(z,extra)
                    trace[cid]={'token_hash':digest(info['tokens']),'conditioning_hash':tensor_hash(z),'shape':list(z.shape),**info}
                plan.append((int(end),lookup[key]))
            result.append((part['weight'],tuple(plan)))
        return tuple(result)
    with mm.cuda_device_context(device) if hasattr(mm,'cuda_device_context') else contextlib.nullcontext():
        pos=expand(positive,False)
        # Do not load unused negative conditions at CFG1. Both plans remain in the report.
        neg=expand(negative,True) if cfg['sampling']['cfg']!=1 else ()
    trace['schedule_calls']=calls
    return ConditioningBundle(spec.config_hash,cfg['family'],canonical(binding),clip_identity(clip),pos,neg,MappingProxyType(entries),canonical(trace))
