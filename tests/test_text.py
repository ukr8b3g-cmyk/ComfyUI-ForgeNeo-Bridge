"""Token/planning tests use deterministic synthetic tokenizers, not trained weights."""
import copy
import re
from types import SimpleNamespace as NS
import pytest
import torch
from fnb.bridge.spec import default_document, finalize, BridgeError
from fnb.bridge.text import AnimaEncoder, ClassicEncoder, plan_prompts, ensure_clip_family
from fnb.bridge.classic import ClassicEngine, apply_emphasis

class Tokenizer:
    def __init__(self, repeat=1):self.repeat=repeat
    def get_vocab(self):return {',</w>':2}
    def __call__(self,texts,**kwargs):
        return {'input_ids':[[2 if token==',' else sum(map(ord,token))%100+3 for token in re.findall(r',|[^,\s]+',s) for _ in range(self.repeat)] for s in texts]}

class Transformer:
    def __init__(self):self.calls=[]
    def __call__(self,tokens,mask,**kwargs):
        self.calls.append((tokens.clone(),dict(kwargs)))
        z=tokens.float().unsqueeze(-1).repeat(1,1,3)/1000+1
        return z,z+kwargs['intermediate_output']/100,torch.ones((tokens.shape[0],3))*7,torch.ones((tokens.shape[0],3))*9

def clip_model(pad=49407):return NS(special_tokens={'start':49406,'end':49407,'pad':pad},transformer=Transformer())

@pytest.mark.parametrize('length',[0,1,74,75,76,149,150,151])
def test_classic_boundaries(length):
    c=ClassicEngine(clip_model(),Tokenizer(),'Original')
    chunks,n=c.tokenize_line('x '*length)
    assert len(chunks)==max(1,(length+74)//75)
    assert n==length
    assert all(len(k.tokens)==77 and len(k.multipliers)==77 for k in chunks)
    assert all(k.tokens[0]==49406 and k.tokens[-1]==49407 for k in chunks)

def test_break_backtrack_and_zero():
    c=ClassicEngine(clip_model(),Tokenizer(),'No norm')
    chunks,_=c.tokenize_line('x BREAK (y:0)')
    assert len(chunks)==2 and chunks[1].multipliers[1]==0
    chunks,_=c.tokenize_line('x '*68+', '+'y '*12)
    assert len(chunks)==2 and chunks[0].tokens[70]==49407
    c2=ClassicEngine(clip_model(),Tokenizer(),'Original',comma_padding_backtrack=0)
    assert c2.tokenize_line('x '*68+', '+'y '*12)[0][0].tokens[70]!=49407

@pytest.mark.parametrize('family,skip,expected',[('sd15',1,-1),('sd15',2,-2),('sd15',4,-4),('sdxl',1,-2),('sdxl',2,-2),('sdxl',4,-4)])
def test_clip_skip_and_pooled(family,skip,expected):
    cfg=default_document(family)['effective'];cfg['text']['clip_skip']=skip
    csm=NS(clip_l=clip_model());tok=NS(clip_l=NS(tokenizer=Tokenizer()))
    if family=='sdxl':csm.clip_g=clip_model(0);tok.clip_g=NS(tokenizer=Tokenizer(2))
    encoder=ClassicEncoder(NS(cond_stage_model=csm,tokenizer=tok),cfg)
    z,extra,info=encoder.encode('x '*40,'cpu')
    call=csm.clip_l.transformer.calls[0][1]
    assert call['intermediate_output']==expected
    assert call['final_layer_norm_intermediate']==(family=='sd15')
    if family=='sdxl':
        assert z.shape==(1,154,6)
        assert torch.equal(extra['pooled_output'],torch.ones((1,3))*7)
        assert torch.count_nonzero(z[:,77:,:3])==0
        assert csm.clip_g.transformer.calls[0][1]['intermediate_output']==expected
        tokens=csm.clip_g.transformer.calls[-1][0]
        assert tokens[0,-1]==0
    else:assert z.shape==(1,77,3)

@pytest.mark.parametrize('mode',['Original','No norm','Ignore','None'])
def test_emphasis(mode):
    z=torch.arange(1,13,dtype=torch.float32).reshape(1,4,3);weights=torch.tensor([[1.,0.,2.,.5]])
    before=z.clone();out=apply_emphasis(mode,z,weights)
    assert torch.equal(z,before)
    if mode=='Original':assert out.mean().item()==pytest.approx(z.mean().item())
    elif mode=='No norm':assert torch.equal(out,z*weights[...,None])
    else:assert torch.equal(out,z)

@pytest.mark.parametrize('skip',[1,2,4])
def test_anima_no_layer_skip(document,skip):
    document['effective']['text']['clip_skip']=skip
    cfg=finalize(document).config
    clip=NS(tokenizer=NS(qwen3_06b=NS(tokenizer=Tokenizer()),t5xxl=NS(tokenizer=Tokenizer(2))))
    encoder=AnimaEncoder(clip,cfg)
    data=encoder.tokens('a (b:2)')
    assert cfg['text']['clip_skip'] is None
    assert data['t5_weights']==[1,1,2,2,1]
    assert data['qwen_mask']==[1,1]
    assert encoder.tokens('')['qwen_ids']==[151643]
    assert encoder.tokens('')['qwen_mask']==[1]
    assert encoder.tokens('')['t5_ids']==[1]

@pytest.mark.parametrize('sampler,calls',[('euler',10),('heun',20),('dpm_2',20)])
def test_schedules_and_and(document,sampler,calls):
    cfg=document['effective'];cfg['sampling']['sampler']=sampler
    cfg['text']['positive_raw']='red [cat:dog:.5] :1.5 AND cup :0.5'
    pos,neg,n=plan_prompts(cfg)
    assert n==calls
    assert [p['weight'] for p in pos]==[1.5,.5]
    assert pos[0]['schedule']==[[calls//2,'red cat'],[calls,'red dog']]
    assert neg[0]['weight'] is None

def test_prompt_lora_requires_explicit_chain(document):
    cfg=document['effective'];cfg['text']['positive_raw']='<lora:missing:1> cat'
    with pytest.raises(BridgeError,match='ASSET_BINDING_MISMATCH'):plan_prompts(cfg)

def test_clip_external_layer_rejected():
    clip=NS(cond_stage_model=NS(clip_l=clip_model()),tokenizer=NS(clip_l=NS(tokenizer=Tokenizer())),layer_idx=-2)
    with pytest.raises(BridgeError,match='UNVERIFIED'):ensure_clip_family(clip,'sd15')
