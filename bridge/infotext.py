"""Loss-preserving infotext -> reviewed GenerationSpec; never assumes Module order."""
from __future__ import annotations
import copy
import json
import re
from .spec import BridgeError, default_document, finalize, issue, parse_json, pointer_set, provenance

ALIASES = {
 'steps':('/sampling/steps',int), 'cfg scale':('/sampling/cfg',float), 'seed':('/noise/seed',str),
 'rng':('/noise/source',lambda x:x.upper()), 'ensd':('/noise/ensd',str),
 'variation seed':('/noise/subseed',str), 'variation seed strength':('/noise/subseed_strength',float),
 'clip skip':('/text/clip_skip',int), 'emphasis':('/text/emphasis',str),
 'eta':('/sampling/eta_ancestral',float), 'eta ddim':('/sampling/eta_ddim',float),
 'sigma churn':('/sampling/s_churn',float), 'sigma tmin':('/sampling/s_tmin',float),
 'sigma tmax':('/sampling/s_tmax',float), 'sigma noise':('/sampling/s_noise',float),
 'schedule min sigma':('/sampling/sigma_min',float), 'schedule max sigma':('/sampling/sigma_max',float),
 'schedule rho':('/sampling/rho',float), 'beta schedule alpha':('/sampling/beta_alpha',float),
 'beta schedule beta':('/sampling/beta_beta',float), 'beta scheduler alpha':('/sampling/beta_alpha',float),
 'beta scheduler beta':('/sampling/beta_beta',float), 'shift':('/sampling/shift',float),
 'skip early cfg':('/guidance/skip_early_cfg',float), 'ngms':('/guidance/ngms',float),
 'extra noise':('/sampling/img2img_extra_noise',float), 'denoising strength':('/sampling/denoise',float),
}
for key in ('s_churn','s_tmin','s_tmax','s_noise'):ALIASES[key]=('/sampling/'+key,float)
SAMPLER_NAMES={'DPM++ 2M':'dpmpp_2m','DPM++ SDE':'dpmpp_sde','DPM++ 2M SDE':'dpmpp_2m_sde','DPM++ 3M SDE':'dpmpp_3m_sde',
 'DPM++ 2s a RF':'dpmpp_2s_ancestral_rf','Euler a':'euler_ancestral','Euler':'euler','ER SDE':'er_sde','LCM':'lcm','LMS':'lms','Heun':'heun',
 'DPM2':'dpm_2','Res Multistep':'res_multistep','Kohaku LoNyu Yog':'kohaku_lonyu_yog','Restart':'restart','UniPC':'unipc','DDIM':'ddim','PLMS':'plms',
 'DPM++ 2M CFG++':'dpmpp_2m_cfg_pp','Euler a CFG++':'euler_ancestral_cfg_pp','Euler CFG++':'euler_cfg_pp'}
SCHEDULER_NAMES={'Automatic':'automatic','Karras':'karras','Exponential':'exponential','Polyexponential':'polyexponential','Normal':'normal','Simple':'simple','Uniform':'uniform',
 'SGM Uniform':'sgm_uniform','SGMUniform':'sgm_uniform','Linear Quadratic':'linear_quadratic','KL Optimal':'kl_optimal','DDIM':'ddim','Align Your Steps':'align_your_steps',
 'Beta':'beta','Turbo':'turbo','Bong Tangent':'bong_tangent','FlowMatchEulerDiscrete':'flow_match','Flux2':'flux2'}


def parse_bool(text):
    if text.lower() not in ('true','false'):raise BridgeError('INVALID_SPEC','Expected True/False')
    return text.lower()=='true'
for key,path in [('discard penultimate sigma','/sampling/discard_penultimate_requested'),('sgm noise multiplier','/sampling/sgm_noise_multiplier_requested'),('ngms all steps','/guidance/ngms_all_steps')]:
    ALIASES[key]=(path,parse_bool)


def split_fields(text):
    fields=[];start=0;quote=False;escape=False;depth=0
    for i,ch in enumerate(text+','):
        if escape:escape=False;continue
        if ch=='\\' and quote:escape=True;continue
        if ch=='"':quote=not quote
        elif not quote:
            if ch in '[{':depth+=1
            if ch in ']}':depth-=1
            if depth<0:raise BridgeError('INVALID_METADATA','Unbalanced metadata value')
            if ch==',' and depth==0:
                field=(text+',')[start:i];start=i+1
                key,sep,value=field.partition(':')
                if not sep:raise BridgeError('INVALID_METADATA','Expected key: value infotext field')
                value=value.strip()
                if value.startswith('"'):
                    value=parse_json(value)
                    if not isinstance(value,str):raise BridgeError('INVALID_METADATA','Expected quoted string')
                fields.append({'key':key.strip(),'value':value})
    if quote or depth:raise BridgeError('INVALID_METADATA','Unclosed metadata value')
    return fields


def parse_loras(text):
    entries=[]
    def replace(match):
        kind, value=match.group(1),match.group(2)
        if kind!='lora' or '@' in value:raise BridgeError('UNSUPPORTED_COMBINATION','Only explicit static LoRA tags are supported')
        pos=[];named={}
        for part in value.split(':'):
            if '=' in part:
                key,val=part.split('=',1);named[key]=val
            else:pos.append(part)
        if not pos or not pos[0] or len(pos)>3 or any(k not in ('te','unet') for k in named):raise BridgeError('INVALID_METADATA','Invalid LoRA syntax')
        te=float(pos[1]) if len(pos)>1 else 1.0
        te=float(named.get('te',te));unet=float(named.get('unet',pos[2] if len(pos)>2 else te))
        entries.append({'name':pos[0],'strength_model':unet,'strength_text_encoder':te})
        return ''
    clean=re.sub(r'<(\w+):([^>]+)>',replace,text)
    return clean,entries


def import_infotext(text, family='anima', policy='strict', tolerant=False):
    if len(text.encode('utf-8'))>16*1024*1024:raise BridgeError('IMPORT_LIMIT_EXCEEDED','Infotext too large')
    p=text.rfind('\nSteps:')
    if p<0:raise BridgeError('NOT_FORGE_METADATA','Missing generation parameters')
    head=text[:p];p2=head.rfind('\nNegative prompt:')
    positive=head if p2<0 else head[:p2]
    negative='' if p2<0 else head[p2+len('\nNegative prompt:'):]
    if negative.startswith(' '):negative=negative[1:]
    fields=split_fields(text[p+1:])
    doc=default_document(family,policy);cfg=doc['effective']
    doc['source'].update(kind='infotext',raw_infotext=text,raw_fields=fields)
    doc['provenance']={f'/{key}':provenance('profile_default',False,evidence='Unobserved defaults; review required') for key in cfg}
    doc['unresolved']=[];doc['unsupported']=[]
    raw_assets=[];seen={}
    def set_value(path,value,key):
        if path in seen and seen[path]!=value:raise BridgeError('INVALID_METADATA',f'Conflicting aliases for {path}')
        seen[path]=value;pointer_set(cfg,path,value);doc['requested'][path]=copy.deepcopy(value)
        doc['provenance'][path]=provenance('metadata_explicit',True,key)
    set_value('/text/positive_raw',positive,'positive prompt');set_value('/text/negative_raw',negative,'negative prompt')
    for f in fields:
        key=f['key'].casefold();value=f['value']
        try:
            if key in ALIASES:
                path,convert=ALIASES[key];v=convert(value)
                # A recorded zero sentinel means model default, not unknown.
                original=v
                if path in ('/sampling/sigma_min','/sampling/sigma_max','/sampling/rho') and v==0:v=None
                set_value(path,v,f['key']);doc['requested'][path]=original
            elif key in ('size','seed resize from'):
                m=re.fullmatch(r'(\d+)[x×](\d+)',value)
                if not m:raise ValueError('Invalid dimensions')
                a,b=map(int,m.groups())
                if key=='size':
                    set_value('/image/width',a,f['key']);set_value('/image/height',b,f['key'])
                    if cfg['sdxl']:
                        for k,v in [('original_width',a),('target_width',a),('original_height',b),('target_height',b)]:cfg['sdxl'][k]=v
                else:set_value('/noise/seed_resize_from',{'width':a,'height':b},f['key'])
            elif key=='sampler':
                name=value
                for suffix in (' Karras',' Exponential'):
                    if name.endswith(suffix):
                        name=name[:-len(suffix)];set_value('/sampling/scheduler',suffix.strip().lower(),f['key'])
                known={k.casefold():v for k,v in SAMPLER_NAMES.items()}
                if name.casefold() not in known:
                    if not tolerant:raise BridgeError('UNSUPPORTED_COMBINATION',f'Unknown sampler {name}')
                    doc['unsupported'].append(issue('UNSUPPORTED_COMBINATION','/sampling/sampler',f'Unknown sampler {name}'))
                    doc['extensions']['unhandled_sampler']=value
                    continue
                set_value('/sampling/sampler',known[name.casefold()],f['key'])
            elif key=='schedule type':
                known={k.casefold():v for k,v in SCHEDULER_NAMES.items()}
                if value.casefold() not in known:
                    if not tolerant:raise BridgeError('UNSUPPORTED_COMBINATION',f'Unknown scheduler {value}')
                    doc['unsupported'].append(issue('UNSUPPORTED_COMBINATION','/sampling/scheduler',f'Unknown scheduler {value}'))
                    doc['extensions']['unhandled_scheduler']=value
                    continue
                set_value('/sampling/scheduler',known[value.casefold()],f['key'])
            elif key=='version':
                doc['source']['exporter_version']=value
                doc['source']['exporter']='forge_neo' if value.lower().startswith('neo') else 'a1111'
            elif key=='model' or key=='vae' or re.fullmatch(r'module \d+',key):
                raw_assets.append({'key':f['key'],'name':value,'role_hint':key if key in ('model','vae') else None})
            elif key in ('model hash','vae hash','lora hashes','ti hashes','hashes','user'):
                doc['extensions'].setdefault('descriptive_metadata',{})[f['key']]=value
            elif key.startswith('hires') or key.startswith(('adetailer','controlnet','reference','anima ref')) or key in ('face restoration','conditional mask weight','pad conds','pad conds v0'):
                doc['unsupported'].append(issue('UNSUPPORTED_COMBINATION','/extensions',f'{f["key"]} requires a separate adapter'))
                if key.startswith('hires'):cfg['mode']='hires_fix'
            else:
                doc['extensions'].setdefault('unknown_fields',[]).append(f)
                doc['unresolved'].append(issue('UNKNOWN_PARAMETER','/extensions',f'Review unhandled parameter {f["key"]}'))
        except (ValueError,TypeError) as exc:
            if isinstance(exc,BridgeError):raise
            raise BridgeError('INVALID_METADATA',f'Invalid value for {f["key"]}') from exc
    if any(f['key'].casefold()=='denoising strength' for f in fields) and cfg['mode']=='txt2img':cfg['mode']='img2img'
    try:_,loras=parse_loras(positive)
    except BridgeError as exc:
        if not tolerant or exc.code!='UNSUPPORTED_COMBINATION':raise
        doc['unsupported'].append(issue(exc.code,'/text/positive_raw',str(exc)))
        loras=[]
    doc['extensions']['raw_assets']=raw_assets
    doc['extensions']['raw_loras']=loras
    if re.search(r'\bembedding:',positive+' '+negative):doc['unsupported'].append(issue('UNSUPPORTED_COMBINATION','/text','Textual Inversion is outside V1'))
    # The GUI must confirm assets and inferred groups before ready. Preserve every raw field.
    return finalize(doc)
