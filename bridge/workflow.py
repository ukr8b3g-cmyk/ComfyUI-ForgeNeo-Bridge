"""Declarative, allow-listed graph plans. Frontend builds a detached graph before loading."""
from __future__ import annotations
from .spec import BridgeError,canonical

ALLOWED={'CheckpointLoaderSimple','UNETLoader','CLIPLoader','DualCLIPLoader','VAELoader','LoraLoader','LoraLoaderModelOnly',
         'ForgeCompatSpec','ForgeCompatText','ForgeCompatNoise','ForgeCompatSampler','VAEDecode','VAEEncode','LoadImage','SaveImage'}


def graph_plan(spec):
    cfg=spec.require_executable();assets={a['asset_id']:a for a in cfg['assets']}
    for a in assets.values():
        if a['relative_name'] is None:raise BridgeError('ASSET_MISSING','Resolve resource names before generating a workflow')
    nodes=[];edges=[]
    def add(kind,values=None,title=None):
        if kind not in ALLOWED:raise BridgeError('UNSUPPORTED_COMBINATION','Node is not in the graph allow-list')
        ident=str(len(nodes)+1);nodes.append({'id':ident,'type':kind,'values':values or {},'title':title or kind});return ident
    def connect(source,slot,target,input):edges.append([source,slot,target,input])
    def one(role):
        selected=[a for a in assets.values() if a['role']==role]
        if len(selected)!=1:raise BridgeError('ASSET_AMBIGUOUS',f'V1 graph planner needs exactly one {role} asset')
        return selected[0]
    m=one('model');v=one('vae');te=[a for a in assets.values() if a['role']=='text_encoder']
    if m['category']=='checkpoints':
        checkpoint=add('CheckpointLoaderSimple',{'ckpt_name':m['relative_name']})
        model=(checkpoint,0)
    elif m['category']=='diffusion_models':model=(add('UNETLoader',{'unet_name':m['relative_name'],'weight_dtype':'default'}),0);checkpoint=None
    else:raise BridgeError('ASSET_MISSING','Model must use checkpoint/diffusion_models category')
    if len(te)==1 and checkpoint and te[0]['category']=='checkpoints' and te[0]['relative_name']==m['relative_name']:clip=(checkpoint,1)
    elif len(te)==1 and te[0]['category']=='text_encoders':clip=(add('CLIPLoader',{'clip_name':te[0]['relative_name'],'type':'anima' if cfg['family']=='anima' else 'stable_diffusion','device':'default'}),0)
    elif len(te)==2 and all(a['category']=='text_encoders' for a in te) and cfg['family']=='sdxl':
        # Explicit component order, not a Module number guess.
        ordered=sorted(te,key=lambda a:a['component'] or '')
        clip=(add('DualCLIPLoader',{'clip_name1':ordered[0]['relative_name'],'clip_name2':ordered[1]['relative_name'],'type':'sdxl','device':'default'}),0)
    else:raise BridgeError('ASSET_AMBIGUOUS','Unsupported text-encoder resource binding')
    if checkpoint and v['category']=='checkpoints' and v['relative_name']==m['relative_name']:vae=(checkpoint,2)
    elif v['category']=='vae':vae=(add('VAELoader',{'vae_name':v['relative_name']}),0)
    else:raise BridgeError('ASSET_AMBIGUOUS','Unsupported VAE resource binding')
    for l in cfg['loras']:
        a=assets[l['asset_id']]
        ident=add('LoraLoader',{'lora_name':a['relative_name'],'strength_model':l['strength_model'],'strength_clip':l['strength_text_encoder']})
        connect(*model,ident,'model');connect(*clip,ident,'clip');model=(ident,0);clip=(ident,1)
    config=add('ForgeCompatSpec',{'source_mode':'spec_json','profile':spec.document()['profile']['id'],'policy':spec.document()['policy'],'payload':canonical(spec.document()),'overrides_json':'{}','image_file':''},'ForgeNeo Bridge — Generation Spec')
    text=add('ForgeCompatText');noise=add('ForgeCompatNoise');sampler=add('ForgeCompatSampler',{'trace_level':'summary'})
    for node in (text,noise,sampler):connect(config,0,node,'spec')
    connect(*clip,text,'clip');connect(*model,noise,'model');connect(*model,sampler,'model')
    connect(text,0,sampler,'conditioning');connect(noise,0,sampler,'latent_image');connect(noise,1,sampler,'noise')
    if cfg['mode']=='img2img':
        images=[a for a in assets.values() if a['role']=='init_image']
        if len(images)!=1:raise BridgeError('ASSET_MISSING','img2img requires one explicit init_image in input category')
        source=add('LoadImage',{'image':images[0]['relative_name']});encode=add('VAEEncode')
        connect(source,0,encode,'pixels');connect(*vae,encode,'vae');connect(encode,0,noise,'input_latent')
    decode=add('VAEDecode');save=add('SaveImage',{'filename_prefix':'ForgeNeo-Bridge'})
    connect(sampler,0,decode,'samples');connect(*vae,decode,'vae');connect(decode,0,save,'images')
    return {'schema_version':'1.0.0','name':'ForgeNeo Bridge '+cfg['family'],'config_hash':spec.config_hash,'nodes':nodes,'edges':edges,'qualification':'not_evaluated'}
