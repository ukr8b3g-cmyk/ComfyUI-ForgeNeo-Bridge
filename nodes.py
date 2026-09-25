"""Four public nodes. Heavy libraries and Comfy runtime code are imported only on execution."""
from __future__ import annotations
import hashlib
from .bridge.spec import PROFILE, BridgeError, GenerationSpec, canonical, default_document, finalize, parse_json, report

CATEGORY = 'ForgeNeo Bridge'
HIDDEN = {'prompt':'PROMPT','unique_id':'UNIQUE_ID','dynprompt':'DYNPROMPT'}


def _spec(value):
    if not isinstance(value, GenerationSpec):raise BridgeError('INVALID_SPEC','Expected ForgeGenerationSpec from Forge Compat Import / Spec')
    return value


class ForgeCompatSpec:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'source_mode':(['manual','image','infotext','spec_json'],),
            'profile':([PROFILE['id']],), 'policy':(['strict','exploratory'],),
            'payload':('STRING',{'multiline':True,'default':canonical(default_document())}),
            'overrides_json':('STRING',{'multiline':True,'default':'{}'}),
        },'optional':{'image_file':('STRING',{'default':''})}}
    RETURN_TYPES=('FORGE_GENERATION_SPEC','STRING')
    RETURN_NAMES=('spec','report')
    FUNCTION='build';CATEGORY=CATEGORY
    DESCRIPTION='Forge Neo parameters and explicit resource bindings. Open the ForgeNeo Bridge editor from the menu to edit without JSON.'
    @classmethod
    def IS_CHANGED(cls,source_mode,profile,policy,payload,overrides_json,image_file=''):
        if source_mode=='image':
            from .bridge.binding import category_path
            p=category_path('input',image_file)
            from .bridge.metadata import MAX_IMAGE
            if p.stat().st_size>MAX_IMAGE:raise BridgeError('IMPORT_LIMIT_EXCEEDED','Image exceeds 128 MiB')
            return hashlib.sha256(p.read_bytes()).hexdigest()
        return hashlib.sha256((source_mode+profile+policy+payload+overrides_json).encode()).hexdigest()
    def build(self,source_mode,profile,policy,payload,overrides_json,image_file=''):
        if profile!=PROFILE['id']:raise BridgeError('INVALID_SPEC','Unknown profile')
        if source_mode in ('manual','spec_json'):
            doc=parse_json(payload)
        elif source_mode in ('image','infotext'):
            from .bridge.infotext import import_infotext
            from .bridge.metadata import read_metadata,MAX_IMAGE
            if source_mode=='image':
                from .bridge.binding import category_path
                p=category_path('input',image_file)
                if p.stat().st_size>MAX_IMAGE:raise BridgeError('IMPORT_LIMIT_EXCEEDED','Image exceeds 128 MiB')
                raw=p.read_bytes();data=read_metadata(raw)
                if data['kind']=='native':raise BridgeError('NATIVE_METADATA_PRIORITY','Open this image through the native Comfy workflow importer')
                payload=data['metadata'].get('parameters','')
            # Family is explicit in the editor; low-level importer cannot identify a model by its filename.
            doc=import_infotext(payload,policy=policy).document()
            if source_mode=='image':doc['source'].update(kind='image',filename=image_file,image_sha256=hashlib.sha256(raw).hexdigest())
        else:raise BridgeError('INVALID_SPEC','Unknown source mode')
        doc['policy']=policy
        result=finalize(doc,parse_json(overrides_json))
        return result,report('ForgeCompatSpec',result,status=result.document()['status'])


class ForgeCompatText:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'spec':('FORGE_GENERATION_SPEC',),'clip':('CLIP',)},'hidden':HIDDEN.copy()}
    RETURN_TYPES=('FORGE_CONDITIONING','STRING');RETURN_NAMES=('conditioning','report')
    FUNCTION='encode';CATEGORY=CATEGORY
    DESCRIPTION='Model-specific Forge encoding and denoiser-call prompt schedule. No global CLIP modifications.'
    def encode(self,spec,clip,prompt=None,unique_id=None,dynprompt=None):
        from .bridge.binding import GraphReader,verify_binding
        from .bridge.text import encode_bundle
        spec=_spec(spec);cfg=spec.require_executable()
        binding=verify_binding(cfg,GraphReader(prompt,dynprompt),unique_id,'clip','text_encoder')
        bundle=encode_bundle(spec,clip,binding)
        return bundle,report('ForgeCompatText',spec,actual=binding,trace=parse_json(bundle.trace_json),not_applied=['clip_skip'] if cfg['family']=='anima' else [])


class ForgeCompatNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'spec':('FORGE_GENERATION_SPEC',),'model':('MODEL',)},'optional':{'input_latent':('LATENT',)},'hidden':HIDDEN.copy()}
    RETURN_TYPES=('LATENT','FORGE_NOISE_BUNDLE','STRING');RETURN_NAMES=('latent','noise','report')
    FUNCTION='generate';CATEGORY=CATEGORY
    DESCRIPTION='Private CPU/GPU/NV noise with immutable replay state. Anima image T=1 only; video/NestedTensor is rejected on this node.'
    def generate(self,spec,model,input_latent=None,prompt=None,unique_id=None,dynprompt=None):
        from .bridge.binding import GraphReader,verify_binding
        from .bridge.noise import generate_noise
        spec=_spec(spec);cfg=spec.require_executable()
        binding=verify_binding(cfg,GraphReader(prompt,dynprompt),unique_id,'model','model')
        latent,noise=generate_noise(spec,model,binding,input_latent)
        return latent,noise,report('ForgeCompatNoise',spec,actual={'shape':noise.shape,'initial_noise_hash':noise.initial_hash,'input_latent_hash':noise.latent_hash,'binding':binding})


class ForgeCompatSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'spec':('FORGE_GENERATION_SPEC',),'model':('MODEL',),'conditioning':('FORGE_CONDITIONING',),
                'noise':('FORGE_NOISE_BUNDLE',),'latent_image':('LATENT',),'trace_level':(['none','summary','tensors'],{'default':'summary'})},'hidden':HIDDEN.copy()}
    RETURN_TYPES=('LATENT','SIGMAS','STRING');RETURN_NAMES=('latent','sigmas','report')
    FUNCTION='sample';CATEGORY=CATEGORY
    DESCRIPTION='Pinned Forge sampling/CFG/sigma semantics. Completion is not a GPU parity certificate. Never silently tiles/reduces resolution on OOM.'
    def sample(self,spec,model,conditioning,noise,latent_image,trace_level='summary',prompt=None,unique_id=None,dynprompt=None):
        from .bridge.binding import GraphReader,verify_binding
        from .bridge.noise import NoiseBundle
        from .bridge.text import ConditioningBundle
        from .bridge.runtime import execute
        spec=_spec(spec);cfg=spec.require_executable()
        if not isinstance(conditioning,ConditioningBundle) or not isinstance(noise,NoiseBundle):raise BridgeError('SPEC_MISMATCH','Invalid execution bundle type')
        reader=GraphReader(prompt,dynprompt)
        binding=verify_binding(cfg,reader,unique_id,'model','model')
        # Revalidate the actual encoder edge, not a user-supplied manifest flag.
        text_id=reader.input(unique_id,'conditioning')[0]
        if reader.get(text_id)['class_type']!='ForgeCompatText':raise BridgeError('UNVERIFIED_PATCH_CHAIN','Unexpected conditioning producer')
        text_binding=verify_binding(cfg,reader,text_id,'clip','text_encoder')
        if text_binding['hash']!=parse_json(conditioning.binding_json)['hash']:raise BridgeError('ASSET_BINDING_MISMATCH','Text assets changed after encoding')
        import latent_preview
        callback=latent_preview.prepare_callback(model,cfg['sampling']['steps'])
        result,sigmas,facts,tensors=execute(spec,model,conditioning,noise,latent_image,binding,trace_level,callback)
        if tensors:
            from .bridge.trace import save_trace
            facts['trace_file']=save_trace(spec,tensors,facts)
        return result,sigmas,report('ForgeCompatSampler',spec,**facts)


NODE_CLASS_MAPPINGS={c.__name__:c for c in (ForgeCompatSpec,ForgeCompatText,ForgeCompatNoise,ForgeCompatSampler)}
NODE_DISPLAY_NAME_MAPPINGS={'ForgeCompatSpec':'Forge Compat Import / Spec','ForgeCompatText':'Forge Compat Text',
                           'ForgeCompatNoise':'Forge Compat Noise','ForgeCompatSampler':'Forge Compat Sampler'}
