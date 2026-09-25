"""Namespaced, local-only import/editor endpoints. Registered without execution hooks."""
from __future__ import annotations
import asyncio
import copy
import hashlib
from .spec import BridgeError, PROFILE, SAMPLERS, SCHEDULERS, canonical, default_document, finalize, parse_json, schema
from .metadata import MAX_IMAGE, read_metadata
from .infotext import import_infotext
from .binding import category_path, file_identity
from .workflow import graph_plan


def catalog():
    import folder_paths
    return {category:folder_paths.get_filename_list(category) for category in ('checkpoints','diffusion_models','text_encoders','vae','loras')}


def review_document(doc, acknowledge=False):
    if type(acknowledge) is not bool:
        raise BridgeError('INVALID_SPEC', 'acknowledge must be a boolean')
    doc = finalize(doc).document()
    # An explicit UI action, never an automatic default assumption on drop.
    if acknowledge:
        for value in doc.get('provenance',{}).values():value['acknowledged']=True
    cfg=doc['effective']
    if cfg is None:raise BridgeError('NEEDS_REVIEW','Select a supported model family')
    for a in cfg['assets']:
        if a['category'] is None or a['relative_name'] is None:continue
        file_identity(a)
        a['resolution']='resolved'
        # This is a plan. Runtime verifies the actual PROMPT loader chain independently.
        a['binding_verification']='sha256_verified' if a['sha256'] else 'known_loader_chain'
    return finalize(doc)


def register_routes(routes):
    from aiohttp import web
    def guarded(fn):
        async def wrapper(request):
            try:return web.json_response(await fn(request),dumps=canonical)
            except BridgeError as exc:return web.json_response({'error':{'code':exc.code,'message':str(exc),'path':exc.path}},status=400)
            except (KeyError,TypeError,ValueError) as exc:return web.json_response({'error':{'code':'INVALID_SPEC','message':str(exc)}},status=400)
        return wrapper
    @routes.get('/forge_neo_bridge/schema')
    @guarded
    async def get_schema(request):return {'schema':schema(),'profile':PROFILE,'samplers':SAMPLERS,'schedulers':SCHEDULERS,'qualification':'not_evaluated'}
    @routes.get('/forge_neo_bridge/defaults/{family}')
    @guarded
    async def get_defaults(request):return default_document(request.match_info['family'])
    @routes.get('/forge_neo_bridge/catalog')
    @guarded
    async def get_catalog(request):return await asyncio.to_thread(catalog)
    @routes.post('/forge_neo_bridge/inspect')
    @guarded
    async def inspect_image(request):
        if request.content_length and request.content_length>MAX_IMAGE:raise BridgeError('IMPORT_LIMIT_EXCEEDED','Image exceeds 128 MiB')
        data=bytearray()
        async for chunk in request.content.iter_chunked(1024*1024):
            data.extend(chunk)
            if len(data)>MAX_IMAGE:raise BridgeError('IMPORT_LIMIT_EXCEEDED','Image exceeds 128 MiB')
        result=await asyncio.to_thread(read_metadata,bytes(data))
        if result['kind']!='forge':return {'kind':result['kind']}
        family=request.query.get('family','anima')
        spec=import_infotext(result['metadata']['parameters'],family=family)
        doc=spec.document();doc['source'].update(kind='image',filename=None,image_sha256=hashlib.sha256(data).hexdigest())
        return {'kind':'forge','spec':doc}
    @routes.post('/forge_neo_bridge/infotext')
    @guarded
    async def inspect_text(request):
        value=parse_json(await request.text())
        return import_infotext(value['text'],value.get('family','anima')).document()
    @routes.post('/forge_neo_bridge/validate')
    @guarded
    async def validate(request):
        value=parse_json(await request.text())
        spec=await asyncio.to_thread(review_document,value['spec'],value.get('acknowledge',False))
        return {'spec':spec.document()}
    @routes.post('/forge_neo_bridge/plan')
    @guarded
    async def plan(request):
        import nodes as core_nodes
        value=parse_json(await request.text())
        spec=await asyncio.to_thread(review_document,value['spec'],value.get('acknowledge',False))
        plan=graph_plan(spec)
        missing=[n['type'] for n in plan['nodes'] if n['type'] not in core_nodes.NODE_CLASS_MAPPINGS]
        if missing:raise BridgeError('NODE_MISSING','Missing registered classes: '+', '.join(sorted(set(missing))))
        return {'spec':spec.document(),'plan':plan}
