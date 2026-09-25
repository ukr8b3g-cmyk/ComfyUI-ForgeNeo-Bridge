"""Import the custom-node package without requiring a running Comfy server."""
import importlib.util
import pathlib
import sys
import pytest
ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fnb',ROOT/'__init__.py',submodule_search_locations=[str(ROOT)])
module=importlib.util.module_from_spec(spec);sys.modules['fnb']=module;spec.loader.exec_module(module)
# pytest sees the hyphenated repository as a root __init__ module. Reuse the
# package already imported exactly as ComfyUI does; do not alter production imports.
sys.modules.setdefault('__init__', module)

@pytest.fixture
def document():
    from fnb.bridge.spec import default_document
    doc=default_document('anima')
    doc['effective']['assets']=[{'asset_id':role,'role':role,'component':None,'category':cat,'relative_name':name,'sha256':None,'short_hash':None,'resolution':'resolved','binding_verification':'known_loader_chain'} for role,cat,name in [('model','diffusion_models','test.safetensors'),('text_encoder','text_encoders','qwen.safetensors'),('vae','vae','test_vae.safetensors')]]
    return doc
