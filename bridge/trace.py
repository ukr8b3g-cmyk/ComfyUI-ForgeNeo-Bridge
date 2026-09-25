"""Opt-in run trace; no arbitrary output path is accepted from metadata."""
from __future__ import annotations
from pathlib import Path
from uuid import uuid4
from .spec import canonical


def save_trace(spec,tensors,report):
    import folder_paths
    from safetensors.torch import save_file
    root=Path(folder_paths.get_output_directory())/'forge_neo_bridge'/'traces'
    root.mkdir(parents=True,exist_ok=True)
    stem=spec.config_hash[:12]+'-'+uuid4().hex
    target=root/(stem+'.safetensors')
    save_file({k:v.detach().contiguous().cpu() for k,v in tensors.items()},str(target),metadata={'profile':spec.document()['profile']['id'],'config_hash':spec.config_hash})
    (root/(stem+'.json')).write_text(canonical(report),encoding='utf-8')
    return 'forge_neo_bridge/traces/'+target.name
