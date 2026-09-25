"""Development-only, reproducible extraction from downloaded, pinned reference trees.
Never called at ComfyUI import/startup. Inputs are source files, not model weights.
"""
import ast
import hashlib
import json
import shutil
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
upstream = Path(sys.argv[1])
out = root / 'vendor/forge'
out.mkdir(parents=True, exist_ok=True)
(out/'__init__.py').write_text('"""Attributed, pinned reference arithmetic. See THIRD_PARTY_NOTICES.md."""\n')
(root/'vendor/__init__.py').write_text('')
manifest = {}

def record(dest, source):
    manifest[dest] = {'source': source, 'sha256': hashlib.sha256((upstream/source).read_bytes()).hexdigest()}

# Use the actual fixed Forge code as oracle; the executable subset keeps its arithmetic.
path='forge/modules_forge/packages/k_diffusion/sampling.py'
src=(upstream/path).read_text();tree=ast.parse(src)
names={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
wanted={'sample_euler','sample_euler_ancestral','sample_euler_ancestral_RF','sample_er_sde','sample_dpmpp_2m','sample_dpmpp_2m_sde','sample_dpm_2','sample_heun',
        'append_zero','get_sigmas_karras','get_sigmas_exponential','get_sigmas_polyexponential','to_d','get_ancestral_step','default_noise_sampler','sigma_to_half_log_snr','half_log_snr_to_sigma','offset_first_sigma_for_snr','_is_const'}
# Walk dependencies, retaining only pure sampler functions (Brownian construction is injected).
for _ in range(8):
    for name in list(wanted):
        for n in ast.walk(names[name]):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in names and n.func.id!='BrownianTreeNoiseSampler':wanted.add(n.func.id)
header='''# SPDX-License-Identifier: AGPL-3.0-only
# Extracted from Forge Neo 710f1e25, modules_forge/packages/k_diffusion/sampling.py.
# Originally derived from ComfyUI / k-diffusion (Katherine Crowson et al.).
# Arithmetic retained. Bridge changes: explicit per-run RNG, required Brownian injection,
# no global monkey patches, no eager torchsde import. See THIRD_PARTY_NOTICES.md.
import math
from functools import partial
import torch
from tqdm.auto import trange

def BrownianTreeNoiseSampler(*args, **kwargs):
    raise ValueError("Bridge requires an explicit private Brownian noise sampler")

def _noise_like(x, extra_args):
    if extra_args is None or "_bridge_rng" not in extra_args:
        raise ValueError("Bridge requires an explicit private RNG")
    return extra_args["_bridge_rng"].next().to(device=x.device, dtype=x.dtype)

'''
chunks=[]
for n in tree.body:
    if getattr(n,'name',None) not in wanted:continue
    lines=src.splitlines();start=min([n.lineno]+[d.lineno for d in getattr(n,'decorator_list',[])])
    text='\n'.join(lines[start-1:n.end_lineno])
    if n.name=='default_noise_sampler':text='def default_noise_sampler(x, extra_args):\n    return lambda sigma, sigma_next: _noise_like(x, extra_args)'
    elif n.name=='to_d':text='def to_d(x, sigma, denoised):\n    # Forge sd_schedulers replaces to_d with this scalar-sigma formula.\n    return (x - denoised) / sigma'
    else:
        text=text.replace('torch.randn_like(x)', '_noise_like(x, extra_args)').replace('default_noise_sampler(x)', 'default_noise_sampler(x, extra_args)')
    chunks.append(text)
(out/'sampling.py').write_text(header+'\n\n'.join(chunks)+'\n',encoding='utf-8');record('sampling.py',path)
# Only pure prompt-planning functions; full model conditioning stays in Bridge.
path='forge/modules/prompt_parser.py';src=(upstream/path).read_text();tree=ast.parse(src)
keep={'get_learned_conditioning_prompt_schedules','SdConditioning','get_multicond_prompt_list','schedule_parser','re_AND','re_weight'}
parts=[]
for n in tree.body:
    ns={getattr(n,'name','')}
    if isinstance(n,ast.Assign):ns|={t.id for t in n.targets if isinstance(t,ast.Name)}
    if ns & keep: parts.append('\n'.join(src.splitlines()[n.lineno-1:n.end_lineno]))
(out/'prompt_parser.py').write_text('# SPDX-License-Identifier: AGPL-3.0-only\n# Forge Neo 710f1e25: pure planning subset (unchanged grammar/step semantics).\nfrom __future__ import annotations\nimport re\nfrom .. import lark\n\n'+'\n\n'.join(parts)+'\n');record('prompt_parser.py',path)
for dest,path in [('parsing.py','forge/backend/text_processing/parsing.py'),('rng_philox.py','forge/modules/rng_philox.py')]:
    (out/dest).write_text('# SPDX-License-Identifier: AGPL-3.0-only\n# Vendored from Forge Neo 710f1e25; original code below.\n'+(upstream/path).read_text());record(dest,path)
shutil.copyfile(upstream/'forge/LICENSE',root/'LICENSE')
(root/'vendor/source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
