"""Immutable, JSON-only ForgeGenerationSpec. No Comfy imports or global RNG state."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROFILE = {"id": "forge-neo@710f1e25", "reference_repository": "Haoming02/sd-webui-forge-classic", "reference_commit": "710f1e25fcac84d880cbccf27d11b2e3276e589e"}
MAX_METADATA = 16 * 1024 * 1024
MAX_SEED = 2**64 - 1

def _finite(value):
    return type(value) is int or (type(value) is float and math.isfinite(value))
# Core-backed selections are usable without claiming Forge numerical parity.
COMFY_SHARED_SAMPLERS = ('ddim', 'unipc', 'euler_cfg_pp', 'euler_ancestral_cfg_pp', 'dpmpp_2m_cfg_pp')
SAMPLERS = ("euler", "euler_ancestral", "er_sde", "dpmpp_2m", "dpmpp_2m_sde", "dpm_2", "heun",
            "lcm", "lms", "dpmpp_sde", "dpmpp_3m_sde", "res_multistep") + COMFY_SHARED_SAMPLERS
SCHEDULERS = ("automatic", "simple", "beta", "karras", "exponential", "polyexponential", "normal", "uniform", "sgm_uniform", "linear_quadratic", "kl_optimal", "ddim", "align_your_steps", "turbo", "bong_tangent", "flow_match")


class BridgeError(ValueError):
    def __init__(self, code: str, message: str, path: str = ""):
        self.code, self.path = code, path
        super().__init__(f"{code}: {message}" + (f" ({path})" if path else ""))


def noise_seed_values(config):
    n=config['noise'];batch=config['image']['batch_size'];variation=n['subseed_strength']!=0
    for key in ('seed','subseed','ensd'):
        if not 0 <= int(n[key]) <= MAX_SEED:
            raise BridgeError('INVALID_SPEC','Seed must be within uint64','/noise/'+key)
    seeds=tuple(int(n['seed'])+(0 if variation else i) for i in range(batch))
    subseeds=tuple(int(n['subseed'])+i for i in range(batch))
    ensd=int(n['ensd']) if config['sampling'].get('adjustments',True) else 0
    for path,value in (('seed',seeds[-1]),('subseed',subseeds[-1] if variation else 0),('ensd',seeds[-1]+ensd)):
        if value > MAX_SEED:
            raise BridgeError('INVALID_SPEC',f'{path} plus active batch/ENSD offset exceeds uint64; reduce the seed or offset','/noise/'+path)
    return seeds,subseeds,ensd


def issue(code: str, path: str, message: str) -> dict:
    return {"code": code, "path": path, "message": message}


def safe_relative(value: str) -> str:
    if not isinstance(value, str) or not value or '\x00' in value or ':' in value:
        raise BridgeError("INVALID_PATH", "Expected an input/category-relative filename")
    path = value.replace('\\', '/')
    if PurePosixPath(path).is_absolute() or PureWindowsPath(value).is_absolute() or any(x in ('.', '..', '') for x in path.split('/')):
        raise BridgeError("INVALID_PATH", "Absolute, UNC, URL and parent paths are forbidden")
    if any(x in path for x in ('\r', '\n')):
        raise BridgeError("INVALID_PATH", "Control characters in filename")
    return path


def _duplicates(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise BridgeError("INVALID_SPEC", f"Duplicate JSON key: {key}")
        out[key] = value
    return out


def parse_json(text: str) -> Any:
    if len(text.encode('utf-8')) > MAX_METADATA:
        raise BridgeError("IMPORT_LIMIT_EXCEEDED", "JSON exceeds 16 MiB")
    try:
        return json.loads(text, object_pairs_hook=_duplicates, parse_constant=lambda x: (_ for _ in ()).throw(BridgeError("INVALID_SPEC", f"Non-finite {x}")))
    except (json.JSONDecodeError, RecursionError) as exc:
        raise BridgeError("INVALID_SPEC", "Invalid or excessively nested JSON") from exc


def canonical(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise BridgeError("INVALID_SPEC", "Only finite JSON values are allowed") from exc


def schema() -> dict:
    return json.loads((ROOT / 'schema/ForgeGenerationSpec.schema.json').read_text(encoding='utf-8'))


def _check(value: Any, rule: dict, path: str = "", depth: int = 0) -> None:
    """Validator for the bundled, closed JSON Schema subset; no runtime dependency install."""
    def fail(msg):
        raise BridgeError("INVALID_SPEC", msg, path)
    if depth > 64:
        fail("Nesting limit exceeded")
    if 'anyOf' in rule:
        for sub in rule['anyOf']:
            try:
                _check(value, sub, path, depth + 1)
                break
            except BridgeError:
                continue
        else:
            fail("No allowed type/shape matches")
    typ = rule.get('type')
    valid = {'object': isinstance(value, dict), 'array': isinstance(value, list), 'string': isinstance(value, str), 'boolean': type(value) is bool,
             'integer': type(value) is int, 'number': type(value) in (int, float) and _finite(value), 'null': value is None}
    if typ and not valid.get(typ, False):
        fail(f"Expected {typ}")
    if 'const' in rule and (value != rule['const'] or type(value) is not type(rule['const'])):
        fail("Wrong constant")
    if 'enum' in rule and value not in rule['enum']:
        fail(f"Value is not in {rule['enum']}")
    if type(value) in (int, float):
        if not _finite(value): fail("Non-finite number")
        for k, ok in [('minimum', lambda x: value >= x), ('maximum', lambda x: value <= x), ('exclusiveMinimum', lambda x: value > x), ('exclusiveMaximum', lambda x: value < x)]:
            if k in rule and not ok(rule[k]): fail(f"Outside {k}")
        if 'multipleOf' in rule and value % rule['multipleOf']: fail("Invalid dimension multiple")
    if isinstance(value, str):
        if 'pattern' in rule and not re.search(rule['pattern'], value): fail("Invalid string format")
        if len(value) > MAX_METADATA: fail("String too long")
    if isinstance(value, dict):
        if any(k not in value for k in rule.get('required', ())): fail("Missing required fields")
        props = rule.get('properties', {})
        for key, child in value.items():
            if not isinstance(key, str): fail("Keys must be strings")
            if 'propertyNames' in rule: _check(key, rule['propertyNames'], path, depth + 1)
            sub = props.get(key, rule.get('additionalProperties', {}))
            if sub is False: fail(f"Unknown field {key}")
            _check(child, sub if isinstance(sub, dict) else {}, path + '/' + key, depth + 1)
    if isinstance(value, list):
        if len(value) > rule.get('maxItems', 100000): fail("Too many array items")
        if len(value) < rule.get('minItems', 0): fail("Too few array items")
        for i, child in enumerate(value): _check(child, rule.get('items', {}), f"{path}/{i}", depth + 1)
    for sub in rule.get('allOf', ()):
        _check(value, sub, path, depth + 1)
    if 'if' in rule:
        try:
            _check(value, rule['if'], path, depth + 1)
        except BridgeError:
            _check(value, rule.get('else', {}), path, depth + 1)
        else:
            _check(value, rule.get('then', {}), path, depth + 1)


def _normalize(value: Any, rule: dict) -> Any:
    if 'anyOf' in rule:
        for sub in rule['anyOf']:
            try: _check(value, sub)
            except BridgeError: continue
            return _normalize(value, sub)
    if type(value) in (int, float) and rule.get('type') == 'number':

        try:
            return 0.0 if value == 0 else float(value)
        except OverflowError as exc:
            raise BridgeError('INVALID_SPEC', 'Number exceeds float64') from exc
    if isinstance(value, dict): return {k: _normalize(v, rule.get('properties', {}).get(k, {})) for k, v in value.items()}
    if isinstance(value, list): return [_normalize(v, rule.get('items', {})) for v in value]
    return value


def pointer_get(data: Any, path: str) -> Any:
    if not isinstance(path, str) or not path.startswith('/'):
        raise BridgeError("INVALID_SPEC", "Invalid JSON Pointer", str(path))
    try:
        for key in path[1:].split('/'):
            key = key.replace('~1', '/').replace('~0', '~')
            data = data[int(key)] if isinstance(data, list) else data[key]
        return data
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise BridgeError("INVALID_SPEC", "Unknown JSON Pointer", path) from exc


def pointer_set(data: dict, path: str, value: Any) -> None:
    pointer_get(data, path)  # Existing, typed canonical fields only.
    parent, _, key = path.rpartition('/')
    target = pointer_get(data, parent) if parent else data
    key = key.replace('~1', '/').replace('~0', '~')
    target[int(key) if isinstance(target, list) else key] = copy.deepcopy(value)


def provenance(origin='manual_input', acknowledged=True, key=None, evidence=None) -> dict:
    return {'origin': origin, 'certainty': 'inferred' if origin == 'profile_default' else 'explicit', 'acknowledged': acknowledged, 'raw_key': key, 'evidence': evidence}


def default_document(family='anima', policy='strict') -> dict:
    if family not in ('anima', 'sd15', 'sdxl'): raise BridgeError('UNSUPPORTED_COMBINATION', 'Unsupported family')
    doc = parse_json((ROOT / f'examples/spec_{family}.json').read_text(encoding='utf-8'))
    doc['policy'], doc['unresolved'], doc['warnings'], doc['config_hash'] = policy, [], [], None
    return doc


def finalize(document: dict, overrides: dict | None = None) -> 'GenerationSpec':
    doc = copy.deepcopy(document)
    doc['status'] = 'needs_review'  # Never trust a supplied ready flag.
    input_schema = schema()
    input_schema['properties']['effective']['anyOf'][0].pop('allOf', None)
    _check(doc, input_schema)
    if doc['profile'] != PROFILE: raise BridgeError('INVALID_SPEC', 'Unknown compatibility profile')
    cfg = doc['effective']
    if cfg is None:
        return GenerationSpec(canonical(doc))
    if not isinstance(overrides or {}, dict): raise BridgeError('INVALID_SPEC', 'Overrides must be a JSON Pointer map')
    for path, value in (overrides or {}).items():
        pointer_set(cfg, path, value)
        doc['requested'][path] = value
        doc['provenance'][path] = provenance('user_override')
    # Report the request without incorrectly applying CLIP skip to Anima.
    request_skip = cfg['text']['clip_skip']
    if cfg['family'] == 'anima':
        if request_skip is not None: doc['requested']['/text/clip_skip'] = request_skip
        cfg['text']['clip_skip'], cfg['text']['layer_policy'] = None, 'anima_last'
    elif cfg['family'] == 'sdxl' and type(request_skip) is int:
        cfg['text']['clip_skip'] = max(2, request_skip)
        cfg['text']['layer_policy'] = 'sdxl_forge'
    _check(doc, schema())
    cfg = _normalize(cfg, schema()['properties']['effective']['anyOf'][0])
    doc['effective'] = cfg
    for group in ('requested', 'provenance'):
        for path in doc[group]: pointer_get(cfg, path)
    n, s = cfg['noise'], cfg['sampling']
    noise_seed_values(cfg)
    for k in ('positive_raw', 'negative_raw'):
        if len(cfg['text'][k]) > 100000: raise BridgeError('IMPORT_LIMIT_EXCEEDED', 'Prompt exceeds 100,000 characters')
    ids = [a['asset_id'] for a in cfg['assets']]
    if len(ids) != len(set(ids)): raise BridgeError('INVALID_SPEC', 'Duplicate asset IDs')
    for a in cfg['assets']:
        if a['relative_name'] is not None: safe_relative(a['relative_name'])
    for lora in cfg['loras']:
        if lora['asset_id'] not in ids: raise BridgeError('INVALID_SPEC', 'LoRA references absent asset')
    if [x['order'] for x in cfg['loras']] != list(range(len(cfg['loras']))):
        raise BridgeError('INVALID_SPEC', 'LoRA order must be contiguous and explicit')
    unresolved = [x for x in doc['unresolved'] if x['code'] not in ('ASSET_BINDING_REQUIRED', 'ASSET_MISSING', 'INFERRED_SETTING', 'UNVERIFIED_PATCH_CHAIN')]
    generated_paths = {'/mode', '/prediction_type', '/sampling', '/noise/seed_resize_from', '/sampling/denoise'}
    unsupported = [x for x in doc['unsupported'] if not (x['code'] == 'UNSUPPORTED_COMBINATION' and x['path'] in generated_paths)]
    roles = {a['role'] for a in cfg['assets']}
    required_roles = {'model', 'text_encoder'} if doc.get('execution_scope') == 'sampling' else {'model', 'text_encoder', 'vae'}
    if not required_roles <= roles:
        unresolved.append(issue('ASSET_BINDING_REQUIRED', '/assets', 'Bind model, text encoder and VAE through known loaders.'))
    for a in cfg['assets']:
        if a['resolution'] not in ('connected', 'resolved') or a['binding_verification'] == 'unverified':
            unresolved.append(issue('ASSET_MISSING', '/assets', f"Review asset {a['asset_id']}"))
    for path, p in doc['provenance'].items():
        if p['certainty'] == 'inferred' and not p['acknowledged']:
            unresolved.append(issue('INFERRED_SETTING', path, 'Confirm the inferred setting; it is not an observed generation value.'))
    if cfg['mode'] not in ('txt2img', 'img2img'):
        unsupported.append(issue('UNSUPPORTED_COMBINATION', '/mode', 'V1 supports txt2img and unmasked img2img only.'))
    if (cfg['family'] == 'anima' and cfg['prediction_type'] != 'flow') or (cfg['family'] != 'anima' and cfg['prediction_type'] != 'epsilon'):
        unsupported.append(issue('UNSUPPORTED_COMBINATION', '/prediction_type', 'Derivative predictor needs its own qualification.'))
    if s['sampler'] not in SAMPLERS or s['scheduler'] not in SCHEDULERS:
        unsupported.append(issue('UNSUPPORTED_COMBINATION', '/sampling', 'Registered inventory item has no V1 executable adapter.'))
    if s['noise_schedule'] != 'Default' or any(s['flow_match_options'].values()):
        unsupported.append(issue('UNSUPPORTED_COMBINATION', '/sampling', 'Non-default predictor/FlowMatch options are not qualified.'))
    if cfg['family'] == 'anima' and any(n['seed_resize_from'].values()):
        unsupported.append(issue('UNSUPPORTED_COMBINATION', '/noise/seed_resize_from', 'Anima seed resize is outside V1.'))
    if s.get('adjustments',True) and cfg['mode'] == 'img2img' and s['img2img_step_mode'] == 'exact_steps' and s['denoise'] == 0:
        unsupported.append(issue('UNSUPPORTED_COMBINATION', '/sampling/denoise', 'Exact-steps/zero denoise has no valid reference schedule.'))
    if s['sigma_min'] is not None and s['sigma_max'] is not None and s['sigma_min'] >= s['sigma_max']:
        raise BridgeError('INVALID_SPEC', 'sigma_min must be less than sigma_max')
    for flag, dst in [('unresolved', unresolved), ('unsupported', unsupported)]:
        doc[flag] = list({canonical(x): x for x in dst}.values())
    doc['status'] = 'unsupported' if doc['unsupported'] else 'needs_review' if doc['unresolved'] else 'ready'
    hash_input = {'profile': doc['profile'], 'effective': cfg}
    if doc.get('execution_scope') == 'sampling':hash_input['execution_scope'] = 'sampling'
    doc['config_hash'] = hashlib.sha256(canonical(hash_input).encode('utf-8')).hexdigest()
    _check(doc, schema())
    return GenerationSpec(canonical(doc))


@dataclass(frozen=True, slots=True)
class GenerationSpec:
    _json: str

    def document(self) -> dict:
        return parse_json(self._json)

    @property
    def config(self) -> dict:
        return self.document()['effective']

    @property
    def config_hash(self) -> str:
        return self.document()['config_hash']

    def require_executable(self) -> dict:
        doc = self.document()
        actual = finalize(doc)
        if actual.config_hash != doc['config_hash']: raise BridgeError('SPEC_MISMATCH', 'Spec integrity mismatch')
        checked = actual.document()
        if checked['unsupported']: raise BridgeError('UNSUPPORTED_COMBINATION', checked['unsupported'][0]['message'])
        if checked['policy'] == 'strict' and checked['unresolved']: raise BridgeError('NEEDS_REVIEW', checked['unresolved'][0]['message'])
        return actual.config


def report(node: str, spec: GenerationSpec | None = None, status='completed', **extra) -> str:
    doc = spec.document() if spec else {}
    return canonical({'schema_version': '1.0.0', 'node': node, 'config_hash': doc.get('config_hash'), 'status': status,
                      'qualification': 'not_evaluated', 'requested': doc.get('requested', {}), 'applied': {}, 'not_applied': [],
                      'inferred': [k for k, v in doc.get('provenance', {}).items() if v['certainty'] == 'inferred'],
                      'unresolved': doc.get('unresolved', []), 'unsupported': doc.get('unsupported', []), **extra})
