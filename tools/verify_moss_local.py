"""Read-only recipe/registered-node check. Does not queue generation or load a canvas."""
from pathlib import Path
import importlib.util
import json
import sys
import urllib.request

root = Path(__file__).resolve().parents[1]
core = Path(sys.argv[1]).resolve()
source = Path(sys.argv[2]).resolve()
output = Path(sys.argv[3]).resolve()
base = 'http://127.0.0.9:8188'
sys.path.insert(0, str(core))
from comfy.cli_args import args
args.cpu = True

package = importlib.util.spec_from_file_location('fnb_local', root/'__init__.py',
                                                submodule_search_locations=[str(root)])
module = importlib.util.module_from_spec(package)
sys.modules[package.name] = module
package.loader.exec_module(module)
from fnb_local.bridge.metadata import inspect_drop, image_dimensions
from fnb_local.bridge.reconstruction import reconstruct, reconstruction_plan


def get(path):
    with urllib.request.urlopen(base+path, timeout=15) as response:
        return json.load(response)


metadata = inspect_drop(source.read_bytes())
assert metadata['kind'] == 'forge'
inventory = get('/forge_neo_bridge/catalog')
doc, positive = reconstruct(metadata['metadata']['parameters'], inventory,
                            image_size=image_dimensions(source.read_bytes()))
plan = reconstruction_plan(doc, positive, inventory)
assert doc['extensions']['reconstruction']['family'] == 'moss'
assert doc['extensions']['reconstruction']['family_evidence'] == 'safetensors_header'
assert not doc['unsupported']
assert not any(x['code'] == 'ASSET_MISSING' for x in doc['unresolved'])
contracts = {n['type']: get('/object_info/'+n['type'])[n['type']] for n in plan['nodes']}
prompt = {n['id']:{'class_type':n['type'], 'inputs':dict(n['values'])} for n in plan['nodes']}
nodes = {n['id']:n for n in plan['nodes']}
for node in plan['nodes']:
    inputs = contracts[node['type']]['input']
    fields = {**inputs.get('required', {}), **inputs.get('optional', {})}
    for key, value in node['values'].items():
        assert key in fields, (node['type'], key)
        allowed = fields[key][0]
        if isinstance(allowed, list):assert value in allowed, (node['type'], key, value)
for source_id, slot, target_id, key in plan['edges']:
    contract = contracts[nodes[source_id]['type']]
    inputs = contracts[nodes[target_id]['type']]['input']
    field = {**inputs.get('required', {}), **inputs.get('optional', {})}[key]
    assert contract['output'][slot] == field[0], (source_id, slot, target_id, key)
    prompt[target_id]['inputs'][key] = [source_id, slot]
for node in plan['nodes']:
    assert set(contracts[node['type']]['input'].get('required', {})) <= set(prompt[node['id']]['inputs'])
output.mkdir(parents=True, exist_ok=True)
(output/'moss.plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
(output/'moss.prompt.json').write_text(json.dumps(prompt, ensure_ascii=False, indent=2), encoding='utf-8')
report = {'status':'PASS', 'scope':'real image metadata, bounded model header and live registered node contracts',
          'source':str(source), 'node_count':len(plan['nodes']), 'edge_count':len(plan['edges']),
          'family_evidence':doc['extensions']['reconstruction']['family_evidence'],
          'components':{key:doc['extensions']['reconstruction'][key] for key in ('model','clip','vae')},
          'notes':plan['notes'], 'gpu_executed':False}
(output/'recipe-check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
