"""Development-only packaging of MIT Lark 1.3.1 into a private namespace.
No runtime pip dependency and no alias/mutation of an installed `lark` package.
"""
import importlib.metadata
from pathlib import Path
import re
import shutil
import lark

assert lark.__version__ == '1.3.1'
root = Path(__file__).resolve().parents[1]
source = Path(lark.__file__).parent
out = root / 'vendor/lark'
for original in source.rglob('*'):
    rel = original.relative_to(source)
    if not original.is_file() or any(p in ('__pycache__', '__pyinstaller', 'tools') for p in rel.parts):
        continue
    if original.suffix not in ('.py', '.lark'):
        continue
    text = original.read_text(encoding='utf-8')
    # Convert the few absolute intra-package imports to relative equivalents.
    dots = '.' * len(rel.parts)
    text = re.sub(r'from lark\.', 'from ' + dots, text)
    text = re.sub(r'from lark import', 'from ' + dots + ' import', text)
    if rel.as_posix() == 'load_grammar.py':
        text = text.replace("FromPackageLoader('lark', IMPORT_PATHS)", 'FromPackageLoader(__package__, IMPORT_PATHS)')
    if rel.as_posix() == 'utils.py':
        text = text.replace('logging.getLogger("lark")', 'logging.getLogger(__package__)')
    target = out / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding='utf-8')
dist = importlib.metadata.distribution('lark')
license_path = next(x for x in dist.files if x.name == 'LICENSE')
shutil.copyfile(dist.locate_file(license_path), out / 'LICENSE')
(out / 'BRIDGE_CHANGES.md').write_text('Lark 1.3.1 (MIT), Erez Shinan and contributors.\nOnly intra-package imports, grammar package lookup, and logger namespace were changed. Tools/pyinstaller are omitted.\nUpstream: https://github.com/lark-parser/lark/tree/1.3.1\n')
