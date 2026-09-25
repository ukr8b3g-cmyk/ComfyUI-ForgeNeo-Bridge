"""Developer/CI-only source download. Never imported by the custom nodes."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import io
import json
import urllib.request
import zipfile

REFERENCES = {
    'forge': ('Haoming02/sd-webui-forge-classic', '710f1e25fcac84d880cbccf27d11b2e3276e589e'),
    'comfy': ('Comfy-Org/ComfyUI', '88ab4a06566454ad89db8f0bedb970d6c08cd1b7'),
}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    parser.add_argument('--with-comfy',action='store_true')
    args=parser.parse_args();manifest={}
    for name,(repo,sha) in REFERENCES.items():
        if name=='comfy' and not args.with_comfy:continue
        request=urllib.request.Request(f'https://codeload.github.com/{repo}/zip/{sha}',headers={'User-Agent':'ForgeNeo-Bridge-CPU-audit'})
        with urllib.request.urlopen(request,timeout=180) as response:data=response.read()
        manifest[name]={'repository':repo,'commit':sha,'archive_sha256':hashlib.sha256(data).hexdigest()}
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for info in archive.infolist():
                parts=PurePosixPath(info.filename).parts[1:]
                if not parts or info.is_dir():continue
                if '..' in parts or any('\\' in p or ':' in p for p in parts):raise ValueError('Unsafe archive member')
                path=args.destination/name/Path(*parts)
                if path.suffix.lower() not in ('.py','.json','.txt','.md','.toml','.model') and path.name not in ('LICENSE','COPYING'):continue
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(archive.read(info))
    args.destination.mkdir(parents=True,exist_ok=True)
    (args.destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
