"""Package the completed repository, including trusted demo bundles, for download.

Run: python scripts/package_project.py
Output: ../AI-Energy-Forecasting.zip
Caches, environments, .git and unneeded raw archives are never included.
Git publication uses .gitignore instead; a ZIP is not a fabricated Git history.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {'.git','.cache','.venv','venv','__pycache__','.pytest_cache','.ruff_cache',
           '.ipynb_checkpoints','node_modules','.idea','.vscode'}
MANIFEST = ROOT / 'outputs/delivery_manifest.json'


def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    files=[]
    for p in ROOT.rglob('*'):
        if not p.is_file() or p==MANIFEST:
            continue
        rel=p.relative_to(ROOT)
        if any(part in EXCLUDE for part in rel.parts):
            continue
        if p.suffix in {'.pyc','.zip','.part','.pem','.key'} or p.name.startswith('.env'):
            continue
        files.append(p)
    files=sorted(files)
    entries=[{'file':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha256(p)} for p in files]
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),
              'project':'AI-Powered Energy Consumption Forecasting System',
              'publication_status':'Local deliverable; not published to a student GitHub account.',
              'scope':'Complete code, measured and synthetic experiments, trusted demo bundles and real local evidence.',
              'file_count_excluding_this_manifest':len(entries),'files':entries}
    MANIFEST.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    target=ROOT.parent/'AI-Energy-Forecasting.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in [*files,MANIFEST]:
            z.write(p,arcname=f'{ROOT.name}/{p.relative_to(ROOT).as_posix()}')
    with zipfile.ZipFile(target) as z:
        bad=z.testzip()
        if bad: raise ValueError(f'Archive CRC failure: {bad}')
    print(f'Packaged {len(files)+1} files → {target} ({target.stat().st_size:,} bytes)')
    print('ZIP SHA256:',sha256(target))
    return target


if __name__=='__main__':
    main()
