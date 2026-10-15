"""Collect the actual implementation and README into a copy-paste source book.

Run: python scripts/export_source.py
This uses only Python's standard library. Edit individual files, then regenerate.
"""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
DESCRIPTIONS={
    'main.py':'CLI orchestration and phase selection; main entry point.',
    'app.py':'Validated Flask JSON API, trusted bundle cache, downloads and dashboard.',
    'generate_data.py':'Deterministic virtual-building data and honest synthetic metadata.',
    'src/__init__.py':'Source package/version marker.',
    'src/config.py':'Visible seeds, horizons, lag/split/fill settings and scenario assumptions.',
    'src/utils.py':'Shared paths, finite metrics, strict JSON, hashes and real logs.',
    'src/data.py':'Official public archive adapter, schema validation, causal cleaning and masks.',
    'src/features.py':'Past-only vectorized training features and identical inference row.',
    'src/models.py':'Naive baselines, Ridge, random forest, scaled MLP and fit warnings.',
    'src/evaluation.py':'Ordered selection, recursive backtests, untouched test reports and final refit.',
    'src/forecast.py':'Recursive future prediction, trusted bundle loading and limited error bands.',
    'src/simulation.py':'Transparent energy-conserving/headroom-limited load-shifting heuristic.',
    'src/visualization.py':'Nine reproducible analytical plots from actual saved evidence.',
    'templates/dashboard.html':'Complete self-contained responsive five-view UI and offline snapshot logic.',
    'requirements.txt':'Pinned tested direct runtime dependencies.',
    'requirements-dev.txt':'Optional automated tests and notebook development environment.',
    'pyproject.toml':'Pytest discovery/warning rules and code-style settings.',
    '.gitignore':'Excludes private credentials, environments, caches and heavy derived artifacts.',
    '.github/workflows/ci.yml':'Ready-to-run CPU checks on Python 3.11/3.12/3.13 after publication.',
    'README.md':'The complete copy-paste GitHub landing page requested in section K.',
}


def main():
    files=[ROOT/'generate_data.py',ROOT/'main.py',ROOT/'app.py']
    files += sorted((ROOT/'src').glob('*.py'))
    files += [ROOT/'templates/dashboard.html']
    files += sorted((ROOT/'scripts').glob('*.py'))
    files += sorted((ROOT/'tests').glob('*.py'))
    files += [ROOT/p for p in ['requirements.txt','requirements-dev.txt','pyproject.toml','.gitignore','.github/workflows/ci.yml','README.md']]
    lines=['# Complete source code — copy-paste ready\n',
           'Generated from the actual project files. Each heading is its target filename, followed by its purpose and full content. For beginner explanations read [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md). Edit source files and regenerate this book, not the other way around. Dataset/model binaries are separate artifacts, not hidden code.\n',
           'The executed notebook is included separately; its code cells appear at the end without bulky embedded image/base64 outputs. All runtime logic is provided below.\n']
    fence=chr(96)*4
    languages={'.py':'python','.html':'html','.toml':'toml','.yml':'yaml','.md':'markdown'}
    for f in files:
        rel=f.relative_to(ROOT).as_posix()
        desc=DESCRIPTIONS.get(rel, 'Executable evidence helper; see its docstring.' if rel.startswith('scripts/') else 'Isolated automated checks; fixtures are not portfolio measurements.')
        language=languages.get(f.suffix,'text')
        lines += [f'## `{rel}`\n',desc+'\n',fence+language,f.read_text(encoding='utf-8').rstrip(),fence+'\n']
    nb_path=ROOT/'notebooks/01_energy_forecasting.ipynb'
    if nb_path.exists():
        nb=json.loads(nb_path.read_text(encoding='utf-8'))
        lines += ['## `notebooks/01_energy_forecasting.ipynb` — code cells\n','An executed reading/learning path using the same modular pipeline. Open the notebook for narrative, actual outputs and figures.\n']
        count=0
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                count+=1
                source=cell['source']
                if isinstance(source,list):source=''.join(source)
                lines += [f'### Code cell {count}\n',fence+'python',source.rstrip(),fence+'\n']
    target=ROOT/'docs/FULL_SOURCE.md'
    target.write_text('\n'.join(lines),encoding='utf-8')
    print(f'Wrote {target} from {len(files)} text files ({target.stat().st_size:,} bytes).')


if __name__=='__main__':
    main()
