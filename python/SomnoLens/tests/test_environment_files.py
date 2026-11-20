"""Quick level: the three environment files agree, and every package the tools import is declared in the right one.

environment.yml (light: every tool, YASA alone in tool 3bis) ⊂ environment-full.yml (+ GSSC and SleepFM) ⊂
environment-dev.yml (+ the test suite): conda has no include, so each file repeats the one below it, with the same pins.
The light file holds every package the tools import except the autoscoring ones, which only the full file holds (a light
installation stays without torch). It replaces the audit of the imports by hand (SPEC.md, Conda environment)."""
import ast
import json
import re
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.quick

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / 'tools'
LIGHT, FULL, DEV = (REPO / name for name in ('environment.yml', 'environment-full.yml', 'environment-dev.yml'))

# The tools' code: the Voila pages, the shared libraries and the batch scripts (not the untracked drafts of tools/).
TOOL_FILES = (sorted(TOOLS.glob('*_voila.ipynb')) + sorted(TOOLS.glob('*_lib.py'))
              + [TOOLS / name for name in ('1_inspect_edf_perdataset.py', '1_inspect_edf_perparticipant.py',
                                           '8_reject_manually_batch.py', 'generate_test_data.py')])
# Import name -> conda (or pip) package name, when they differ.
PACKAGE_OF = {'sklearn': 'scikit-learn', 'yaml': 'pyyaml', 'IPython': 'ipython',
              'mne': 'mne-base', 'matplotlib': 'matplotlib-base', 'tkinter': 'tk'}
# Imported only by tool 3bis's GSSC and SleepFM: in the full file, never in the light one (torch from pip).
AUTOSCORING = {'torch', 'gssc', 'h5py', 'einops', 'loguru', 'pyyaml', 'importlib_resources'}
# SleepFM's official code, downloaded by tool 3bis itself into its model folder (sleepfm_lib): not a package.
NOT_PACKAGES = {'models'}


def packages(path):
    """{package name: spec} of an environment file, the pip ones included ('yasa': 'yasa=0.8.*', 'gssc': 'gssc==0.0.9')."""
    found = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        m = re.match(r'^\s+-\s+([A-Za-z0-9_.\-]+)\s*([=<>!~][^\s#]*)?', line)
        if m and m.group(1) not in ('conda-forge', 'pip:'):
            found[m.group(1).lower()] = (m.group(1) + (m.group(2) or '')).lower()
    return found


def imported_modules(path):
    """The top-level modules a .py file or a notebook imports (absolute imports only)."""
    if path.suffix == '.ipynb':
        cells = json.loads(path.read_text(encoding='utf-8'))['cells']
        sources = [''.join(c['source']) if isinstance(c['source'], list) else c['source']
                   for c in cells if c['cell_type'] == 'code']
    else:
        sources = [path.read_text(encoding='utf-8')]
    modules = set()
    for source in sources:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                modules |= {a.name.split('.')[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split('.')[0])
    return modules


def third_party_imports():
    """{package name: [files importing it]} over the tools' code, the stdlib and the tools' own modules left out."""
    own = {p.stem for p in TOOLS.glob('*.py')}
    found = {}
    for path in TOOL_FILES:
        for module in imported_modules(path):
            if module in own or module in NOT_PACKAGES or (module in sys.stdlib_module_names and module != 'tkinter'):
                continue
            found.setdefault(PACKAGE_OF.get(module, module).lower(), []).append(path.name)
    return found


def test_each_file_holds_the_one_below_it_with_the_same_pins():
    light, full, dev = packages(LIGHT), packages(FULL), packages(DEV)
    assert len(light) > 20                             # guard: the parser read the file
    for name, below, above in (('environment-full.yml', light, full), ('environment-dev.yml', full, dev)):
        missing = sorted(set(below) - set(above))
        assert not missing, f'{name} misses {missing}'
        different = [f'{below[p]} / {above[p]}' for p in below if below[p] != above[p]]
        assert not different, f'{name} pins differently: {different}'


def test_every_package_a_tool_imports_is_declared():
    light, full = packages(LIGHT), packages(FULL)
    imports = third_party_imports()
    assert len(imports) > 10                           # guard: the tools' imports were found
    missing_light = {p: f for p, f in imports.items() if p not in AUTOSCORING and p not in light}
    assert not missing_light, f'imported by the tools, absent from environment.yml: {missing_light}'
    missing_full = {p: f for p, f in imports.items() if p in AUTOSCORING and p not in full}
    assert not missing_full, f'imported by tool 3bis, absent from environment-full.yml: {missing_full}'


def test_the_light_file_holds_no_autoscoring_package():
    assert not AUTOSCORING & set(packages(LIGHT)), 'environment.yml must stay without torch, GSSC and SleepFM'
    assert AUTOSCORING <= set(packages(FULL))


def test_every_pin_is_set():
    """Versions pinned in every file (SPEC.md, Conda environment): only pip itself floats."""
    for path in (LIGHT, FULL, DEV):
        floating = [spec for name, spec in packages(path).items() if name != 'pip' and not re.search(r'[=<>~]', spec)]
        assert not floating, f'{path.name}: not pinned {floating}'
