"""The hypnogram pre-choice of tools 3, 3bis, 5, 6 and 7 in their notebooks, on a study that also holds the
hypnograms the toolkit writes (user report of 2026-11-02: tool 3 took tool 3bis's _hypnogram_gssc.txt, tools 5-7
tool 0's rescored hypnogram). Tool 3 must take the raw export, the others the remapped hypnogram; then the names
recorded in config_param/file_names.json are followed.

Only the file names matter to the pre-choice (made when the study folder is picked), so the recordings are empty
.edf files and nothing is run."""
import sys
from pathlib import Path

import pytest

from nbdriver import NotebookSession

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
import study_files_lib as SF     # noqa: E402

IDS = ['s1', 's2', 's10']        # 's1' starts 's10': its files must not give a false suffix '0_...'
COMPANIONS = ['_Hypnogram_Export.txt', '_Hypnogram_remapped.txt', '_hypnogram_gssc.txt', '_hypno_yasa.txt',
              '_Hypnogram_remapped_rescored.txt', '_ScoredEvents_Export.txt', '_Summary_Export.txt']
# tool: notebook, study-folder chooser, hypnogram suffix field, the pre-choice expected
TOOLS = {
    '3': ('tools/3_remap_hypno_voila.ipynb', 'chooser', 'suffix_box', '_Hypnogram_Export.txt'),
    '3bis': ('tools/3bis_autoscoring_voila.ipynb', 'fc_data', 'hypno_suffix', '_Hypnogram_remapped.txt'),
    '5': ('tools/5_compute_sleep_macrostructure_voila.ipynb', 'fc_data', 'txt_hypno_suffix', '_Hypnogram_remapped.txt'),
    '6': ('tools/6_quality_overview_voila.ipynb', 'fc_folder', 'hypno_suffix', '_Hypnogram_remapped.txt'),
    '7': ('tools/7_preprocessing_voila.ipynb', 'fc_edf', 'hypno_suffix', '_Hypnogram_remapped.txt'),
}
PARAMS = [pytest.param(tool, marks=getattr(pytest.mark, f'tool{tool}')) for tool in TOOLS]


def make_study(folder, extra=()):
    for fid in IDS:
        (folder / f'{fid}.edf').write_bytes(b'')
        for suffix in COMPANIONS + list(extra):
            (folder / f'{fid}{suffix}').write_text('W\n', encoding='utf-8')
    return folder


def prechoice(tool, study, extra_values=()):
    """Pick the study folder; return the hypnogram suffix field and the extra widget values asked for."""
    notebook, chooser, field, _ = TOOLS[tool]
    with NotebookSession(notebook) as nb:
        nb.pick(chooser, study)
        return nb.value(f'{field}.value'), [nb.value(v) for v in extra_values]


@pytest.mark.parametrize('tool', PARAMS)
def test_the_toolkit_hypnograms_are_never_pre_chosen(tool, tmp_path):
    study = make_study(tmp_path)
    extra = {'3': ['output_suffix_box.value'],
             '3bis': ['out_suffix.value', 'out_suffix_yasa.value', 'out_suffix_sleepfm.value']}.get(tool, [])
    suffix, values = prechoice(tool, study, extra)
    assert suffix == TOOLS[tool][3]
    # a study keeps its names: the former defaults found beside the recordings, else the short ones
    if tool == '3':
        assert values == ['_Hypnogram_remapped.txt']
    if tool == '3bis':
        assert values == ['_hypnogram_gssc.txt', '_hypno_yasa.txt', '_hypno_sleepfm.txt']


@pytest.mark.parametrize('tool', PARAMS)
def test_the_names_recorded_by_tool_3_are_followed(tool, tmp_path):
    # shorter than the former default and as frequent as the export: only the registry can make them win
    study = make_study(tmp_path, extra=['_raw.txt', '_stages.txt'])
    assert SF.write_names(study, '3', {'hypnogram_scored': '_raw.txt', 'hypnogram_remapped': '_stages.txt'}) is None
    suffix, _ = prechoice(tool, study)
    assert suffix == ('_raw.txt' if tool == '3' else '_stages.txt')
