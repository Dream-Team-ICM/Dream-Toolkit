"""The rules of tools/study_files_lib.py: the names registry (config_param/file_names.json), the counting of the
companion suffixes and the pre-choice of the hypnogram suffix (tools 0, 3, 3bis, 5, 6, 7)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
import study_files_lib as SF     # noqa: E402

pytestmark = [pytest.mark.quick, pytest.mark.tool0, pytest.mark.tool3, pytest.mark.tool3bis,
              pytest.mark.tool5, pytest.mark.tool6, pytest.mark.tool7]


def _counts(ids, suffixes, folder='data'):
    recs = [Path(folder) / f'{i}.edf' for i in ids]
    files = [Path(folder) / f'{i}{s}' for i in ids for s in suffixes]
    return SF.companion_suffix_counts(recs, files)


def test_a_file_belongs_to_the_longest_recording_name():
    ids = ['1', '2', '10', '20']
    counts = _counts(ids, ['_Hypnogram_remapped.txt'])
    assert counts == {'_Hypnogram_remapped.txt': 4}           # no false '0_Hypnogram_remapped.txt'
    assert SF.pick_hypno_suffix(counts, 'expert')[:2] == ('_Hypnogram_remapped.txt', 4)
    # a hypnogram kept in another folder still counts (tools 5-7 find it anywhere in the study by its name)
    assert SF.companion_suffix_counts([Path('edf/1.edf')], [Path('hypno/1_hypno.txt')]) == {'_hypno.txt': 1}


def test_expert_prefers_the_most_specific_and_never_an_event_export():
    edfs = [Path(f'p{i}.edf') for i in range(4)]
    txt = ([Path(f'p{i}_Hypnogram_Export.txt') for i in range(4)]
           + [Path(f'p{i}_Hypnogram_remapped.txt') for i in range(3)]
           + [Path(f'p{i}_ScoredEvents_Export.txt') for i in range(4)])
    counts = SF.companion_suffix_counts(edfs, txt)
    assert SF.pick_hypno_suffix(counts, 'expert') == ('_Hypnogram_remapped.txt', 3, 'detected')  # >= 50 %, longest
    assert SF.pick_hypno_suffix({'_ScoredEvents_Export.txt': 4}, 'expert') == (None, 0, 'none')
    assert SF.pick_hypno_suffix({}, 'expert') == (None, 0, 'none')
    assert SF.select_event_suffix({'_event_xml.csv': 4, '_ev.csv': 4, '_x.csv': 1}) == ('_ev.csv', 4)
    assert SF.select_event_suffix({}) == (None, 0)


def test_the_toolkit_hypnograms_never_win_the_pre_choice():
    """User report of 2026-11-02: tool 3 took _hypnogram_gssc.txt (shorter than the export), tools 5-7 the
    rescored hypnogram of tool 0 (longer than the remapped one)."""
    sufs = ['_Hypnogram_Export.txt', '_Hypnogram_remapped.txt', '_hypnogram_gssc.txt', '_hypno_yasa.txt',
            '_Hypnogram_remapped_rescored.txt', '_Summary_Export.txt', '_StudyLog_Export.txt']
    counts = _counts(['s1', 's2', 's3'], sufs)
    assert SF.pick_hypno_suffix(counts, 'scored')[0] == '_Hypnogram_Export.txt'
    assert SF.pick_hypno_suffix(counts, 'expert')[0] == '_Hypnogram_remapped.txt'
    # only automatic hypnograms beside the recordings: nothing pre-chosen (the user picks one knowingly)
    only_auto = _counts(['s1', 's2'], ['_hypno_gssc.txt', '_hypnogram_sleepfm.txt'])
    assert SF.pick_hypno_suffix(only_auto, 'expert') == (None, 0, 'none')
    # tool 3bis sets its own (possibly renamed) outputs aside
    counts = _counts(['s1', 's2'], ['_hypno_remapped.txt', '_my_gssc_scoring_long.txt'])
    assert SF.pick_hypno_suffix(counts, 'expert', own_outputs=['_my_gssc_scoring_long.txt'])[0] == '_hypno_remapped.txt'


def test_a_name_recorded_in_the_registry_is_followed(tmp_path):
    counts = _counts(['s1', 's2'], ['_raw.txt', '_stages.txt', '_Hypnogram_remapped.txt', '_hypno_remapped.txt'])
    # without the registry: the longest wins (the former default, from an older run of tool 3)
    assert SF.pick_hypno_suffix(counts, 'expert')[0] == '_Hypnogram_remapped.txt'
    assert SF.write_names(tmp_path, '3', {'hypnogram_scored': '_raw.txt', 'hypnogram_remapped': '_stages.txt'}) is None
    names, warn = SF.load_names(tmp_path)
    assert warn is None and SF.recorded(names, 'hypnogram_remapped') == '_stages.txt'
    assert SF.pick_hypno_suffix(counts, 'expert', names) == ('_stages.txt', 2, 'recorded')
    assert SF.pick_hypno_suffix(counts, 'scored', names) == ('_raw.txt', 2, 'recorded')
    # a recorded name with no file left falls back on the detection, the recorded raw export set aside
    counts = _counts(['s1', 's2'], ['_raw.txt', '_Hypnogram_remapped.txt'])
    assert SF.pick_hypno_suffix(counts, 'expert', names) == ('_Hypnogram_remapped.txt', 2, 'detected')


def test_the_registry_keeps_other_keys_dates_and_an_unreadable_file(tmp_path):
    assert SF.load_names(tmp_path) == ({}, None)                             # absent: nothing recorded, no warning
    SF.write_names(tmp_path, '3bis', {'hypnogram_gssc': '_hypno_gssc.txt'})
    first = json.loads(SF.names_path(tmp_path).read_text(encoding='utf-8'))
    stamp = SF.names_path(tmp_path).stat().st_mtime_ns
    SF.write_names(tmp_path, '3bis', {'hypnogram_gssc': '_hypno_gssc.txt'})  # same value: file untouched
    assert SF.names_path(tmp_path).stat().st_mtime_ns == stamp
    SF.write_names(tmp_path, '3', {'hypnogram_remapped': '_hypno_remapped.txt'})
    names, _ = SF.load_names(tmp_path)
    assert names['hypnogram_gssc'] == first['hypnogram_gssc']                # another tool's key kept as is
    assert names['hypnogram_remapped']['tool'] == '3'
    SF.names_path(tmp_path).write_text('{not json', encoding='utf-8')
    names, warn = SF.load_names(tmp_path)
    assert names == {} and 'unreadable' in warn
    assert 'not updated' in SF.write_names(tmp_path, '3', {'hypnogram_remapped': '_x.txt'})
    assert SF.names_path(tmp_path).read_text(encoding='utf-8') == '{not json'  # never overwritten


def test_the_notes_under_the_suffix():
    assert SF.choice_notes('detected', 'expert') == []
    assert SF.choice_notes('recorded', 'expert')[0][0] == 'muted'
    assert 'raw hypnogram export' in SF.choice_notes('none', 'scored')[0][1]
    assert SF.choice_notes('detected', 'expert', 'unreadable') == [('warn', 'unreadable')]


def test_a_study_keeps_its_output_names():
    old = {'_Hypnogram_Export.txt': 3, '_Hypnogram_remapped.txt': 3}
    new = {'_Hypnogram_Export.txt': 3}
    args = (SF.HYPNO_REMAPPED_DEFAULT, SF.HYPNO_REMAPPED_FORMER)
    assert SF.output_suffix({}, 'hypnogram_remapped', old, *args) == '_Hypnogram_remapped.txt'
    assert SF.output_suffix({}, 'hypnogram_remapped', new, *args) == '_hypno_remapped.txt'
    names = {'hypnogram_remapped': {'value': '_mine.txt'}}
    assert SF.output_suffix(names, 'hypnogram_remapped', old, *args) == '_mine.txt'
