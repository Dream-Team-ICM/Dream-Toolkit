"""The rules of tools/study_files_lib.py: the names registry (config_param/file_names.json), the counting of the
companion suffixes and the pre-choice of the hypnogram suffix (tools 0, 3, 3bis, 5, 6, 7)."""
import json
import os
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


def test_an_input_is_found_where_it_was_recorded_else_at_its_default_location(tmp_path):
    study = tmp_path / 'study'
    (study / 'config_param').mkdir(parents=True)
    path, how, folder = SF.find_input(study, {}, 'channel_config')
    assert (path, how, folder) == (None, '', study / 'config_param')    # 'Choose another…' opens in config_param/
    assert SF.find_input(study, {}, 'quality_summary') == (None, '', study)
    former = study / 'config_param' / 'remap_reref_persubject.json'
    former.write_text('{}', encoding='utf-8')
    assert SF.find_input(study, {}, 'channel_config')[:2] == (former, 'default location')   # a study keeps its name
    new = study / 'config_param' / 'channel_config.json'
    new.write_text('{}', encoding='utf-8')
    assert SF.find_input(study, {}, 'channel_config')[0] == new
    mine = tmp_path / 'elsewhere' / 'my_channels.json'
    mine.parent.mkdir()
    mine.write_text('{}', encoding='utf-8')
    names = {'channel_config': {'value': str(mine), 'tool': '2'}}
    assert SF.find_input(study, names, 'channel_config') == (mine, 'recorded by tool 2', mine.parent)
    names = {'channel_config': {'value': 'config_param/gone.json', 'tool': '2'}}   # recorded, but no longer there
    assert SF.find_input(study, names, 'channel_config')[0] == new


def test_the_light_read_of_an_input_says_what_it_holds(tmp_path):
    def write(name, text):
        (tmp_path / name).write_text(text, encoding='utf-8')
        return tmp_path / name

    cfg = {'s1': {'remap': {'C3': 'C3'}, 'ref_channels': ['M2']}, 's2': {'remap': {}, 'ref_channels': []}}
    assert SF.check_channel_config(write('c.json', json.dumps(cfg))) == ('ok', '2 participants')
    assert SF.check_channel_config(write('c.json', '{}'))[0] == 'warn'
    kind, text = SF.check_channel_config(write('c.json', json.dumps({'Arousal': 'arousal'})))
    assert kind == 'warn' and "'remap'" in text
    assert SF.check_event_remap(write('e.json', '{"Arousal": "arousal", "Eveil": "arousal", "x": null,}')) == \
        ('ok', '3 labels → 1 event types')                                # one trailing comma repaired, like the tools
    assert SF.check_event_remap(write('e.json', json.dumps(cfg)))[0] == 'warn'
    assert SF.check_quality_summary(write('q.tsv', 'file_id\tchannel\ns1\tC3\ns1\tO1\ns2\tC3\n')) == ('ok', '2 recordings')
    assert SF.check_quality_summary(write('q.tsv', 'a\tb\n1\t2\n'))[0] == 'warn'
    assert SF.check_sleep_metrics_table(write('g.tsv', 'file_id\tTST\ns1\t400\ns2\t420\n')) == ('ok', '2 participants')
    assert SF.check_sleep_metrics_table(write('g.tsv', 'a\tb\n1\t2\n'))[0] == 'warn'


def test_a_study_keeps_the_name_of_its_configuration_files(tmp_path):
    study = tmp_path / 'study'
    (study / 'config_param').mkdir(parents=True)
    args = ('channel_config', SF.CHANNEL_CONFIG_DEFAULT, SF.CHANNEL_CONFIG_FORMER)
    assert SF.config_file_name(study, {}, *args) == 'channel_config.json'             # a new study
    former = study / 'config_param' / 'remap_reref_persubject.json'
    former.write_text('{}', encoding='utf-8')
    assert SF.config_file_name(study, {}, *args) == 'remap_reref_persubject.json'     # keeps the name it has
    (study / 'config_param' / 'channel_config.json').write_text('{}', encoding='utf-8')
    assert SF.config_file_name(study, {}, *args) == 'channel_config.json'             # the one find_input finds first
    names = {'channel_config': {'value': 'config_param/mine.json', 'tool': '2'}}
    assert SF.config_file_name(study, names, *args) == 'mine.json'
    names = {'channel_config': {'value': str(tmp_path / 'elsewhere.json'), 'tool': '2'}}   # not in config_param/
    assert SF.config_file_name(study, names, *args) == 'channel_config.json'


def _run(study, name, n=2):
    folder = study / 'derivatives' / name
    folder.mkdir(parents=True)
    for i in range(n):
        (folder / f's{i}_all-epo.fif').write_bytes(b'')
    return folder


def test_the_runs_of_tool_7_are_found_from_the_study_folder(tmp_path):
    study = tmp_path / 'study'
    study.mkdir()
    assert SF.epoch_folders(study) == []
    first = _run(study, 'preproc_epo')
    (study / 'derivatives' / 'preproc_epo_empty').mkdir()                 # no epochs: not a run
    (study / 'derivatives' / 'rejection_auto').mkdir()
    assert SF.epoch_folders(study) == [(first, 2)]
    second = _run(study, 'preproc_epo_v2', 3)
    for f in first.iterdir():
        os.utime(f, (1000, 1000))
    assert SF.epoch_folders(study) == [(second, 3), (first, 2)]           # the most recent first
    for f in second.iterdir():
        os.utime(f, (500, 500))
    assert SF.epoch_folders(study) == [(first, 2), (second, 3)]
    assert SF.epoch_folder_options(study, [(second, 3)]) == [
        ('derivatives/preproc_epo_v2  (3 participants)', str(second))]
    outside = _run(tmp_path / 'out', 'preproc_epo', 1)                    # tool 7's output folder, as it noted it
    names = {'preproc_output': {'value': str((tmp_path / 'out').resolve()), 'tool': '7'}}
    assert (outside, 1) in SF.epoch_folders(study, names)
    names = {'preproc_output': {'value': '.', 'tool': '7'}}               # noted, then back to the study folder
    assert SF.epoch_folders(study, names) == [(first, 2), (second, 3)]


def test_the_reports_of_a_run_are_paired_by_name(tmp_path):
    study = tmp_path / 'study'
    first, second = _run(study, 'preproc_epo'), _run(study, 'preproc_epo_v2')
    assert SF.paired_reports_folder(first) is None
    (study / 'reports_preproc').mkdir()
    assert SF.paired_reports_folder(first) == study / 'reports_preproc'
    assert SF.paired_reports_folder(second) is None                       # never another run's reports
    (study / 'reports_preproc_v2').mkdir()
    assert SF.paired_reports_folder(second) == study / 'reports_preproc_v2'
    assert SF.paired_reports_folder(study / 'derivatives' / 'other') is None


def test_the_suffix_of_a_run(tmp_path):
    assert SF.run_suffix(tmp_path / 'derivatives' / 'preproc_epo') == ''
    assert SF.run_suffix(str(tmp_path / 'derivatives' / 'preproc_epo_v2')) == '_v2'
    assert SF.run_suffix(tmp_path / 'derivatives' / 'my_epochs') == ''      # chosen by hand


def test_the_next_run_suffix_is_the_first_free_one(tmp_path):
    root = tmp_path / 'study'
    (root / 'derivatives' / 'preproc_epo').mkdir(parents=True)
    assert not SF.run_exists(root, '_v2') and SF.run_exists(root, '')
    assert SF.next_run_suffix(root) == '_v2'
    (root / 'derivatives' / 'preproc_epo_v2').mkdir()
    assert SF.run_exists(root, '_v2') and SF.next_run_suffix(root) == '_v3'
    (root / 'reports_preproc_v3').mkdir()                                 # a run may hold only its reports
    assert SF.next_run_suffix(root) == '_v4'
    assert SF.next_run_suffix(root, used={'_v4', '_nonotch'}) == '_v5'    # a run deleted by hand, still in the registry


def test_a_typed_suffix_must_fit_a_folder_name():
    assert SF.run_suffix_from_text('_v2') == '_v2'
    assert SF.run_suffix_from_text(' nonotch ') == '_nonotch'
    assert SF.run_suffix_from_text('-a_1') == '-a_1'
    for bad in ('', '   ', '_', 'a b', 'a/b', 'é', 'a.b', 'x' * 21, None):
        assert SF.run_suffix_from_text(bad) == '', bad


def test_a_run_lists_only_its_own_decisions(tmp_path):
    study = tmp_path / 'study'
    first, second = _run(study, 'preproc_epo'), _run(study, 'preproc_epo_v2')
    for name in ('rejection_auto', 'rejection_auto_v2', 'rejection_manual_v2', 'rejection_auto_v22'):
        folder = study / 'derivatives' / name
        folder.mkdir()
        (folder / 's0_epoch_decision.tsv').write_text('x', encoding='utf-8')
    assert [f.name for f, _, _ in SF.decision_folders(first)] == ['rejection_auto']
    assert sorted(f.name for f, _, _ in SF.decision_folders(second)) == ['rejection_auto_v2', 'rejection_manual_v2']


def test_the_most_recent_decision_comes_first(tmp_path):
    study = tmp_path / 'study'
    epochs = _run(study, 'preproc_epo')
    assert SF.decision_folders(epochs) == []
    assert SF.decision_folder_options(study, []) == [('none: no decision', SF.NO_DECISION)]
    for name, date in (('rejection_auto', 1000), ('rejection_manual', 2000)):
        folder = study / 'derivatives' / name
        folder.mkdir()
        table = folder / 's0_epoch_decision.tsv'
        table.write_text('x', encoding='utf-8')
        os.utime(table, (date, date))
    (study / 'derivatives' / 'rejection_empty').mkdir()
    found = SF.decision_folders(epochs)
    assert [f.name for f, _, _ in found] == ['rejection_manual', 'rejection_auto'] and found[0][2] == 1
    options = SF.decision_folder_options(study, found)
    assert 'tool 8,' in options[0][0] and 'tool 8bis,' in options[1][0] and options[-1][1] == SF.NO_DECISION


def test_the_light_read_of_a_folder_says_what_it_holds(tmp_path):
    study = tmp_path / 'study'
    epochs = _run(study, 'preproc_epo')
    assert SF.check_epoch_folder(epochs) == ('ok', '2 participants')
    assert SF.check_epoch_folder(study)[0] == 'ok'                        # recursive, like the tools
    (tmp_path / 'empty').mkdir()
    for check in (SF.check_epoch_folder, SF.check_reports_folder, SF.check_decision_folder):
        assert check(tmp_path / 'empty')[0] == 'warn'


def test_a_path_is_recorded_relative_to_the_study_when_inside_it(tmp_path):
    study = tmp_path / 'study'
    (study / 'config_param').mkdir(parents=True)
    assert SF.path_to_record(study, study / 'config_param' / 'x.json') == 'config_param/x.json'
    outside = tmp_path / 'out' / 'table.tsv'
    assert SF.path_to_record(study, outside) == str(outside.resolve())


SIDECAR = {'file_id': 'p', 'sfreq_hz': 256.0, 'epoch_length_s': 30, 'channels': ['C3', 'O1'], 'bad_channels': ['Fp1'],
           'custom_stages': [], 'methods_run': ['amplitude'], 'resample': {'applied': False, 'target_freq_hz': None},
           'filter': {'applied': True, 'l_freq_hz': 0.1, 'h_freq_hz': 50.0}, 'notch': {'applied': True, 'freq_hz': 50.0},
           'rejection_thresholds': {'flat_ptp_uV': 1.0}, 'event_rejection': {'applied': False, 'event_types': []},
           'psd_smoothing': {'enabled': True}, 'manual_events': False, 'reference': ['M2'], 'hypnogram_file': 'p_hypno.txt'}


def _fingerprint(tmp_path, **changes):
    path = tmp_path / 'p_preprocessing_params.json'
    path.write_text(json.dumps(dict(SIDECAR, **changes)), encoding='utf-8')
    return SF.preproc_fingerprint(path)


def test_the_fingerprint_follows_tool_7s_settings(tmp_path):
    same = _fingerprint(tmp_path)
    assert len(same) == 16
    # What only informs, or a key added to the sidecar later, does not shift it.
    assert _fingerprint(tmp_path, sfreq_hz=128.0, file_id='q', custom_stages=['IS'], a_later_key=1) == same
    for changes in ({'notch': {'applied': False, 'freq_hz': None}}, {'channels': ['C3']}, {'bad_channels': []},
                    {'rejection_thresholds': {'flat_ptp_uV': 2.0}}, {'reference': []}, {'hypnogram_file': 'p_other.txt'},
                    {'epoch_length_s': 10}, {'manual_events': True}):
        assert _fingerprint(tmp_path, **changes) != same, changes


def test_without_a_readable_sidecar_there_is_no_fingerprint(tmp_path):
    assert SF.preproc_fingerprint(tmp_path / 'absent.json') == ''
    assert SF.preproc_fingerprint(None) == ''
    broken = tmp_path / 'broken.json'
    broken.write_text('{not json', encoding='utf-8')
    assert SF.preproc_fingerprint(broken) == ''


def test_the_fingerprint_is_read_back_from_a_decision_record(tmp_path):
    record = tmp_path / 'p_autoreject_decision.tsv'
    record.write_text('file_id\tpreproc_fingerprint\tn\np\tabc\t1\n', encoding='utf-8')
    assert SF.recorded_preproc_fingerprint(record) == 'abc'
    record.write_text('file_id\tn\np\t1\n', encoding='utf-8')          # written before the column existed
    assert SF.recorded_preproc_fingerprint(record) == ''
    assert SF.recorded_preproc_fingerprint(tmp_path / 'absent.tsv') == ''


def test_a_record_is_out_of_date_only_when_both_sides_have_a_fingerprint():
    assert SF.preproc_changed('abc', 'def')
    assert not SF.preproc_changed('abc', 'abc')
    assert not SF.preproc_changed('', 'def')
    assert not SF.preproc_changed('abc', '')
