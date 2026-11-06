"""'Skip already processed' ticked: a participant already processed with other settings is not skipped silently. The run
stops before any analysis and asks (reprocess them, or keep them). Each scenario opens a NEW session on a study already
processed, the real case: the participant selector then knows who is done before the run starts.

Each test works on a light private copy of the chain's outputs, so the golden comparison is unaffected.
"""
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

import minidb
import pipeline
import snapshot
from nbdriver import NotebookSession
from pipeline import select_subset

QUESTION = 'were made with other settings'


@pytest.fixture
def src(chain):
    if chain['dataset'] != 'synthetic':
        pytest.skip('these scenarios run on the synthetic database only')
    return chain['data']


@pytest.fixture
def ids(src):
    return minidb.PARTICIPANTS['synthetic']


def tsv(path):
    return pd.read_csv(path, sep='\t', dtype={'file_id': str})


def mtimes(folder, *patterns):
    return {f.name: f.stat().st_mtime_ns for pattern in patterns for f in Path(folder).rglob(pattern)}


def light_copy(src, dst, dirs=()):
    """The chain's inputs (recordings, hypnograms, events, config_param, participants.tsv) + some output folders: the
    derivatives (~330 MB) are never copied whole."""
    dst.mkdir(parents=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    for d in ('config_param',) + tuple(dirs):
        shutil.copytree(src / d, dst / d)
    return dst


# =====================================================================================  tool 6
T6_OUTPUTS = ('*_quality_metrics.tsv', '*_quality_by_stage.tsv', '*_quality_overview.html')


@pytest.fixture
def t6_data(src, tmp_path):
    return light_copy(src, tmp_path / 'data', ['reports_quality_overview'])


def open_tool6(nb, data):
    nb.pick('fc_folder', data)
    nb.pick('fc_config', Path(data) / 'config_param' / 'remap_reref_persubject.json')


@pytest.mark.tool6
def test_tool6_same_settings_are_skipped_without_a_question(t6_data):
    reports = t6_data / 'reports_quality_overview'
    before = mtimes(reports, *T6_OUTPUTS)
    log = pipeline.run_tool6(t6_data)
    assert QUESTION not in log, log[-2000:]
    assert 'Participants processed     : 0' in log
    assert mtimes(reports, *T6_OUTPUTS) == before


@pytest.mark.tool6
def test_tool6_other_preparation_asks_before_anything_runs(t6_data, ids):
    reports = t6_data / 'reports_quality_overview'
    before = mtimes(reports, *T6_OUTPUTS)
    with NotebookSession(pipeline.T6) as nb:
        open_tool6(nb, t6_data)
        # a resampling asked after tables made without it: the header says the files are above the target
        nb.set('cb_resample', True)
        nb.set('txt_target_freq', 128)
        nb.click('btn_run')
        log = nb.widget_text()
        assert f'{len(ids)} of the {len(ids)} processed participant(s) {QUESTION}' in log, log[-2000:]
        assert 'signal preparation : none → resampled to 128 Hz' in log
        assert 'Analysis complete' not in log
        assert 'just below' in nb.value('participant_selector.run_note.value')
        nb.set('cb_resample', False)
        nb.set('hp_check', True)
        nb.set('hp_freq', 0.5)
        nb.click('btn_run')
        log = nb.widget_text()
        assert 'signal preparation : none → high-pass 0.5 Hz' in log
        assert nb.value('participant_selector.btn_keep_other.description') == 'Keep them, only rebuild the tables'
        assert mtimes(reports, *T6_OUTPUTS) == before                  # nothing ran while waiting for the answer
        nb.click('participant_selector.btn_keep_other')                # keep: the tables are only rebuilt
        log = nb.widget_text()
        assert nb.value('participant_selector.run_note.value') == ''
    assert 'Analysis complete' in log and 'Participants processed     : 0' in log, log[-2000:]
    assert 'Kept as they are, with their recorded settings' in log     # all kept, nobody new: nothing is mixed
    assert mtimes(reports, *T6_OUTPUTS) == before
    assert (tsv(reports / 'quality_summary.tsv')['signal_preparation'] == 'none').all()


@pytest.mark.tool6
def test_tool6_the_page_around_the_question_tells_where_the_run_stands(t6_data):
    with NotebookSession(pipeline.T6) as nb:
        open_tool6(nb, t6_data)
        nb.click('btn_run')
        assert nb.value('progress_label.value') == 'Done.'
        nb.set('hp_check', True)
        nb.click('btn_run')                               # the question: nothing says the previous run is done
        assert QUESTION in nb.widget_text()
        assert nb.value('progress_label.value') == ''
        assert nb.value('progress_ch.value') == 0
        nb.set('hp_check', False)
        nb.click('btn_run')                               # no difference any more: the note goes with the question
        assert QUESTION not in nb.widget_text()
        assert nb.value('participant_selector.run_note.value') == ''


@pytest.mark.tool6
def test_tool6_reprocess_makes_again_the_participants_concerned(t6_data, ids):
    reports = t6_data / 'reports_quality_overview'
    target, others = ids[3], ids[:3]
    with NotebookSession(pipeline.T6) as nb:
        open_tool6(nb, t6_data)
        select_subset(nb, [target], skip=True)
        nb.set('hp_check', True)
        nb.set('hp_freq', 0.5)
        nb.click('btn_run')
        assert f'1 of the 1 processed participant(s) {QUESTION}' in nb.widget_text()
        nb.click('participant_selector.btn_redo_other')
        log = nb.widget_text()
    assert f'Made again (other settings): {target} (signal preparation)' in log, log[-2000:]
    q = tsv(reports / 'quality_summary.tsv')
    assert (q.loc[q['file_id'] == target, 'signal_preparation'] == 'high-pass 0.5 Hz').all()
    assert (q.loc[q['file_id'].isin(others), 'signal_preparation'] == 'none').all()

    # Next session with the same high-pass: only the three others are still different.
    with NotebookSession(pipeline.T6) as nb:
        open_tool6(nb, t6_data)
        nb.set('hp_check', True)
        nb.set('hp_freq', 0.5)
        nb.click('btn_run')
        log = nb.widget_text()
        assert f'3 of the 4 processed participant(s) {QUESTION}' in log, log[-2000:]
        assert nb.value('participant_selector.btn_redo_other.description') == 'Reprocess 3 participant(s)'


@pytest.mark.tool6
def test_tool6_a_threshold_counts_only_through_the_flags(t6_data, ids):
    """Fp1 of the clipped recording has flat_pct 4.9 %, the only value between 3.5 and 10: the dead channel (99 %) stays
    flagged whatever the threshold, the others stay below."""
    clipped = ids[0]
    reports = t6_data / 'reports_quality_overview'
    before = mtimes(reports, *T6_OUTPUTS)
    with NotebookSession(pipeline.T6) as nb:              # a threshold moved, no flag moved: nothing to ask
        open_tool6(nb, t6_data)
        nb.set('thresh_flat', 4.0)
        nb.click('btn_run')
        log = nb.widget_text()
    assert QUESTION not in log, log[-2000:]
    with NotebookSession(pipeline.T6) as nb:              # a flag moved on one participant: only this one
        open_tool6(nb, t6_data)
        nb.set('thresh_flat', 10.0)
        nb.click('btn_run')
        log = nb.widget_text()
    assert f'1 of the {len(ids)} processed participant(s) {QUESTION}' in log, log[-2000:]
    assert 'flag threshold flat_pct : 3.5 → 10' in log            # the former threshold, recorded
    assert 'flags of Fp1 : bounds_pct, flat_pct, hist_extreme → bounds_pct, hist_extreme' in log
    assert 'Keeping them mixes settings' in log                   # the three others have this run's settings
    assert f'(1 participant(s): {clipped})' in log
    assert mtimes(reports, *T6_OUTPUTS) == before


@pytest.mark.tool6
def test_tool6_tables_written_before_the_record_are_not_reported_as_different(t6_data, ids):
    reports = t6_data / 'reports_quality_overview'
    for path in reports.rglob('*_quality_metrics.tsv'):
        tsv(path).drop(columns=['signal_preparation', 'flag_thresholds']).to_csv(path, sep='\t', index=False)
    with NotebookSession(pipeline.T6) as nb:
        open_tool6(nb, t6_data)
        nb.set('hp_check', True)
        nb.set('hp_freq', 0.5)
        nb.click('btn_run')
        log = nb.widget_text()
    assert QUESTION not in log, log[-2000:]
    assert f'{len(ids)} participant(s) written before the signal preparation was recorded' in log


# =====================================================================================  tool 7
T7_OUTPUTS = ('*_preprocessing_report.html', '*_epoch_channel_rejection.tsv', '*_rejection_summary.tsv',
              '*_preprocessing_params.json')


@pytest.fixture
def t7_data(src, tmp_path):
    """Reports + params sidecars: enough to compare and to rebuild the tables, no .fif."""
    data = light_copy(src, tmp_path / 'data', ['reports_quality_overview', 'reports_preproc'])
    epochs = data / 'derivatives' / 'preproc_epo'
    epochs.mkdir(parents=True)
    for f in (src / 'derivatives' / 'preproc_epo').rglob('*_preprocessing_params.json'):
        shutil.copy2(f, epochs / f.name)
    return data


def open_tool7(nb, data):
    """The same picks as the chain (pipeline.run_tool7), the session staying open for the answer."""
    nb.pick('fc_edf', data)
    nb.pick('fc_quality', Path(data) / 'reports_quality_overview' / 'quality_summary.tsv')
    nb.pick('fc_config', Path(data) / 'config_param' / 'remap_reref_persubject.json')
    nb.pick('fc_events', Path(data) / 'config_param' / 'event_remap.json')
    nb.pick('fc_out', data)
    nb.set('cb_event_reject', True)
    nb.click('btn_load')


def t7_outputs(data):
    return {**mtimes(Path(data) / 'reports_preproc', *T7_OUTPUTS), **mtimes(Path(data) / 'derivatives', *T7_OUTPUTS)}


def sidecar_path(data, fid, epochs='preproc_epo'):
    return Path(data) / 'derivatives' / epochs / f'{fid}_preprocessing_params.json'


def edit_sidecar(data, fid, **changes):
    path = sidecar_path(data, fid)
    sidecar = json.loads(path.read_text(encoding='utf-8'))
    sidecar.update(changes)
    path.write_text(json.dumps(sidecar), encoding='utf-8')


@pytest.mark.tool7
def test_tool7_same_settings_are_skipped_without_a_question(t7_data):
    before = t7_outputs(t7_data)
    log = pipeline.run_tool7(t7_data)
    assert QUESTION not in log, log[-2000:]
    assert 'Processed this run    : 0' in log
    assert t7_outputs(t7_data) == before


@pytest.mark.tool7
def test_tool7_other_settings_ask_before_anything_runs_then_keep(t7_data, ids):
    before = t7_outputs(t7_data)
    with NotebookSession(pipeline.T7) as nb:
        open_tool7(nb, t7_data)
        nb.set('txt_flat', 2.0)
        nb.set('cb_notch', False)
        nb.click('btn_run')
        log = nb.widget_text()
        assert f'{len(ids)} of the {len(ids)} processed participant(s) {QUESTION}' in log, log[-2000:]
        assert 'flat threshold (µV) : 1 → 2' in log
        assert 'notch : 50 Hz → off' in log
        assert 'Run complete' not in log
        assert 'just below' in nb.value('participant_selector.run_note.value')
        assert not nb.value('btn_run.disabled')
        assert t7_outputs(t7_data) == before                           # nothing ran while waiting for the answer
        assert nb.value('participant_selector.btn_keep_other.description') == 'Keep them, only rebuild the tables'
        nb.click('participant_selector.btn_keep_other')                # keep: the tables are only rebuilt
        log = nb.widget_text()
        assert nb.value('participant_selector.run_note.value') == ''
    assert 'Run complete' in log and 'Processed this run    : 0' in log, log[-2000:]
    assert 'Kept as they are, with their recorded settings' in log     # all kept, nobody new: nothing is mixed
    assert t7_outputs(t7_data) == before


@pytest.mark.tool7
def test_tool7_a_threshold_of_a_method_not_run_is_no_difference(t7_data):
    with NotebookSession(pipeline.T7) as nb:
        open_tool7(nb, t7_data)
        nb.set('cb_flat', False)
        nb.set('txt_flat', 2.0)
        nb.click('btn_run')
        log = nb.widget_text()
    assert 'flagging methods' in log, log[-2000:]
    assert 'flat threshold' not in log


@pytest.mark.tool7
def test_tool7_reprocess_makes_again_the_participants_concerned(t7_data, ids):
    target, others = ids[3], ids[:3]
    before = json.loads(sidecar_path(t7_data, target).read_text(encoding='utf-8'))
    with NotebookSession(pipeline.T7) as nb:
        open_tool7(nb, t7_data)
        select_subset(nb, [target], skip=True)
        nb.set('cb_notch', False)
        nb.click('btn_run')
        assert f'1 of the 1 processed participant(s) {QUESTION}' in nb.widget_text()
        nb.click('participant_selector.btn_redo_other')
        log = nb.widget_text()
    assert f'Made again (other settings): {target} (notch)' in log, log[-2000:]
    assert 'Processed this run    : 1' in log
    written = json.loads(sidecar_path(t7_data, target).read_text(encoding='utf-8'))
    assert written['notch'] == {'applied': False, 'freq_hz': None}
    # Same channels: the reference (M2, listed in the editor, not in the sidecar) stayed the reference.
    assert (written['channels'], written['bad_channels']) == (before['channels'], before['bad_channels'])
    # The three additive keys of the sidecar, the ones an older sidecar does not hold.
    assert written['manual_events'] is False
    config = json.loads((t7_data / 'config_param' / 'remap_reref_persubject.json').read_text(encoding='utf-8'))
    assert written['reference'] == config[target]['ref_channels']
    assert written['hypnogram_file'].startswith(target)
    for fid in others:
        assert json.loads(sidecar_path(t7_data, fid).read_text(encoding='utf-8'))['notch']['applied'] is True

    # Next session with the same settings: nothing left on this one (the new keys read back the same), the three others
    # are still different.
    with NotebookSession(pipeline.T7) as nb:
        open_tool7(nb, t7_data)
        nb.set('cb_notch', False)
        nb.click('btn_run')
        log = nb.widget_text()
        assert f'3 of the 4 processed participant(s) {QUESTION}' in log, log[-2000:]
        assert nb.value('participant_selector.btn_redo_other.description') == 'Reprocess 3 participant(s)'


@pytest.mark.tool7
def test_tool7_the_editor_starts_with_the_channels_processed(t7_data, ids):
    """The sidecar decides for a participant already processed: a channel unticked by hand comes back unticked, a channel
    tool 6 flags and the study kept stays ticked. So reopening the study with the defaults asks nothing."""
    fid = ids[0]                                                       # tool 6 flags its Fp1, kept out by the chain
    edit_sidecar(t7_data, fid, channels=['Fp1', 'C3'], bad_channels=['O1'])
    with NotebookSession(pipeline.T7) as nb:
        open_tool7(nb, t7_data)
        assert nb.value(f'participant_channels[{fid!r}]') == {'C3': True, 'Fp1': True, 'M2': True, 'O1': False}
        nb.set('dd_edit', fid)
        assert 'processed with in this run' in nb.value('lbl_flag_reason.value')
        nb.click('btn_run')
        assert QUESTION not in nb.widget_text()
        # Reset goes back to what the sidecar says, not to tool 6's flags.
        nb.run(f"participant_channels[{fid!r}]['O1'] = True")
        nb.click('btn_reset_sel')
        assert nb.value(f"participant_channels[{fid!r}]['O1']") is False
        # A change in the editor is a difference.
        nb.run(f"participant_channels[{fid!r}]['O1'] = True")
        nb.click('btn_run')
        log = nb.widget_text()
    assert f'1 of the {len(ids)} processed participant(s) {QUESTION}' in log, log[-2000:]
    assert 'channels kept : C3, Fp1 → C3, Fp1, O1' in log


@pytest.mark.tool7
def test_tool7_a_participant_not_processed_starts_with_the_flags_of_tool_6(t7_data, ids):
    fid = ids[0]
    for pattern in (f'{fid}_preprocessing_report.html', f'{fid}_epoch_channel_rejection.tsv'):
        for path in (t7_data / 'reports_preproc').rglob(pattern):
            path.unlink()
    with NotebookSession(pipeline.T7) as nb:
        open_tool7(nb, t7_data)
        assert nb.value(f"participant_channels[{fid!r}]['Fp1']") is False  # from tool 6's flags, the sidecar is not read
        nb.set('dd_edit', fid)
        assert "Unticked from tool 6" in nb.value('lbl_flag_reason.value')


@pytest.mark.tool7
def test_tool7_a_second_run_does_not_disturb_the_first(t7_data, ids):
    """Another run keeps its own settings: no question and no mixed-method warning from it, and the channels of the editor
    follow the run chosen, except for a participant modified in this session."""
    fid, modified = ids[0], ids[2]
    deriv = t7_data / 'derivatives'
    for participant in ids:                                            # the menu finds a run through its .fif
        (deriv / 'preproc_epo' / f'{participant}_all-epo.fif').touch()
    shutil.copytree(t7_data / 'reports_preproc', t7_data / 'reports_preproc_v2')
    shutil.copytree(deriv / 'preproc_epo', deriv / 'preproc_epo_v2')
    for path in (deriv / 'preproc_epo_v2').glob('*_preprocessing_params.json'):
        sidecar = json.loads(path.read_text(encoding='utf-8'))
        sidecar['methods_run'] = [m for m in sidecar['methods_run'] if m != 'flat']
        sidecar['channels'] = ['C3']
        sidecar['bad_channels'] = [ch for ch in sidecar['bad_channels'] + ['O1', 'Fp1'] if ch != 'C3']
        path.write_text(json.dumps(sidecar), encoding='utf-8')
    with NotebookSession(pipeline.T7) as nb:
        open_tool7(nb, t7_data)
        nb.set('run_menu', str(deriv / 'preproc_epo_v2'))
        assert nb.value(f"participant_channels[{fid!r}]['O1']") is False
        nb.run(f"participant_channels[{modified!r}]['C3'] = False")       # modified in this session: stays
        nb.set('run_menu', str(deriv / 'preproc_epo'))
        assert nb.value(f"participant_channels[{fid!r}]['O1']") is True
        assert nb.value(f"participant_channels[{modified!r}]['C3']") is False
        nb.click('btn_reset_sel')
        nb.click('btn_run')
        log = nb.widget_text()
    assert 'DIFFERENT' not in log, log[-2000:]
    assert 'Processed this run    : 0' in log


# =====================================================================================  tool 9
T9_OUTPUTS = ('sim*_spectral_stage.tsv', 'sim*_bandpower_epoch.tsv', 'sim*_psd_epoch.npz', 'sim*_spectral_settings.json',
              'sim*_spectral_report.html')
T9_TABLES = ('spectral_stage', 'aperiodic_stage', 'psd_stage', 'spectral_stage_third')


@pytest.fixture
def t9_data(src, tmp_path):
    data = light_copy(src, tmp_path / 'data', ['reports_features_spectral', 'reports_rejection_auto'])
    for folder in ('preproc_epo', 'rejection_auto', 'features_spectral'):
        shutil.copytree(src / 'derivatives' / folder, data / 'derivatives' / folder)
    return data


def t9_folders(data):
    return Path(data) / 'derivatives' / 'preproc_epo', Path(data) / 'derivatives' / 'rejection_auto'


def t9_outputs(data):
    return {**mtimes(Path(data) / 'derivatives' / 'features_spectral', *T9_OUTPUTS),
            **mtimes(Path(data) / 'reports_features_spectral', *T9_OUTPUTS)}


def open_tool9(nb, data):
    """The same picks and options as the chain (pipeline.run_tool9), the session staying open for the answer."""
    raw_dir, decision_dir = t9_folders(data)
    nb.pick('fc_data', data)
    nb.pick('fc_raw', raw_dir)
    nb.pick('fc_decision', decision_dir)
    nb.pick('fc_subj', Path(data) / 'participants.tsv')
    nb.set('dd_join_col', 'participant_id')
    nb.click('btn_scan')
    nb.set('cb_avg_lin', True)
    nb.set('cb_thirds', True)


def feature_file(data, fid, name):
    return Path(data) / 'derivatives' / 'features_spectral' / f'{fid}_{name}'


@pytest.mark.tool9
def test_tool9_same_settings_are_skipped_without_a_question(t9_data):
    before = t9_outputs(t9_data)
    log = pipeline.run_tool9(t9_data, *t9_folders(t9_data))
    assert QUESTION not in log, log[-2000:]
    assert '0 processed' in log
    assert t9_outputs(t9_data) == before


@pytest.mark.tool9
def test_tool9_a_widget_without_effect_asks_nothing(t9_data):
    with NotebookSession(pipeline.T9) as nb:
        open_tool9(nb, t9_data)
        nb.set('txt_bandwidth', 5.0)               # not used with Welch
        nb.set('txt_n_jobs', 1)                    # changes no value
        nb.click('btn_run')
        log = nb.widget_text()
    assert QUESTION not in log, log[-2000:]


@pytest.mark.tool9
def test_tool9_other_settings_ask_before_anything_runs_then_keep(t9_data, ids):
    before = t9_outputs(t9_data)
    with NotebookSession(pipeline.T9) as nb:
        open_tool9(nb, t9_data)
        nb.set('txt_min_epochs', 5)
        nb.set('txt_win_s', 2.0)
        nb.click('btn_run')
        log = nb.widget_text()
        assert f'{len(ids)} of the {len(ids)} processed participant(s) {QUESTION}' in log, log[-2000:]
        assert 'to re-aggregate: min epochs per cell : 20 → 5' in log
        assert 'to recompute: Welch window (s) : 4 → 2' in log
        assert f'the full computation (PSD, fit) for {len(ids)}, a re-aggregation only for 0' in log
        assert 'just below' in nb.value('participant_selector.run_note.value')
        assert t9_outputs(t9_data) == before                           # nothing ran while waiting for the answer
        assert nb.value('participant_selector.btn_keep_other.description') == 'Keep them, only rebuild the tables'
        nb.click('participant_selector.btn_keep_other')
        log = nb.widget_text()
        assert nb.value('participant_selector.run_note.value') == ''
    assert '0 processed' in log, log[-2000:]
    assert 'Kept as they are, with their recorded settings' in log     # all kept, nobody new: nothing is mixed
    assert t9_outputs(t9_data) == before


@pytest.mark.tool9
def test_tool9_reprocess_takes_the_path_each_setting_needs(t9_data, ids):
    first, second = ids[0], ids[1]
    npz_before = feature_file(t9_data, first, 'psd_epoch.npz').stat().st_mtime_ns

    # An aggregation setting: re-aggregated, the spectra stay as they are.
    with NotebookSession(pipeline.T9) as nb:
        open_tool9(nb, t9_data)
        select_subset(nb, [first], skip=True)
        nb.set('txt_min_epochs', 5)
        nb.click('btn_run')
        assert 'the full computation (PSD, fit) for 0, a re-aggregation only for 1' in nb.widget_text()
        nb.click('participant_selector.btn_redo_other')
        log = nb.widget_text()
    assert f'Made again (other settings): {first} (to re-aggregate: min epochs per cell)' in log, log[-2000:]
    assert '1 processed' in log
    assert feature_file(t9_data, first, 'psd_epoch.npz').stat().st_mtime_ns == npz_before
    assert json.loads(feature_file(t9_data, first, 'spectral_settings.json').read_text(encoding='utf-8'))['min_epochs'] == 5

    # A setting of the computation: the spectra are made again.
    npz_second = feature_file(t9_data, second, 'psd_epoch.npz').stat().st_mtime_ns
    with NotebookSession(pipeline.T9) as nb:
        open_tool9(nb, t9_data)
        select_subset(nb, [second], skip=True)
        nb.set('txt_win_s', 2.0)
        nb.click('btn_run')
        assert 'the full computation (PSD, fit) for 1, a re-aggregation only for 0' in nb.widget_text()
        nb.click('participant_selector.btn_redo_other')
        log = nb.widget_text()
    assert f'Made again (other settings): {second} (to recompute: Welch window (s))' in log, log[-2000:]
    assert feature_file(t9_data, second, 'psd_epoch.npz').stat().st_mtime_ns != npz_second


@pytest.mark.tool9
def test_tool9_a_re_aggregation_equals_a_full_run(t9_data, ids):
    fid = ids[0]
    with NotebookSession(pipeline.T9) as nb:
        open_tool9(nb, t9_data)
        select_subset(nb, [fid], skip=True)
        nb.set('txt_min_epochs', 200)
        nb.click('btn_run')
        nb.click('participant_selector.btn_redo_other')
    reaggregated = {name: tsv(feature_file(t9_data, fid, f'{name}.tsv')) for name in T9_TABLES}
    assert reaggregated['spectral_stage']['power_db_from_log_mean'].isna().any()   # the minimum took effect

    with NotebookSession(pipeline.T9) as nb:
        open_tool9(nb, t9_data)
        select_subset(nb, [fid], skip=False)
        nb.set('txt_min_epochs', 200)
        nb.click('btn_run')
        assert '1 processed' in nb.widget_text()
    for name, table in reaggregated.items():
        # the per-epoch spectrum is stored in float32: psd_stage agrees to ~1e-7 relative
        diffs = snapshot.compare_tables(tsv(feature_file(t9_data, fid, f'{name}.tsv')), table, atol=1e-5)
        assert not diffs, f'{name}: ' + '; '.join(diffs)


@pytest.mark.tool9
def test_tool9_a_decision_changed_is_asked_about_through_what_it_says(t9_data, ids):
    fid = ids[0]
    path = t9_folders(t9_data)[1] / f'{fid}_epoch_decision.tsv'
    table = tsv(path)

    # A reason reworded: the aggregates would not move.
    table.assign(reject_reason='reworded').to_csv(path, sep='\t', index=False)
    with NotebookSession(pipeline.T9) as nb:
        open_tool9(nb, t9_data)
        nb.click('btn_run')
        assert QUESTION not in nb.widget_text()

    # Epochs rejected: this participant only, by a re-aggregation.
    newly = table.index[table['in_scope'] & ~table['rejected']][:25]
    table.loc[newly, 'rejected'] = True
    table.to_csv(path, sep='\t', index=False)
    npz_before = feature_file(t9_data, fid, 'psd_epoch.npz').stat().st_mtime_ns
    with NotebookSession(pipeline.T9) as nb:
        open_tool9(nb, t9_data)
        nb.click('btn_run')
        log = nb.widget_text()
        assert f'1 of the {len(ids)} processed participant(s) {QUESTION}' in log, log[-2000:]
        assert 'to re-aggregate: decision : as it was at the last run → changed since' in log
        assert 'Keeping them mixes settings' in log
        nb.click('participant_selector.btn_redo_other')
        log = nb.widget_text()
    assert '1 processed' in log, log[-2000:]
    assert feature_file(t9_data, fid, 'psd_epoch.npz').stat().st_mtime_ns == npz_before

    # Nothing left to ask.
    log = pipeline.run_tool9(t9_data, *t9_folders(t9_data))
    assert QUESTION not in log, log[-2000:]


@pytest.mark.tool9
def test_tool9_outputs_written_before_the_record_compare_only_the_computation(t9_data, ids):
    for path in (t9_data / 'derivatives' / 'features_spectral').glob('*_spectral_settings.json'):
        path.unlink()
    with NotebookSession(pipeline.T9) as nb:
        open_tool9(nb, t9_data)
        nb.set('txt_min_epochs', 5)                # not recorded: nothing to compare it with
        nb.click('btn_run')
        log = nb.widget_text()
        assert QUESTION not in log, log[-2000:]
        assert f'{len(ids)} participant(s) written before their settings were all recorded' in log
        nb.set('txt_win_s', 2.0)                   # the computation stays compared, through the .npz
        nb.click('btn_run')
        assert 'to recompute: Welch window (s) : 4 → 2' in nb.widget_text()


# =====================================================================================  tool 8bis
T8BIS_OUTPUTS = ('sim*_autoreject_decision.tsv', 'sim*_autoreject_report.html', 'sim*_epoch_decision.tsv',
                 'sim*_channel_decision.tsv')


def epoch_stubs(src, data):
    """Tool 7's params sidecars + an empty .fif per participant: enough for what finds and compares, nothing reads a signal."""
    epochs = Path(data) / 'derivatives' / 'preproc_epo'
    epochs.mkdir(parents=True, exist_ok=True)
    for f in (Path(src) / 'derivatives' / 'preproc_epo').glob('*_preprocessing_params.json'):
        shutil.copy2(f, epochs / f.name)
        (epochs / f.name.replace('_preprocessing_params.json', '_all-epo.fif')).touch()


@pytest.fixture
def t8bis_data(src, tmp_path):
    data = light_copy(src, tmp_path / 'data', ['reports_preproc', 'reports_rejection_auto'])
    epoch_stubs(src, data)
    shutil.copytree(src / 'derivatives' / 'rejection_auto', data / 'derivatives' / 'rejection_auto')
    return data


def t8bis_outputs(data):
    return {**mtimes(Path(data) / 'derivatives' / 'rejection_auto', *T8BIS_OUTPUTS),
            **mtimes(Path(data) / 'reports_rejection_auto', *T8BIS_OUTPUTS)}


def open_tool8bis(nb, data):
    nb.pick('fc_raw', Path(data) / 'derivatives' / 'preproc_epo')
    nb.pick('fc_reports', Path(data) / 'reports_preproc')
    nb.click('btn_scan')


@pytest.mark.tool8bis
def test_tool8bis_same_settings_are_skipped_without_a_question(t8bis_data, ids):
    before = t8bis_outputs(t8bis_data)
    log = pipeline.run_tool8bis(t8bis_data)
    assert QUESTION not in log, log[-2000:]
    assert f'skipped={len(ids)}' in log
    assert t8bis_outputs(t8bis_data) == before


@pytest.mark.tool8bis
def test_tool8bis_other_settings_ask_before_anything_runs_then_keep(t8bis_data, ids):
    before = t8bis_outputs(t8bis_data)
    with NotebookSession(pipeline.T8BIS) as nb:
        open_tool8bis(nb, t8bis_data)
        nb.set('ft_ch_pct', 50.0)
        nb.click('btn_run')
        log = nb.widget_text()
        assert f'{len(ids)} of the {len(ids)} processed participant(s) {QUESTION}' in log, log[-2000:]
        assert 'channel reject (%) : 20 → 50' in log
        assert 'Done.' not in log
        assert 'just below' in nb.value('participant_selector.run_note.value')
        assert t8bis_outputs(t8bis_data) == before                     # nothing ran while waiting for the answer
        assert nb.value('participant_selector.btn_keep_other.description') == 'Keep them, only rebuild the tables'
        nb.click('participant_selector.btn_keep_other')
        log = nb.widget_text()
        assert nb.value('participant_selector.run_note.value') == ''
    assert f'skipped={len(ids)}' in log, log[-2000:]
    assert 'Kept as they are, with their recorded settings' in log     # all kept, nobody new: nothing is mixed
    assert t8bis_outputs(t8bis_data) == before


@pytest.mark.tool8bis
def test_tool8bis_reprocess_decides_again_the_participants_concerned(t8bis_data, ids):
    target, others = ids[3], ids[:3]
    with NotebookSession(pipeline.T8BIS) as nb:
        open_tool8bis(nb, t8bis_data)
        select_subset(nb, [target], skip=True)
        nb.set('ft_ch_pct', 50.0)
        nb.click('btn_run')
        assert f'1 of the 1 processed participant(s) {QUESTION}' in nb.widget_text()
        nb.click('participant_selector.btn_redo_other')
        log = nb.widget_text()
    assert f'Made again (other settings): {target} (channel reject (%))' in log, log[-2000:]
    reports = t8bis_data / 'reports_rejection_auto'
    assert tsv(reports / f'{target}_autoreject_decision.tsv')['channel_reject_pct'].iloc[0] == 50.0
    for fid in others:
        assert tsv(reports / f'{fid}_autoreject_decision.tsv')['channel_reject_pct'].iloc[0] == 20.0
    # The new record reads back the same, the three others are still different.
    with NotebookSession(pipeline.T8BIS) as nb:
        open_tool8bis(nb, t8bis_data)
        nb.set('ft_ch_pct', 50.0)
        nb.click('btn_run')
        assert f'3 of the 4 processed participant(s) {QUESTION}' in nb.widget_text()


# =====================================================================================  what follows tool 7
OTHER_NOTCH = {'notch': {'applied': False, 'freq_hz': None}}


@pytest.mark.tool8bis
def test_tool8bis_asks_when_tool7_ran_again_with_other_settings(t8bis_data, ids):
    fid = ids[0]
    # The same settings (a rate is not one of them): nothing is out of date.
    edit_sidecar(t8bis_data, fid, sfreq_hz=128.0)
    log = pipeline.run_tool8bis(t8bis_data)
    assert QUESTION not in log and 'decided before tool 7' not in log, log[-2000:]

    edit_sidecar(t8bis_data, fid, **OTHER_NOTCH)
    with NotebookSession(pipeline.T8BIS) as nb:
        open_tool8bis(nb, t8bis_data)
        assert '1 participant(s) decided before tool 7 was run again' in nb.value('lbl_scan.value')
        nb.click('btn_run')
        log = nb.widget_text()
    assert f'1 of the {len(ids)} processed participant(s) {QUESTION}' in log, log[-2000:]
    assert 'tool 7 settings : as at the last decision → changed since' in log

    # A record written before the column: no alert, no reprocessing after the update.
    record = t8bis_data / 'reports_rejection_auto' / f'{fid}_autoreject_decision.tsv'
    tsv(record).drop(columns=['preproc_fingerprint']).to_csv(record, sep='\t', index=False)
    log = pipeline.run_tool8bis(t8bis_data)
    assert QUESTION not in log and 'decided before tool 7' not in log, log[-2000:]


@pytest.fixture
def t8_data(src, tmp_path):
    data = light_copy(src, tmp_path / 'data', ['reports_preproc', 'reports_rejection_manual'])
    epoch_stubs(src, data)
    shutil.copytree(src / 'derivatives' / 'rejection_manual', data / 'derivatives' / 'rejection_manual')
    return data


def tool8_badge(data, fid):
    with NotebookSession(pipeline.T8) as nb:
        nb.pick('fc_data', data)
        nb.pick('fc_raw', Path(data) / 'derivatives' / 'preproc_epo')
        nb.pick('fc_reports', Path(data) / 'reports_preproc')
        nb.set('dd_part', fid)
        return nb.value('lbl_done.value')


@pytest.mark.tool8
def test_tool8_only_warns_when_tool7_ran_again_with_other_settings(t8_data):
    fid = minidb.manual_review_participant('synthetic')
    record = t8_data / 'reports_rejection_manual' / f'{fid}_manualreject_decision.tsv'
    before = record.stat().st_mtime_ns
    badge = tool8_badge(t8_data, fid)
    assert 'already processed' in badge and 'run again' not in badge
    edit_sidecar(t8_data, fid, sfreq_hz=128.0)                 # the same settings
    assert 'run again' not in tool8_badge(t8_data, fid)
    edit_sidecar(t8_data, fid, **OTHER_NOTCH)
    assert 'Tool 7 was run again with other settings' in tool8_badge(t8_data, fid)
    assert record.stat().st_mtime_ns == before                 # a manual review is never made again
    tsv(record).drop(columns=['preproc_fingerprint']).to_csv(record, sep='\t', index=False)
    assert 'run again' not in tool8_badge(t8_data, fid)        # a record written before the column


@pytest.mark.tool9
def test_tool9_asks_when_tool7_ran_again_with_other_settings(t9_data, ids):
    fid = ids[0]
    edit_sidecar(t9_data, fid, sfreq_hz=128.0)                 # the same settings
    log = pipeline.run_tool9(t9_data, *t9_folders(t9_data))
    assert QUESTION not in log and 'decision(s) made before tool 7' not in log, log[-2000:]

    edit_sidecar(t9_data, fid, **OTHER_NOTCH)
    with NotebookSession(pipeline.T9) as nb:
        open_tool9(nb, t9_data)
        assert f'1 decision(s) made before tool 7 was run again with other settings: {fid}' in nb.widget_text()
        nb.click('btn_run')
        log = nb.widget_text()
    assert f'1 of the {len(ids)} processed participant(s) {QUESTION}' in log, log[-2000:]
    assert 'to recompute: tool 7 settings : as at the last computation → changed since' in log
    assert 'the full computation (PSD, fit) for 1, a re-aggregation only for 0' in log

    # 'Re-aggregate only' keeps the spectra: they are still those of the former run of tool 7.
    log = pipeline.run_tool9(t9_data, *t9_folders(t9_data), subset=[fid], reaggregate=True)
    assert '1 processed' in log, log[-2000:]
    log = pipeline.run_tool9(t9_data, *t9_folders(t9_data))
    assert 'to recompute: tool 7 settings' in log, log[-2000:]

    # Outputs written before the key: no alert, no reprocessing after the update.
    path = feature_file(t9_data, fid, 'spectral_settings.json')
    settings = json.loads(path.read_text(encoding='utf-8'))
    del settings['preproc_fingerprint']
    path.write_text(json.dumps(settings), encoding='utf-8')
    log = pipeline.run_tool9(t9_data, *t9_folders(t9_data))
    assert QUESTION not in log, log[-2000:]
