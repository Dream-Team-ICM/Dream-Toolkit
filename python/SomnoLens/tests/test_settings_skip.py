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
