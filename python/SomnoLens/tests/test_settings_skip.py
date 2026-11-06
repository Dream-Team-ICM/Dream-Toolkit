"""'Skip already processed' ticked: a participant already processed with other settings is not skipped silently. The run
stops before any analysis and asks (reprocess them, or keep them). Each scenario opens a NEW session on a study already
processed, the real case: the participant selector then knows who is done before the run starts.

Each test works on a light private copy of the chain's outputs, so the golden comparison is unaffected.
"""
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
    assert 'Kept as they are, so the tables mix settings' in log
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
