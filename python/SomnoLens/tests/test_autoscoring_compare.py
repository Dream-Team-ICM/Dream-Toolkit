"""Tool 3bis's comparison with the expert hypnogram across nights (tools/plan_3bis_comparison.md): the database report of
step 1 with a night where the expert scored a single stage and a night compared before the macro-F1, the balanced
accuracy and the MCC were added, both ways of the Section 1 option. Then Section 4 in the toolkit's layout, on a copy
of the synthetic study: its files, the same metrics as step 1, a second comparison replacing the first."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
import autoscoring_lib as A          # noqa: E402
import minidb                        # noqa: E402
import pipeline                      # noqa: E402
from nbdriver import NotebookSession  # noqa: E402

pytestmark = [pytest.mark.tool3bis]
T3BIS = 'tools/3bis_autoscoring_voila.ipynb'
NEW_COLUMNS = ['macro_f1', 'balanced_accuracy', 'mcc', 'n_stages_expert', 'single_stage_expert']
INPUT_SUFFIXES = ('.edf', '.txt', '.csv', '.xml', '.tsv')


def write_night(rep, fid, expert, model, old=False):
    """A night's comparison table and long confusion matrix, as step 1 writes them (`old`: without the new columns)."""
    res, _, cm_long = A.compare_scoring(np.array(expert), np.array(model))
    row = {'file_id': fid, 'n_epochs_gssc': len(model), 'n_epochs_expert': len(expert),
           'n_compared': res['n_compared'], 'n_excluded_expert': res['n_excluded_expert'],
           'accuracy': res['accuracy'], 'kappa': res['kappa'], **{f'f1_{s}': res[f'f1_{s}'] for s in A.STAGES},
           **({} if old else {k: res[k] for k in NEW_COLUMNS}),
           'mean_confidence': 0.9, 'scoring_mode': 'consensus', 'eeg_used': 'C3', 'eog_used': 'none',
           'status': 'compared'}
    pd.DataFrame([row]).to_csv(rep / f'{fid}_autoscoring_comparison.tsv', sep='\t', index=False, float_format='%.4f')
    cm_long.insert(0, 'file_id', fid)
    cm_long.to_csv(rep / f'{fid}_confusion.tsv', sep='\t', index=False)


def test_database_report_with_a_single_stage_night_and_a_night_compared_before(tmp_path):
    rep = tmp_path / 'reports_autoscoring_gssc'
    rep.mkdir()
    night = ['W'] * 10 + ['N1'] * 5 + ['N2'] * 20 + ['N3'] * 10 + ['R'] * 15
    scored = ['W'] * 9 + ['N1'] * 6 + ['N2'] * 18 + ['N3'] * 12 + ['R'] * 15
    write_night(rep, 'night', night, scored)
    write_night(rep, 'nap_awake', ['W'] * 40, ['W'] * 37 + ['N1'] * 3)      # the expert scored W only
    write_night(rep, 'former', night, scored, old=True)
    with NotebookSession(T3BIS) as nb:
        nb.run(f'rebuild_database_summary(Path(r"{tmp_path}"), "GSSC", True)')
        report = (rep / 'global_autoscoring_report.html').read_text(encoding='utf-8')
        assert '1 participant(s) where the expert scored a single stage: left out of the kappa and the MCC' in report
        assert '1 participant(s) compared before the MCC, the macro-F1 and the balanced accuracy' in report
        summary = pd.read_csv(rep / 'global_autoscoring_summary.tsv', sep='\t')
        assert len(summary) == 3 and summary['macro_f1'].isna().sum() == 1      # the former night: missing
        pooled = pd.read_csv(rep / 'global_autoscoring_pooled.tsv', sep='\t')
        # the pooled values keep every epoch, the single-stage nap included
        cm = sum(A.compare_scoring(np.array(e), np.array(m))[1]
                 for e, m in [(night, scored), (['W'] * 40, ['W'] * 37 + ['N1'] * 3), (night, scored)])
        expected = A.agreement_metrics(cm)
        assert pooled.loc[0, 'n_nights'] == 3 and pooled.loc[0, 'n_compared'] == 160
        assert pooled.loc[0, 'kappa'] == pytest.approx(expected['kappa'], abs=1e-4)      # 4 decimals in the TSV
        assert pooled.loc[0, 'macro_f1'] == pytest.approx(expected['macro_f1'], abs=1e-4)
        nb.run(f'rebuild_database_summary(Path(r"{tmp_path}"), "GSSC", False)')
        report = (rep / 'global_autoscoring_report.html').read_text(encoding='utf-8')
        assert 'kept in the kappa and the MCC, where they count as 0 whatever the model' in report


def study_inputs(chain, dest):
    """The study INPUTS of the chain (recordings, companions, config_param) copied to `dest`."""
    data = Path(chain['data'])
    dest.mkdir(parents=True, exist_ok=True)
    for f in data.iterdir():
        if f.is_file() and f.suffix.lower() in INPUT_SUFFIXES:
            shutil.copy2(f, dest / f.name)
    shutil.copytree(data / 'config_param', dest / 'config_param')
    return dest


def read_tsv(path):
    return pd.read_csv(path, sep='\t', dtype={'file_id': str, 'stage_expert': str, 'stage_model': str})


def test_section4_writes_the_toolkit_layout_and_a_compare_replaces_the_previous_one(chain, tmp_path):
    pytest.importorskip('gssc')
    if chain['dataset'] != 'synthetic':
        pytest.skip('one dataset is enough: the layout and the rules do not depend on the recordings')
    study = study_inputs(chain, tmp_path / 'study')
    first, second = minidb.PARTICIPANTS['synthetic'][:2]
    deriv = study / 'derivatives' / 'autoscoring_comparison'
    rep = study / 'reports_autoscoring_comparison'
    with NotebookSession(T3BIS) as nb:
        nb.pick('fc_data', study)
        nb.click('btn_scan')
        pipeline.select_subset(nb, [first, second], skip=False)
        nb.click('btn_run')
        nb.click('btn_compare_find')
        assert nb.value("[k for k, cb in cmp_checks.items() if cb.value]") == ['GSSC']
        nb.click('btn_compare')
        for fid in (first, second):
            assert next(deriv.rglob(f'{fid}_comparison_epochs.tsv'), None) is not None, fid
            for suffix in ('_comparison.tsv', '_confusion.tsv', '_comparison_report.html'):
                assert next(rep.rglob(f'{fid}{suffix}'), None) is not None, fid + suffix
        for name in ('global_comparison_summary.tsv', 'global_comparison_pooled.tsv', 'global_confusion.tsv',
                     'global_comparison_report.html'):
            assert (rep / name).exists(), name
        assert not (study / 'autoscore_outputs').exists()                       # Score_EDF's layout no longer written
        epochs = read_tsv(next(deriv.rglob(f'{first}_comparison_epochs.tsv')))
        assert set(epochs['model']) == {'gssc'}
        assert set(epochs['stage_expert'].fillna('')) <= {'W', 'N1', 'N2', 'N3', 'R', ''}
        assert (epochs['epoch_idx'] == (epochs['onset_s'] / 30).round().astype(int)).all()   # from the recording start
        # The synthetic nights last exactly 480 epochs: gssc 0.0.9 leaves out the last one (said in step 1's notes),
        # which Section 4 counts as not predicted; its metrics are then step 1's.
        params = json.loads(next(study.rglob(f'{first}_scoring_params.json')).read_text(encoding='utf-8'))
        assert any('GSSC scored 479 of the 480 epochs' in n for n in params['notes'])
        s4 = read_tsv(rep / 'global_comparison_summary.tsv').set_index('file_id')
        s1 = read_tsv(study / 'reports_autoscoring_gssc' / 'global_autoscoring_summary.tsv').set_index('file_id')
        assert (s4['n_not_predicted'] == 1).all()
        cols = ['n_compared', 'accuracy', 'kappa', 'mcc', 'macro_f1', 'balanced_accuracy',
                *[f'f1_{s}' for s in A.STAGES]]
        np.testing.assert_allclose(s4.loc[[first, second], cols].astype(float),
                                   s1.loc[[first, second], cols].astype(float), atol=1e-4)
        # A comparison is one set of nights: comparing the first night alone replaces the two-night comparison.
        pipeline.select_subset(nb, [first], skip=True)
        nb.click('btn_compare_find')
        nb.click('btn_compare')
        assert not list(deriv.rglob(f'{second}_*')) and not list(rep.rglob(f'{second}_*'))
        assert read_tsv(rep / 'global_comparison_summary.tsv')['file_id'].tolist() == [first]
