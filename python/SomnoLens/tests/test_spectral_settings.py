"""What separates a participant already processed from the run's settings in tools/spectral_lib.py (tool 9's question when
Skip is ticked): what takes the full computation, what takes a re-aggregation only, and what makes no difference."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
import spectral_lib as SL     # noqa: E402

pytestmark = [pytest.mark.quick, pytest.mark.tool9]

BANDS = [('delta', 0.5, 4.0), ('theta', 4.0, 8.0)]
CFG = dict(psd=dict(SL.DEF_PSD), fit=dict(SL.DEF_FIT), do_fit=True, smooth=dict(SL.DEF_SMOOTH), bands=BANDS,
           measures=['power_db', 'power_rel'], stages=['W', 'N2', 'N3'], spaces=['log'], min_epochs=20, thirds=False,
           exclude_bad_fit=False, n_jobs=-1)


def computed(**changes):
    """The cfg_json of a _psd_epoch.npz: the keys of the computation, after a JSON round trip (tuples -> lists)."""
    keys = ('psd', 'fit', 'do_fit', 'smooth', 'bands', 'measures')
    return dict(json.loads(json.dumps({k: CFG[k] for k in keys})), **changes)


def run_cfg(section=None, **changes):
    cfg = copy.deepcopy(CFG)
    (cfg[section] if section else cfg).update(changes)
    return cfg


def test_the_same_computation_makes_no_difference():
    assert SL.computation_differences(computed(), run_cfg()) == []


@pytest.mark.parametrize('section, changes, row', [
    ('psd', {'method': 'multitaper'}, ('PSD method', 'welch', 'multitaper')),
    ('psd', {'fmax': 40.0}, ('PSD fmax (Hz)', '45', '40')),
    ('psd', {'win_s': 2.0}, ('Welch window (s)', '4', '2')),
    ('psd', {'overlap_pct': 25.0}, ('Welch overlap (%)', '50', '25')),
    ('psd', {'window': 'hamming'}, ('Welch window shape', 'hann', 'hamming')),
    ('fit', {'aperiodic_mode': 'knee'}, ('aperiodic mode', 'fixed', 'knee')),
    ('fit', {'r2_min': 0.8}, ('poor fit if R2 <', '0.9', '0.8')),
    ('smooth', {'lowess': False}, ('smoothing before the fit', 'median 3 Hz + LOWESS 2 Hz', 'median 3 Hz')),
    ('smooth', {'enabled': False}, ('smoothing before the fit', 'median 3 Hz + LOWESS 2 Hz', 'off')),
])
def test_each_computation_setting_is_reported_with_both_values(section, changes, row):
    assert SL.computation_differences(computed(), run_cfg(section, **changes)) == [row]


def test_a_widget_without_effect_is_no_difference():
    # With Welch, the bandwidth is not used.
    assert SL.computation_differences(computed(), run_cfg('psd', bandwidth=5.0)) == []
    # With multitaper, the window and the overlap are not used.
    multitaper = computed(psd=dict(computed()['psd'], method='multitaper'))
    assert SL.computation_differences(multitaper, run_cfg('psd', method='multitaper', win_s=2.0, overlap_pct=10.0,
                                                          window='boxcar')) == []
    assert SL.computation_differences(multitaper, run_cfg('psd', method='multitaper', bandwidth=4.0)) == [
        ('multitaper bandwidth (Hz)', '2', '4')]
    # Without a fit on one side, its settings and the smoothing are not compared: only the fit itself.
    no_fit = run_cfg(do_fit=False, fit=dict(CFG['fit'], r2_min=0.5), smooth=dict(CFG['smooth'], enabled=False))
    assert SL.computation_differences(computed(), no_fit) == [('1/f fit', 'on', 'off')]
    assert SL.computation_differences(computed(do_fit=False), no_fit) == []


def test_bands_and_measures_are_compared_as_sets():
    assert SL.computation_differences(computed(), run_cfg(bands=list(reversed(BANDS)),
                                                          measures=['power_rel', 'power_db'])) == []
    assert SL.computation_differences(computed(), run_cfg(bands=BANDS + [('sigma', 12.0, 16.0)])) == [
        ('bands', 'delta 0.5-4 Hz; theta 4-8 Hz', 'delta 0.5-4 Hz; theta 4-8 Hz; sigma 12-16 Hz')]
    assert SL.computation_differences(computed(), run_cfg(measures=['power_db'])) == [
        ('measures', 'power_db, power_rel', 'power_db')]


META = pd.DataFrame({'epoch_idx': [0, 1, 2, 3], 'stage': ['W', 'N2', 'N2', 'N3']})
IN_SCOPE = np.array([True, True, True, True])
NOT_REJECTED = np.array([False, False, False, False])


def settings(cfg=None, source='auto', rejected=NOT_REJECTED, dropped=('Fp1',), meta=META):
    return SL.aggregation_settings(cfg or CFG, source, meta, IN_SCOPE, rejected, list(dropped))


def test_the_same_aggregation_makes_no_difference():
    assert SL.aggregation_differences(settings(), settings()) == []
    # What is written to the sidecar reads back the same.
    assert SL.aggregation_differences(json.loads(json.dumps(settings())), settings()) == []


@pytest.mark.parametrize('changes, row', [
    ({'stages': ['W', 'N2']}, ('stages', 'N2, N3, W', 'N2, W')),
    ({'spaces': ['log', 'lin']}, ('averaging space', 'log', 'lin, log')),
    ({'min_epochs': 5}, ('min epochs per cell', '20', '5')),
    ({'thirds': True}, ('night thirds', 'off', 'on')),
    ({'exclude_bad_fit': True}, ('exclude poorly-fitted epochs', 'off', 'on')),
])
def test_each_aggregation_setting_is_reported_with_both_values(changes, row):
    assert SL.aggregation_differences(settings(), settings(run_cfg(**changes))) == [row]


def test_a_stage_of_the_rest_of_the_database_does_not_concern_a_participant():
    assert SL.aggregation_differences(settings(), settings(run_cfg(stages=['W', 'N2', 'N3', 'R', 'IS']))) == []
    # A stage all of whose epochs are rejected has no row for this participant: the same.
    all_n3_rejected = np.array([False, False, False, True])
    assert settings(rejected=all_n3_rejected)['stages'] == ['W', 'N2']


def test_excluding_poor_fits_counts_only_when_a_fit_ran():
    assert settings(run_cfg(exclude_bad_fit=True, do_fit=False))['exclude_bad_fit'] is False
    assert settings(run_cfg(exclude_bad_fit=True))['exclude_bad_fit'] is True


def test_the_decision_counts_through_what_it_says():
    changed = np.array([False, True, False, False])
    assert SL.aggregation_differences(settings(), settings(rejected=changed)) == [
        ('decision', 'as it was at the last run', 'changed since')]
    assert SL.aggregation_differences(settings(), settings(dropped=('Fp1', 'O1'))) == [
        ('decision', 'as it was at the last run', 'changed since')]
    # Another source says it: the fingerprint would only repeat it.
    assert SL.aggregation_differences(settings(), settings(source='manual', rejected=changed)) == [
        ('decision', 'auto', 'manual')]
    # The order of the channels dropped changes nothing.
    assert settings(dropped=('O1', 'Fp1'))['decision_fingerprint'] == settings(dropped=('Fp1', 'O1'))['decision_fingerprint']
