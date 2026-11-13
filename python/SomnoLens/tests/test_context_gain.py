"""Phase 4 ter of tools/plan_brainvision.md: a context channel recorded through an input with its own gain the file does not
declare (an auxiliary input behind a BIP2AUX adapter reads 100 times too large), corrected in tool 2 ('Divide by', the
additive `context_gain` block of the channel configuration) and divided wherever the channel is read, before any
derivation: tool 3bis (the models' EOG, chin EMG, ECG), tool 7's context companion, tool 0, the channel checks. On
BrainVision recordings written by hand with invented values (tests/test_autoscoring_formats.py)."""
import json
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
import autoscoring_lib as A          # noqa: E402
import channel_check_lib as CC       # noqa: E402
import edf_io_lib as EIO             # noqa: E402
import pipeline                      # noqa: E402
from nbdriver import NotebookSession                                   # noqa: E402
from test_autoscoring_formats import SFREQ, ecg_like, entry, recording, row   # noqa: E402

T0 = 'tools/0_live_explore_1file_voila.ipynb'
T2 = 'tools/2_select&remap_channels_edf_voila.ipynb'
T3BIS = 'tools/3bis_autoscoring_voila.ipynb'
GAIN = {'emg': 100, 'ecg': 100}


def gained(gain, **kwargs):
    """Tool 2's entry of test_autoscoring_formats with a context_gain block."""
    return {**entry(**kwargs), 'context_gain': dict(gain)}


def aux_rows(duration_s, factor=1.0, seed=3):
    """An ECG of 1 mV R waves and a chin EMG of 6 uV, both multiplied by `factor` (an adapter's gain)."""
    n = int(duration_s * SFREQ)
    return {'ECG': ecg_like(duration_s, 1000) * factor, 'EMG': np.random.default_rng(seed).normal(0, 6, n) * factor}


# ---------------------------------------------------------------------------------------------------------
# The reading (quick)
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.quick
@pytest.mark.tool2
@pytest.mark.tool3bis
@pytest.mark.tool7
@pytest.mark.tool0
def test_the_context_gain_block_of_tool_2():
    cfg = gained({'emg': 100, 'ecg': 1, 'eog_right': 'x', 'eog_left': 10})
    assert EIO.context_gain(cfg) == {'emg': 100.0}           # 1 = as recorded, a text is ignored, no EOG-L declared
    assert EIO.context_divisors(cfg) == {'EMG': 100.0}
    assert EIO.context_gain(gained({'emg': -2, 'ecg': 0})) == {}
    assert EIO.context_gain(entry()) == {} and EIO.context_gain(None) == {}


@pytest.mark.quick
@pytest.mark.tool3bis
@pytest.mark.tool7
@pytest.mark.tool0
def test_a_context_channel_is_divided_before_any_derivation(tmp_path):
    path, uv = recording(tmp_path, scale={'IO': 100}, rows=aux_rows(600, 100))
    n = uv.shape[1]
    cfg = gained({'eog_right': 100, 'emg': 100, 'ecg': 100})
    # the models' EOG: divided, then referenced like the EEG (IO / 100 - TP9)
    data, names, notes, _ = A.load_eog(path, cfg, SFREQ, n, 'Right only', A.REFERENCE_CONTRALATERAL)
    np.testing.assert_allclose(data[0] * 1e6, row(uv, 'IO') / 100 - row(uv, 'TP9'), rtol=1e-5, atol=1e-6)
    assert 'EOG-R (IO) divided by 100 (tool 2)' in notes
    # YASA's and SleepFM's read
    _, _, extra, _, notes, _, _ = A.read_signals(path, cfg, ['C4'], A.REFERENCE_CONTRALATERAL,
                                                 {'eog': 'IO', 'emg': 'EMG'}, eog_keys={'eog': 'EOG-R'})
    np.testing.assert_allclose(extra['eog'] * 1e6, row(uv, 'IO') / 100 - row(uv, 'TP9'), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(extra['emg'] * 1e6, row(uv, 'EMG') / 100, rtol=1e-5, atol=1e-6)
    assert 'emg (EMG) divided by 100 (tool 2)' in notes
    # tool 7's companion and tool 0 (read_context_raw): the same channels, 100 times smaller
    ctx = entry()['context_channels']
    as_recorded = EIO.read_context_raw(path, ctx, SFREQ, entry())
    divided = EIO.read_context_raw(path, ctx, SFREQ, cfg)
    assert divided.ch_names == as_recorded.ch_names == ['EOG-R', 'EMG', 'ECG']
    np.testing.assert_allclose(divided.get_data() * 100, as_recorded.get_data(), rtol=1e-6, atol=1e-12)


@pytest.mark.quick
@pytest.mark.tool3bis
@pytest.mark.tool7
@pytest.mark.tool0
def test_the_checks_judge_the_signal_after_the_correction(tmp_path):
    path, _ = recording(tmp_path, scale={'IO': 20}, rows=aux_rows(600, 100))
    eog = {'EOG-R': 'IO'}
    before = A.channel_checks(path, entry(), {}, eog, ('ecg', 'emg'))
    assert len(before) == 3, before
    assert 'about 100 times the usual 1 mV' in before[1] and 'about 100 times the usual 5 uV' in before[2]
    assert A.channel_checks(path, gained({'eog_right': 20, **GAIN}), {}, eog, ('ecg', 'emg')) == []
    # a factor that corrects only part of the gain: the warning says what it judged
    partly = CC.context_amplitude_warnings(path, gained({'ecg': 10}), ('ecg',))
    assert len(partly) == 1 and 'after the correction set in tool 2 (divided by 10), about 10 times' in partly[0]


@pytest.mark.quick
@pytest.mark.tool2
def test_tool_2_proposes_the_factor_and_never_sets_it(tmp_path):
    path, _ = recording(tmp_path, rows=aux_rows(600, 100))
    measured = CC.context_amplitude(path, 'ECG', 'ecg')
    assert CC.decade(measured['ecg_mV']) == 100 and 60 < measured['rate'] < 80
    assert CC.context_amplitude(path, 'absent', 'ecg') is None
    ecg = [{'ecg_mV': 59.4, 'rate': 70}, {'ecg_mV': 124.0, 'rate': 65}, None]
    kind, text = CC.context_gain_advice('ecg', 'ECG', 'ECG', ecg)
    assert kind == 'warn' and 'R waves of 59.4-124 mV in 2/2 recording(s), about 100 times the usual 1 mV' in text
    assert text.endswith('Set Divide by to 100?')
    assert CC.context_gain_advice('ecg', 'ECG', 'ECG', ecg, 100) == (
        'ok', 'ECG (ECG) divided by 100: R waves of 0.594-1.24 mV in 2 recording(s) (usual: 0.1-5 mV).')
    kind, text = CC.context_gain_advice('ecg', 'ECG', 'ECG', ecg, 10)        # a factor too small: the rest proposed
    assert kind == 'warn' and 'Set Divide by to 100?' in text
    kind, text = CC.context_gain_advice('emg', 'Chin EMG', 'EMG', [{'emg_uV': 4.0}, {'emg_uV': 6.5}])
    assert kind == 'ok' and 'RMS of 4-6.5 uV in 2 recording(s) (usual: up to 100 uV)' in text
    assert CC.context_gain_advice('ecg', 'ECG', 'ECG', [None, None])[0] == 'info'
    kind, text = CC.context_gain_advice('ecg', 'ECG', 'ECG', [{'ecg_mV': np.nan, 'rate': np.nan}])
    assert kind == 'info' and 'no regular beat found' in text


@pytest.mark.quick
@pytest.mark.tool3bis
def test_tool_3bis_compares_the_correction_its_model_receives():
    cfg = gained({'eog_right': 2, **GAIN})
    assert A.model_context_gain('GSSC', cfg) == {'eog_right': 2.0}
    assert A.model_context_gain('YASA', cfg) == {'eog_right': 2.0, 'emg': 100.0}
    assert A.model_context_gain('SleepFM', cfg) == {'eog_right': 2.0, 'emg': 100.0, 'ecg': 100.0}
    before = A.stored_scoring_settings('YASA', {'eeg_used': 'C4-M1', 'eog_choice': 'Right', 'emg_choice': 'x'})
    assert before['context channels divided by'] == 'none'               # a file written before the correction
    now = dict(before, **{'context channels divided by': A.context_gain_text(A.model_context_gain('YASA', cfg))})
    assert A.scoring_settings_mismatch(before, now) == 'context channels divided by none → emg 100, eog_right 2'


# ---------------------------------------------------------------------------------------------------------
# The tools (standard): two recordings of a same signal, the second with its ECG and EMG 100 times too large and the
# factor set in tool 2
# ---------------------------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def twin_study(tmp_path_factory):
    """bvA as it should be, bvB the same signal with its ECG and chin EMG 100 times too large, corrected in tool 2."""
    data = tmp_path_factory.mktemp('gain') / 'data'
    stages = ['W'] * 20 + ['N1'] * 10 + ['N2'] * 30
    for pid, factor in (('bvA', 1), ('bvB', 100)):
        recording(data, pid, duration_s=len(stages) * 30, seed=1, rows=aux_rows(len(stages) * 30, factor))
        (data / f'{pid}_hypno_remapped.txt').write_text('\n'.join(stages) + '\n', encoding='utf-8')
    (data / 'config_param').mkdir()
    configs = {'bvA': entry(), 'bvB': gained(GAIN)}
    (data / 'config_param' / 'channel_config.json').write_text(json.dumps(configs, indent=2), encoding='utf-8')
    return data


@pytest.mark.tool2
def test_tool2_checks_the_amplitude_and_saves_the_factor(tmp_path):
    for pid in ('bv01', 'bv02'):
        recording(tmp_path, pid, seed=int(pid[-1]), rows=aux_rows(600, 100))
    with NotebookSession(T2) as nb:
        nb.pick('chooser', tmp_path)
        nb.click('run_button')
        label = nb.value('list(STATE.ui["context_gain"])')[0]
        field = f'STATE.ui["context_gain"][{label!r}]'
        advice = f'STATE.ui["context_advice"][{label!r}].value'
        before = nb.value(advice)
        nb.set(f'{field}["ecg"]', 100.0)
        nb.set(f'{field}["emg"]', 100.0)
        after = nb.value(advice)
        assert nb.value(f'{field}["ecg"].value') == 100.0           # what the user typed, never changed by the check
        nb.click_labelled('Save context selection')
        nb.click_labelled('Save selection')
        nb.click_labelled('Save remapping')
        nb.click_labelled('Save re-reference')
        nb.click_labelled('Preview & Save')
        saved = json.loads(Path(nb.value('STATE.json_path')).read_text(encoding='utf-8'))
    assert 'about 100 times the usual 1 mV' in before and 'Set Divide by to 100?' in before, before
    assert 'about 100 times the usual 5 uV' in before
    assert 'ECG (ECG) divided by 100: R waves of' in after and 'Set Divide by' not in after, after
    for pid in ('bv01', 'bv02'):
        assert saved[pid]['context_gain'] == {'emg': 100, 'ecg': 100}


@pytest.mark.tool2
def test_tool2_writes_no_factor_left_at_1(tmp_path):
    recording(tmp_path, 'bv01', rows=aux_rows(600))
    with NotebookSession(T2) as nb:
        nb.pick('chooser', tmp_path)
        nb.click('run_button')
        label = nb.value('list(STATE.ui["context_gain"])')[0]
        advice = nb.value(f'STATE.ui["context_advice"][{label!r}].value')
        for button in ('Save context selection', 'Save selection', 'Save remapping', 'Save re-reference', 'Preview & Save'):
            nb.click_labelled(button)
        saved = json.loads(Path(nb.value('STATE.json_path')).read_text(encoding='utf-8'))
    assert 'R waves of' in advice and 'Set Divide by' not in advice, advice
    assert 'context_gain' not in saved['bv01'] and 'context_channels' in saved['bv01']


@pytest.mark.tool3bis
def test_tool3bis_scores_the_corrected_emg_like_the_right_one(twin_study):
    data = twin_study
    with NotebookSession(T3BIS) as nb:
        nb.pick('fc_data', data)
        nb.set('cb_gssc', False)
        nb.set('cb_yasa', True)
        nb.click('btn_scan')
        nb.set('dd_yasa_eog', 'Right')
        nb.click('btn_run')
        text = nb.widget_text()
    assert 'Done: 2 scoring(s) done, 0 failed.' in text, text[-3000:]
    params = {pid: json.loads((data / 'derivatives' / 'autoscoring_yasa' / f'{pid}_scoring_params.json')
                              .read_text(encoding='utf-8')) for pid in ('bvA', 'bvB')}
    assert 'context_gain' not in params['bvA']
    assert params['bvB']['context_gain'] == {'emg': 100.0}                 # YASA receives no ECG
    assert 'emg (EMG) divided by 100 (tool 2)' in params['bvB']['notes']
    assert 'channel_quality_warnings' not in params['bvB']                 # the EMG judged after the correction
    hypno = {pid: (data / f'{pid}_hypno_yasa.txt').read_text(encoding='utf-8') for pid in ('bvA', 'bvB')}
    assert hypno['bvA'] == hypno['bvB']
    proba = {pid: pd.read_csv(data / 'derivatives' / 'autoscoring_yasa' / f'{pid}_hypnodensity.tsv', sep='\t')
             for pid in ('bvA', 'bvB')}
    # The two files round the EMG differently (float32 in mV, 1 and 100 times larger): YASA's classifier is a set of trees,
    # so a difference of 1e-7 can fall on the other side of a split. The stages are the same, the probabilities within 0.01.
    num = proba['bvA'].select_dtypes('number').columns
    np.testing.assert_allclose(proba['bvA'][num].to_numpy(), proba['bvB'][num].to_numpy(), atol=0.01)


@pytest.mark.tool7
def test_tool7_divides_the_context_companion_and_records_it(twin_study):
    data = twin_study
    with NotebookSession(pipeline.T7) as nb:      # no tool 6: the channels come from tool 2's configuration
        nb.pick('fc_edf', data)
        nb.click('btn_load')
        nb.click('btn_run')
        text = nb.widget_text()
    assert 'Traceback' not in text, text[-3000:]
    assert '[bvB] Context channels divided by the factor set in tool 2: emg 100, ecg 100' in text
    assert '[bvA] Context channels divided' not in text and 'about 100 times' not in text
    epochs_dir = data / 'derivatives' / 'preproc_epo'
    ctx = {pid: mne.read_epochs(epochs_dir / f'{pid}_context-epo.fif', preload=True, verbose=False)
           for pid in ('bvA', 'bvB')}
    for ch in ('EMG', 'ECG'):
        np.testing.assert_allclose(ctx['bvB'].get_data(picks=[ch]), ctx['bvA'].get_data(picks=[ch]), rtol=1e-4, atol=1e-10)
    side = {pid: json.loads((epochs_dir / f'{pid}_preprocessing_params.json').read_text(encoding='utf-8'))
            for pid in ('bvA', 'bvB')}
    assert 'context_gain' not in side['bvA'] and side['bvB']['context_gain'] == {'emg': 100.0, 'ecg': 100.0}


@pytest.mark.tool0
def test_tool0_shows_the_corrected_context_channels(twin_study):
    data = twin_study
    largest = "float(np.abs(STATE['ctx']['data_uV'][:, STATE['ctx']['labels'].index('{}')]).max())"
    seen = {}
    with NotebookSession(T0) as nb:
        nb.pick('fc_data', data)
        for pid in ('bvA', 'bvB'):
            nb.set('dd_part', pid)
            nb.click('btn_run1')
            assert nb.value("STATUS['1']") == 'up_to_date', nb.widget_text()[-2000:]
            seen[pid] = {ch: nb.value(largest.format(ch)) for ch in ('EMG', 'ECG')}
            seen[pid]['notes'] = nb.value("STATE['notes']")
    for ch in ('EMG', 'ECG'):
        assert seen['bvB'][ch] == pytest.approx(seen['bvA'][ch], rel=1e-4)
    assert any('Divided by the factor set in tool 2' in n and 'EMG by 100' in n for n in seen['bvB']['notes'])
    assert not any('Divided by the factor' in n for n in seen['bvA']['notes'])
