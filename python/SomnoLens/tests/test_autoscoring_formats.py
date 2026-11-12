"""Tool 3bis on any recording format: the reading of autoscoring_lib through recording_lib and the checks of the channels
the models receive (channel_check_lib), on BrainVision recordings written by hand with invented values. The mastoids of
a 10-10 cap (TP9 / TP10), the online reference recreated only where a derivation needs it, the EOG of an EEG input
referenced like the EEG, the belt found for SleepFM, the checks. Then tool 3bis itself on such recordings, and the proof
that a same signal is scored the same in EDF and in BrainVision."""
import json
import shutil
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
import autoscoring_lib as A          # noqa: E402
import channel_check_lib as CC       # noqa: E402
import minidb                        # noqa: E402
from brainvision_files import recorder_comment, write_brainvision    # noqa: E402
from nbdriver import NotebookSession                                 # noqa: E402

T3BIS = 'tools/3bis_autoscoring_voila.ipynb'
SFREQ = 250
# 5 EEG inputs and an eye electrode on an EEG input (all against the online reference Cz, not stored), then the auxiliary
# inputs, which carry a gradient in the [Comment]
NAMES = ['Fp1', 'C3', 'C4', 'TP9', 'TP10', 'IO', 'EMG', 'ECG', 'Respi']
RES = [0.05] * 6 + [1, 1, 1]
UNITS = ['µV'] * 6 + ['mV', 'mV', 'UA']
GRADIENTS = {'EMG': '1[mV/mV]', 'ECG': '1[mV/mV]', 'Respi': '1[mV/UA]'}
COORDS = {'Fp1': (1, -92, -72), 'C3': (1, -46, 0), 'C4': (1, 46, 0), 'TP9': (1, -115, -18), 'TP10': (1, 115, 18),
          'IO': (0, 0, 0), 'EMG': (0, 0, 0), 'ECG': (0, 0, 0), 'Respi': (0, 0, 0)}
CONTEXT = {'eog_right': 'IO', 'emg': 'EMG', 'ecg': 'ECG'}


def recording(folder, name='rec', duration_s=600, scale=None, seed=0, comment=True, rows=None):
    """A recording where every channel is a noise of 20 uV, `scale` = {channel: factor}, `rows` = {channel: values in uV}
    in place of the noise. Returns (.vhdr path, the values in uV)."""
    rng = np.random.default_rng(seed)
    uv = rng.normal(0, 20, (len(NAMES), int(duration_s * SFREQ)))
    for ch, values in (rows or {}).items():
        uv[NAMES.index(ch)] = values
    for ch, factor in (scale or {}).items():
        uv[NAMES.index(ch)] *= factor
    stored = uv / np.array(RES)[:, None]
    stored[6:] = uv[6:] * 1e-3          # the auxiliary inputs, in mV at a resolution of 1
    text = recorder_comment(NAMES, RES, UNITS, gradients=GRADIENTS) if comment else None
    path = write_brainvision(folder, name, stored, NAMES, sfreq=SFREQ, resolutions=RES, units=UNITS, coordinates=COORDS,
                             comment=text)
    # what the file holds once read back (float32)
    read = stored.astype(np.float32).astype(float) * np.array(RES)[:, None]
    read[6:] = stored[6:].astype(np.float32).astype(float) * 1e3
    return path, read


def entry(ref_channels=(), remap=None):
    """Tool 2's entry of a participant: the EEG under their own names, Cz (the online reference) kept, the context."""
    remap = remap or {ch: ch for ch in ['Fp1', 'C3', 'C4', 'TP9', 'TP10', 'Cz']}
    return {'remap': dict(remap), 'ref_channels': list(ref_channels), 'context_channels': dict(CONTEXT)}


def row(values, ch):
    return values[NAMES.index(ch)]


# ---------------------------------------------------------------------------------------------------------
# Electrodes, mastoids, reference
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.quick
@pytest.mark.tool3bis
@pytest.mark.tool0
def test_tp9_tp10_are_the_mastoids_of_a_10_10_cap_and_m1_m2_come_first():
    assert A.contralateral_refs('C3', ['C3', 'TP9', 'TP10']) == ['TP10']
    assert A.contralateral_refs('C4', ['C4', 'TP9', 'TP10']) == ['TP9']
    assert A.contralateral_refs('C3', ['C3', 'M2', 'TP10']) == ['M2']           # an EDF with M1 / M2 is unchanged
    assert A.derivation_label('C3', ['TP10']) == 'C3-M2'
    assert (A.reference_text(A.REFERENCE_CONTRALATERAL, {'C3': ['TP10'], 'C4': ['TP9']}, entry())
            == 'contralateral mastoid: C3-M2, C4-M1 (M1 = TP9, M2 = TP10)')
    assert (A.reference_text(A.REFERENCE_CONTRALATERAL, {'C3': ['M2']}, entry())
            == 'contralateral mastoid: C3-M2')


@pytest.mark.quick
@pytest.mark.tool3bis
@pytest.mark.tool0
def test_the_electrodes_offered_leave_out_mastoids_eog_and_context_channels():
    cfg = entry(remap={'Fp1': 'Fp1', 'C3': 'C3', 'TP9': 'TP9', 'TP10': 'TP10', 'Cz': 'Cz', 'IO': 'IO', 'A1': 'M1'})
    assert A.eeg_options(cfg) == ['Fp1', 'C3', 'Cz']


@pytest.mark.quick
@pytest.mark.tool3bis
@pytest.mark.tool0
def test_the_online_reference_is_recreated_only_for_a_derivation_that_needs_it(tmp_path):
    path, uv = recording(tmp_path)
    raw, present, refs, problem = A.load_eeg_contralateral(path, entry(), ['C3', 'C4'])
    assert (present, refs, problem) == (['C3', 'C4'], {'C3': ['TP10'], 'C4': ['TP9']}, '')
    data = raw.get_data() * 1e6
    np.testing.assert_allclose(data[raw.ch_names.index('C3')], row(uv, 'C3') - row(uv, 'TP10'), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(data[raw.ch_names.index('C4')], row(uv, 'C4') - row(uv, 'TP9'), rtol=1e-5, atol=1e-6)
    # Cz, a midline electrode, is the online reference: recreated (zeros), then on both mastoids
    raw, present, refs, _ = A.load_eeg_contralateral(path, entry(), ['Cz'])
    assert refs == {'Cz': ['TP9', 'TP10']}
    np.testing.assert_allclose(raw.get_data()[0] * 1e6, -(row(uv, 'TP9') + row(uv, 'TP10')) / 2, rtol=1e-5, atol=1e-6)
    # tool 2's reference: linked mastoids
    raw, present = A.load_eeg(path, entry(['TP9', 'TP10']), ['C3'])
    np.testing.assert_allclose(raw.get_data()[0] * 1e6, row(uv, 'C3') - (row(uv, 'TP9') + row(uv, 'TP10')) / 2,
                               rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------------------------------------
# The EOG
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.quick
@pytest.mark.tool3bis
@pytest.mark.tool0
def test_an_eog_on_an_eeg_input_gets_the_reference_of_the_eeg(tmp_path):
    path, uv = recording(tmp_path)
    n = uv.shape[1]
    data, names, notes, description = A.load_eog(path, entry(), SFREQ, n, 'Right only', A.REFERENCE_CONTRALATERAL)
    assert names == ['EOG-R'] and notes == []
    assert description == 'EOG-R = IO - TP9 (IO recorded against the online reference Cz)'
    np.testing.assert_allclose(data[0] * 1e6, row(uv, 'IO') - row(uv, 'TP9'), rtol=1e-5, atol=1e-6)
    # tool 2's reference
    data, _, _, description = A.load_eog(path, entry(['TP9', 'TP10']), SFREQ, n, 'Right only', A.REFERENCE_DECLARED)
    assert description.startswith('EOG-R = IO - mean of TP9, TP10')
    np.testing.assert_allclose(data[0] * 1e6, row(uv, 'IO') - (row(uv, 'TP9') + row(uv, 'TP10')) / 2, rtol=1e-5, atol=1e-6)
    # YASA's and SleepFM's read: the same derivation
    sfreq, eeg, extra, _, _, refs, description = A.read_signals(
        path, entry(), ['C4'], A.REFERENCE_CONTRALATERAL, {'eog': 'IO', 'emg': 'EMG'}, eog_keys={'eog': 'EOG-R'})
    assert list(eeg) == ['C4-M1'] and refs == {'C4': ['TP9']}
    assert description == 'EOG-R = IO - TP9 (IO recorded against the online reference Cz)'
    np.testing.assert_allclose(extra['eog'] * 1e6, row(uv, 'IO') - row(uv, 'TP9'), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(extra['emg'] * 1e6, row(uv, 'EMG'), rtol=1e-5, atol=1e-6)     # the EMG untouched


@pytest.mark.quick
@pytest.mark.tool3bis
@pytest.mark.tool0
def test_an_eog_is_left_as_recorded_when_nothing_says_it_is_against_the_online_reference(tmp_path):
    path, uv = recording(tmp_path / 'a')
    eog = {'EOG-R': 'IO'}
    # the L - R derivation of two EOG: the common term cancels
    assert A.eog_reference(path, entry(), eog, A.REFERENCE_CONTRALATERAL, bipolar_pair=True) == ({}, '', [])
    # tool 2 declared no reference: the EEG stays against the online reference, so does the EOG
    assert A.eog_reference(path, entry(), eog, A.REFERENCE_DECLARED) == ({}, '', [])
    # no mastoid of the other side: said
    refs, description, notes = A.eog_reference(path, entry(remap={'C3': 'C3', 'TP10': 'TP10'}), eog,
                                               A.REFERENCE_CONTRALATERAL)
    assert refs == {} and description == '' and 'used as recorded' in notes[0]
    # a recording that names no online reference (no [Comment]), and an EOG on an auxiliary input
    path, _ = recording(tmp_path / 'b', comment=False)
    assert A.eog_reference(path, entry(), eog, A.REFERENCE_CONTRALATERAL) == ({}, '', [])
    path, _ = recording(tmp_path / 'c')
    assert A.eog_reference(path, entry(), {'EOG-R': 'EMG'}, A.REFERENCE_CONTRALATERAL) == ({}, '', [])


@pytest.mark.quick
@pytest.mark.tool3bis
def test_a_single_belt_is_given_to_sleepfm_as_abdominal(tmp_path):
    path, _ = recording(tmp_path)
    assert A.find_resp_channels(path) == {'Abdominal': 'Respi'}


# ---------------------------------------------------------------------------------------------------------
# Channel checks
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.quick
@pytest.mark.tool3bis
@pytest.mark.tool0
def test_the_median_comparison():
    stds = np.ones((10, 5)) * 20
    stds[:, 1] = 200          # noisy
    stds[:3, 2] = 0.2         # nearly flat on 30 % of the epochs
    noisy, dead = CC.compared_with_median(stds, [0, 1, 2, 3, 4])
    assert noisy.tolist() == [0, 1, 0, 0, 0] and dead.tolist() == [0, 0, 0.3, 0, 0]
    assert CC.compared_with_median(stds[:, :2], [0, 1]) is None          # no median with 2 channels
    assert [CC.decade(v) for v in (72, 124, 59.4, 0.03)] == [100, 100, 100, 0.01]


@pytest.mark.quick
@pytest.mark.tool3bis
@pytest.mark.tool0
def test_a_noisy_electrode_a_dead_mastoid_and_a_noisy_eog_are_said(tmp_path):
    path, _ = recording(tmp_path / 'bad', scale={'C3': 10, 'TP10': 0.01, 'IO': 20})
    warnings = A.channel_checks(path, entry(), {'C3': ['TP10'], 'C4': ['TP9']}, {'EOG-R': 'IO'})
    assert len(warnings) == 3, warnings
    assert warnings[0].startswith('C3 (electrode): noisy on 100% of 20 epochs')
    assert warnings[1].startswith('TP10 (reference of C3): nearly flat on 100% of 20 epochs')
    assert warnings[2].startswith('EOG-R (IO): median standard deviation of')
    path, _ = recording(tmp_path / 'good')
    assert A.channel_checks(path, entry(), {'C3': ['TP10'], 'C4': ['TP9']}, {'EOG-R': 'IO'}) == []


def ecg_like(duration_s, amplitude_uv, rate=70):
    """Beats of `amplitude_uv` (a narrow Gaussian, the QRS) at `rate` per minute, on a small noise."""
    t = np.arange(int(duration_s * SFREQ)) / SFREQ
    x = np.random.default_rng(1).normal(0, 0.01 * amplitude_uv, t.size)
    for beat in np.arange(0.3, duration_s, 60 / rate):
        x += amplitude_uv * np.exp(-0.5 * ((t - beat) / 0.012) ** 2)
    return x


@pytest.mark.quick
@pytest.mark.tool3bis
@pytest.mark.tool0
def test_an_ecg_and_a_chin_emg_far_from_their_amplitude(tmp_path):
    t = np.arange(600 * SFREQ) / SFREQ
    mains = 30 * np.sin(2 * np.pi * 50 * t) + np.random.default_rng(2).normal(0, 3, t.size)
    path, _ = recording(tmp_path / 'good', rows={'ECG': ecg_like(600, 1000), 'EMG': np.random.default_rng(3).normal(0, 6, t.size)})
    assert A.channel_checks(path, entry(), {}, {}, ('ecg', 'emg')) == []
    path, _ = recording(tmp_path / 'gain', rows={'ECG': ecg_like(600, 100_000), 'EMG': mains})
    warnings = A.channel_checks(path, entry(), {}, {}, ('ecg', 'emg'))
    assert len(warnings) == 2, warnings
    assert 'about 100 times the usual 1 mV' in warnings[0]
    assert warnings[1].startswith('chin EMG (EMG): the mains (50 Hz) holds')
    path, _ = recording(tmp_path / 'noise')               # a noise on the ECG input: no heart beat
    assert 'does not look like an ECG' in A.channel_checks(path, entry(), {}, {}, ('ecg',))[0]


# ---------------------------------------------------------------------------------------------------------
# Tool 3bis
# ---------------------------------------------------------------------------------------------------------
def write_config(study, ids):
    folder = Path(study) / 'config_param'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'channel_config.json').write_text(json.dumps({pid: entry() for pid in ids}, indent=2), encoding='utf-8')


@pytest.mark.tool3bis
def test_tool3bis_on_brainvision_recordings(tmp_path):
    """GSSC and YASA on two recordings: C3-M2, C4-M1 on TP10 / TP9, IO referenced like the EEG, the noisy C3 of the first
    recording said, nothing to say on the second (its parameters file has no check key)."""
    recording(tmp_path, 'bv01', duration_s=1800, seed=1, scale={'C3': 10})
    recording(tmp_path, 'bv02', duration_s=1800, seed=2)
    for pid in ('bv01', 'bv02'):
        (tmp_path / f'{pid}_hypno_remapped.txt').write_text('W\n' * 60, encoding='utf-8')
    write_config(tmp_path, ['bv01', 'bv02'])
    with NotebookSession(T3BIS) as nb:
        nb.pick('fc_data', tmp_path)
        nb.set('cb_yasa', True)
        nb.click('btn_scan')
        scan = nb.value('scan_info.value')
        emg_choices = nb.value('list(dd_yasa_emg.options)')
        nb.set('dd_yasa_eog', 'Right')
        nb.click('btn_run')
        text = nb.widget_text()
    assert '2 recording(s): 2 configured in tool 2' in scan, scan
    assert 'Cz' not in emg_choices                       # the online reference is not in the files
    assert 'Done: 4 scoring(s) done, 0 failed.' in text, text[-3000:]
    assert 'channel check: C3 (electrode): noisy' in text
    for pid in ('bv01', 'bv02'):
        assert (tmp_path / f'{pid}_hypno_gssc.txt').is_file() and (tmp_path / f'{pid}_hypno_yasa.txt').is_file()
    gssc = json.loads((tmp_path / 'derivatives' / 'autoscoring_gssc' / 'bv01_scoring_params.json').read_text(encoding='utf-8'))
    assert gssc['eeg_reference'] == 'contralateral mastoid: C3-M2, C4-M1 (M1 = TP9, M2 = TP10)'
    assert gssc['eog_reference'] == 'EOG-R = IO - TP9 (IO recorded against the online reference Cz)'
    assert gssc['channel_quality_warnings'][0].startswith('C3 (electrode): noisy')
    yasa = json.loads((tmp_path / 'derivatives' / 'autoscoring_yasa' / 'bv02_scoring_params.json').read_text(encoding='utf-8'))
    assert yasa['eeg_used'] == 'C4-M1' and yasa['eog_used'] == 'IO' and yasa['emg_used'] == 'EMG'
    assert 'channel_quality_warnings' not in yasa
    report = (tmp_path / 'reports_autoscoring_gssc' / 'bv01_autoscoring_report.html').read_text(encoding='utf-8')
    assert 'Channel checks' in report and 'EOG reference (model)' in report


@pytest.mark.tool3bis
def test_tool3bis_scores_a_same_signal_the_same_in_edf_and_in_brainvision(tmp_path):
    """One synthetic night as EDF and as BrainVision (no [Comment]: nothing to recreate, the EOG used as recorded): the same
    GSSC hypnodensity, to the float32 precision of the BrainVision file."""
    pid = minidb.PARTICIPANTS['synthetic'][0]
    edf_study = minidb.build(tmp_path / 'edf', 'synthetic')
    for other in minidb.PARTICIPANTS['synthetic'][1:]:
        for path in edf_study.glob(f'{other}*'):
            path.unlink()
    bv_study = tmp_path / 'bv'
    shutil.copytree(edf_study / 'config_param', bv_study / 'config_param')
    raw = mne.io.read_raw_edf(edf_study / f'{pid}.edf', preload=True, verbose=False)
    write_brainvision(bv_study, pid, raw.get_data() * 1e6, raw.ch_names, sfreq=raw.info['sfreq'],
                      resolutions=[1.0] * len(raw.ch_names))
    shutil.copy2(edf_study / f'{pid}_Hypnogram_remapped.txt', bv_study)
    densities = []
    for study in (edf_study, bv_study):
        with NotebookSession(T3BIS) as nb:
            nb.pick('fc_data', study)
            nb.click('btn_scan')
            nb.click('btn_run')
            text = nb.widget_text()
        assert 'Done: 1 scoring(s) done, 0 failed.' in text, text[-3000:]
        densities.append(pd.read_csv(next(study.rglob(f'{pid}_hypnodensity.tsv')), sep='\t'))
    edf, bv = densities
    assert (edf['stage_gssc'] == bv['stage_gssc']).mean() > 0.99
    probs = [f'p_{s}' for s in ('W', 'N1', 'N2', 'N3', 'R')]
    np.testing.assert_allclose(bv[probs].values, edf[probs].values, atol=2e-3)
