"""The rest of the chain on BrainVision recordings written by hand with invented values (phase 4 of
tools/plan_brainvision.md): the reading functions of edf_io_lib / macrostructure_lib for any format, then tools 4, 5, 6, 7
and 8bis on a small nap study (few N2 epochs, an eye electrode on an EEG input, a scored-event text export)."""
import json
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
import autoreject_lib as AR          # noqa: E402
import edf_io_lib as EIO             # noqa: E402
import macrostructure_lib as ML      # noqa: E402
import recording_lib as REC          # noqa: E402
import pipeline                      # noqa: E402
from brainvision_files import recorder_comment, write_brainvision    # noqa: E402
from nbdriver import NotebookSession                                 # noqa: E402

T4 = 'tools/4_remap_events_edf_voila.ipynb'
SFREQ = 250
# EEG inputs (against the online reference Cz, not stored) and an eye electrode on an EEG input, then the auxiliary inputs
NAMES = ['Fp1', 'C3', 'C4', 'O1', 'O2', 'TP9', 'TP10', 'IO', 'EMG', 'ECG', 'Respi']
RES = [0.05] * 8 + [1, 1, 1]
UNITS = ['µV'] * 8 + ['mV', 'mV', 'UA']
GRADIENTS = {'EMG': '1[mV/mV]', 'ECG': '1[mV/mV]', 'Respi': '1[mV/UA]'}
# Positions for the EEG electrodes only, as Recorder writes them: MNE then types the eye electrode (no position) misc
COORDS = {'Fp1': (1, -92, -72), 'C3': (1, -46, 0), 'C4': (1, 46, 0), 'O1': (1, -92, 72), 'O2': (1, 92, -72),
          'TP9': (1, -115, -18), 'TP10': (1, 115, 18), 'IO': (0, 0, 0), 'EMG': (0, 0, 0), 'ECG': (0, 0, 0),
          'Respi': (0, 0, 0)}
EEG = ['Fp1', 'C3', 'C4', 'O1', 'O2', 'TP9', 'TP10', 'Cz']
N_EPOCHS = 80                        # a 40-min nap
START = '20260102030405123456'       # 2026-01-02 03:04:05 in the .vmrk
EVENT_CLOCKS = ['3:14:05', '3:24:10']   # 600 s and 1205 s after the start
EVENT_LABEL = 'Micro-éveil 2 ARO SPONT'


def nap_stages():
    """Mostly wake, a few N1 / N2 epochs (a nap)."""
    stages = ['W'] * N_EPOCHS
    stages[30:40] = ['N1'] * 10
    stages[40:50] = ['N2'] * 10
    return stages


def write_recording(folder, name, seed=0, extra_segments=()):
    """A nap recording: noise of 20 uV on every input, the eye electrode equal to TP9 (so IO - TP9 is zero). Returns the
    .vhdr path and the values in uV."""
    rng = np.random.default_rng(seed)
    uv = rng.normal(0, 20, (len(NAMES), N_EPOCHS * 30 * SFREQ))
    uv[NAMES.index('IO')] = uv[NAMES.index('TP9')]
    stored = uv / np.array(RES)[:, None]
    stored[8:] = uv[8:] * 1e-3
    path = write_brainvision(folder, name, stored, NAMES, sfreq=SFREQ, resolutions=RES, units=UNITS, start=START,
                             coordinates=COORDS, comment=recorder_comment(NAMES, RES, UNITS, gradients=GRADIENTS),
                             extra_segments=extra_segments)
    return path, uv


def entry(ref_channels=('TP9',)):
    """Tool 2's entry: the EEG, Cz (the online reference) and, as some studies do, the eye electrode kept in the montage
    under another name (MNE types it misc: it has no position)."""
    return {'config': 'config. 1', 'remap': {**{ch: ch for ch in EEG}, 'IO': 'EOG_R'}, 'ref_channels': list(ref_channels),
            'context_channels': {'eog_right': 'IO', 'emg': 'EMG', 'ecg': 'ECG'}}


def write_events(folder, name):
    """The Compumedics French text export (UTF-16 with BOM): clock time, epoch, stage, label, duration, ..."""
    lines = [f'{clock},{20 + i},Eveil,{EVENT_LABEL},0:05.0,-,-,Dos' for i, clock in enumerate(EVENT_CLOCKS)]
    (Path(folder) / f'{name}_ScoredEvents_Export.txt').write_bytes(('\r\n'.join(lines) + '\r\n').encode('utf-16'))


# ---------------------------------------------------------------------------------------------------------
# The reading functions, any format (quick)
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.quick
@pytest.mark.tool5
@pytest.mark.tool7
@pytest.mark.tool0
def test_the_start_of_a_paused_recording_is_not_used(tmp_path):
    path, _ = write_recording(tmp_path / 'a', 'rec')
    assert REC.read_start_datetime(path) == pd.Timestamp('2026-01-02 03:04:05').to_pydatetime()
    write_events(tmp_path / 'a', 'rec')
    events, source = EIO.load_events(path)
    assert source == 'txt' and events['Start'].tolist() == [600.0, 1205.0]
    paused, _ = write_recording(tmp_path / 'b', 'rec', extra_segments=(1000,))
    write_events(tmp_path / 'b', 'rec')
    assert REC.read_start_datetime(paused) is None           # its samples no longer follow the clock
    assert EIO.load_events(paused) == (None, None)           # so the clock times of the export are not placed
    with pytest.raises(ValueError, match='paused'):
        ML.read_header_info(paused)


@pytest.mark.quick
@pytest.mark.tool5
@pytest.mark.tool0
def test_the_header_of_tool_5_for_a_brainvision_recording(tmp_path):
    path, _ = write_recording(tmp_path, 'rec')
    header = ML.read_header_info(path)
    assert header['start_dt'] == pd.Timestamp('2026-01-02 03:04:05').to_pydatetime()
    assert header['duration_s'] == N_EPOCHS * 30 and header['ch_names'] == NAMES       # the online reference left out
    assert header['sfreq_per_ch']['C3'] == SFREQ


@pytest.mark.quick
@pytest.mark.tool7
@pytest.mark.tool0
def test_the_reads_of_tools_0_and_7(tmp_path):
    path, uv = write_recording(tmp_path, 'rec')
    raw = EIO.read_eeg_raw(path, entry()['remap'])
    assert raw.ch_names == EEG[:-1] + ['EOG_R', 'Cz'] and not raw.get_data(picks=['Cz']).any()   # Cz recreated
    assert set(raw.get_channel_types()) == {'eeg'}                       # the eye electrode too, like an EDF read
    assert EIO.phys_bounds_for_remap(path, entry()['remap'])[0] == {}     # no declared range
    # the context companion: the eye electrode gets tool 2's reference (TP9), like the EEG; here IO = TP9
    with_ref = EIO.read_context_raw(path, entry()['context_channels'], SFREQ, entry())
    as_recorded = EIO.read_context_raw(path, entry()['context_channels'], SFREQ, entry(ref_channels=()))
    assert with_ref.ch_names == ['EOG-R', 'EMG', 'ECG'] == as_recorded.ch_names
    assert np.abs(with_ref.get_data(picks=['EOG-R'])).max() < 1e-9 < np.abs(as_recorded.get_data(picks=['EOG-R'])).max()
    np.testing.assert_allclose(with_ref.get_data(picks=['ECG']), as_recorded.get_data(picks=['ECG']))


@pytest.mark.quick
@pytest.mark.tool8bis
@pytest.mark.tool0
def test_the_reference_stages_of_tool_8bis():
    stages = ['W', 'N1', 'N2', 'N3', 'R']
    assert AR.default_reference_stages(stages) == ['N2', 'N3', 'R']
    assert AR.default_reference_stages(stages, [400, 350, 15]) == ['N2', 'N3', 'R']     # nights: unchanged
    assert AR.default_reference_stages(stages, [4, 20, 13]) == stages                   # naps: every stage
    assert AR.default_reference_stages(['W', 'N1'], [0, 0]) == ['W', 'N1']


# ---------------------------------------------------------------------------------------------------------
# The tools on a nap study (standard)
# ---------------------------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def nap_study(tmp_path_factory):
    """Two nap recordings with their hypnograms and event exports, tool 2's file and tool 4's event remap, a participant
    table: the layout the test chain uses."""
    data = tmp_path_factory.mktemp('naps') / 'data'
    ids = ['bv01', 'bv02']
    for k, pid in enumerate(ids):
        write_recording(data, pid, seed=k)
        (data / f'{pid}_hypno_remapped.txt').write_text('\n'.join(nap_stages()) + '\n', encoding='utf-8')
        write_events(data, pid)
    (data / 'config_param').mkdir()
    (data / 'config_param' / 'remap_reref_persubject.json').write_text(
        json.dumps({pid: entry() for pid in ids}, indent=2), encoding='utf-8')
    (data / 'config_param' / 'event_remap.json').write_text(
        json.dumps({EVENT_LABEL: 'arousal_spontaneous'}, ensure_ascii=False), encoding='utf-8')
    pd.DataFrame({'participant_id': ids, 'age': [30, 40]}).to_csv(data / 'participants.tsv', sep='\t', index=False)
    return data, ids


@pytest.mark.tool4
def test_tool4_finds_the_events_of_brainvision_recordings(nap_study):
    data, ids = nap_study
    with NotebookSession(T4) as nb:
        nb.pick('chooser', data)
        nb.click('run_scan_button')
        text = nb.widget_text()
    assert f'Scanned {len(ids)} recording(s): {len(ids)} with events, 0 failed.' in text, text[-2000:]
    assert f'event source: {len(ids)} from TXT' in text


@pytest.mark.tool5
def test_tool5_on_brainvision_naps(nap_study):
    data, ids = nap_study
    text = pipeline.run_tool5(data)
    assert 'Traceback' not in text, text[-3000:]
    table = pd.read_csv(data / 'reports_features_macrostructure' / 'global_sleep_metrics.tsv', sep='\t')
    assert sorted(table['file_id']) == ids
    assert (table['recording_start'] == '2026-01-02 03:04:05').all()       # from the .vmrk


@pytest.fixture(scope='module')
def preprocessed(nap_study):
    """Tools 6 then 7 (event flagging on) on the nap study."""
    data, ids = nap_study
    text6 = pipeline.run_tool6(data)
    text7 = pipeline.run_tool7(data, events=True)
    return data, ids, text6, text7


@pytest.mark.tool7
def test_tool7_on_brainvision_naps(preprocessed):
    data, ids, text6, text7 = preprocessed
    assert 'Traceback' not in text6 + text7, (text6 + text7)[-3000:]
    epochs_dir = data / 'derivatives' / 'preproc_epo'
    for pid in ids:
        epochs = mne.read_epochs(epochs_dir / f'{pid}_all-epo.fif', preload=False, verbose=False)
        assert len(epochs) == N_EPOCHS and 'Cz' in epochs.ch_names          # the online reference, re-referenced
        assert 'EOG_R' in epochs.ch_names and set(epochs.get_channel_types()) == {'eeg'}   # flagged like the EEG
        context = mne.read_epochs(epochs_dir / f'{pid}_context-epo.fif', preload=True, verbose=False)
        assert context.ch_names == ['EOG-R', 'EMG', 'ECG'] and len(context) == N_EPOCHS
        assert np.abs(context.get_data(picks=['EOG-R'])).max() < 1e-9        # IO - TP9, tool 2's reference
        onsets = pd.read_csv(epochs_dir / f'{pid}_event_onsets.tsv', sep='\t')
        assert onsets['onset_s'].tolist() == [600.0, 1205.0]                 # the export's clock times, placed
    assert 'channel check: ECG (ECG): does not look like an ECG' in text7   # the checks ran (a noise on the ECG input)


@pytest.mark.tool8bis
def test_tool8bis_ticks_every_stage_on_naps(preprocessed):
    data, ids, _, _ = preprocessed
    with NotebookSession(pipeline.T8BIS) as nb:
        nb.pick('fc_raw', data / 'derivatives' / 'preproc_epo')
        nb.pick('fc_reports', data / 'reports_preproc')
        nb.click('btn_scan')
        scan = nb.value('lbl_scan.value')
        ticked = nb.value("{st: cb.value for st, cb in S['stage_checks'].items()}")
    assert 'Few epochs in N2 / N3 / R (median 10 per participant' in scan, scan
    assert ticked == {'W': True, 'N1': True, 'N2': True}
