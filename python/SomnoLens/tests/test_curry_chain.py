"""The tools on Curry 9 recordings written by tests/curry_files.py, with invented values (phase 6 of
tools/plan_brainvision.md: the Curry twins removed, every tool reads a .cdt through recording_lib). Quick: the reading
functions of edf_io_lib / macrostructure_lib on a .cdt. Standard: tools 1, 4, 5, 6, 7 and 0 on a small nap study (32
EEG would be a cap; four are enough here, with an EOG, an EMG, an ECG and a belt in the group of the other channels).
The two real Curry recordings of tools/test_data_curry were compared with the former twins' reading once, by hand: same
channels, signal, positions, start, impedances, hypnograms and events."""
import json
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
import edf_io_lib as EIO             # noqa: E402
import macrostructure_lib as ML      # noqa: E402
import pipeline                      # noqa: E402
from curry_files import write_curry  # noqa: E402
from nbdriver import NotebookSession  # noqa: E402

T0 = 'tools/0_live_explore_1file_voila.ipynb'
T1 = 'tools/1_inspect_edf_voila.ipynb'
T4 = 'tools/4_remap_events_edf_voila.ipynb'
SFREQ = 250
EEG = ['Fp1', 'C3', 'C4', 'O1', 'O2', 'TP9', 'TP10']
OTHERS = ['EOG_R', 'EMG', 'ECG', 'Resp']
N_EPOCHS = 80                                    # a 40-min nap
START = (2026, 1, 2, 22, 4, 5, 123)              # 22:04:05.123 in the .dpo
EVENT_CLOCKS = ['22:14:05', '22:24:10']          # 599.877 s and 1204.877 s after the start (the milliseconds kept)
EVENT_ONSETS = [599.877, 1204.877]
EVENT_LABEL = 'Micro-éveil 2 ARO SPONT'
# one measuring session, in Ohm: C4 above tool 1's 20 kOhm, O2 never measured, the auxiliary inputs not measured
IMPEDANCES = [[4000, 6000, 35000, 5000, -1, 3000, 3000, -1, -1, -1, -1]]


def nap_stages():
    stages = ['W'] * N_EPOCHS
    stages[30:40] = ['N1'] * 10
    stages[40:50] = ['N2'] * 10
    return stages


def write_recording(folder, name, seed=0):
    """A nap: noise of 20 uV everywhere, the EOG equal to TP9. Returns the .cdt path."""
    rng = np.random.default_rng(seed)
    uv = rng.normal(0, 20, (len(EEG) + len(OTHERS), N_EPOCHS * 30 * SFREQ))
    uv[len(EEG) + OTHERS.index('EOG_R')] = uv[EEG.index('TP9')]
    return write_curry(folder, name, uv, EEG, OTHERS, sfreq=SFREQ, start=START, impedances=IMPEDANCES)


def entry():
    """Tool 2's entry: the EEG, TP9 as reference, the context channels."""
    return {'config': 'config. 1', 'remap': {ch: ch for ch in EEG}, 'ref_channels': ['TP9'],
            'context_channels': {'eog_right': 'EOG_R', 'emg': 'EMG', 'ecg': 'ECG'}}


def write_events(folder, name):
    """The Compumedics French text export (UTF-16 with BOM): clock time, epoch, stage, label, duration, ..."""
    lines = [f'{clock},{20 + i},Eveil,{EVENT_LABEL},0:05.0,-,-,Dos' for i, clock in enumerate(EVENT_CLOCKS)]
    (Path(folder) / f'{name}_ScoredEvents_Export.txt').write_bytes(('\r\n'.join(lines) + '\r\n').encode('utf-16'))


# ---------------------------------------------------------------------------------------------------------
# The reading functions on a .cdt (quick)
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.quick
@pytest.mark.tool5
@pytest.mark.tool7
@pytest.mark.tool0
def test_the_reads_of_tools_0_5_7_on_a_curry_recording(tmp_path):
    path = write_recording(tmp_path, 'rec')
    raw = EIO.read_eeg_raw(path, entry()['remap'])
    assert raw.ch_names == EEG and set(raw.get_channel_types()) == {'eeg'}
    assert EIO.phys_bounds_for_remap(path, entry()['remap'])[0] == {}       # no declared range
    context = EIO.read_context_raw(path, entry()['context_channels'], SFREQ, entry())
    assert context.ch_names == ['EOG-R', 'EMG', 'ECG']
    write_events(tmp_path, 'rec')
    events, source = EIO.load_events(path)
    assert source == 'txt' and np.allclose(events['Start'], EVENT_ONSETS)  # against the start of the .dpo, to the ms
    header = ML.read_header_info(path)
    assert header['start_dt'] == pd.Timestamp('2026-01-02 22:04:05.123').to_pydatetime()
    assert header['duration_s'] == N_EPOCHS * 30 and header['ch_names'] == EEG + OTHERS


# ---------------------------------------------------------------------------------------------------------
# The tools on a nap study (standard)
# ---------------------------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def curry_study(tmp_path_factory):
    """Two Curry naps with their hypnograms and event exports, tool 2's file, tool 4's event remap, a participant table:
    the layout the test chain uses."""
    data = tmp_path_factory.mktemp('curry') / 'data'
    ids = ['cur01', 'cur02']
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


@pytest.mark.tool1
def test_tool1_on_curry(curry_study):
    data, ids = curry_study
    with NotebookSession(T1) as nb:
        nb.pick('chooser', data)
        nb.click('run_button')
        text = nb.widget_text()
    assert 'Traceback' not in text, text[-3000:]
    out = data / 'reports_inspection'
    full = pd.read_csv(out / 'FULL_summary_table_edf.tsv', sep='\t')
    assert sorted(full['subject'].unique()) == ids and set(full['channel']) == set(EEG + OTHERS)
    assert pd.isna(full['res_theoretical']).all()                          # float samples: no fixed step
    assert 'Not applicable to 2 Curry recording(s)' in text
    assert 'Curry (2 recording(s))' in text and 'names no online reference' in text
    flags = pd.read_csv(out / 'impedance_flags.tsv', sep='\t')
    status = flags[flags['subject'] == 'cur01'].set_index('electrode')['status'].to_dict()
    assert status == {'C4': 'above 20 kOhm', 'O2': 'not measured'}           # the auxiliary inputs are not listed
    notes = pd.read_csv(out / 'recording_notes.tsv', sep='\t')
    assert set(notes['format']) == {'Curry'}


@pytest.mark.tool4
def test_tool4_finds_the_events_of_curry_recordings(curry_study):
    data, ids = curry_study
    with NotebookSession(T4) as nb:
        nb.pick('chooser', data)
        nb.click('run_scan_button')
        text = nb.widget_text()
    assert f'Scanned {len(ids)} recording(s): {len(ids)} with events, 0 failed.' in text, text[-2000:]


@pytest.mark.tool5
def test_tool5_on_curry_naps(curry_study):
    data, ids = curry_study
    text = pipeline.run_tool5(data)
    assert 'Traceback' not in text, text[-3000:]
    table = pd.read_csv(data / 'reports_features_macrostructure' / 'global_sleep_metrics.tsv', sep='\t')
    assert sorted(table['file_id']) == ids
    assert table['recording_start'].astype(str).str.startswith('2026-01-02 22:04:05').all()      # from the .dpo


@pytest.fixture(scope='module')
def preprocessed(curry_study):
    """Tools 6 then 7 (event flagging on) on the Curry naps."""
    data, ids = curry_study
    with NotebookSession(pipeline.T6) as nb:
        nb.pick('fc_folder', data)
        acquisition = nb.value('acq_info.value')
        hp = nb.value('hp_check.value')
    text6 = pipeline.run_tool6(data)
    text7 = pipeline.run_tool7(data, events=True)
    return data, ids, text6, text7, acquisition, hp


@pytest.mark.tool6
def test_tool6_on_curry_naps(preprocessed):
    data, ids, text6, _, acquisition, hp = preprocessed
    assert 'Traceback' not in text6, text6[-3000:]
    assert 'not stated' in acquisition and hp is False        # a .dpo states no filter: the high-pass stays unticked
    for pid in ids:
        metrics = pd.read_csv(data / 'reports_quality_overview' / f'{pid}_quality_metrics.tsv', sep='\t')
        assert sorted(metrics['channel']) == sorted(EEG) and metrics['bounds_pct'].isna().all()   # no declared range


@pytest.mark.tool7
def test_tool7_on_curry_naps(preprocessed):
    data, ids, text6, text7, _, _ = preprocessed
    assert 'Traceback' not in text7, text7[-3000:]
    epochs_dir = data / 'derivatives' / 'preproc_epo'
    for pid in ids:
        epochs = mne.read_epochs(epochs_dir / f'{pid}_all-epo.fif', preload=False, verbose=False)
        assert len(epochs) == N_EPOCHS and epochs.ch_names == [ch for ch in EEG if ch != 'TP9']   # TP9: the reference
        assert epochs.get_montage() is not None                      # the positions of the .dpo, for tool 8's maps
        context = mne.read_epochs(epochs_dir / f'{pid}_context-epo.fif', preload=True, verbose=False)
        assert context.ch_names == ['EOG-R', 'EMG', 'ECG'] and len(context) == N_EPOCHS
        onsets = pd.read_csv(epochs_dir / f'{pid}_event_onsets.tsv', sep='\t')
        assert np.allclose(onsets['onset_s'], EVENT_ONSETS)


@pytest.mark.tool0
def test_tool0_loads_a_curry_recording(curry_study):
    data, _ = curry_study
    with NotebookSession(T0) as nb:
        nb.pick('fc_data', data)
        nb.set('dd_part', 'cur01')
        nb.set('dd_hypno', '_hypno_remapped.txt')
        nb.click('btn_run1')
        assert nb.value("STATUS['1']") == 'up_to_date', nb.widget_text()[-2000:]
        assert nb.value("STATE['P']['ch_names']") == EEG
        assert nb.value("len(STATE['events'])") == 2
