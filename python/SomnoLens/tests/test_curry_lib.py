"""The Curry reader (tools/curry_lib.py) and its branch of recording_lib, on recordings written by tests/curry_files.py
(phase 6 of tools/plan_brainvision.md: the Curry twins removed, Curry read like EDF and BrainVision). Quick."""
import datetime
import sys
from pathlib import Path

import mne
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
import curry_lib as CL              # noqa: E402
import recording_lib as RL          # noqa: E402
import study_files_lib as SF        # noqa: E402
from curry_files import write_curry  # noqa: E402

pytestmark = pytest.mark.quick

EEG = ['Fp1', 'C3', 'C4', 'O1']
OTHERS = ['EOG_R', 'EMG']
SFREQ = 250


def recording(folder, **kwargs):
    data = np.random.default_rng(0).normal(0, 20, (len(EEG) + len(OTHERS), SFREQ * 60))
    kwargs.setdefault('impedances', [[4000, -1, 7000, 3000, -1, -2], [4500, -1, 25000, 3000, -1, -2]])
    return write_curry(folder, 'cur01', data, EEG, OTHERS, sfreq=SFREQ, **kwargs), data


def test_the_header_is_read_from_the_dpo_alone(tmp_path):
    path, _ = recording(tmp_path)
    header = CL.read_cdt_header(path)
    assert header['format'] == 'curry' and header['channel'] == EEG + OTHERS
    # the name first (an EOG may sit in any group), then EEG for the EEG group of the .dpo
    assert header['transducer_type'] == ['EEG', 'EEG', 'EEG', 'EEG', 'EOG', 'EMG']
    assert header['channel_group'] == ['EEG'] * 4 + ['other'] * 2
    assert (header['sfreq'], header['n_times'], header['duration_s']) == (SFREQ, SFREQ * 60, 60.0)
    assert header['sampling_frequency'] == [str(SFREQ)] * 6
    # no declared range (float samples): no bounds_pct, the ±500 µV cap; no filter stated
    assert set(header['physical_min']) == {''} and set(header['prefiltering']) == {''}
    assert (header['start_date'], header['start_time']) == ('02.01.26', '22.04.05')


def test_the_impedances_are_in_kohm_from_the_last_measure(tmp_path):
    path, _ = recording(tmp_path)
    header = CL.read_cdt_header(path)
    # Ohm in the file; the last measured session; an EEG channel never measured listed without a value; the auxiliary
    # inputs never measured left out (they would be flagged in every recording)
    assert header['impedance_kohm'] == {'Fp1': 4.5, 'C3': None, 'C4': 25.0, 'O1': 3.0}
    assert header['impedance_text'] == {'C3': 'not measured'}
    path, _ = recording(tmp_path / 'b', impedances=[[-2, 5000, 5000, 5000, 8000, -2]])
    header = CL.read_cdt_header(path)
    assert header['impedance_text'] == {'Fp1': 'disabled'} and header['impedance_kohm']['EOG_R'] == 8.0


def test_the_signal_reads_back_as_written(tmp_path):
    path, data = recording(tmp_path, events=[(500, 1), (900, 2)])
    assert len(mne.io.read_raw_curry(str(path), verbose='ERROR').annotations) == 2      # the native markers exist
    raw = CL.read_raw_cdt(path, preload=True)
    assert raw.ch_names == EEG + OTHERS and raw.info['sfreq'] == SFREQ
    assert np.allclose(raw.get_data() * 1e6, data.astype('float32'), atol=1e-6)
    assert len(raw.annotations) == 0                    # ... but not carried (as for BrainVision)
    assert raw.get_channel_types() == ['eeg'] * 4 + ['misc'] * 2
    assert raw.get_montage() is not None and raw.get_montage().ch_names == EEG       # positions of the EEG group
    # include: in the order of the file, a name absent from the file ignored (as read_raw_edf)
    assert CL.read_raw_cdt(path, include=['C4', 'Fp1', 'X']).ch_names == ['Fp1', 'C4']


def test_recording_lib_reads_a_cdt_like_the_other_formats(tmp_path):
    path, _ = recording(tmp_path)
    assert RL.file_format(path) == 'curry' and RL.FORMAT_NAMES['curry'] == 'Curry'
    assert RL.read_header(path)['format'] == 'curry'
    assert RL.read_start_datetime(path) == datetime.datetime(2026, 1, 2, 22, 4, 5, 123000)   # to the millisecond
    assert RL.read_raw(path, include=['C3', 'EMG']).ch_names == ['C3', 'EMG']
    assert RL.online_reference(path) == (None, [])     # a .dpo does not say what the channels were recorded against
    assert RL.read_acquisition(path) == (f'{SFREQ} Hz', 'not stated', False)     # tool 6's high-pass stays unticked
    assert RL.channel_table(RL.read_header(path))['channel'].tolist() == EEG + OTHERS


def test_a_cdt_without_its_dpo_fails_clearly(tmp_path):
    path, _ = recording(tmp_path)
    CL.dpo_path(path).unlink()
    with pytest.raises(FileNotFoundError, match='cur01.cdt.dpo'):
        RL.read_header(path)
    assert RL.read_start_datetime(path) is None


def test_the_cdt_is_found_never_its_companions(tmp_path):
    recording(tmp_path / 'sub', events=[(500, 1)])
    (tmp_path / 'sub' / 'cur01_Hypnogram_Export.txt').write_text('W\n')
    (tmp_path / 'sub' / '._cur01.cdt').write_bytes(b'')
    assert [f.name for f in SF.list_recordings(tmp_path)] == ['cur01.cdt']
    # a name in two formats: the .edf first, then the .vhdr, then the .cdt; the one set aside is said
    for name in ('cur01.edf', 'cur02.vhdr', 'cur02.cdt'):
        (tmp_path / name).write_bytes(b'')
    assert [f.name for f in SF.list_recordings(tmp_path)] == ['cur01.edf', 'cur02.vhdr']
    assert sorted(f.name for f in SF.recording_name_conflicts(tmp_path)) == ['cur01.cdt', 'cur02.cdt']
    assert 'cur02.cdt' in SF.recording_conflicts_text(tmp_path)
