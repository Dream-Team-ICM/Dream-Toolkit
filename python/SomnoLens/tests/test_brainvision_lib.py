"""Reading BrainVision recordings (tools/brainvision_lib.py, recording_lib.py, study_files_lib.list_recordings), on
hand-written files: the signal at the right scale in every data format, the header in the shape of the EDF one, what the
free-text [Comment] gives (and that its absence costs nothing), the online reference, the start of the recording."""
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
import brainvision_lib as BV     # noqa: E402
import edf_io_lib as EIO         # noqa: E402
import recording_lib as REC      # noqa: E402
import study_files_lib as SF     # noqa: E402
from brainvision_files import recorder_comment, write_brainvision    # noqa: E402

pytestmark = pytest.mark.quick

SFREQ = 500
N_TIMES = 2000
# The kinds of channels of a real montage: EEG (TP9 / FT10 are absent from the toolkit's list of EEG names), an eye
# electrode on an EEG input without a position, then the auxiliary inputs
NAMES = ['Fp1', 'C3', 'C4', 'TP9', 'TP10', 'FT10', 'IO', 'Left Thumb', 'EMG', 'ECG', 'Respi']
RES = [0.0488281] * 7 + [1, 1, 1, 1]
UNITS = ['µV'] * 7 + ['mV', 'mV', 'mV', 'UA']
COORDS = {'Fp1': (1, -90, -72), 'C3': (1, -45, 0), 'C4': (1, 45, 0), 'TP9': (1, -113, 18), 'TP10': (1, 113, -18),
          'FT10': (1, 113, 18), 'IO': (0, 130, 74), 'Left Thumb': (0, 0, 0), 'EMG': (0, 0, 0), 'ECG': (0, 0, 0),
          'Respi': (0, 0, 0)}


def stored_values(n_ch, n_times=N_TIMES, integers=False):
    rng = np.random.default_rng(0)
    values = rng.normal(0, 300, (n_ch, n_times))
    return np.round(values) if integers else values.astype(np.float32).astype(float)


def full_recording(folder, name='rec', **kwargs):
    """A recording with the layout of the real ones."""
    comment = kwargs.pop('comment', recorder_comment(
        NAMES, RES, UNITS, no_software_filter=('Respi',),
        impedances=[{'Fp1': 4, 'C3': 'Out of Range!', 'TP9': 11, 'Cz': 3, 'Gnd': 1}],
        gradients={'Left Thumb': '1[mV/mV]', 'EMG': '1[mV/mV]', 'ECG': '1[mV/mV]', 'Respi': '1[mV/UA]'}))
    stored = kwargs.pop('stored', stored_values(len(NAMES)))
    path = write_brainvision(folder, name, stored, NAMES, resolutions=RES, units=UNITS, coordinates=COORDS,
                             comment=comment, **kwargs)
    return path, stored


# ---------------------------------------------------------------------------------------------------------
# The signal
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('data_format', ['IEEE_FLOAT_32', 'INT_16', 'INT_32'])
@pytest.mark.parametrize('orientation', ['MULTIPLEXED', 'VECTORIZED'])
def test_the_signal_is_read_at_the_scale_of_the_header(tmp_path, data_format, orientation):
    stored = stored_values(3, integers=data_format != 'IEEE_FLOAT_32')
    path = write_brainvision(tmp_path, 'rec', stored, ['C3', 'C4', 'ECG'], resolutions=[0.1, 0.0488281, 1],
                             units=['µV', 'µV', 'mV'], data_format=data_format, orientation=orientation)
    data = BV.read_raw_bv(path, preload=True).get_data()
    expected = np.vstack([stored[0] * 0.1e-6, stored[1] * 0.0488281e-6, stored[2] * 1e-3])
    assert np.allclose(data, expected, rtol=1e-5, atol=1e-12)


def test_a_read_stays_lazy_and_loses_the_task_markers(tmp_path):
    path, _ = full_recording(tmp_path, markers=[('S  1', 100), ('S  2', 200)])
    raw = BV.read_raw_bv(path)
    assert not raw.preload
    assert len(raw.annotations) == 0


def test_include_keeps_the_order_of_the_file_and_ignores_unknown_names(tmp_path):
    path, _ = full_recording(tmp_path)
    raw = BV.read_raw_bv(path, include=['ECG', 'C3', 'nothing'])
    assert raw.ch_names == ['C3', 'ECG']
    assert not raw.preload
    with pytest.raises(ValueError, match='none of the channels'):
        BV.read_raw_bv(path, include=['nothing'])


@pytest.mark.parametrize('missing', ['.eeg', '.vmrk'])
def test_a_missing_companion_is_named(tmp_path, missing):
    path, _ = full_recording(tmp_path)
    path.with_suffix(missing).unlink()
    for read in (BV.read_vhdr_header, BV.read_raw_bv):
        with pytest.raises(FileNotFoundError, match='renamed file by file'):
            read(path)


# ---------------------------------------------------------------------------------------------------------
# The header
# ---------------------------------------------------------------------------------------------------------
def small_edf(path, start=datetime.datetime(2026, 1, 2, 3, 4, 5)):
    import edfio
    signals = [edfio.EdfSignal(np.random.default_rng(i).normal(0, 20, 2560), sampling_frequency=256, label=label,
                               physical_dimension='uV', physical_range=(-500, 500))
               for i, label in enumerate(['C3', 'C4'])]
    edf = edfio.Edf(signals, recording=edfio.Recording(startdate=start.date()), starttime=start.time())
    edf.write(path)
    return path


def test_the_header_has_the_shape_of_the_edf_header(tmp_path):
    edf_header = EIO.read_edf_header_custom(small_edf(tmp_path / 'small.edf'))
    path, _ = full_recording(tmp_path)
    header = BV.read_vhdr_header(path)
    assert set(edf_header) <= set(header)
    n_ch = header['n_channels']
    for key in ('channel', 'transducer_type', 'dimension', 'physical_min', 'physical_max', 'digital_min', 'digital_max',
                'prefiltering', 'samples_per_record', 'reserved', 'sampling_frequency', 'in_file', 'resolution',
                'gradient'):
        assert len(header[key]) == n_ch, key
    assert header['sampling_frequency'][0] == '500'
    assert header['n_times'] == N_TIMES
    # the EDF formula for the duration stays exact
    assert header['n_data_records'] * header['duration_data_record'] == pytest.approx(N_TIMES / SFREQ)
    assert header['dimension'][:2] == ['uV', 'uV'] and header['dimension'][8:] == ['mV', 'mV', 'UA', 'uV']
    # the toolkit's table works, with no EDF physical range
    table = EIO.header_channel_table(header)
    assert list(table['channel']) == header['channel']
    assert table['res_theoretical'].isna().all()
    assert table.set_index('channel').loc['C3', 'highpass'] == pytest.approx(0.01, abs=1e-5)


def test_channel_types_follow_the_toolkit_rule(tmp_path):
    path, _ = full_recording(tmp_path)
    header = BV.read_vhdr_header(path)
    assert dict(zip(header['channel'], header['transducer_type'])) == {
        'Fp1': 'EEG', 'C3': 'EEG', 'C4': 'EEG', 'TP9': 'EEG', 'TP10': 'EEG', 'FT10': 'EEG', 'IO': 'EOG',
        'Left Thumb': '', 'EMG': 'EMG', 'ECG': 'ECG', 'Respi': '', 'Cz': 'EEG'}
    types = EIO.detect_channel_types(header['channel'], header['transducer_type'])
    assert types['TP9'] == 'EEG' and types['TP10'] == 'EEG' and types['FT10'] == 'EEG'
    assert types['IO'] == 'EOG' and types['Left Thumb'] == 'other' and types['Respi'] == 'other'
    context = EIO.detect_context_channels(pd.DataFrame({'channel': header['channel'],
                                                        'transducer_type': header['transducer_type']}))
    assert context == {'eog_left': 'IO', 'eog_right': None, 'emg': 'EMG', 'ecg': 'ECG'}


def test_without_coordinates_only_the_channels_in_uv_are_eeg(tmp_path):
    names = ['C3', 'TP9', 'Left Thumb', 'VEOG', 'ECG']
    units = ['µV', 'µV', 'mV', 'µV', 'mV']
    path = write_brainvision(tmp_path, 'rec', stored_values(5), names, units=units)
    header = BV.read_vhdr_header(path)
    assert header['transducer_type'] == ['EEG', 'EEG', '', 'EOG', 'ECG']


def test_with_coordinates_the_position_decides_not_the_unit(tmp_path):
    names = ['C3', 'C4', 'Sensor']
    coords = {'C3': (1, -45, 0), 'C4': (1, 45, 0), 'Sensor': (0, 0, 0)}
    path = write_brainvision(tmp_path, 'rec', stored_values(3), names, units=['V', 'µV', 'µV'], coordinates=coords)
    assert BV.read_vhdr_header(path)['transducer_type'] == ['EEG', 'EEG', '']


def test_the_units_are_read_whatever_the_order_of_the_sections():
    text = ('[Coordinates]\nCh1=1,-45,0\nCh2=1,45,0\n\n[Channel Infos]\nCh1=C3,,0.1,µV\nCh2=ECG,,1,mV\n\n'
            '[Comment]\nCh1=not a channel\n')
    assert BV._channel_infos(text) == {'C3': (0.1, 'uV'), 'ECG': (1.0, 'mV')}


def test_a_comma_and_an_old_code_page(tmp_path):
    names = ['Left,Thumb', 'C3']
    path = write_brainvision(tmp_path, 'rec', stored_values(2), names, codepage='ANSI', vhdr_encoding='cp1252')
    header = BV.read_vhdr_header(path)
    assert header['channel'] == names
    assert header['dimension'] == ['uV', 'uV']
    assert BV.read_raw_bv(path).ch_names == names


# ---------------------------------------------------------------------------------------------------------
# The [Comment]
# ---------------------------------------------------------------------------------------------------------
def test_filters_hardware_and_software_together(tmp_path):
    path, _ = full_recording(tmp_path)
    header = BV.read_vhdr_header(path)
    by_name = dict(zip(header['channel'], header['prefiltering']))
    # 15.9155 s is a 0.01 Hz high-pass; the hardware is DC to 140 Hz
    assert by_name['C3'] == 'HP:0.01Hz LP:140Hz'
    assert EIO.extract_filter_value(by_name['C3'], 'HP') == pytest.approx(0.01)
    # a long name touching the next column, and a channel with its software filter off
    assert by_name['Left Thumb'] == 'HP:0.01Hz LP:140Hz'
    assert by_name['Respi'] == 'LP:140Hz'
    assert EIO.extract_filter_value(by_name['Respi'], 'HP') is None


def test_a_low_cutoff_in_hz_is_not_converted(tmp_path):
    names = ['C3', 'C4']
    comment = recorder_comment(names, [0.1, 0.1], ['µV', 'µV'], software_low=0.3, low_unit='Hz')
    path = write_brainvision(tmp_path, 'rec', stored_values(2), names, comment=comment)
    assert BV.read_vhdr_header(path)['prefiltering'][:2] == ['HP:0.3Hz LP:140Hz'] * 2


def test_an_amplifier_without_a_unit_column(tmp_path):
    names = ['C3', 'C4']
    comment = recorder_comment(names, [0.1, 0.1], ['µV', 'µV'], unit_column=False, software_low=None)
    path = write_brainvision(tmp_path, 'rec', stored_values(2), names, comment=comment)
    assert BV.read_vhdr_header(path)['prefiltering'][:2] == ['LP:140Hz'] * 2


def test_impedances_and_gradients(tmp_path):
    path, _ = full_recording(tmp_path)
    header = BV.read_vhdr_header(path)
    assert header['impedance_kohm'] == {'Fp1': 4.0, 'C3': None, 'TP9': 11.0, 'Cz': 3.0, 'Gnd': 1.0}
    assert header['impedance_time'] == '12:00:00'
    gradient = dict(zip(header['channel'], header['gradient']))
    assert gradient['ECG'] == (1.0, 'mV/mV') and gradient['Respi'] == (1.0, 'mV/UA') and gradient['C3'] is None


def test_the_last_impedance_measure_is_kept(tmp_path):
    names = ['C3', 'C4']
    comment = recorder_comment(names, [0.1, 0.1], ['µV', 'µV'], impedances=[{'C3': 50, 'C4': 60}, {'C3': 5, 'C4': 6}])
    path = write_brainvision(tmp_path, 'rec', stored_values(2), names, comment=comment)
    assert BV.read_vhdr_header(path)['impedance_kohm'] == {'C3': 5.0, 'C4': 6.0}


@pytest.mark.parametrize('comment', [
    None,                                                      # no [Comment] at all
    '',                                                        # an empty one
    'Some text written by another program\n\nand nothing else\n',
    'A m p l i f i e r  S e t u p\n====\nChannels\n--------\nno table here\nReference Channel Name =\n',
    'Impedance [kOhm] at 12:00:00 :\n',
])
def test_a_comment_absent_or_written_differently_costs_nothing(tmp_path, comment):
    path = write_brainvision(tmp_path, 'rec', stored_values(2), ['C3', 'C4'], comment=comment)
    header = BV.read_vhdr_header(path)
    assert header['channel'] == ['C3', 'C4']
    assert header['online_reference'] is None
    assert header['prefiltering'] == ['', '']
    assert header['impedance_kohm'] == {}
    assert BV.read_raw_bv(path, include=['C3', 'Cz']).ch_names == ['C3']


def comment_cut_in_its_table(names, resolutions, units):
    comment = recorder_comment(names, resolutions, units)
    return comment[:comment.index(f'2     {names[1]}')]


def comment_with_another_name(names, resolutions, units):
    return recorder_comment(names, resolutions, units).replace(names[1], 'XX')


@pytest.mark.parametrize('make_comment', [comment_cut_in_its_table, comment_with_another_name])
def test_a_filter_table_mne_stops_on_costs_nothing_and_is_reported(tmp_path, make_comment):
    # MNE raises an IndexError or an AssertionError without a message on these tables
    names, resolutions, units = ['C3', 'C4', 'ECG'], [0.1, 0.1, 1], ['µV', 'µV', 'mV']
    stored = stored_values(3)
    path = write_brainvision(tmp_path, 'rec', stored, names, resolutions=resolutions, units=units,
                             comment=make_comment(names, resolutions, units), codepage='ANSI', vhdr_encoding='cp1252',
                             markers=[('S  1', 100)])
    header = BV.read_vhdr_header(path)
    assert header['channel'][:3] == names and header['n_times'] == N_TIMES
    assert 'could not be read by MNE' in header['comment_warning']
    assert header['dimension'][:3] == ['uV', 'uV', 'mV']
    assert header['start_date'] == '02.01.26'
    raw = BV.read_raw_bv(path, include=['C3', 'ECG'])
    expected = np.vstack([stored[0] * 0.1e-6, stored[2] * 1e-3])
    assert np.allclose(raw.get_data(), expected, rtol=1e-5, atol=1e-12)
    assert len(raw.annotations) == 0


def test_a_readable_comment_gives_no_warning(tmp_path):
    path, _ = full_recording(tmp_path)
    assert BV.read_vhdr_header(path)['comment_warning'] == ''


def test_a_header_naming_its_files_by_dollar_b(tmp_path):
    stored = stored_values(2)
    path = write_brainvision(tmp_path, 'rec', stored, ['C3', 'C4'])
    text = path.read_text(encoding='utf-8')
    path.write_text(text.replace('DataFile=rec.eeg', 'DataFile=$b.eeg').replace('MarkerFile=rec.vmrk', 'MarkerFile=$b.vmrk'),
                    encoding='utf-8')
    assert BV.read_vhdr_header(path)['n_times'] == N_TIMES
    assert BV.read_vmrk_info(path)['n_segments'] == 1
    assert np.allclose(BV.read_raw_bv(path).get_data(), stored * 0.1e-6, rtol=1e-5, atol=1e-12)


def test_read_comment_info_never_raises():
    names = ['C3', 'C4']
    for text in ('', '[Comment]', '[Comment]\n#     Name\n1 C3\n2', '[Comment]\nS o f t w a r e\n#     Low Cutoff\n1 a\n',
                 '[Comment]\n#     Name   Resolution / Unit\n1     C3   x y DC abc Off\n'):
        info = BV.read_comment_info(text, names)
        assert info['online_reference'] is None and len(info['highpass']) == 2


# ---------------------------------------------------------------------------------------------------------
# The online reference
# ---------------------------------------------------------------------------------------------------------
def test_the_reference_is_listed_but_not_as_a_channel_of_the_file(tmp_path):
    path, _ = full_recording(tmp_path)
    header = BV.read_vhdr_header(path)
    assert header['online_reference'] == 'Cz'
    assert header['channel'][-1] == 'Cz' and header['in_file'][-1] is False and all(header['in_file'][:-1])
    assert header['prefiltering'][-1] == 'online reference, not in the file'
    assert header['sampling_frequency'][-1] == '500'


def test_the_reference_is_recreated_only_on_request(tmp_path):
    path, stored = full_recording(tmp_path)
    assert 'Cz' not in BV.read_raw_bv(path).ch_names
    assert 'Cz' not in BV.read_raw_bv(path, include=['C3', 'C4']).ch_names
    assert 'Cz' not in BV.read_raw_bv(path, include=['C3', 'Cz'], add_online_reference=False).ch_names

    raw = BV.read_raw_bv(path, include=['C3', 'TP9', 'TP10', 'Cz'])
    assert raw.ch_names == ['C3', 'TP9', 'TP10', 'Cz']
    assert raw.get_channel_types() == ['eeg'] * 4
    assert np.all(raw.get_data(picks=['Cz']) == 0)
    # after a reference to the mastoids it is the signal the online reference really carried
    raw.set_eeg_reference(['TP9', 'TP10'], verbose='ERROR')
    expected = -(stored[NAMES.index('TP9')] + stored[NAMES.index('TP10')]) / 2 * 0.0488281e-6
    assert np.allclose(raw.get_data(picks=['Cz'])[0], expected, atol=1e-10)


@pytest.mark.parametrize('reference, xyz', [('Cz', (0, 0, 0.095)), ('FCz', None)])
def test_the_recreated_reference_has_a_position_on_the_same_head(tmp_path, reference, xyz):
    names = ['C3', 'C4', 'TP9', 'TP10']
    coords = {'C3': (1, -45, 0), 'C4': (1, 45, 0), 'TP9': (1, -113, 18), 'TP10': (1, 113, -18)}
    comment = recorder_comment(names, [0.1] * 4, ['µV'] * 4, reference=reference)
    path = write_brainvision(tmp_path, 'rec', stored_values(4), names, coordinates=coords, comment=comment)
    raw = BV.read_raw_bv(path, include=names + [reference])
    assert raw.ch_names[-1] == reference
    positions = raw.get_montage().get_positions()['ch_pos']
    assert np.all(np.isfinite(positions[reference]))
    assert np.linalg.norm(positions[reference]) == pytest.approx(np.linalg.norm(positions['C3']), rel=0.02)
    if xyz:
        assert np.allclose(positions[reference], xyz, atol=1e-3)
    # the other positions did not move
    assert np.allclose(positions['C3'], BV.read_raw_bv(path).get_montage().get_positions()['ch_pos']['C3'])


def test_a_reference_the_file_holds_is_not_recreated(tmp_path):
    names = ['C3', 'Cz']
    comment = recorder_comment(names, [0.1, 0.1], ['µV', 'µV'], reference='Cz')
    path = write_brainvision(tmp_path, 'rec', stored_values(2), names, comment=comment)
    header = BV.read_vhdr_header(path)
    assert header['channel'] == names and all(header['in_file'])
    raw = BV.read_raw_bv(path, include=names)
    assert raw.ch_names == names and not raw.preload


# ---------------------------------------------------------------------------------------------------------
# The start of the recording
# ---------------------------------------------------------------------------------------------------------
def test_the_start_is_cut_to_the_second(tmp_path):
    path, _ = full_recording(tmp_path)
    info = BV.read_vmrk_info(path)
    assert info['start_datetime'] == datetime.datetime(2026, 1, 2, 3, 4, 5)
    assert info['n_segments'] == 1
    header = BV.read_vhdr_header(path)
    assert header['start_date'] == '02.01.26' and header['start_time'] == '03.04.05'


@pytest.mark.parametrize('start', [None, '00000000000000000000'])
def test_no_usable_date_gives_no_start(tmp_path, start):
    path, _ = full_recording(tmp_path, start=start)
    assert BV.read_vmrk_info(path)['start_datetime'] is None
    assert BV.read_vhdr_header(path)['start_date'] == ''
    assert REC.read_start_datetime(path) is None


def test_a_recording_pause_is_counted_and_task_markers_are_not(tmp_path):
    path, _ = full_recording(tmp_path, extra_segments=[1000], markers=[('S  1', 100), ('S  2', 200)])
    info = BV.read_vmrk_info(path)
    assert info['n_segments'] == 2 and info['n_markers'] == 4
    assert BV.read_vhdr_header(path)['n_segments'] == 2


def test_a_missing_marker_file_gives_no_start_without_raising(tmp_path):
    path, _ = full_recording(tmp_path)
    path.with_suffix('.vmrk').unlink()
    assert BV.read_vmrk_info(path) == {'start_datetime': None, 'n_segments': 0, 'n_markers': 0}


# ---------------------------------------------------------------------------------------------------------
# One reading for every format
# ---------------------------------------------------------------------------------------------------------
def test_the_edf_branch_is_the_current_reading(tmp_path):
    import mne
    path = small_edf(tmp_path / 'small.edf')
    assert REC.file_format(path) == 'edf'
    assert REC.read_header(path) == EIO.read_edf_header_custom(path)
    assert REC.read_start_datetime(path) == EIO.read_edf_start_datetime(path)
    direct = mne.io.read_raw_edf(str(path), preload=False, encoding='latin-1', include=['C4'], verbose=False)
    raw = REC.read_raw(path, include=['C4'])
    assert raw.ch_names == direct.ch_names == ['C4'] and not raw.preload
    assert np.array_equal(raw.get_data(), direct.get_data())
    assert REC.read_raw(path).ch_names == ['C3', 'C4']


def test_the_brainvision_branch(tmp_path):
    path, _ = full_recording(tmp_path)
    assert REC.file_format(path) == 'brainvision'
    assert REC.read_header(path)['format'] == 'brainvision'
    assert REC.read_start_datetime(path) == datetime.datetime(2026, 1, 2, 3, 4, 5)
    assert REC.read_raw(path, include=['C3']).ch_names == ['C3']
    assert REC.read_raw(path, include=['C3', 'TP9', 'Cz']).ch_names == ['C3', 'TP9', 'Cz']
    assert REC.read_raw(path, include=['C3', 'TP9', 'Cz'], add_online_reference=False).ch_names == ['C3', 'TP9']


def test_any_other_file_is_refused(tmp_path):
    for read in (REC.file_format, REC.read_header, REC.read_start_datetime, REC.read_raw):
        with pytest.raises(ValueError, match='not a recording format'):
            read(tmp_path / 'rec.eeg')


# ---------------------------------------------------------------------------------------------------------
# Finding the recordings
# ---------------------------------------------------------------------------------------------------------
def touch(folder, *names):
    for name in names:
        path = Path(folder) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'')


def test_the_recordings_are_found_by_their_vhdr_never_their_eeg(tmp_path):
    touch(tmp_path, 'sub/a.vhdr', 'sub/a.vmrk', 'sub/a.eeg', 'sub/._a.vhdr', 'sub/a_Hypnogram.txt', 'lone.eeg', 'b.VHDR',
          'c.edf', 'sub/._c.edf')
    assert [f.name for f in SF.list_recordings(tmp_path)] == ['b.VHDR', 'c.edf', 'a.vhdr']
    assert SF.recording_name_conflicts(tmp_path) == []


def test_an_edf_study_is_listed_as_before(tmp_path):
    touch(tmp_path, 'g2/2.edf', 'g1/1.EDF', '._3.edf', '10.edf', 'g1/1.edf.XML', 'g1/1_hypno.txt')
    assert SF.list_recordings(tmp_path) == EIO.list_edf_files(tmp_path)


def test_a_name_in_two_formats_keeps_the_edf_and_says_so(tmp_path):
    touch(tmp_path, 'a.edf', 'data/a.vhdr', 'data/b.vhdr')
    assert [f.name for f in SF.list_recordings(tmp_path)] == ['a.edf', 'b.vhdr']
    assert [f.name for f in SF.recording_name_conflicts(tmp_path)] == ['a.vhdr']
