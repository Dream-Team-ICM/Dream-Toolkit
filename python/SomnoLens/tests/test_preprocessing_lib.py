"""What separates a participant's params sidecar from the run's settings in tools/preprocessing_lib.py (tool 7's question
when Skip is ticked), and the channels the editor starts with for a participant already processed."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
import preprocessing_lib as PL     # noqa: E402

pytestmark = [pytest.mark.quick, pytest.mark.tool7]

SIDECAR = {
    'file_id': 'p', 'sfreq_hz': 256.0, 'epoch_length_s': 30, 'channels': ['C3', 'O1'], 'bad_channels': ['Fp1'],
    'custom_stages': [], 'methods_run': ['amplitude', 'gradient', 'flat', '1f_r2', '1f_error', 'event'],
    'resample': {'applied': False, 'target_freq_hz': None},
    'filter': {'applied': True, 'l_freq_hz': 0.1, 'h_freq_hz': 50.0},
    'notch': {'applied': True, 'freq_hz': 50.0},
    'rejection_thresholds': {'amplitude_ptp_uV': {'W': 250.0, 'N2': 300.0}, 'flat_ptp_uV': 1.0,
                             'gradient_uV_per_sample': 100.0, '1f_mae_max': 0.15, '1f_r2_min': 0.95,
                             '1f_fit_range_hz': [2.0, 45.0]},
    'event_rejection': {'applied': True, 'event_types': ['hypopnea', 'plm']},
    'psd_smoothing': {'enabled': True, 'method': 'median+lowess', 'median_span_hz': 1.0, 'lowess_span_hz': 2.0},
    'manual_events': False, 'reference': ['M2'], 'hypnogram_file': 'p_hypno.txt',
}
CURRENT = {key: SIDECAR[key] for key in ('epoch_length_s', 'channels', 'resample', 'filter', 'notch', 'rejection_thresholds',
                                         'event_rejection', 'psd_smoothing', 'manual_events', 'reference', 'hypnogram_file')}
CURRENT['methods_run'] = ['amplitude', 'gradient', 'flat', '1f_r2', '1f_error']     # no event: see settings_differences


def differences(**changes):
    current = copy.deepcopy(CURRENT)
    current.update(changes)
    return PL.settings_differences(SIDECAR, current)


def with_threshold(**changes):
    return dict(copy.deepcopy(CURRENT['rejection_thresholds']), **changes)


def test_same_settings_make_no_difference():
    assert differences() == []


def test_the_event_method_counts_only_through_the_types_asked_for():
    # A file without any of the types has no 'event' in methods_run, under the same settings.
    sidecar = dict(SIDECAR, methods_run=[m for m in SIDECAR['methods_run'] if m != 'event'])
    assert PL.settings_differences(sidecar, copy.deepcopy(CURRENT)) == []
    assert differences(event_rejection={'applied': False, 'event_types': []}) == [('event flagging', 'hypopnea, plm', 'off')]
    assert differences(event_rejection={'applied': True, 'event_types': ['plm']}) == [
        ('event flagging', 'hypopnea, plm', 'plm')]


@pytest.mark.parametrize('changes, row', [
    ({'epoch_length_s': 10}, ('epoch length', '30 s', '10 s')),
    ({'resample': {'applied': True, 'target_freq_hz': 128.0}}, ('resampling', 'off', '128 Hz')),
    ({'filter': {'applied': False, 'l_freq_hz': None, 'h_freq_hz': None}}, ('band-pass filter', '0.1-50 Hz', 'off')),
    ({'filter': {'applied': True, 'l_freq_hz': 0.3, 'h_freq_hz': 50.0}}, ('band-pass filter', '0.1-50 Hz', '0.3-50 Hz')),
    ({'notch': {'applied': False, 'freq_hz': None}}, ('notch', '50 Hz', 'off')),
    ({'manual_events': True}, ('manual annotations', 'off', 'on')),
    ({'channels': ['C3']}, ('channels kept', 'C3, O1', 'C3')),
    ({'reference': 'average'}, ('reference', 'M2', 'average')),
    ({'reference': []}, ('reference', 'M2', 'none')),
    ({'hypnogram_file': 'p_other.txt'}, ('hypnogram', 'p_hypno.txt', 'p_other.txt')),
])
def test_each_setting_is_reported_with_both_values(changes, row):
    assert differences(**changes) == [row]


def test_a_channel_the_sidecar_does_not_name_is_not_compared():
    assert differences(channels=['C3', 'M2', 'O1']) == []          # the reference has left the epochs before the sidecar
    assert differences(channels=['C3', 'M2', 'O1', 'Fp1']) == [('channels kept', 'C3, O1', 'C3, Fp1, O1')]


def test_the_hypnogram_name_is_compared_without_its_case():
    if Path('A') == Path('a'):        # Windows: the scan finds the file whatever the case of its name
        assert differences(hypnogram_file='P_HYPNO.TXT') == []


@pytest.mark.parametrize('changes, row', [
    ({'flat_ptp_uV': 2.0}, ('flat threshold (µV)', '1', '2')),
    ({'gradient_uV_per_sample': 80.0}, ('gradient threshold (µV/sample)', '100', '80')),
    ({'1f_mae_max': 0.2}, ('1/f error max', '0.15', '0.2')),
    ({'1f_r2_min': 0.9}, ('1/f R² min', '0.95', '0.9')),
    ({'1f_fit_range_hz': [1.0, 40.0]}, ('1/f fit range', '2-45 Hz', '1-40 Hz')),
    ({'amplitude_ptp_uV': {'W': 250.0, 'N2': 350.0}}, ('amplitude threshold N2 (µV)', '300', '350')),
])
def test_a_threshold_is_reported_with_both_values(changes, row):
    assert differences(rejection_thresholds=with_threshold(**changes)) == [row]


def test_a_threshold_counts_only_if_its_method_ran_on_both_sides():
    without_flat = [m for m in CURRENT['methods_run'] if m != 'flat']
    assert differences(methods_run=without_flat, rejection_thresholds=with_threshold(flat_ptp_uV=2.0)) == [
        ('flagging methods', 'amplitude, gradient, flat, 1f_r2, 1f_error', 'amplitude, gradient, 1f_r2, 1f_error')]
    sidecar = dict(SIDECAR, methods_run=['amplitude', 'gradient', 'flat'])
    assert PL.settings_differences(sidecar, dict(CURRENT, methods_run=['amplitude', 'gradient', 'flat'],
                                                 rejection_thresholds=with_threshold(**{'1f_mae_max': 0.5}),
                                                 psd_smoothing={'enabled': False})) == []


def test_a_custom_stage_declared_later_does_not_concern_a_participant():
    ptp = {'W': 250.0, 'N2': 300.0, 'IS': 400.0}
    assert differences(rejection_thresholds=with_threshold(amplitude_ptp_uV=ptp)) == []


def test_the_psd_smoothing_in_words():
    assert differences(psd_smoothing={'enabled': False, 'method': 'lowess', 'median_span_hz': None,
                                      'lowess_span_hz': 2.0}) == [('PSD smoothing', 'median 1 Hz + LOWESS 2 Hz', 'off')]
    assert differences(psd_smoothing={'enabled': True, 'method': 'lowess', 'median_span_hz': None,
                                      'lowess_span_hz': 3.0}) == [
        ('PSD smoothing', 'median 1 Hz + LOWESS 2 Hz', 'LOWESS 3 Hz')]


def test_an_older_sidecar_compares_what_it_holds():
    old = {key: SIDECAR[key] for key in ('file_id', 'channels', 'methods_run', 'resample', 'filter', 'rejection_thresholds')}
    # Not held and did not exist: absent = off, and the epochs were 30 s long.
    assert PL.settings_differences(old, copy.deepcopy(CURRENT)) == [
        ('notch', 'off', '50 Hz'), ('PSD smoothing', 'off', 'median 1 Hz + LOWESS 2 Hz')]
    old_same = dict(old, notch=SIDECAR['notch'], psd_smoothing=SIDECAR['psd_smoothing'])
    # Not held, unknown: the events, the manual annotations, the reference and the hypnogram are not compared.
    assert PL.settings_differences(old_same, dict(copy.deepcopy(CURRENT), manual_events=True, reference=[],
                                                  hypnogram_file='other.txt',
                                                  event_rejection={'applied': False, 'event_types': []})) == []


def test_the_channels_the_editor_starts_with():
    start = {'Fp1': False, 'C3': True, 'M2': True, 'O1': True}
    # Fp1, flagged by tool 6, was kept in this run; O1 was unticked by hand: the sidecar decides, in the order of `start`.
    # M2, the reference, has left the epochs before the sidecar is written: it stays ticked to be the reference again.
    assert PL.recorded_channels({'channels': ['Fp1', 'C3'], 'bad_channels': ['O1']}, start) == {
        'Fp1': True, 'C3': True, 'M2': True, 'O1': False}
    assert list(PL.recorded_channels({'channels': ['O1', 'C3'], 'bad_channels': []}, start)) == ['Fp1', 'C3', 'M2', 'O1']
    # A channel only the sidecar names is added.
    assert PL.recorded_channels({'channels': ['C3', 'F4'], 'bad_channels': ['F3']}, {'C3': True, 'O1': True}) == {
        'C3': True, 'O1': True, 'F4': True, 'F3': False}
    # Nothing recorded: `start` stays.
    assert PL.recorded_channels({}, start) == start


def test_the_recorded_settings_leave_out_what_belongs_to_the_participant():
    settings = PL.recorded_settings(SIDECAR, 'p')
    assert settings['methods'] == ['amplitude', 'gradient', 'flat', '1f_r2', '1f_error']       # no event method
    assert settings['hypnogram_suffix'] == '_hypno.txt'
    assert not {'file_id', 'sfreq_hz', 'channels', 'bad_channels', 'reference', 'custom_stages'} & set(settings)
    # Two participants made with the same settings give the same set.
    other = dict(SIDECAR, file_id='q', channels=['C3'], bad_channels=['O1'], reference=['M1'], hypnogram_file='q_hypno.txt',
                 methods_run=[m for m in SIDECAR['methods_run'] if m != 'event'])
    assert PL.recorded_settings(other, 'q') == settings


def test_the_recorded_settings_of_an_older_sidecar_hold_what_it_has():
    old = {'file_id': 'p', 'methods_run': ['amplitude'], 'filter': SIDECAR['filter']}
    assert PL.recorded_settings(old, 'p') == {'filter': SIDECAR['filter'], 'methods': ['amplitude']}
