"""The signal-preparation label of tools/quality_lib.py (tool 6, tool 0's step 5): what it says, and what makes a
participant's stored table different from the run's settings (preparation, flags at the run's thresholds)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
import quality_lib as QL     # noqa: E402

pytestmark = [pytest.mark.quick, pytest.mark.tool6]


def test_signal_preparation_label():
    assert QL.signal_preparation_label() == 'none'
    assert QL.signal_preparation_label(resampled_to=128.0) == 'resampled to 128 Hz'
    assert QL.signal_preparation_label(highpassed_at=0.5) == 'high-pass 0.5 Hz'
    assert QL.signal_preparation_label(128.0, 0.5) == 'resampled to 128 Hz, high-pass 0.5 Hz'


def test_signal_preparation_mismatch():
    # same settings, or a table written before the column existed: nothing to report
    assert QL.signal_preparation_mismatch('none', False, None, False, None) == ''
    assert QL.signal_preparation_mismatch('high-pass 0.5 Hz', False, None, True, 0.5) == ''
    assert QL.signal_preparation_mismatch(float('nan'), False, None, True, 0.5) == ''
    assert QL.signal_preparation_mismatch('', False, None, True, 0.5) == ''
    # a different high-pass is always reported
    assert (QL.signal_preparation_mismatch('high-pass 0.5 Hz', False, None, False, None)
            == 'made with high-pass 0.5 Hz, this run: none')
    assert QL.signal_preparation_mismatch('none', False, None, True, 0.3) == 'made with none, this run: high-pass 0.3 Hz'
    # a stored resampling is compared with the run's
    assert QL.signal_preparation_mismatch('resampled to 128 Hz', True, 128, False, None) == ''
    assert QL.signal_preparation_mismatch('resampled to 128 Hz', False, None, False, None) != ''
    assert QL.signal_preparation_mismatch('resampled to 128 Hz', True, 256, False, None) != ''
    # not resampled while this run resamples: the file may already have been at the target rate (not told)
    assert QL.signal_preparation_mismatch('none', True, 128, False, None) == ''
    # the header tells: different only when this run would really resample the file
    assert QL.signal_preparation_mismatch('none', True, 128, False, None, native_sf=256.0) != ''
    assert QL.signal_preparation_mismatch('none', True, 128, False, None, native_sf=128.0) == ''
    assert QL.signal_preparation_mismatch('none', False, None, False, None, native_sf=256.0) == ''


THRESHOLDS = {'flat_pct': 3.5, 'bounds_pct': 1.0, 'n_peaks': 2, 'std_low': 5.0, 'hist_extreme_pct': 1.0}


def stored_table(**changes):
    """A participant's {file_id}_quality_metrics.tsv: C3 flagged by flat_pct, O1 not flagged."""
    table = pd.DataFrame([
        {'file_id': 'p', 'channel': 'C3', 'std_uV': 20.0, 'flat_pct': 5.0, 'bounds_pct': 0.0, 'hist_extreme_pct': 0.0,
         'n_peaks': 1, 'signal_preparation': 'none', 'flag_thresholds': QL.flag_thresholds_text(THRESHOLDS),
         'suspect_reason': 'flat_pct=5.0% > 3.5%', 'exclude': True},
        {'file_id': 'p', 'channel': 'O1', 'std_uV': 20.0, 'flat_pct': 1.0, 'bounds_pct': 0.0, 'hist_extreme_pct': 0.0,
         'n_peaks': 1, 'signal_preparation': 'none', 'flag_thresholds': QL.flag_thresholds_text(THRESHOLDS),
         'suspect_reason': float('nan'), 'exclude': False}])
    return table.assign(**changes)


def differences(table, thresholds=THRESHOLDS, do_resample=False, target=None, do_highpass=False, hp=None, native=None):
    return QL.settings_differences(table, do_resample, target, do_highpass, hp, thresholds, native_sf=native)


def test_flag_thresholds_text_reads_back():
    text = QL.flag_thresholds_text(THRESHOLDS)
    assert text == 'bounds_pct:1+flat_pct:3.5+hist_extreme_pct:1+n_peaks:2+std_low:5'
    assert QL.parse_flag_thresholds(text) == THRESHOLDS
    assert QL.parse_flag_thresholds(float('nan')) == {}


def test_same_settings_make_no_difference():
    assert differences(stored_table()) == []


def test_flag_criteria_ignore_the_threshold_in_the_text():
    assert QL.flag_criteria(['n_peaks=3 (multimodal)', 'flat_pct=5.0% > 3.5%']) == ['flat_pct', 'n_peaks']
    # the same channel stays flagged for the same cause, under another threshold: no flag moved
    assert differences(stored_table(), dict(THRESHOLDS, flat_pct=4.0)) == []


def test_a_threshold_counts_when_it_moves_a_flag():
    assert differences(stored_table(), dict(THRESHOLDS, flat_pct=6.0)) == [
        ('flag threshold flat_pct', '3.5', '6'), ('flags of C3', 'flat_pct', 'none')]
    assert differences(stored_table(), dict(THRESHOLDS, flat_pct=0.5)) == [
        ('flag threshold flat_pct', '3.5', '0.5'), ('flags of O1', 'none', 'flat_pct')]


def test_the_preparation_is_reported_with_both_values():
    assert differences(stored_table(), do_highpass=True, hp=0.5) == [('signal preparation', 'none', 'high-pass 0.5 Hz')]
    assert differences(stored_table(signal_preparation='high-pass 0.5 Hz')) == [
        ('signal preparation', 'high-pass 0.5 Hz', 'none')]
    assert differences(stored_table(), do_resample=True, target=128, native=256.0) == [
        ('signal preparation', 'none', 'resampled to 128 Hz')]


def test_a_table_written_before_the_records_compares_only_its_flags():
    old = stored_table().drop(columns=['signal_preparation', 'flag_thresholds'])
    assert QL.recorded_preparation(old) == ''
    assert differences(old, do_highpass=True, hp=0.5) == []
    assert differences(old, dict(THRESHOLDS, flat_pct=6.0)) == [('flags of C3', 'flat_pct', 'none')]


def test_the_recorded_settings_read_back():
    table = stored_table(signal_preparation='resampled to 128 Hz, high-pass 0.5 Hz')
    assert QL.recorded_settings(table) == {'preparation_known': True, 'highpass': 0.5, 'resampled_to': 128.0,
                                           'thresholds': THRESHOLDS}
    assert QL.recorded_settings(stored_table()) == {'preparation_known': True, 'highpass': None, 'resampled_to': None,
                                                    'thresholds': THRESHOLDS}
    old = stored_table().drop(columns=['signal_preparation', 'flag_thresholds'])
    assert QL.recorded_settings(old) == {'preparation_known': False, 'highpass': None, 'resampled_to': None, 'thresholds': {}}
