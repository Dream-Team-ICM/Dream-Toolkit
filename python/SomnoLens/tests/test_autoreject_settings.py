"""What separates a participant's decision record from the run's settings in tools/autoreject_lib.py (tool 8bis's question
when Skip is ticked), the fingerprint of tool 7's settings included."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
import autoreject_lib as AR     # noqa: E402

pytestmark = [pytest.mark.quick, pytest.mark.tool8bis]

SETTINGS = dict(channel_reject_pct=20.0, epoch_reject_pct=20.0, stages_of_interest='N2+N3+R',
                methods_used='amplitude+flat', event_types_used='', event_thresholds='', preproc_fingerprint='abc')


@pytest.fixture
def record(tmp_path):
    path = tmp_path / 'p_autoreject_decision.tsv'
    pd.DataFrame([dict(SETTINGS, file_id='p')]).to_csv(path, sep='\t', index=False)
    return path


def differences(record, **changes):
    return AR.settings_differences(record, dict(SETTINGS, **changes))


def test_the_same_settings_make_no_difference(record):
    assert differences(record) == []


@pytest.mark.parametrize('changes, row', [
    ({'channel_reject_pct': 50.0}, ('channel reject (%)', '20', '50')),
    ({'epoch_reject_pct': 12.5}, ('epoch reject (%)', '20', '12.5')),
    ({'stages_of_interest': 'N2'}, ('reference stages', 'N2+N3+R', 'N2')),
    ({'methods_used': 'amplitude'}, ('flagging methods', 'amplitude+flat', 'amplitude')),
    ({'event_types_used': 'plm'}, ('event types', 'none', 'plm')),
    ({'event_thresholds': 'plm:10'}, ('event thresholds', 'none', 'plm:10')),
    ({'preproc_fingerprint': 'def'}, ('tool 7 settings', 'as at the last decision', 'changed since')),
])
def test_each_setting_is_reported_with_both_values(record, changes, row):
    assert differences(record, **changes) == [row]


def test_a_fingerprint_missing_on_one_side_is_no_difference(record, tmp_path):
    assert differences(record, preproc_fingerprint='') == []                 # no readable sidecar now
    old = tmp_path / 'old_autoreject_decision.tsv'
    pd.read_csv(record, sep='\t').drop(columns=['preproc_fingerprint', 'event_thresholds']).to_csv(old, sep='\t', index=False)
    assert AR.settings_differences(old, SETTINGS) == []                      # a record written before these columns


def test_an_unreadable_record_counts_as_different(tmp_path):
    assert AR.settings_differences(tmp_path / 'absent.tsv', SETTINGS) == [('decision record', 'unreadable', 'this run')]
