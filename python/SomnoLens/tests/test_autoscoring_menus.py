"""Tool 3bis's menus (phase 5 ter of tools/plan_brainvision.md): only the channels tool 2 declares, '(n/N)' after a choice
that not every recording allows, and what a recording lacking a chosen channel gets (autoscoring_lib)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
import autoscoring_lib as A     # noqa: E402

pytestmark = [pytest.mark.quick, pytest.mark.tool3bis]


def entry(eeg=('C3', 'C4', 'F3'), **context):
    return {'remap': {ch: ch for ch in list(eeg) + ['M1', 'M2']}, 'ref_channels': [], 'context_channels': dict(context)}


# A study whose single EOG (an eye electrode, IO) is declared on the right, and one recording without F3 nor EMG
CONFIGS = {'p1': entry(eog_right='IO', emg='EMG'), 'p2': entry(eog_right='IO', emg='EMG'),
           'p3': entry(eeg=('C3', 'C4'), eog_right='IO')}


def test_an_eog_menu_offers_only_what_tool_2_declares():
    counts = A.eog_counts(CONFIGS, A.GSSC_EOG_ROLES)
    assert counts == {'L − R bipolar': 0, 'L and R separately': 0, 'Left only': 0, 'Right only': 3, 'None': 3}
    assert A.menu_options(counts, 3) == [('Right only', 'Right only'), ('None', 'None')]
    # the former default (bipolar) is no longer offered: the first choice every recording allows
    assert A.menu_default(counts, 3, 'L − R bipolar', A.GSSC_EOG_PREFERRED) == 'Right only'
    yasa = A.eog_counts(CONFIGS, A.YASA_EOG_ROLES)
    assert A.menu_default(yasa, 3, 'Left', A.YASA_EOG_PREFERRED) == 'Right'


def test_a_choice_not_every_recording_allows_is_marked():
    configs = dict(CONFIGS, p4=entry(eog_left='E1', eog_right='E2'))
    counts = A.eog_counts(configs, A.GSSC_EOG_ROLES)
    assert A.menu_options(counts, 4)[:2] == [('L − R bipolar  (1/4)', 'L − R bipolar'),
                                            ('L and R separately  (1/4)', 'L and R separately')]
    assert A.menu_default(counts, 4, 'L − R bipolar', A.GSSC_EOG_PREFERRED) == 'Right only'      # 4/4
    assert A.menu_default(counts, 4, 'Right only', A.GSSC_EOG_PREFERRED) == 'Right only'         # kept
    assert A.eeg_counts(CONFIGS) == {'C3': 3, 'C4': 3, 'F3': 2}
    # no EOG declared anywhere: None only
    none = A.eog_counts({'p': entry()}, A.YASA_EOG_ROLES)
    assert A.menu_options(none, 1) == [('None', 'None')] and A.menu_default(none, 1, 'Left', A.YASA_EOG_PREFERRED) == 'None'


def test_what_a_recording_lacking_a_chosen_channel_gets():
    lines = A.missing_channel_lines(CONFIGS, ['GSSC', 'YASA'], gssc_eeg=['C3', 'F3'], gssc_one=False,
                                    gssc_eog='Right only', yasa_eeg='F3', yasa_eog='Right')
    assert lines == [('GSSC', 'electrode F3 absent from 1 recording(s) (p3): scored without it.'),
                     ('YASA', 'electrode F3 absent from 1 recording(s) (p3): not scored (never substituted).'),
                     ('YASA', 'no chin EMG declared in tool 2 for 1 recording(s) (p3): scored without EMG.')]
    one = A.missing_channel_lines(CONFIGS, ['GSSC'], gssc_eeg=['F3'], gssc_one=True, gssc_eog='L − R bipolar')
    assert one[0][1].endswith('not scored (one-electrode mode: never substituted).')
    assert one[1][1] == 'EOG L − R bipolar: one side not declared in tool 2 for 3 recording(s) (p1, p2, p3): scored with the EOG declared.'
    hand = A.missing_channel_lines(CONFIGS, ['YASA'], yasa_eeg='C4', yasa_eog='None', yasa_emg='Chin',
                                   file_channels={'p1': ['Chin'], 'p2': [], 'p3': ['Chin']})
    assert hand == [('YASA', 'channel Chin absent from 1 recording(s) (p2): scored without EMG.')]
    assert A.missing_channel_lines(CONFIGS, ['SleepFM']) == []


def test_the_channels_sleepfm_will_read():
    text = A.sleepfm_channels_text(entry(eeg=('C3', 'C4', 'Fz'), eog_right='IO', emg='EMG', ecg='ECG'),
                                   ['C3', 'C4', 'Fz', 'IO', 'EMG', 'ECG', 'Respi'])
    assert text == 'EEG C3, C4; EOG IO; chin EMG EMG; ECG ECG; respiration Abdominal = Respi'
    assert A.sleepfm_channels_text(entry(eeg=('Fz',)), []).endswith('respiration none found by name')
