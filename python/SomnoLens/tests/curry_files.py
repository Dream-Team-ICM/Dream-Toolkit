"""Hand-written Curry 9 recordings for the tests: a .cdt (float32, every channel of one sample, then the next sample) and
its .cdt.dpo, with the blocks that MNE (through curryreader) and tools/curry_lib.py read, and invented values. Never a
real recording.

The .dpo writes each channel group four times under the name its readers look for: `LABELS START` / `LABELS END` (an
empty block of key = value pairs), then `LABELS START_LIST` ... `LABELS END_LIST` with one label a row; curryreader
takes the labels between the third and the fourth mention, curry_lib takes the list. Same for the sensor positions
(EEG group only, as MNE expects one position per EEG channel)."""
from pathlib import Path

import mne
import numpy as np


def standard_positions_mm(eeg_names):
    """{channel: (x, y, z) in mm} from MNE's standard_1005 montage (the unit of a .dpo)."""
    positions = mne.channels.make_standard_montage('standard_1005').get_positions()['ch_pos']
    return {ch: tuple(float(v) * 1000 for v in positions[ch]) for ch in eeg_names}


def _list_block(name, rows):
    """A group block as Curry writes it: the empty key = value block, then the list."""
    return [f'{name} START', f'{name} END', '', f'{name} START_LIST\t# Do not edit!'] + list(rows) + \
        [f'{name} END_LIST', '']


def write_curry(folder, name, data_uV, eeg_names, other_names=(), sfreq=500, start=(2026, 1, 2, 22, 4, 5, 123),
                positions=None, impedances=None, other_unit='uV', events=()):
    """Write <name>.cdt and <name>.cdt.dpo in `folder`. `data_uV` = (n_channels, n_times) in µV, the EEG group first, then
    the others. `positions` = {EEG channel: (x, y, z) mm} (default: standard_1005). `impedances` = a list of measuring
    sessions, each one value per channel in Ohm (-1 not measured, -2 disabled); default: one session at 5 kOhm for the
    EEG group, -1 for the others. `events` = [(sample, code)], written in a .cdt.ceo (the native markers). `start` =
    (year, month, day, hour, minute, second, millisecond). Returns the .cdt path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    eeg_names, other_names = list(eeg_names), list(other_names)
    names = eeg_names + other_names
    data_uV = np.asarray(data_uV, dtype=float)
    assert data_uV.shape[0] == len(names)
    n_times = data_uV.shape[1]
    cdt = folder / f'{name}.cdt'
    data_uV.T.astype('<f4').tofile(cdt)       # one sample of every channel after the other

    positions = positions if positions is not None else standard_positions_mm(eeg_names)
    if impedances is None:
        impedances = [[5000.0] * len(eeg_names) + [-1.0] * len(other_names)]
    year, month, day, hour, minute, second, millisecond = start
    lines = ['ASCII_PARAMETERS START', '   Program = tests/curry_files.py', 'ASCII_PARAMETERS END', '',
             'DATA_PARAMETERS START',
             '   FileVersion        = 90',
             f'   NumSamples         = {n_times}',
             f'   NumChannels        = {len(names)}',
             '   NumTrials          = 1',
             f'   SampleFreqHz       = {sfreq:g}',
             f'   SampleTimeUsec     = {1e6 / sfreq:g}',
             '   TriggerOffsetUsec  = 0',
             '   DataFormat         = BINARY',
             '   DataSampOrder      = SAMPLE',
             '   DataByteOrder      = LITTLE_ENDIAN',
             f'   StartYear          = {year}',
             f'   StartMonth         = {month}',
             f'   StartDay           = {day}',
             f'   StartHour          = {hour}',
             f'   StartMin           = {minute}',
             f'   StartSec           = {second}',
             f'   StartMillisec      = {millisecond}',
             '   AmplifierInfo      = ',
             'DATA_PARAMETERS END', '',
             'DEVICE_PARAMETERS START',
             f'   NumChanThisGroup   = {len(eeg_names)}',
             '   DataUnit           = uV',
             'DEVICE_PARAMETERS END', '']
    if other_names:
        lines += ['DEVICE_PARAMETERS_OTHERS START',
                  f'   NumChanThisGroup   = {len(other_names)}',
                  f'   DataUnit           = {other_unit}',
                  'DEVICE_PARAMETERS_OTHERS END', '']
    lines += _list_block('LABELS', eeg_names)
    if other_names:
        lines += _list_block('LABELS_OTHERS', other_names)
    lines += _list_block('SENSORS', [' '.join(f'{v:.3f}' for v in positions[ch]) for ch in eeg_names])
    lines += ['IMPEDANCE_VALUES START_LIST\t# Do not edit!']
    lines += [' '.join(f'{v:g}' for v in session) for session in impedances]
    lines += ['IMPEDANCE_VALUES END_LIST', '']
    # curryreader needs a final newline (it looks for blocks that may be absent from the end of the text)
    (folder / f'{name}.cdt.dpo').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    if events:
        rows = [' '.join(str(v) for v in (sample, 0, code, 0, sample, sample, 0, 0, 0, 0, 0)) for sample, code in events]
        ceo = (['NUMBER_LIST START_LIST'] + rows + ['NUMBER_LIST END_LIST', 'REMARK_LIST START_LIST'] +
               [f'Trigger {code}' for _, code in events] + ['REMARK_LIST END_LIST'])
        (folder / f'{name}.cdt.ceo').write_text('\n'.join(ceo) + '\n', encoding='utf-8')
    return cdt
