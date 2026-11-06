"""Hand-written BrainVision recordings for the tests: the three files Recorder writes, with the layout of its [Comment]
(actiCHamp) and invented values. Never a real recording."""
from pathlib import Path

import numpy as np

FORMATS = {'IEEE_FLOAT_32': '<f4', 'INT_16': '<i2', 'INT_32': '<i4'}


def recorder_comment(names, resolutions, units, reference='Cz', software_low=15.9155, no_software_filter=(),
                     impedances=None, unit_column=True, low_unit='s', gradients=None):
    """The [Comment] text of BrainVision Recorder for these channels. `gradients` = {channel: text} adds the two columns of
    the auxiliary inputs. `impedances` = a list of blocks, each {electrode: value or text}."""
    head = '#     Name      Phys. Chn.    '
    head += 'Resolution / Unit' if unit_column else 'Resolution [µV]'
    head += f'   Low Cutoff [{low_unit}]   High Cutoff [Hz]   Notch [Hz]    Gradient         Offset'
    lines = ['', 'BrainVision Recorder Professional   -   V. 1.23.0003', '', '',
             'A m p l i f i e r  S e t u p', '============================', f'Number of channels: {len(names)}', '',
             'Channels', '--------', head]
    for i, (name, res, unit) in enumerate(zip(names, resolutions, units), start=1):
        unit_text = f'{res} {unit}' if unit_column else f'{res}'
        grad = (gradients or {}).get(name)
        tail = f'           {grad}       0[mV] = 0[mV]' if grad else ''
        # a long name touches the next column, as in the real files
        lines.append(f'{i:<6}{name:<12}{i:<11}{unit_text}             DC              140              Off{tail}')
    lines += ['']
    if reference:
        lines += [f'Reference Channel Name = {reference}', 'Reference Phys. Chn.   = 18', '']
    if software_low is not None:
        lines += ['', 'S o f t w a r e  F i l t e r s', '==============================',
                  f'#     Low Cutoff [{low_unit}]   High Cutoff [Hz]   Notch [Hz]']
        for i, name in enumerate(names, start=1):
            low = 'Off' if name in no_software_filter else software_low
            lines.append(f'{i:<7}{low:<17}Off                Off')
        lines += ['']
    for block in impedances or []:
        lines += ['', 'Impedance [kOhm] at 12:00:00 :']
        lines += [f'{name + ":":<10}{value:>4}' if not isinstance(value, str) else f'{name + ":":<10}{value}'
                  for name, value in block.items()]
        lines += ['']
    return '\n'.join(lines) + '\n'


def write_brainvision(folder, name, stored, ch_names, sfreq=500, resolutions=None, units=None, coordinates=None,
                      comment=None, data_format='IEEE_FLOAT_32', orientation='MULTIPLEXED', start='20260102030405123456',
                      extra_segments=(), markers=(), codepage='UTF-8', vhdr_encoding='utf-8'):
    """Write <name>.vhdr, .vmrk, .eeg in `folder`. `stored` = (n_channels, n_times), the values as the file holds them
    (the physical value is value x resolution, in the channel's unit). `coordinates` = {channel: (r, theta, phi)}, all
    channels listed (r = 0 for a channel without a position) or None for no [Coordinates] section. Returns the .vhdr path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    resolutions = resolutions if resolutions is not None else [0.1] * len(ch_names)
    units = units if units is not None else ['µV'] * len(ch_names)
    stored = np.asarray(stored)
    values = stored.T if orientation == 'MULTIPLEXED' else stored
    values.astype(FORMATS[data_format]).tofile(folder / f'{name}.eeg')

    channel_lines = [f"Ch{i}={ch.replace(',', chr(92) + '1')},,{res},{unit}"
                     for i, (ch, res, unit) in enumerate(zip(ch_names, resolutions, units), start=1)]
    parts = ['BrainVision Data Exchange Header File Version 1.0', '; Data created by the tests', '',
             '[Common Infos]', f'Codepage={codepage}', f'DataFile={name}.eeg', f'MarkerFile={name}.vmrk',
             'DataFormat=BINARY', f'DataOrientation={orientation}', f'NumberOfChannels={len(ch_names)}',
             f'SamplingInterval={1e6 / sfreq:g}', '', '[Binary Infos]', f'BinaryFormat={data_format}', '',
             '[Channel Infos]'] + channel_lines + ['']
    if coordinates is not None:
        parts += ['[Coordinates]'] + [f'Ch{i}={",".join(str(v) for v in coordinates[ch])}'
                                      for i, ch in enumerate(ch_names, start=1)] + ['']
    if comment is not None:
        parts += ['[Comment]', comment]
    (folder / f'{name}.vhdr').write_bytes('\n'.join(parts).encode(vhdr_encoding))

    stamp = f',{start}' if start else ''
    marks = [f'Mk1=New Segment,,1,1,0{stamp}']
    for pos in extra_segments:
        marks.append(f'Mk{len(marks) + 1}=New Segment,,{pos},1,0{stamp}')
    for desc, pos in markers:
        marks.append(f'Mk{len(marks) + 1}=Stimulus,{desc},{pos},1,0')
    vmrk = ['BrainVision Data Exchange Marker File Version 1.0', '', '[Common Infos]', f'Codepage={codepage}',
            f'DataFile={name}.eeg', '', '[Marker Infos]'] + marks + ['']
    (folder / f'{name}.vmrk').write_bytes('\n'.join(vmrk).encode(vhdr_encoding))
    return folder / f'{name}.vhdr'
