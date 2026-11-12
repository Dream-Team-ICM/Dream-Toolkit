"""Reading BrainVision recordings (.vhdr + .vmrk + .eeg, the three files BrainVision Recorder writes).

The signal, the sampling rate, the channel names and types come from MNE, which opens a recording without reading its
signal and copes with the data formats and code pages. Only what MNE does not expose is read here, as text: the channel
units, the free-text [Comment] of the .vhdr (online reference, filters, impedances, gradients) and the .vmrk (recording
start, recording pauses).

The [Comment] has no guaranteed format: everything taken from it is optional. Absent or unreadable, it is simply not
reported and nothing stops. The task markers of the .vmrk are ignored for now (SPEC, Task triggers).
"""
import datetime
import math
import re
import tempfile
import warnings
from pathlib import Path

import mne

# Names that say what a channel records, whatever input of the amplifier it uses. IO / SO / LO are the usual names of the
# electrodes around the eyes (infra-orbital, supra-orbital, outer canthus).
EOG_NAME_RE = r'eog|^(IO|SO|LO)\d*[LR]?$|^(LOC|ROC)$'
EMG_NAME_RE = r'emg|chin|menton'
ECG_NAME_RE = r'ecg|ekg'


def _read_text(path):
    """A BrainVision text file, decoded as MNE does: its Codepage, ANSI meaning cp1252, else Latin-1."""
    raw = Path(path).read_bytes()
    found = re.search(r'Codepage=(.+)', raw.decode('ascii', 'ignore'), re.IGNORECASE)
    codepage = found.group(1).strip() if found else 'utf-8'
    if codepage.upper() == 'ANSI':
        codepage = 'cp1252'
    try:
        return raw.decode(codepage)
    except (UnicodeDecodeError, LookupError):
        return raw.decode('latin-1')


def _companion_name(text, key, vhdr_path):
    """The file a .vhdr names under `key` (DataFile, MarkerFile), or None. '$b' stands for the name of the .vhdr."""
    found = re.search(rf'^{key}=(.+?)\s*$', text, re.MULTILINE)
    return found.group(1).replace('$b', Path(vhdr_path).stem) if found else None


def _check_companions(vhdr_path):
    """A clear message when the .eeg or the .vmrk a .vhdr names is not beside it: the three files of a recording refer to
    one another by name, so renaming only one of them breaks the recording."""
    vhdr_path = Path(vhdr_path)
    text = _read_text(vhdr_path)
    for key in ('DataFile', 'MarkerFile'):
        name = _companion_name(text, key, vhdr_path)
        if name and not (vhdr_path.parent / name).is_file():
            raise FileNotFoundError(f"{vhdr_path.name} names {key}={name}, which is not beside it (a BrainVision recording "
                                    'cannot be renamed file by file: the .vhdr holds the names of the two others)')


def _open_raw(vhdr_path):
    """(Raw, warning). MNE reads the filter table of the [Comment] and stops on one it does not understand (a table cut
    short, another name than in [Channel Infos]), and it does not know the '$b' some .vhdr use for their own name. Neither
    should cost us the recording: it is then opened through a copy of the header without its [Comment], with the full paths
    of the two other files. The warning says so, empty otherwise."""
    vhdr_path = Path(vhdr_path)
    text = _read_text(vhdr_path)
    warning = ''
    if '$b' not in text:
        try:
            return mne.io.read_raw_brainvision(str(vhdr_path), preload=False, verbose='ERROR'), ''
        except (AssertionError, IndexError):
            warning = 'the [Comment] of the .vhdr could not be read by MNE: its filters are not reported to it'
    header = re.sub(r'^Codepage=.*$', 'Codepage=UTF-8', text.split('[Comment]')[0], flags=re.MULTILINE)
    for key in ('DataFile', 'MarkerFile'):
        full = vhdr_path.parent / _companion_name(text, key, vhdr_path)
        header = re.sub(rf'^{key}=.*$', lambda _: f'{key}={full}', header, flags=re.MULTILINE)
    with tempfile.TemporaryDirectory(prefix='bvh') as folder:
        copy = Path(folder) / vhdr_path.name
        copy.write_text(header, encoding='utf-8')
        return mne.io.read_raw_brainvision(str(copy), preload=False, verbose='ERROR'), warning


def _channel_infos(text):
    """{channel name: (resolution or None, unit)} from [Channel Infos], with MNE's defaults (no unit means µV). A comma in
    a name is coded '\\1' in the file. µV is written 'uV', as in the EDF headers."""
    infos = {}
    section = re.search(r'^\[Channel Infos\](.*?)(?=^\[|\Z)', text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
    for found in re.finditer(r'^Ch\d+=(.*)$', section.group(1) if section else '', re.MULTILINE):
        props = found.group(1).strip().split(',')
        if len(props) < 4 or props[3] == '':
            props = (props + ['', '', ''])[:3] + ['µV']
        try:
            resolution = float(props[2])
        except ValueError:
            resolution = None
        unit = props[3].replace('\xc2', '').strip().replace('µ', 'u').replace('μ', 'u')
        infos[props[0].replace(r'\1', ',')] = (resolution, unit)
    return infos


def _value_or_none(token):
    """A number from the [Comment], or None for DC / Off / NaN and anything else that is not a number."""
    try:
        value = float(token.replace(',', '.'))
    except (ValueError, AttributeError):
        return None
    return None if math.isnan(value) else value


def _in_hz(token, in_seconds):
    """A filter setting in Hz. A low cut-off is often given as a time constant t in seconds: f = 1 / (2 pi t)."""
    value = _value_or_none(token)
    if value is None or value <= 0:
        return None
    return 1.0 / (2 * math.pi * value) if in_seconds else value


def _filter_rows(lines, start):
    """The rows under a table header of the [Comment]: [(row number, text after it)], until the first other line."""
    rows = []
    for line in lines[start + 1:]:
        found = re.match(r'\s*(\d+)\s+(.*)$', line)
        if not found:
            break
        rows.append((int(found.group(1)), found.group(2)))
    return rows


def _table_header_index(lines, pattern, after=0):
    return next((i for i in range(after, len(lines)) if re.match(pattern, lines[i])), None)


def read_comment_info(text, ch_names):
    """What the [Comment] of a .vhdr says, as a dict: online_reference (name or None), highpass / lowpass / notch (one
    value in Hz or None per channel of `ch_names`: the setting that really limits the signal, hardware and software
    together), dc_coupled (one bool per channel: the [Comment] says the hardware has no high-pass and no software filter
    adds one; False when it says nothing, which is not the same), gradient (one (value, unit) or None per channel),
    impedance_kohm ({electrode: kOhm, None without a measure}, from the last measure in the file), impedance_text
    ({electrode: what the file says instead of a number, e.g. 'Out of Range!', 'not connected'}), impedance_time.
    Never raises."""
    n_ch = len(ch_names)
    info = {'online_reference': None, 'highpass': [None] * n_ch, 'lowpass': [None] * n_ch, 'notch': [None] * n_ch,
            'dc_coupled': [False] * n_ch, 'gradient': [None] * n_ch, 'impedance_kohm': {}, 'impedance_text': {},
            'impedance_time': None}
    if '[Comment]' not in text:
        return info
    comment = text.split('[Comment]', 1)[1]
    lines = comment.splitlines()
    try:
        found = re.search(r'^\s*Reference Channel Name\s*=\s*(\S.*?)\s*$', comment, re.MULTILINE)
        info['online_reference'] = found.group(1) if found else None

        # Hardware table: # Name Phys.Chn. Resolution [Unit] LowCutoff HighCutoff Notch [Gradient Offset]. A long name can
        # touch the next column ('Left Thumb67'), so the row is cut at the known channel name.
        i_hw = _table_header_index(lines, r'#\s+Name')
        if i_hw is not None:
            head = lines[i_hw]
            shift = 3 if 'Resolution / Unit' in head else 2
            low_s, high_s = 'Low Cutoff [s]' in head, 'High Cutoff [s]' in head
            for row, body in _filter_rows(lines, i_hw):
                k = row - 1
                if not (0 <= k < n_ch and body.startswith(ch_names[k])):
                    continue
                tokens = body[len(ch_names[k]):].split()
                if len(tokens) < shift + 3:
                    continue
                info['highpass'][k] = _in_hz(tokens[shift], low_s)
                info['dc_coupled'][k] = tokens[shift].upper() == 'DC' or _value_or_none(tokens[shift]) == 0
                info['lowpass'][k] = _in_hz(tokens[shift + 1], high_s)
                info['notch'][k] = _value_or_none(tokens[shift + 2])
                for token in tokens[shift + 3:]:
                    gradient = re.fullmatch(r'([-+\d.,eE]+)\[([^\]]+)\]', token)
                    if gradient:
                        info['gradient'][k] = (_value_or_none(gradient.group(1)), gradient.group(2))
                        break

        # Software filters come on top of the hardware ones: the higher high-pass and the lower low-pass limit the signal.
        i_sw = next((i for i, line in enumerate(lines) if 'S o f t w a r e' in line), None)
        i_sw = None if i_sw is None else _table_header_index(lines, r'#\s+Low Cutoff', i_sw)
        if i_sw is not None:
            head = lines[i_sw]
            low_s, high_s = 'Low Cutoff [s]' in head, 'High Cutoff [s]' in head
            for row, body in _filter_rows(lines, i_sw):
                k = row - 1
                tokens = body.split()
                if not (0 <= k < n_ch) or len(tokens) < 3:
                    continue
                low, high = _in_hz(tokens[0], low_s), _in_hz(tokens[1], high_s)
                if low is not None:
                    info['highpass'][k] = max(low, info['highpass'][k] or 0.0)
                    info['dc_coupled'][k] = False
                if high is not None:
                    info['lowpass'][k] = min(high, info['lowpass'][k] or math.inf)
                if _value_or_none(tokens[2]) is not None:
                    info['notch'][k] = _value_or_none(tokens[2])

        for i, line in enumerate(lines):
            found = re.match(r'\s*Impedance\s*\[(k?)Ohm\]\s*(?:at\s*(\S+))?', line, re.IGNORECASE)
            if not found:
                continue
            block, texts = {}, {}
            for row in lines[i + 1:]:
                cell = re.match(r'\s*(.+?)\s*:\s*(.*?)\s*$', row)
                if not cell:
                    break
                value = _value_or_none(cell.group(2))
                block[cell.group(1)] = value if value is None or found.group(1) else value / 1000.0
                if value is None:
                    texts[cell.group(1)] = cell.group(2)
            if block:
                info['impedance_kohm'], info['impedance_text'], info['impedance_time'] = block, texts, found.group(2)
    except Exception:
        pass   # a [Comment] written differently must never cost us the recording: what was understood is kept
    return info


def read_vmrk_info(vhdr_path):
    """From the marker file a .vhdr names: start_datetime (naive, cut to the second as the EDF start and the scored-event
    exports are, None when the file gives no date), n_segments (the 'New Segment' markers: more than one means the
    recording was paused and the samples no longer follow the clock), n_markers. Never raises."""
    info = {'start_datetime': None, 'n_segments': 0, 'n_markers': 0}
    try:
        vhdr_path = Path(vhdr_path)
        name = _companion_name(_read_text(vhdr_path), 'MarkerFile', vhdr_path)
        text = _read_text(vhdr_path.parent / name)
    except Exception:
        return info
    markers = re.findall(r'^Mk\d+=(.*)$', text, re.MULTILINE)
    info['n_markers'] = len(markers)
    segments = [m.strip() for m in markers if m.startswith('New Segment,')]
    info['n_segments'] = len(segments)
    if segments:
        stamp = segments[0].split(',')[5] if segments[0].count(',') >= 5 else ''
        if re.fullmatch(r'\d{14,}', stamp):
            try:
                info['start_datetime'] = datetime.datetime.strptime(stamp[:14], '%Y%m%d%H%M%S')
            except ValueError:      # the all-zero date some writers use
                pass
    return info


def _transducer_type(name, mne_type, unit, has_positions):
    """The type of a channel, in the EDF header's field, so that the toolkit's 'transducer type OR name' rule works: the
    name first (an EOG electrode sits on an EEG input), then EEG for what MNE types EEG. In a file with electrode positions
    MNE gives that type to the channels that have one, which settles it. Without positions it gives it to every channel
    in volts, so only µV counts as EEG (the sensors of the auxiliary inputs are in mV). This keeps TP9, TP10, FT9, FT10,
    absent from the toolkit's list of EEG names."""
    if re.search(EOG_NAME_RE, name, re.IGNORECASE):
        return 'EOG'
    if re.search(EMG_NAME_RE, name, re.IGNORECASE):
        return 'EMG'
    if re.search(ECG_NAME_RE, name, re.IGNORECASE):
        return 'ECG'
    return 'EEG' if mne_type == 'eeg' and (has_positions or unit == 'uV') else ''


def _prefiltering(highpass, lowpass, notch):
    """The filters of a channel, written like the EDF 'prefiltering' field that edf_io_lib.extract_filter_value reads."""
    parts = [f'{tag}:{value:.4g}Hz' for tag, value in (('HP', highpass), ('LP', lowpass), ('NOTCH', notch))
             if value is not None]
    return ' '.join(parts)


def read_vhdr_header(vhdr_path):
    """A BrainVision header in the shape of edf_io_lib.read_edf_header_custom (global fields + one list per channel field),
    without reading the signal. No EDF digital range here: physical_* and digital_* are empty. One data record holds one
    sample (n_data_records = n_times), which keeps the exact duration and the sampling frequency formula.

    The online reference is listed last when the [Comment] names it and the file does not hold it (it can be recreated,
    see read_raw_bv): `in_file` is False for that row.

    Added fields: format, binary_format, sfreq, n_times, duration_s, in_file, resolution, gradient, dc_coupled,
    online_reference, impedance_kohm, impedance_text, impedance_time, n_segments, data_file, comment_warning. Raises when
    the recording cannot be opened (fatal for this recording)."""
    vhdr_path = Path(vhdr_path)
    _check_companions(vhdr_path)
    raw, comment_warning = _open_raw(vhdr_path)
    text = _read_text(vhdr_path)
    names = list(raw.ch_names)
    infos = _channel_infos(text)
    comment = read_comment_info(text, names)
    vmrk = read_vmrk_info(vhdr_path)
    sfreq = float(raw.info['sfreq'])
    units = [infos.get(ch, (None, 'uV'))[1] for ch in names]
    has_positions = raw.get_montage() is not None
    transducers = [_transducer_type(ch, t, u, has_positions) for ch, t, u in zip(names, raw.get_channel_types(), units)]
    resolution = [infos.get(ch, (None, 'uV'))[0] for ch in names]
    prefilter = [_prefiltering(h, l, n) for h, l, n in zip(comment['highpass'], comment['lowpass'], comment['notch'])]
    gradient = list(comment['gradient'])
    dc_coupled = list(comment['dc_coupled'])
    in_file = [True] * len(names)

    reference = comment['online_reference']
    if reference and reference not in names:
        names.append(reference)
        units.append('uV')
        transducers.append('EEG')
        resolution.append(None)
        gradient.append(None)
        dc_coupled.append(False)
        prefilter.append('online reference, not in the file')
        in_file.append(False)

    n_ch = len(names)
    start = vmrk['start_datetime']
    freq = str(int(sfreq)) if sfreq == int(sfreq) else str(sfreq)
    binary_format = re.search(r'^\s*BinaryFormat\s*=\s*(\S+)', text, re.MULTILINE | re.IGNORECASE)
    return {
        'version': 'BrainVision',
        'patient_id': '',
        'recording_id': '',
        'start_date': start.strftime('%d.%m.%y') if start else '',
        'start_time': start.strftime('%H.%M.%S') if start else '',
        'header_bytes': 0,
        'n_data_records': int(raw.n_times),
        'duration_data_record': 1.0 / sfreq,
        'n_channels': n_ch,
        'channel': names,
        'transducer_type': transducers,
        'dimension': units,
        'physical_min': [''] * n_ch,
        'physical_max': [''] * n_ch,
        'digital_min': [''] * n_ch,
        'digital_max': [''] * n_ch,
        'prefiltering': prefilter,
        'samples_per_record': ['1'] * n_ch,
        'reserved': [''] * n_ch,
        'sampling_frequency': [freq] * n_ch,
        'format': 'brainvision',
        'binary_format': binary_format.group(1).upper() if binary_format else '',
        'sfreq': sfreq,
        'n_times': int(raw.n_times),
        'duration_s': raw.n_times / sfreq,
        'in_file': in_file,
        'resolution': resolution,
        'gradient': gradient,
        'dc_coupled': dc_coupled,
        'online_reference': reference,
        'impedance_kohm': comment['impedance_kohm'],
        'impedance_text': comment['impedance_text'],
        'impedance_time': comment['impedance_time'],
        'n_segments': vmrk['n_segments'],
        'data_file': _companion_name(text, 'DataFile', vhdr_path),
        'comment_warning': comment_warning,
    }


def read_vhdr_acquisition(vhdr_path, keep_labels=None):
    """What tool 6 shows of a recording, restricted to the channels in `keep_labels` when given: (sampling frequency text,
    high-pass text, dc_coupled). Written like the texts of its EDF reading: every distinct high-pass, 'none/DC' when the
    [Comment] says there is none, 'not stated' when it does not say. dc_coupled = all these channels are DC-coupled."""
    header = read_vhdr_header(vhdr_path)
    kept = [i for i, ch in enumerate(header['channel'])
            if header['in_file'][i] and (keep_labels is None or ch in keep_labels)]
    highpass = set()
    for i in kept:
        found = re.search(r'HP:([\d.eE+-]+)Hz', header['prefiltering'][i])
        highpass.add(f'{float(found.group(1)):g} Hz' if found else 'none/DC' if header['dc_coupled'][i] else 'not stated')
    return (f"{header['sampling_frequency'][0]} Hz", ' / '.join(sorted(highpass)) if highpass else '?',
            bool(kept) and all(header['dc_coupled'][i] for i in kept))


def _reference_position(name):
    """Where an electrode sits on MNE's easycap-M1 montage, written with the same spherical angles and head radius as the
    [Coordinates] of a .vhdr. None for a name it does not know."""
    positions = mne.channels.make_standard_montage('easycap-M1').get_positions()['ch_pos']
    return next((xyz for ch, xyz in positions.items() if ch.lower() == name.lower()), None)


def channels_on_online_reference(vhdr_path):
    """(online reference, [channels recorded against it]) from the .vhdr, or (None, []) when the [Comment] names no
    reference. A channel of an EEG input is recorded against the online reference: its unit is µV and the [Comment] gives
    it no gradient (the auxiliary inputs carry one, and their own unit). An EOG electrode on an EEG input therefore also
    holds the EEG of the reference electrode, inverted. Read as text, without opening the recording. Never raises."""
    try:
        text = _read_text(vhdr_path)
        infos = _channel_infos(text)
        names = list(infos)
        comment = read_comment_info(text, names)
    except Exception:
        return None, []
    reference = comment['online_reference']
    if not reference:
        return None, []
    return reference, [ch for k, ch in enumerate(names) if infos[ch][1] == 'uV' and comment['gradient'][k] is None]


def recreate_online_reference(raw, reference):
    """Recreate the online reference (MNE's documented method): a channel of zeros, which becomes a real signal once the
    data are re-referenced. It needs the data in memory. MNE leaves its position empty, so it is set here, which the
    topographies of tool 8 need."""
    raw.load_data()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Location for this channel')   # the position is set just below
        mne.add_reference_channels(raw, [reference], copy=False)
    position = _reference_position(reference)
    montage = raw.get_montage()
    if position is None or montage is None:
        return raw
    known = montage.get_positions()
    ch_pos = {ch: xyz for ch, xyz in known['ch_pos'].items() if ch != reference}
    ch_pos[reference] = position
    raw.set_montage(mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame=known['coord_frame']),
                    on_missing='ignore', verbose='ERROR')
    return raw


def read_raw_bv(vhdr_path, include=None, add_online_reference=True, preload=False):
    """A BrainVision recording as an MNE Raw, without its task markers (the .fif written later do not carry them).

    `include` works like read_raw_edf's: the channels to keep, in the order of the file, the names absent from the file
    being ignored. The online reference is recreated only when `include` names it and the file does not hold it: a read
    that does not ask for it stays lazy. Tool 6, which does not re-reference, passes add_online_reference=False (it would
    be a flat channel)."""
    vhdr_path = Path(vhdr_path)
    _check_companions(vhdr_path)
    raw, _ = _open_raw(vhdr_path)
    raw.set_annotations(None, verbose='ERROR')
    if include is None:
        return raw.load_data() if preload else raw
    keep = [ch for ch in raw.ch_names if ch in include]
    if not keep:
        raise ValueError(f'{vhdr_path.name}: none of the channels asked for is in the file ({list(include)})')
    reference = None
    if add_online_reference:
        reference = read_comment_info(_read_text(vhdr_path), raw.ch_names)['online_reference']
    raw.pick(keep)
    if reference and reference in include and reference not in keep:
        return recreate_online_reference(raw, reference)
    return raw.load_data() if preload else raw
