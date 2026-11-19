"""Reading Curry 9 recordings (Neuroscan / Compumedics Curry: a .cdt signal file beside its .cdt.dpo parameter file).

The signal comes from MNE (mne.io.read_raw_curry, with the curryreader package). MNE reads it lazily, but opening a
recording loads the whole file once (about 15 s for 7 GB), so a header that does not need the signal is read here, from
the .dpo: a text file of named blocks (`NAME START` ... `NAME END` for key = value pairs, `NAME START_LIST` ...
`NAME END_LIST` for lists), holding the two channel groups (EEG, then the 'others': EOG, EMG, ECG, sensors), the start of
the recording, the impedances and the electrode positions. A .dpo states no filter.

Formerly tools_curry/curry_header.py and curry_io.py, used by the Curry twins of the tools (removed in phase 6 of
tools/plan_brainvision.md): a Curry recording is now read through recording_lib, like EDF and BrainVision.
"""
import datetime
import re
from pathlib import Path

import chardet
import mne

import brainvision_lib   # the names that say what a channel records (EOG / EMG / ECG), one rule for every format

IMPEDANCE_NOT_MEASURED = -1.0   # what a .dpo writes for an electrode not measured in that session
IMPEDANCE_DISABLED = -2.0       # ... and for a disabled / unconnected input


def dpo_path(cdt_path):
    """The parameter file of a .cdt: the same name followed by .dpo."""
    return Path(str(cdt_path) + '.dpo')


def _read_text(path):
    """A .dpo as text: UTF-8 with a BOM, UTF-16, else what chardet finds, Latin-1 at worst."""
    raw = Path(path).read_bytes()
    if raw[:3] == b'\xef\xbb\xbf':
        return raw[3:].decode('utf-8')
    if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
        return raw.decode('utf-16')
    encoding = chardet.detect(raw).get('encoding') or 'latin-1'
    try:
        return raw.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        return raw.decode('latin-1', errors='replace')


def _parse_blocks(text):
    """The blocks of a .dpo: {name: {key: value}} for the key = value blocks (the first of a repeated name kept), and
    {name + '_list': rows} for the lists that are read (LABELS and LABELS_OTHERS: one label a row; SENSORS: (x, y, z)
    in mm; IMPEDANCE_VALUES: one row a measuring session, one value a channel, in Ohm)."""
    result = {}
    for found in re.finditer(r'^(\w[\w\s]*?)\s+START\s*$(.+?)^\1\s+END\s*$', text, re.MULTILINE | re.DOTALL):
        name = found.group(1).strip()
        pairs = {}
        for line in found.group(2).splitlines():
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, _, value = line.partition('=')
                pairs[key.strip()] = value.strip()
        result.setdefault(name, pairs)
    for found in re.finditer(r'^(\w[\w\s]*?)\s+START_LIST.*?$(.+?)^\1\s+END_LIST\s*$', text, re.MULTILINE | re.DOTALL):
        name = found.group(1).strip()
        rows = [ln.strip() for ln in found.group(2).splitlines() if ln.strip() and not ln.strip().startswith('#')]
        if name in ('LABELS', 'LABELS_OTHERS'):
            result[name + '_list'] = rows
        elif name == 'SENSORS':
            coords = []
            for row in rows:
                parts = row.split()
                try:
                    coords.append(tuple(float(p) for p in parts[:3]))
                except ValueError:
                    pass
            result['SENSORS_list'] = coords
        elif name == 'IMPEDANCE_VALUES':
            sessions = []
            for row in rows:
                try:
                    sessions.append([float(v) for v in row.split()])
                except ValueError:
                    pass
            result['IMPEDANCE_VALUES_list'] = sessions
    return result


def _start_datetime(params):
    """The start of the recording from the StartYear ... StartMillisec fields, naive, to the millisecond (the precision
    of the file: the clock times of the scored-event exports are converted against it). None when absent."""
    try:
        fields = [int(params.get(key, 0) or 0) for key in
                  ('StartYear', 'StartMonth', 'StartDay', 'StartHour', 'StartMin', 'StartSec', 'StartMillisec')]
        if fields[0] and fields[1] and fields[2]:
            return datetime.datetime(*fields[:6], fields[6] * 1000)
    except (ValueError, TypeError):
        pass
    return None


def read_dpo(cdt_path):
    """What the .dpo of a .cdt says: sfreq, n_samples, start_datetime (or None), eeg_labels, other_labels, eeg_unit,
    other_unit, sensor_xyz (one (x, y, z) per EEG channel, mm), impedances (one row per session, in Ohm, the sentinels
    -1 not measured and -2 disabled). Raises when the .dpo is missing (fatal for this recording)."""
    path = dpo_path(cdt_path)
    if not path.is_file():
        raise FileNotFoundError(f'{Path(cdt_path).name}: its Curry parameter file {path.name} is not beside it')
    blocks = _parse_blocks(_read_text(path))
    params = blocks.get('DATA_PARAMETERS', {})
    return {
        'sfreq': float(params.get('SampleFreqHz', 0) or 0),
        'n_samples': int(float(params.get('NumSamples', 0) or 0)),
        'start_datetime': _start_datetime(params),
        'eeg_labels': blocks.get('LABELS_list', []),
        'other_labels': blocks.get('LABELS_OTHERS_list', []),
        'eeg_unit': str(blocks.get('DEVICE_PARAMETERS', {}).get('DataUnit', '')).strip(),
        'other_unit': str(blocks.get('DEVICE_PARAMETERS_OTHERS', {}).get('DataUnit', '')).strip(),
        'sensor_xyz': blocks.get('SENSORS_list', []),
        'impedances': blocks.get('IMPEDANCE_VALUES_list', []),
    }


def impedances_kohm(dpo):
    """({channel: last measured impedance in kOhm, None without a measure}, {channel: 'not measured' / 'disabled'}),
    in the shape of a BrainVision header. Every EEG channel is listed (one never measured is worth a look); a channel of
    the other group only with a measure (its auxiliary inputs are usually not measured: listed, they would be flagged in
    every recording)."""
    labels = dpo['eeg_labels'] + dpo['other_labels']
    n_eeg = len(dpo['eeg_labels'])
    sessions = dpo['impedances']
    kohm, text = {}, {}
    for k, label in enumerate(labels):
        values = [row[k] for row in sessions if k < len(row)]
        measured = [v for v in values if v not in (IMPEDANCE_NOT_MEASURED, IMPEDANCE_DISABLED)]
        if not measured and k >= n_eeg:
            continue
        kohm[label] = measured[-1] / 1000.0 if measured else None       # the file is in Ohm
        if not measured:
            text[label] = 'disabled' if values and values[-1] == IMPEDANCE_DISABLED else 'not measured'
    return kohm, text


def _transducer_type(name, in_eeg_group):
    """The type of a channel, in the EDF header's field, so that the toolkit's 'transducer type OR name' rule works: the
    name first (an EOG electrode may sit in the EEG group), then EEG for the EEG group of the .dpo."""
    if re.search(brainvision_lib.EOG_NAME_RE, name, re.IGNORECASE):
        return 'EOG'
    if re.search(brainvision_lib.EMG_NAME_RE, name, re.IGNORECASE):
        return 'EMG'
    if re.search(brainvision_lib.ECG_NAME_RE, name, re.IGNORECASE):
        return 'ECG'
    return 'EEG' if in_eeg_group else ''


def read_cdt_header(cdt_path):
    """A Curry header in the shape of edf_io_lib.read_edf_header_custom and brainvision_lib.read_vhdr_header (global
    fields + one list per channel field), from the .dpo only, the signal not read. No EDF digital range (the signal is
    stored as float): physical_* and digital_* are empty, so no bounds_pct and the ±500 µV display cap, the rules of a
    format without a declared range. No filter in a .dpo: prefiltering is empty ('not stated'). One data record holds one
    sample (n_data_records = n_times). Raises when the .dpo is missing."""
    dpo = read_dpo(cdt_path)
    names = dpo['eeg_labels'] + dpo['other_labels']
    n_eeg = len(dpo['eeg_labels'])
    n_ch = len(names)
    sfreq = dpo['sfreq']
    start = dpo['start_datetime']
    units = [dpo['eeg_unit'] or 'uV'] * n_eeg + [dpo['other_unit'] or 'uV'] * (n_ch - n_eeg)
    kohm, text = impedances_kohm(dpo)
    freq = str(int(sfreq)) if sfreq == int(sfreq) else str(sfreq)
    return {
        'version': 'Curry',
        'patient_id': '',
        'recording_id': '',
        'start_date': start.strftime('%d.%m.%y') if start else '',
        'start_time': start.strftime('%H.%M.%S') if start else '',
        'header_bytes': 0,
        'n_data_records': dpo['n_samples'],
        'duration_data_record': 1.0 / sfreq if sfreq else 0.0,
        'n_channels': n_ch,
        'channel': names,
        'transducer_type': [_transducer_type(ch, k < n_eeg) for k, ch in enumerate(names)],
        'dimension': units,
        'physical_min': [''] * n_ch,
        'physical_max': [''] * n_ch,
        'digital_min': [''] * n_ch,
        'digital_max': [''] * n_ch,
        'prefiltering': [''] * n_ch,
        'samples_per_record': ['1'] * n_ch,
        'reserved': [''] * n_ch,
        'sampling_frequency': [freq] * n_ch,
        'format': 'curry',
        'binary_format': 'FLOAT',
        'sfreq': sfreq,
        'n_times': dpo['n_samples'],
        'duration_s': dpo['n_samples'] / sfreq if sfreq else 0.0,
        'in_file': [True] * n_ch,
        'resolution': [None] * n_ch,
        'gradient': [None] * n_ch,
        'dc_coupled': [False] * n_ch,
        'online_reference': None,
        'impedance_kohm': kohm,
        'impedance_text': text,
        'impedance_time': None,
        'n_segments': 1,
        'data_file': Path(cdt_path).name,
        'comment_warning': '',
        'channel_group': ['EEG'] * n_eeg + ['other'] * (n_ch - n_eeg),
    }


def read_cdt_start_datetime(cdt_path):
    """The start of the recording (naive, to the millisecond), None when the .dpo does not give it or cannot be read."""
    try:
        return read_dpo(cdt_path)['start_datetime']
    except Exception:
        return None


def read_cdt_acquisition(cdt_path, keep_labels=None):
    """What tool 6 shows of a recording, as brainvision_lib.read_vhdr_acquisition: (sampling frequency text, high-pass
    text, dc_coupled). A .dpo states no filter: 'not stated', and dc_coupled False (tool 6's high-pass stays unticked)."""
    header = read_cdt_header(cdt_path)
    kept = [ch for ch in header['channel'] if keep_labels is None or ch in keep_labels]
    return f"{header['sampling_frequency'][0]} Hz", 'not stated' if kept else '?', False


def read_raw_cdt(cdt_path, include=None, preload=False):
    """A Curry recording as an MNE Raw, without its event annotations (the .cdt.ceo markers: the .fif written later do not
    carry them, as for BrainVision). `include` works like read_raw_edf's: the channels to keep, in the order of the file,
    the names absent from the file being ignored. MNE types the EEG group 'eeg' (with its positions) and the other group
    'misc'."""
    raw = mne.io.read_raw_curry(str(cdt_path), preload=preload, verbose='ERROR')
    raw.set_annotations(None)
    if include is not None:
        wanted = set(include)
        raw.pick([ch for ch in raw.ch_names if ch in wanted])
    return raw
