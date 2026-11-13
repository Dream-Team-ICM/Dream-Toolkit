"""One way to read a recording, whatever its format: the tools call these functions and the extension of the file decides.

The EDF branch is exactly the reading the tools already do, so an EDF study gives the same results. A study may hold
only one format: study_files_lib.list_recordings finds the recordings.

Also here, because it depends on how the channels were recorded: the reference an EOG gets (eog_reference), with the
names of tool 2's configuration it relies on (EOG names, mastoids).
"""
import re
from pathlib import Path

import mne
import pandas as pd

import brainvision_lib
from brainvision_lib import channels_on_online_reference, read_raw_bv, read_vhdr_header, read_vmrk_info
from edf_io_lib import read_edf_header_custom, read_edf_start_datetime


def file_format(path):
    """'edf' or 'brainvision', from the extension of the file. Raises for anything else."""
    suffix = Path(path).suffix.lower()
    if suffix == '.edf':
        return 'edf'
    if suffix == '.vhdr':
        return 'brainvision'
    raise ValueError(f'{Path(path).name}: not a recording format the toolkit reads (.edf, .vhdr)')


def read_header(path, fallback_encoding=None):
    """The header of a recording, without its signal, in the shape of edf_io_lib.read_edf_header_custom."""
    if file_format(path) == 'edf':
        return read_edf_header_custom(path, fallback_encoding=fallback_encoding)
    return read_vhdr_header(path)


def channel_table(header):
    """A header as a table with one row per channel. The impedances of a BrainVision header are per electrode (the ground
    is one, it is not a channel), so they stay out of it."""
    return pd.DataFrame({key: value for key, value in header.items() if not isinstance(value, dict)})


def read_start_datetime(path):
    """The start of the recording as a naive datetime, or None when the file does not give it. None too for a BrainVision
    recording that was paused (several 'New Segment' markers): after a pause its samples no longer follow the clock, so
    clock times (scored-event exports, clock axes) cannot be placed on it."""
    if file_format(path) == 'edf':
        return read_edf_start_datetime(path)
    info = read_vmrk_info(path)
    return info['start_datetime'] if info['n_segments'] <= 1 else None


def read_raw(path, include=None, add_online_reference=True, preload=False):
    """A recording as an MNE Raw. `include` = the channels to keep, the names absent from the file being ignored.
    add_online_reference concerns BrainVision only (brainvision_lib.read_raw_bv)."""
    if file_format(path) == 'edf':
        return mne.io.read_raw_edf(str(path), preload=preload, encoding='latin-1', include=include, verbose=False)
    return read_raw_bv(path, include=include, add_online_reference=add_online_reference, preload=preload)


def online_reference(path):
    """(online reference, [channels recorded against it]) when the recording says so, else (None, []). An EDF header does
    not say which reference its channels were recorded against: (None, []), and an EDF is used as it is."""
    if file_format(path) == 'edf':
        return None, []
    return channels_on_online_reference(path)


def flat_online_reference(path, cfg_entry):
    """The harmonized name of the online reference when it would stay flat in the montage of tool 2, else None.
    A BrainVision online reference (e.g. Cz) is recreated as a channel of zeros. Re-referenced, it becomes a real signal
    (Cz - TP9), so it is kept when tool 2 declares a reference (a list of channels or 'average'). Without one it would
    stay at 0 and be flagged in every epoch: tools 7 and 0 leave it out. None for an EDF (no online reference) and when
    tool 2 did not keep the online reference in the montage."""
    online, _ = online_reference(path)
    target = (cfg_entry or {}).get('remap', {}).get(online) if online else None
    ref = (cfg_entry or {}).get('ref_channels', [])
    if not target or ref == 'average' or (isinstance(ref, list) and ref):
        return None
    return target


def recreate_online_reference(raw, name):
    """Add the online reference `name` to a Raw whose data are loaded (a channel of zeros, which becomes a signal once
    the data are re-referenced): for a read that first opens every channel lazily, then loads only those it needs."""
    return brainvision_lib.recreate_online_reference(raw, name)


# ---------------------------------------------------------------------------------------------------------
# The reference of an EOG
# ---------------------------------------------------------------------------------------------------------
# EOG names that tool 2 may have kept in the remap: never an EEG electrode
EOG_NAME_RE = re.compile(r'(EOG|^LOC$|^ROC$|^E1$|^E2$)', re.IGNORECASE)
# 10-20 names of the left mastoid / earlobe, then the 10-10 electrode over the mastoid (a cap without M1 / M2). The
# first one found is used, so a recording with M1 / M2 never takes TP9 / TP10.
LEFT_MASTOIDS = ('M1', 'A1', 'TP9')
RIGHT_MASTOIDS = ('M2', 'A2', 'TP10')


def _reference_words(refs):
    """'TP9', 'mean of TP9, TP10', 'average of 63 EEG channels': the reference of a derivation, in a report."""
    if len(refs) == 1:
        return refs[0]
    return f'mean of {", ".join(refs)}' if len(refs) <= 3 else f'average of {len(refs)} EEG channels'


def eog_reference(path, cfg_entry, eog, contralateral, bipolar_pair=False):
    """How each EOG gets the reference of the EEG: ({label: harmonized names of the channels whose mean is subtracted},
    description for a parameters file or a report ('' when every EOG is left as recorded), notes).
    `eog` = {label: original name}, label 'EOG-L' (left eye) or 'EOG-R' (right eye). `contralateral` = the reference the
    sleep-scoring models learnt (tool 3bis's default), else tool 2's declared reference.

    An EOG electrode on an EEG input is recorded against the online reference (Cz on many 10-10 caps): it then also
    holds the EEG of that electrode, inverted (spindles, K-complexes and slow waves are large at the vertex). Subtracting
    a mastoid removes it: (eye - Cz) - (M - Cz) = eye - M. Such an EOG gets the reference of the EEG: the mastoid of the
    other side when contralateral (left eye on M2, right eye on M1), tool 2's reference otherwise. Left as recorded: an
    EOG on an auxiliary (already bipolar) input; an EDF, whose header does not say against what a channel was recorded;
    the L - R derivation of two EOG (`bipolar_pair`: the common term cancels); tool 2 declaring no reference; an EOG
    without a mastoid of the other side (said in the notes)."""
    online, on_reference = online_reference(path)
    eog = {label: orig for label, orig in eog.items() if orig in on_reference}
    if not eog or bipolar_pair:
        return {}, '', []
    names = list(cfg_entry.get('remap', {}).values())
    upper = {n.upper(): n for n in names}
    declared = cfg_entry.get('ref_channels', [])
    refs, notes = {}, []
    for label, orig in eog.items():
        if contralateral:
            other_side = RIGHT_MASTOIDS if label == 'EOG-L' else LEFT_MASTOIDS
            mastoid = next((upper[m] for m in other_side if m in upper), None)
            if mastoid is None:
                notes.append(f'{label} ({orig}) is recorded against the online reference {online} and no mastoid of '
                             'the other side is configured in tool 2: used as recorded')
            else:
                refs[label] = [mastoid]
        elif declared == 'average':
            refs[label] = [ch for ch in names if not EOG_NAME_RE.search(ch)]
        elif isinstance(declared, list) and declared:
            refs[label] = [ch for ch in declared if ch in names]
        # tool 2 declared no reference: the EEG stays against the online reference, and so does the EOG
    refs = {label: r for label, r in refs.items() if r}
    description = '; '.join(f'{label} = {eog[label]} - {_reference_words(r)}' for label, r in refs.items())
    if description:
        description += f' ({", ".join(eog[label] for label in refs)} recorded against the online reference {online})'
    return refs, description, notes
