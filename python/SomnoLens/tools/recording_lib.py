"""One way to read a recording, whatever its format: the tools call these functions and the extension of the file decides.

The EDF branch is exactly the reading the tools already do, so an EDF study gives the same results. A study may hold
only one format: study_files_lib.list_recordings finds the recordings.
"""
from pathlib import Path

import mne
import pandas as pd

from brainvision_lib import read_raw_bv, read_vhdr_header, read_vmrk_info
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
    """The start of the recording as a naive datetime, or None when the file does not give it."""
    if file_format(path) == 'edf':
        return read_edf_start_datetime(path)
    return read_vmrk_info(path)['start_datetime']


def read_raw(path, include=None, add_online_reference=True, preload=False):
    """A recording as an MNE Raw. `include` = the channels to keep, the names absent from the file being ignored.
    add_online_reference concerns BrainVision only (brainvision_lib.read_raw_bv)."""
    if file_format(path) == 'edf':
        return mne.io.read_raw_edf(str(path), preload=preload, encoding='latin-1', include=include, verbose=False)
    return read_raw_bv(path, include=include, add_online_reference=add_online_reference, preload=preload)
