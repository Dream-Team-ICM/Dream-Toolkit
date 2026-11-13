"""High-density montages (phase 5 of tools/plan_brainvision.md): above HD_CHANNEL_THRESHOLD (12) channels the figures switch
to their dense layout; at or below it they are drawn as before (that the test chain's outputs do not move is proven by
tests/refactor_check.py). Quick: each figure changed, at 62 channels (a 64-electrode cap less Cz, the online reference,
and one more), against the same figure at 4 channels. Standard: tool 6 on a 20-channel BrainVision study."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import mne                        # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
import pytest                     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
import preprocessing_lib as PL        # noqa: E402
import qc_rejected_epochs_lib as L    # noqa: E402
import quality_lib as QL              # noqa: E402
import spectral_lib as SL             # noqa: E402
import pipeline                       # noqa: E402
from brainvision_files import recorder_comment, write_brainvision    # noqa: E402

SFREQ = 128
EEG62 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'C4', 'T8', 'TP9', 'CP5',
         'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'AF7', 'AF3', 'AF4', 'AF8',
         'F5', 'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8', 'FT10', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz',
         'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3', 'POz', 'PO4', 'PO8']


def names(n_ch):
    return EEG62[:n_ch]


def visible_axes(fig):
    return [ax for ax in fig.axes if ax.get_visible()]


def panel_rows(axes):
    return len({round(ax.get_position().y0, 3) for ax in axes})


# ---------------------------------------------------------------------------------------------------------
# The figures (quick)
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.quick
@pytest.mark.tool7
@pytest.mark.tool0
def test_tool7_heatmap_keeps_a_readable_hypnogram():
    """Above 12 channels the hypnogram strip keeps about 1 inch, as in tool 8bis; at 4 channels the figure is the same."""
    def hypnogram_height(n_ch):
        combined = np.zeros((120, n_ch), dtype=int)
        combined[10:20, 1] = 1
        fig = PL.plot_rejection_heatmap(combined, names(n_ch), np.array(['W'] * 40 + ['N2'] * 80), 'x')
        size, height = tuple(fig.get_size_inches()), fig.axes[0].get_position().height * fig.get_figheight()
        plt.close(fig)
        return size, height
    assert hypnogram_height(4)[0] == (20.0, 3.8)                   # max(3, min(0.45 * 4 + 2, 8)): as before
    for n_ch in (30, 62):
        assert 0.7 < hypnogram_height(n_ch)[1] < 1.2, n_ch          # was 0.12 inch at 62 channels


@pytest.mark.quick
@pytest.mark.tool9
@pytest.mark.tool0
def test_tool9_figures_become_a_grid_of_8_columns():
    rng = np.random.default_rng(0)
    freqs = np.arange(1.0, 31.0)
    n_ep = 20
    masks = {'N2': np.ones(n_ep, dtype=bool)}
    for n_ch, n_rows in ((4, 1), (62, 8)):
        fig = SL.plot_psd_per_stage(freqs, masks, names(n_ch), [], 1, 'PSD', 'PSD (dB/Hz)',
                                    values_db=rng.normal(size=(n_ep, n_ch, len(freqs))))
        panels = visible_axes(fig)
        assert len(panels) == n_ch and panel_rows(panels) == n_rows, n_ch
        assert fig.get_size_inches()[0] == 20.0
        plt.close(fig)
    meta = pd.DataFrame({'epoch_idx': range(n_ep), 'stage': ['N2'] * n_ep})
    res = {'exponent': rng.normal(1, 0.1, (n_ep, 62)), 'offset': rng.normal(size=(n_ep, 62))}
    fig = SL.plot_aperiodic_over_night(meta, res, names(62), [])
    panels = visible_axes(fig)
    assert len(panels) == 124 and panel_rows(panels) == 16          # exponent and offset rows, 8 blocks of 8 channels
    assert [panels[0].get_ylabel(), panels[8].get_ylabel()] == ['Exponent', 'Offset']
    plt.close(fig)


def make_epochs(ch_names, montage=True):
    """Three 30-s epochs of noise; positions from the standard 10-05 montage (a name outside it gets none)."""
    info = mne.create_info(ch_names, SFREQ, 'eeg')
    data = np.random.default_rng(1).normal(0, 20e-6, (3, len(ch_names), 30 * SFREQ))
    epochs = mne.EpochsArray(data, info, verbose=False)
    if montage:
        epochs.set_montage(mne.channels.make_standard_montage('standard_1005'), on_missing='ignore')
    epochs.metadata = pd.DataFrame({'stage': ['N2'] * 3, 'reject_flag': [False] * 3, 'reject_method': ['none'] * 3})
    return epochs


@pytest.mark.quick
@pytest.mark.tool8
@pytest.mark.tool0
def test_the_topomaps_use_the_electrodes_that_have_a_position():
    """An eye electrode on an EEG input (IO, no position) no longer removes the topomaps of a BrainVision cap."""
    def topomap_titles(P):
        freqs, psds = L.compute_psds(P['epochs'], fmax=30.0)
        fig = L.plot_epoch_detail(P, 0, freqs, psds, L.DEFAULT_THRESHOLDS, table_rows=L.DEFAULT_TABLE_ROWS)
        titles = [ax.get_title() for ax in fig.axes if ax.get_title().startswith(('Peak-to-peak', '1/f MAE'))]
        plt.close(fig)
        return titles
    cap = L.participant_from_epochs(make_epochs(EEG62 + ['IO']))
    assert len(L.positioned_channels(cap)) == 62
    titles = topomap_titles(cap)
    assert len(titles) == 2 and all('no position: IO' in t for t in titles), titles
    assert topomap_titles(L.participant_from_epochs(make_epochs(EEG62, montage=False))) == []   # an EDF: none


def make_raw(n_ch):
    info = mne.create_info(names(n_ch), SFREQ, 'eeg')
    return mne.io.RawArray(np.random.default_rng(2).normal(0, 20e-6, (n_ch, 60 * SFREQ)), info, verbose=False)


@pytest.mark.quick
@pytest.mark.tool6
@pytest.mark.tool0
def test_tool6_overlays_name_only_the_flagged_channels_on_a_dense_montage():
    figures = (lambda raw, flagged: QL.plot_overlay_timeseries(raw, SFREQ, flagged=flagged),
               lambda raw, flagged: QL.plot_overlay_psd(raw, SFREQ, flagged=flagged),
               lambda raw, flagged: QL.plot_overlay_distribution(raw, flagged=flagged))
    for make in figures:
        fig = make(make_raw(4), ['F7'])
        assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == names(4)   # as before
        plt.close(fig)
        fig = make(make_raw(62), ['F7'])
        assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == \
            ['61 channels within thresholds', 'F7 (flagged)']
        plt.close(fig)


@pytest.mark.quick
@pytest.mark.tool6
@pytest.mark.tool0
def test_tool6_scatter_by_electrode_on_a_dense_montage():
    rng = np.random.default_rng(3)
    rows = pd.DataFrame({'file_id': 'x', 'channel': EEG62, 'exclude': [False] * 61 + [True],
                         **{m: rng.normal(size=62) for m in QL.OVERVIEW_KEY_METRICS}})
    rows['bounds_pct'] = np.nan                     # no declared physical range (BrainVision)
    fig = QL.plot_participant_electrode_scatter(rows)
    panels = visible_axes(fig)
    assert len(panels) == len(QL.OVERVIEW_KEY_METRICS) - 1
    assert 'At EDF bounds (%)' not in [ax.get_title() for ax in panels]
    assert {label.get_rotation() for label in panels[0].get_xticklabels()} == {90.0}
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------
# Tool 6 on a 20-channel BrainVision study (standard)
# ---------------------------------------------------------------------------------------------------------
def spherical(ch_names):
    """BrainVision [Coordinates] (r, theta, phi) in degrees from MNE's standard_1005 positions."""
    pos = mne.channels.make_standard_montage('standard_1005').get_positions()['ch_pos']
    out = {}
    for ch in ch_names:
        x, y, z = pos[ch] / np.linalg.norm(pos[ch])
        out[ch] = (1, round(float(np.degrees(np.arccos(z))), 1), round(float(np.degrees(np.arctan2(y, x))), 1))
    return out


@pytest.mark.tool6
def test_tool6_reports_of_a_dense_cap(tmp_path):
    """20 electrodes, C3 dead: the participant report details C3 alone, the database report has one figure per metric
    and no per-electrode section."""
    data = tmp_path / 'data'
    chans = names(20)
    rng = np.random.default_rng(4)
    for k, pid in enumerate(('hd01', 'hd02')):
        uv = rng.normal(0, 20, (len(chans), 20 * 30 * SFREQ))
        uv[chans.index('C3')] *= 0.01
        write_brainvision(data, pid, uv / 0.05, chans, sfreq=SFREQ, resolutions=[0.05] * len(chans),
                          coordinates=spherical(chans), comment=recorder_comment(chans, [0.05] * len(chans),
                                                                                 ['µV'] * len(chans)))
        (data / f'{pid}_hypno_remapped.txt').write_text('\n'.join(['W'] * 10 + ['N2'] * 10) + '\n', encoding='utf-8')
    (data / 'config_param').mkdir()
    cfg = {pid: {'config': 'config. 1', 'remap': {ch: ch for ch in chans}, 'ref_channels': []} for pid in ('hd01', 'hd02')}
    (data / 'config_param' / 'remap_reref_persubject.json').write_text(json.dumps(cfg), encoding='utf-8')
    text = pipeline.run_tool6(data)
    assert 'Traceback' not in text, text[-3000:]
    reports = data / 'reports_quality_overview'
    report = (reports / 'hd01_quality_overview.html').read_text(encoding='utf-8')
    assert 'Per-channel detail below for' in report and 'C3: histogram' in report and 'Fz: histogram' not in report
    overview = (reports / 'dataset_overview.html').read_text(encoding='utf-8')
    assert 'Distributions by electrode (one figure per metric' in overview and 'Electrode: Fz' not in overview
