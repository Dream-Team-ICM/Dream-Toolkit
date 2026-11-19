"""High-density montages (phases 5 and 5 bis of tools/plan_brainvision.md): above HD_CHANNEL_THRESHOLD (12) channels the
figures switch to their dense layout; at or below it they are drawn as before (that the test chain's outputs do not move is
proven by tests/refactor_check.py). Quick: each figure changed, at 62 channels (a 64-electrode cap less Cz, the online
reference, and one more), against the same figure at 4 channels; phase 5 bis also checks what changed for every montage
(tool 7's heatmap aligned on its epochs, the shares of clean epochs of tool 9). Standard: tool 6 on a 20-channel
BrainVision study."""
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
        fig = L.plot_epoch_detail(P, 0, freqs, psds, L.DEFAULT_THRESHOLDS)
        titles = [ax.get_title() for ax in fig.axes
                  if ax.get_title().startswith(('Peak-to-peak', 'Gradient', '1/f MAE', '1/f R²'))]
        plt.close(fig)
        return titles
    cap = L.participant_from_epochs(make_epochs(EEG62 + ['IO']))
    assert len(L.positioned_channels(cap)) == 62
    titles = topomap_titles(cap)
    assert len(titles) == 4 and all('no position: IO' in t for t in titles), titles      # the four metrics of the table
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
# Phase 5 bis: the feedback of the visual inspection (quick)
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.quick
@pytest.mark.tool7
@pytest.mark.tool0
def test_tool7_heatmap_hypnogram_spans_the_epochs_of_the_matrix():
    """The colour bar has its own column: the hypnogram and the matrix start and end on the same epochs, at any density.
    Above 12 channels the figure is at most 14 inches wide."""
    for n_ch in (4, 62):
        combined = np.zeros((120, n_ch), dtype=int)
        fig = PL.plot_rejection_heatmap(combined, names(n_ch), np.array(['W'] * 40 + ['N2'] * 80), 'x')
        hyp, matrix = fig.axes[0].get_position(), fig.axes[2].get_position()
        assert hyp.x0 == pytest.approx(matrix.x0) and hyp.width == pytest.approx(matrix.width), n_ch
        assert fig.get_size_inches()[0] == (20.0 if n_ch == 4 else 14.0)
        plt.close(fig)


@pytest.mark.quick
@pytest.mark.tool8bis
@pytest.mark.tool0
def test_tool8bis_decision_figure_is_smaller_on_a_dense_montage():
    import autoreject_lib as AR

    def size(n_ch):
        epochs, chans = list(range(100)), names(n_ch)
        dec = {'channels': chans, 'stage_by_epoch': {e: 'N2' for e in epochs},
               'reject_epoch': {e: False for e in epochs}, 'rejected_channels': []}
        fig = AR.plot_decision_heatmap(pd.DataFrame(0, index=epochs, columns=chans), dec, 'x', [])
        inches = tuple(fig.get_size_inches())
        plt.close(fig)
        return inches
    assert size(4) == pytest.approx((100 / 6, 1.0 + 0.4 + 0.88 + 0.7))          # as before
    # was 16.7 x 14.1 inches; phase 5 ter: lower hypnogram and decision strip, so that it holds on one page
    assert size(62) == pytest.approx((14.0, 0.6 + 0.2 + 0.14 * 62 + 0.7))


@pytest.mark.quick
@pytest.mark.tool8
@pytest.mark.tool0
def test_the_detail_table_lists_every_channel_and_scrolls():
    P = L.participant_from_epochs(make_epochs(EEG62))
    P['data_uV'][0, 5] *= 10                                       # F7, ten times the others: over the p-p threshold
    freqs, psds = L.compute_psds(P['epochs'], fmax=30.0)
    values = L.epoch_channel_values(P, 0, freqs, psds, L.DEFAULT_THRESHOLDS)
    assert 'ptp' in values['breached'][5]
    html = L.epoch_metric_table_html(P, 0, values)
    assert all(f';">{ch}</td>' in html or f'left;">{ch}</td>' in html for ch in EEG62)   # every channel, in a row
    assert 'overflow-y:auto' in html and 'position:sticky' in html                   # a box that scrolls, its header stays
    assert 'color:#c0392b; font-weight:bold;">F7</td>' in html


@pytest.mark.quick
@pytest.mark.tool6
@pytest.mark.tool0
def test_tool6_flags_a_channel_far_from_the_others_above_12_channels():
    thresholds = {'flat_pct': 3.5, 'bounds_pct': 1.0, 'n_peaks': 2, 'std_low': 5.0, 'hist_extreme_pct': 1.0}
    assert QL.comparison_flag(100.0, 0.0, 62)[0].startswith('noisy_vs_median=100% of the epochs > 20%')
    assert QL.comparison_flag(0.0, 50.0, 62)[0].startswith('flat_vs_median=50%')
    assert QL.comparison_flag(100.0, 0.0, 12) == [] and QL.comparison_flag(np.nan, np.nan, 62) == []
    for n_ch in (12, 20):
        raw = make_raw(n_ch)
        raw._data[0] *= 10                                             # Fp1, ten times the others
        _, ch_flags, rows, _ = QL.compute_channel_metrics(raw, SFREQ, 'x', thresholds, None, [], {}, {})
        assert rows[0]['comparison_alert'].startswith('noisy'), n_ch
        if n_ch > 12:
            assert rows[0]['exclude'] and QL.flag_criteria(ch_flags['Fp1']) == ['noisy_vs_median']
            # a table made before (the alert apart): Skip asks
            stored = pd.DataFrame(rows).assign(suspect_reason='', exclude=False)
            assert ('flags of Fp1', 'none', 'noisy_vs_median') in QL.settings_differences(
                stored, False, None, False, None, thresholds)
        else:
            assert not rows[0]['exclude'] and ch_flags['Fp1'] == []                 # a PSG: an alert apart, as before


@pytest.mark.quick
@pytest.mark.tool9
@pytest.mark.tool0
def test_tool9_says_the_share_of_clean_epochs():
    meta = pd.DataFrame({'epoch_idx': range(3), 'stage': ['N2'] * 3})
    fig = SL.plot_clean_hypnogram(meta, [], 'x', n_total=8, stage_totals={'N2': 6, 'W': 2})
    assert [t.get_text() for t in fig.axes[1].texts] == ['3 (50 %)']
    plt.close(fig)
    rows = []
    for fid, excluded in (('p1', False), ('p2', True)):
        for ch in ('C3', 'C4'):
            for band in ('delta', 'sigma'):
                rows += [{'file_id': fid, 'stage': 'N2', 'channel': ch, 'band': band, 'n_epochs': 30,
                          'n_epochs_rejected': 10, 'n_epochs_out_of_scope': 0, 'participant_excluded': excluded},
                         {'file_id': fid, 'stage': 'W', 'channel': ch, 'band': band, 'n_epochs': 5,
                          'n_epochs_rejected': 5, 'n_epochs_out_of_scope': 0, 'participant_excluded': excluded}]
    stage_df = pd.DataFrame(rows)
    html = SL.clean_epochs_html(stage_df)
    assert '30 / 40 (75 %)' in html and '5 / 10 (50 %)' in html and '70 %' in html and '(excluded)' in html
    fig = SL.plot_group_counts(stage_df[~stage_df['participant_excluded']])
    assert [t.get_text() for t in fig.axes[0].texts] == ['70 %']                # (30 + 5) / (40 + 10)
    plt.close(fig)


@pytest.mark.quick
@pytest.mark.tool9
@pytest.mark.tool0
def test_tool9_overlays_the_psd_of_a_dense_montage():
    rng = np.random.default_rng(5)
    freqs = np.arange(1.0, 31.0)
    masks = {'N2': np.ones(20, dtype=bool), 'W': np.ones(20, dtype=bool)}
    fig = SL.plot_psd_overlay(freqs, masks, names(62), [], 1, 'PSD', 'PSD (dB/Hz)',
                              values_db=rng.normal(size=(20, 62, 30)))
    panels = visible_axes(fig)
    assert len(panels) == 2 and len(panels[0].lines) == 63                     # every channel and their mean
    plt.close(fig)

    def group_items(n_ch):
        psd_df = pd.DataFrame([{'file_id': f, 'stage': 'N2', 'channel': ch, 'freq_hz': fr, 'psd_db_from_log': 1.0}
                               for f in ('p1', 'p2') for ch in names(n_ch) for fr in freqs])
        items = SL.group_psd_items(psd_df, [], 'Group mean PSD per stage')
        SL.close_figs(items)
        return items
    items = group_items(62)
    assert [kind for kind, _, _ in items] == ['fig', 'html']
    assert items[1][2].startswith('<details>') and 'data:image/png;base64,' in items[1][2]
    assert [kind for kind, _, _ in group_items(4)] == ['fig']                   # as before
    # the log-log panels of a dense grid: 30 Hz left out, it overlapped 45 Hz
    freqs = np.linspace(0.5, 45.0, 90)
    psds = np.exp(rng.normal(size=(20, 62, 90)))
    fig = SL.plot_loglog_with_fit(freqs, psds, np.zeros((20, 62, 90)), masks, names(62), [], 1, (2.0, 45.0))
    assert 30 not in list(visible_axes(fig)[0].get_xticks())
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------
# Phase 5 ter: the second visual inspection (quick)
# ---------------------------------------------------------------------------------------------------------
def detail_figure(P):
    freqs, psds = L.compute_psds(P['epochs'], fmax=30.0)
    return L.plot_epoch_detail(P, 0, freqs, psds, L.DEFAULT_THRESHOLDS)


@pytest.mark.quick
@pytest.mark.tool8
@pytest.mark.tool0
def test_the_topomaps_are_scaled_on_their_thresholds():
    """One electrode 100 times too large no longer squeezes the others into one colour; a flat one gets a cyan ring."""
    P = L.participant_from_epochs(make_epochs(EEG62))
    P['data_uV'][0, 5] *= 100                         # F7
    P['data_uV'][0, 6] *= 0                           # F8, flat
    fig = detail_figure(P)
    maps = {ax.get_title().split('\n')[0]: ax for ax in fig.axes if ax.images}
    thr = L.DEFAULT_THRESHOLDS
    assert maps['Peak-to-peak (µV)'].images[0].get_clim() == (0, 2 * thr['amplitude_ptp_uV']['N2'])
    assert maps['1/f R²'].images[0].get_clim() == pytest.approx((2 * thr['1f_r2_min'] - 1, 1))
    ptp = maps['Peak-to-peak (µV)']
    assert 'red > ' in ptp.get_title() and 'cyan < ' in ptp.get_title()
    edges = [line.get_markeredgecolor() for line in ptp.get_lines()]
    assert L.TOPO_FLAT_COLOR in edges and L.TOPO_RING_COLOR in edges       # F8 cyan, F7 red
    plt.close(fig)


@pytest.mark.quick
@pytest.mark.tool8
@pytest.mark.tool0
def test_the_clean_reference_is_drawn_over_the_channels():
    for n_ch in (4, 62):
        fig = detail_figure(L.participant_from_epochs(make_epochs(EEG62[:n_ch])))
        ax = next(a for a in fig.axes if a.get_title().startswith('PSD vs clean'))     # the topomaps come first
        lines = {line.get_label(): line for line in ax.get_lines()}
        median = next(line for label, line in lines.items() if label.startswith('clean N2 median'))
        assert median.get_color() == 'black' and median.get_zorder() > max(
            line.get_zorder() for label, line in lines.items() if not label.startswith(('clean', '_child')))
        assert 'clean N2 IQR' in lines and lines['clean N2 IQR'].get_linestyle() == ':'
        assert not ax.collections or n_ch > 12        # no grey fill under the reference (only this epoch's band, dense)
        plt.close(fig)


@pytest.mark.quick
@pytest.mark.tool9
@pytest.mark.tool0
def test_tool9_draws_every_per_channel_figure_on_the_mean_of_the_channels_above_12():
    rng = np.random.default_rng(5)
    n_ep, freqs = 20, np.linspace(0.5, 30.0, 60)
    meta = pd.DataFrame({'epoch_idx': range(n_ep), 'stage': ['N2'] * n_ep})
    masks = {'N2': np.ones(n_ep, dtype=bool)}
    for n_ch, n_items in ((4, 1), (62, 2)):
        psds = np.exp(rng.normal(size=(n_ep, n_ch, len(freqs))))
        res = {k: rng.normal(1, 0.1, (n_ep, n_ch)) for k in ('offset', 'knee', 'exponent', 'r2', 'mae', 'n_peaks')}
        res['fit_ok'] = np.ones((n_ep, n_ch), dtype=bool)
        bands = {'delta': {'power_db': rng.normal(size=(n_ep, n_ch))}}
        for items in (SL.loglog_items('x', freqs, psds, np.zeros_like(psds), masks, names(n_ch), [], 1, (2.0, 30.0), res),
                      SL.aperiodic_night_items('x', meta, res, names(n_ch), []),
                      SL.band_power_items('x', bands, names(n_ch), masks, [], 'power_db', 'Power (dB/Hz)')):
            assert len(items) == n_items, n_ch
            if n_items == 2:
                assert visible_axes(items[0][2])[0].get_title().startswith('mean of the 62 channels')
                assert items[1][0] == 'html' and '<details>' in items[1][2]
            plt.close('all')
        rows = pd.DataFrame([{'file_id': f, 'stage': 'N2', 'band': 'delta', 'channel': ch, 'power_db_from_log_mean': v,
                              'exponent_mean': v, 'offset_mean': v}
                             for f in ('a', 'b') for ch, v in zip(names(n_ch), rng.normal(size=n_ch))])
        assert len(SL.group_band_items('x', rows, 'power_db_from_log_mean', 'Power', [])) == n_items
        assert len(SL.group_aperiodic_items('x', rows, [])) == n_items
        plt.close('all')


@pytest.mark.quick
@pytest.mark.tool9
@pytest.mark.tool0
def test_a_long_title_wraps_on_a_narrow_figure():
    n_ep, freqs = 10, np.linspace(0.5, 30.0, 60)
    values = np.random.default_rng(6).normal(size=(n_ep, 62, len(freqs)))
    title = 'Mean PSD per stage after removing the aperiodic component'
    fig = SL.plot_psd_overlay(freqs, {'W': np.ones(n_ep, dtype=bool)}, names(62), [], 1, title, 'dB', values_db=values)
    assert '\n' in fig.get_suptitle() and fig.get_suptitle().replace('\n', ' ') == f'{title}: every channel'
    plt.close(fig)
    fig = SL.plot_psd_overlay(freqs, {s: np.ones(n_ep, dtype=bool) for s in ('W', 'N1', 'N2')}, names(62), [], 1, title,
                              'dB', values_db=values)
    assert '\n' not in fig.get_suptitle()           # wide enough: one line, as before
    plt.close(fig)


@pytest.mark.quick
@pytest.mark.tool9
def test_tool9_says_how_many_participants_reach_the_minimum_of_epochs():
    line = SL.min_epochs_html({'a': {'W': 30, 'N2': 10}, 'b': {'W': 25, 'N2': 25}}, 20, ['W', 'N2', 'R'])
    assert 'W 2/2 (median 27.5), N2 1/2 (median 17.5), R 0/2 (median 0)' in line
    assert 'For naps or short recordings, a lower minimum includes them' in line and '(N2, R)' in line
    assert 'lower minimum' not in SL.min_epochs_html({'a': {'W': 30}}, 20, ['W'])
    rows = pd.DataFrame({'file_id': ['a', 'b', 'a'], 'stage': ['W', 'W', 'N2'], 'n_epochs': [30, 25, 10]})
    html = SL.missing_stages_html(rows, ['W', 'N2', 'R'], 20)
    assert 'W' not in html.split('<li>')[1] and '<b>N2</b>: 1 participant(s) have kept epochs' in html
    assert '<b>R</b>: no participant has a kept epoch' in html
    assert SL.missing_stages_html(rows, ['W'], 20) is None


@pytest.mark.quick
@pytest.mark.tool8
@pytest.mark.tool0
def test_tool0_keeps_its_signal_in_float32_and_tool8_in_float64():
    """Tool 0's explorer reads its signal 50 epochs at a time into float32 (the float64 values rounded once); tool 8 keeps
    float64."""
    n_ep = 120
    data = np.random.default_rng(2).normal(0, 20e-6, (n_ep, 4, 30 * SFREQ))

    def epochs():
        ep = mne.EpochsArray(data, mne.create_info(names(4), SFREQ, 'eeg'), verbose=False)
        ep.metadata = pd.DataFrame({'stage': ['N2'] * n_ep, 'reject_flag': [False] * n_ep, 'reject_method': ['none'] * n_ep})
        return ep
    P64, P32 = L.participant_from_epochs(epochs()), L.participant_from_epochs(epochs(), dtype=np.float32)
    assert P64['data_uV'].dtype == np.float64 and P32['data_uV'].dtype == np.float32
    assert np.array_equal(P32['data_uV'], P64['data_uV'].astype(np.float32))


@pytest.mark.quick
@pytest.mark.tool8
@pytest.mark.tool0
def test_the_psd_of_a_few_epochs_is_the_psd_of_every_epoch():
    """What lets tool 0 compute the detail panel's PSD on demand: each spectrum depends on its epoch only (Welch, then tool
    7's smoothing spectrum by spectrum)."""
    ep = make_epochs(EEG62[:4])
    smoothing = {'enabled': True, 'method': 'median+lowess', 'median_span_hz': 3.0, 'lowess_span_hz': 2.0}
    for kw in ({}, {'smoothing': smoothing}):
        _, every = L.compute_psds(ep, fmax=30.0, **kw)
        _, some = L.compute_psds(ep[[0, 2]], fmax=30.0, **kw)
        assert np.allclose(some, every[[0, 2]], rtol=1e-12, atol=0), kw


@pytest.mark.quick
@pytest.mark.tool8
@pytest.mark.tool0
def test_tool0_detail_reference_is_a_sample_of_the_stage_on_a_dense_montage():
    """Tool 0 computes the PSD of the clean reference when a stage is first shown: every clean epoch of the stage on a PSG,
    48 spread over the night on a 62-channel cap; the legend says how many of the stage's clean epochs it shows."""
    stages = np.array(['W'] * 100 + ['N2'] * 400)
    reject = np.zeros(500, dtype=bool)
    reject[[150, 151]] = True
    for n_ch, n_ref in ((6, 398), (62, 48)):
        idx, n_clean = L.detail_reference_epochs({'stages': stages, 'reject_flag': reject, 'ch_names': names(n_ch)}, 120)
        assert n_clean == 398 and len(idx) == n_ref, n_ch
        assert set(stages[idx]) == {'N2'} and not reject[idx].any() and (idx[0], idx[-1]) == (100, 499), n_ch
    P = L.participant_from_epochs(make_epochs(EEG62[:4]))
    freqs, psds = L.compute_psds(P['epochs'], fmax=30.0)
    fig = L.plot_epoch_detail(P, 0, freqs, psds, L.DEFAULT_THRESHOLDS, n_clean=10)
    assert 'clean N2 median (n=3 of 10 epochs)' in [line.get_label() for line in fig.axes[0].get_lines()]
    plt.close(fig)


@pytest.mark.quick
@pytest.mark.tool6
@pytest.mark.tool0
def test_tool6_overlays_say_why_a_channel_is_flagged():
    reasons = {'F7': ['flat_pct=5.0% > 3.5%', 'n_peaks=2 (bimodal)']}
    for n_ch, expected in ((4, ['Fp1', 'Fp2', 'F7: flat_pct=5.0%; n_peaks=2', 'F3']),
                           (62, ['61 channels within thresholds', 'F7: flat_pct=5.0%; n_peaks=2'])):
        fig = QL.plot_overlay_psd(make_raw(n_ch), SFREQ, flagged=reasons)
        assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == expected
        plt.close(fig)


# ---------------------------------------------------------------------------------------------------------
# Phase 5 quater: the third visual inspection (quick)
# ---------------------------------------------------------------------------------------------------------
def flagged_cap(n_ep=8):
    """A 62-channel N2 night where each epoch has one channel flagged (amplitude): tool 7's 'any' rule rejects them all."""
    info = mne.create_info(EEG62, SFREQ, 'eeg')
    data = np.random.default_rng(2).normal(0, 20e-6, (n_ep, 62, 30 * SFREQ))
    epochs = mne.EpochsArray(data, info, verbose=False)
    epochs.metadata = pd.DataFrame({'stage': ['N2'] * n_ep, 'reject_flag': [True] * n_ep,
                                    'reject_method': ['amplitude'] * n_ep, 'flag_amplitude': [True] * n_ep})
    flags = np.zeros((n_ep, 62), dtype=bool)
    flags[np.arange(n_ep), np.arange(n_ep) % 62] = True
    return L.participant_from_epochs(epochs), {'amplitude': flags}


@pytest.mark.quick
@pytest.mark.tool8
@pytest.mark.tool0
def test_a_dense_cap_takes_its_clean_reference_from_the_clean_channels():
    """Every epoch has a flagged channel: no epoch is clean, the clean channels of each epoch make the reference."""
    n_ep = 8
    P, pair_flags = flagged_cap(n_ep)
    freqs, psds = L.compute_psds(P['epochs'], fmax=30.0)
    pairs = L.clean_pairs(P['meta'], ['amplitude'], [], pair_flags, P['ch_names'])
    assert (pairs.sum(axis=1) == 61).all()

    def psd_axis(fig):
        return next(a for a in fig.axes if a.get_title().startswith('PSD vs clean'))
    fig = L.plot_epoch_detail(P, 0, freqs, psds, L.DEFAULT_THRESHOLDS, topomap=False)      # the epochs with no flag
    ax = psd_axis(fig)
    assert not any(line.get_label().startswith('clean N2 median') for line in ax.get_lines())
    assert any('No clean N2 reference: 0 epochs with no flagged channel' in t.get_text() for t in ax.texts)
    plt.close(fig)
    fig = L.plot_epoch_detail(P, 0, freqs, psds, L.DEFAULT_THRESHOLDS, topomap=False, clean_pairs=pairs)
    ax = psd_axis(fig)
    median = next(line for line in ax.get_lines() if line.get_label().startswith('clean N2 median'))
    assert median.get_label() == f'clean N2 median (clean channels of {n_ep} epochs)'
    expected = np.median([psds[e][pairs[e]].mean(axis=0) for e in range(n_ep)], axis=0)
    assert np.allclose(median.get_ydata(), expected, rtol=1e-12)
    fig.canvas.draw()                                     # the legend under the plot
    assert ax.get_legend().get_window_extent().y1 < ax.get_window_extent().y0
    plt.close(fig)
    # tool 8's Section 2 and tool 0's sample of the reference follow the same rule
    fig = L.plot_psd_overlays_grid(freqs, psds, P['stages'], P['reject_flag'], P['meta']['reject_method'].values,
                                   ['N2'], clean_pairs=pairs)
    assert f'clean median (clean channels of {n_ep} epochs)' in [line.get_label() for line in fig.axes[0].get_lines()]
    plt.close(fig)
    assert L.detail_reference_epochs(P, 0)[1] == 0
    half = pairs.copy()
    half[0, :40] = False                                  # epoch 0: 21 clean channels of 62, under half
    idx, n_clean = L.detail_reference_epochs(P, 0, clean_pairs=half)
    assert list(idx) == list(range(1, n_ep)) and n_clean == n_ep - 1


@pytest.mark.quick
@pytest.mark.tool8
@pytest.mark.tool0
def test_clean_pairs_follow_the_channel_layer_of_tool8():
    """A method without a per-channel column (event, 1/f of recomputed flags) flags the whole epoch; a dropped channel
    is never clean."""
    meta = pd.DataFrame({'stage': ['N2'] * 3, 'flag_amplitude': [True, False, False], 'flag_event': [False, True, False],
                         'flag_1f_r2': [False, False, True]})
    pair_flags = {'amplitude': np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=bool)}
    pairs = L.clean_pairs(meta, ['amplitude', 'event', '1f_r2'], [], pair_flags, ['A', 'B', 'C'], dropped_channels=['C'])
    assert pairs.tolist() == [[False, True, False], [False, False, False], [False, False, False]]


@pytest.mark.quick
@pytest.mark.tool3bis
@pytest.mark.tool5
@pytest.mark.tool6
@pytest.mark.tool7
@pytest.mark.tool8
@pytest.mark.tool8bis
@pytest.mark.tool9
@pytest.mark.tool0
def test_every_hypnogram_is_drawn_the_same_way():
    """One look (edf_io_lib.draw_hypnogram): grey line, REM red at the same width, no frame at the top and on the right."""
    import autoreject_lib as AR
    import autoscoring_compare_lib as AC
    import autoscoring_lib as AL
    import edf_io_lib as EIO
    import macrostructure_lib as ML
    stages = np.array(['W'] * 4 + ['N1'] * 2 + ['N2'] * 6 + ['N3'] * 4 + ['R'] * 4)
    n = len(stages)

    def check(ax, where):
        line = ax.get_lines()[0]                          # the step line, drawn first
        assert (line.get_color(), line.get_linewidth()) == (EIO.HYPNO_LINE_COLOR, EIO.HYPNO_LINE_WIDTH), where
        rem = [x for x in ax.get_lines() if x.get_color() == EIO.HYPNO_REM_COLOR]
        assert rem and all(x.get_linewidth() == EIO.HYPNO_LINE_WIDTH for x in rem), where
        assert not ax.spines['top'].get_visible() and not ax.spines['right'].get_visible(), where
        assert [t.get_text() for t in ax.get_yticklabels()] == ['N3', 'N2', 'N1', 'REM', 'W'], where

    combined = np.zeros((n, 4), dtype=int)
    fig = PL.plot_rejection_heatmap(combined, names(4), stages, 'x')
    check(fig.axes[0], 'tool 7')
    assert fig._suptitle is None and fig.axes[0].get_title().startswith('x: ')     # the title on the hypnogram
    piv = pd.DataFrame(combined, columns=names(4), index=pd.Index(range(n), name='epoch_idx'))
    dec = AR.auto_reject_decision(piv, pd.Series(stages, index=piv.index), ['N2'], 20, 20)
    figs = {'tool 8bis': AR.plot_decision_heatmap(piv, dec, 'x', []),
            'tool 8 heatmap': L.plot_channel_flag_heatmap({'ch_names': names(4), 'stages': stages},
                                                          {'amplitude': combined.astype(bool)}, ['amplitude'],
                                                          np.ones(n, dtype=bool)),
            'tool 8 review strip': L.plot_review_strip({'stages': stages}, np.zeros(n, dtype=bool), set()),
            'tools 6 and 0': QL.render_hypnospectrogram(np.arange(1.0, 5.0), np.arange(n) * 30 / 3600 + 1 / 240,
                                                         np.random.default_rng(0).normal(size=(4, n)), stages, []),
            'tool 5': ML.plot_night_overview(stages, []),
            'tool 3bis': AL.plot_hypnograms(stages, stages, 'x'),
            'tool 9': SL.plot_clean_hypnogram(pd.DataFrame({'epoch_idx': range(n), 'stage': stages}), [], 'x')}
    figs['tool 7'] = fig
    for where, f in figs.items():
        check(f.axes[0], where)
    fig, ax = plt.subplots()
    AC._manual_hypnogram(ax, np.arange(n) * 0.5, [{'W': 'Wake', 'R': 'REM'}.get(s, s) for s in stages])
    check(ax, 'tool 3bis, comparison with the expert')
    plt.close('all')


@pytest.mark.quick
@pytest.mark.tool9
@pytest.mark.tool0
def test_tool9_aperiodic_and_band_figures_draw_filled_points():
    rng = np.random.default_rng(4)
    rows = pd.DataFrame([{'file_id': f'p{p}', 'stage': st, 'channel': 'C3', 'band': b, 'exponent_mean': rng.normal(),
                          'offset_mean': rng.normal(), 'power': rng.normal()}
                         for p in range(5) for st in ('W', 'N2', 'R') for b in ('delta', 'alpha')])
    for fig in (SL.plot_group_aperiodic(rows.drop_duplicates(['file_id', 'stage']), []),
                SL.plot_group_band(rows, 'power', 'Power', [])):
        ax = fig.axes[0]
        assert not ax.patches                             # no box any more
        assert ax.collections and all(c.get_alpha() == SL.POINT_ALPHA for c in ax.collections)
        assert sum(len(c.get_offsets()) for c in ax.collections) == len(ax.collections) * 5    # a point per participant
        plt.close(fig)
    meta = pd.DataFrame({'epoch_idx': range(6), 'stage': ['N2'] * 6})
    fig = SL.plot_aperiodic_over_night(meta, {'exponent': rng.normal(size=(6, 1)), 'offset': rng.normal(size=(6, 1))},
                                       ['C3'], [])
    assert all(c.get_alpha() == 0.9 for c in fig.axes[0].collections)
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
    # phase 5 bis: above 12 channels, the comparison with the median channel is a flag too (C3, nearly flat)
    summary = pd.read_csv(reports / 'quality_summary.tsv', sep='\t')
    c3 = summary[(summary['file_id'] == 'hd01') & (summary['channel'] == 'C3')].iloc[0]
    assert c3['exclude'] and 'flat_vs_median' in c3['suspect_reason']
    # and the page says what the reports of a dense montage detail, above Run
    from nbdriver import NotebookSession
    with NotebookSession(pipeline.T6) as nb:
        nb.pick('fc_folder', data)
        nb.pick('fc_config', data / 'config_param' / 'remap_reref_persubject.json')
        info = nb.value('hd_info.value')
        nb.set('cb_all_channels', True)
        ticked = nb.value('hd_info.value')
    assert 'High-density montage (20 channels)' in info and 'only the <b>flagged channels</b>' in info
    assert 'every channel is detailed' in ticked
