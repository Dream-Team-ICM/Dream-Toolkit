"""The inspection tools (1, 2, 3, 6) on BrainVision recordings, written by hand with invented values, plus the proof that
a same signal gives the same quality metrics in EDF and in BrainVision (tool 6)."""
import json
import shutil
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

import minidb
from brainvision_files import recorder_comment, write_brainvision
from nbdriver import NotebookSession

T1 = 'tools/1_inspect_edf_voila.ipynb'
T2 = 'tools/2_select&remap_channels_edf_voila.ipynb'
T3 = 'tools/3_remap_hypno_voila.ipynb'
T6 = 'tools/6_quality_overview_voila.ipynb'

SFREQ = 250
# 4 EEG inputs + an EOG on an EEG input, then the sensors of the auxiliary inputs. Cz is the online reference, not stored.
NAMES = ['Fp1', 'C3', 'TP9', 'TP10', 'IO', 'Left Thumb', 'EMG', 'ECG', 'Respi']
RESOLUTIONS = [0.05, 0.05, 0.05, 0.05, 0.05, 1.0, 1.0, 1.0, 1.0]
UNITS = ['µV', 'µV', 'µV', 'µV', 'µV', 'mV', 'mV', 'mV', 'UA']
SPHERICAL = {'Fp1': (1, -92, -72), 'C3': (1, -46, 0), 'TP9': (1, -115, -18), 'TP10': (1, 115, 18), 'IO': (1, 92, -60),
             'Left Thumb': (0, 0, 0), 'EMG': (0, 0, 0), 'ECG': (0, 0, 0), 'Respi': (0, 0, 0)}
IMPEDANCES = {'Fp1': 5, 'C3': 35, 'TP9': 'Out of Range!', 'TP10': 'not connected', 'IO': 12, 'Cz': 3, 'Gnd': 4}
GRADIENTS = {name: '1[mV/mV]' for name in ['Left Thumb', 'EMG', 'ECG', 'Respi']}


def write_small(folder, name, duration_s=120, seed=0, **kwargs):
    """A short BrainVision recording. `kwargs` go to recorder_comment, except `comment=None` for no [Comment] at all."""
    rng = np.random.default_rng(seed)
    uv = rng.normal(0, 20, (len(NAMES), int(duration_s * SFREQ)))
    stored = uv / np.array(RESOLUTIONS)[:, None]
    stored[5:] = uv[5:] * 1e-3          # the sensors, in mV at a resolution of 1
    comment = kwargs.pop('comment', 'written')
    if comment == 'written':
        kwargs.setdefault('impedances', [IMPEDANCES])
        kwargs.setdefault('gradients', GRADIENTS)
        comment = recorder_comment(NAMES, RESOLUTIONS, UNITS, **kwargs)
    return write_brainvision(folder, name, stored, NAMES, sfreq=SFREQ, resolutions=RESOLUTIONS, units=UNITS,
                             coordinates=SPHERICAL, comment=comment)


def write_config(study, ids, ref_channels, remap=None):
    """Tool 2's file, written by hand (the widgets of its sections are tested once, in test_tool2_whole_page)."""
    remap = remap or {'Fp1': 'Fp1', 'C3': 'C3', 'TP9': 'M1', 'TP10': 'M2', 'Cz': 'Cz'}
    config = {pid: {'config': 'config. 1', 'remap': dict(remap), 'ref_channels': ref_channels} for pid in ids}
    folder = Path(study) / 'config_param'
    folder.mkdir(exist_ok=True)
    path = folder / 'channel_config.json'
    path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    return path


@pytest.fixture
def study(tmp_path):
    """Two good recordings, one with no [Comment], one paused and one whose .eeg is gone."""
    data = tmp_path / 'study' / 'data'
    write_small(data, 'bv01', seed=1)
    write_small(data, 'bv02', seed=2)
    write_small(data, 'bv03_nocomment', seed=3, comment=None)
    broken = write_small(data, 'bv04_broken', seed=4)
    broken.with_suffix('.eeg').unlink()
    return tmp_path / 'study'


# ----------------------------------------------------------------------------------------------- tool 1
@pytest.mark.tool1
def test_tool1_on_brainvision(study):
    with NotebookSession(T1) as nb:
        nb.pick('chooser', study)
        info = nb.value('existing_files_info.value')
        nb.set('cb_ecg', True)
        nb.set('cb_emg', True)
        nb.click('run_button')
        text = nb.widget_text()
    assert '0 / 4 recording(s) already inspected' in info

    # The recording whose .eeg is gone is recorded as failed, with a clear message, and the others are read
    out = study / 'reports_inspection'
    failed = pd.read_csv(out / 'failed_edf_read.tsv', sep='\t')
    assert [Path(p).name for p in failed['path']] == ['bv04_broken.vhdr']
    assert 'cannot be renamed file by file' in text
    full = pd.read_csv(out / 'FULL_summary_table_edf.tsv', sep='\t', dtype={'subject': str})
    assert sorted(full['subject'].unique()) == ['bv01', 'bv02', 'bv03_nocomment']
    # The online reference is not inspected as a channel
    assert 'Cz' not in set(full['channel'])
    assert set(full['sampling_frequency']) == {SFREQ}

    # What is read per channel: filters, resolution and impedance, from the [Comment] when there is one
    fp1 = full[(full['subject'] == 'bv01') & (full['channel'] == 'Fp1')].iloc[0]
    assert fp1['lowpass'] == 140 and fp1['highpass'] == pytest.approx(0.01, abs=1e-4)
    assert fp1['resolution'] == 0.05 and fp1['impedance_kohm'] == 5
    assert pd.isna(fp1['res_theoretical'])      # float32 samples have no fixed step
    no_comment = full[(full['subject'] == 'bv03_nocomment') & (full['channel'] == 'Fp1')].iloc[0]
    assert pd.isna(no_comment['lowpass']) and pd.isna(no_comment['impedance_kohm'])
    emg = full[(full['subject'] == 'bv01') & (full['channel'] == 'EMG')].iloc[0]
    assert emg['gradient'] == '1 mV/mV' and emg['transducer_type'] == 'EMG'

    # The checks that do not apply say so, and flag nothing
    for table in ('EEG_bad_dynamic_range_edf.tsv', 'EEG_bad_resolution_edf.tsv', 'EEG_inverted_polarity_edf.tsv'):
        assert pd.read_csv(out / table, sep='\t').empty, table
    assert 'Inverted polarity: not applicable' in text and 'Dynamic range: not applicable' in text
    assert 'Resolution: not applicable' in text
    assert 'No EEG with a dynamic range' not in text and 'No inverted polarity was detected in EEGs' not in text
    assert 'Not applicable to 3 BrainVision recording(s)' in text
    assert not (out / 'anonymization_check_edf.tsv').exists()

    # Section 1.3: the reference, the gain of the auxiliary inputs, the impedances
    assert 'Online reference: Cz' in text and 'not in the file' in text
    assert 'The [Comment] names no online reference' in text
    assert 'ECG, EMG' in text and 'sit on auxiliary inputs' in text
    flags = pd.read_csv(out / 'impedance_flags.tsv', sep='\t')
    assert sorted(flags[flags['subject'] == 'bv01']['electrode']) == ['C3', 'TP10', 'TP9']
    status = flags[flags['subject'] == 'bv01'].set_index('electrode')['status'].to_dict()
    assert status == {'C3': 'above 20 kOhm', 'TP9': 'Out of Range!', 'TP10': 'not connected'}   # what the file says
    notes = pd.read_csv(out / 'recording_notes.tsv', sep='\t')
    assert notes.set_index('subject').loc['bv01', 'n_segments'] == 1


@pytest.mark.tool1
def test_tool1_says_a_recording_was_paused(tmp_path):
    uv = np.random.default_rng(0).normal(0, 20, (1, 5000))
    write_brainvision(tmp_path, 'paused', uv / 0.05, ['C3'], sfreq=SFREQ, resolutions=[0.05], extra_segments=[2500])
    with NotebookSession(T1) as nb:
        nb.pick('chooser', tmp_path)
        nb.click('run_button')
        text = nb.widget_text()
    assert '1 recording(s) were paused' in text and 'paused' in text


# ----------------------------------------------------------------------------------------------- tool 2
@pytest.mark.tool2
def test_tool2_whole_page(study):
    """The scan, the sections built one after the other with their default choices, the JSON and its test, on the channels
    of a montage with an online reference. The default re-reference is None, so the flat reference is warned about."""
    (study / 'data' / 'bv04_broken.vhdr').unlink()                  # tested with tool 1 and below
    (study / 'data' / 'bv04_broken.vmrk').unlink()
    (study / 'data' / 'bv03_nocomment.vhdr').unlink()               # another configuration (no Cz)
    (study / 'data' / 'bv03_nocomment.vmrk').unlink()
    (study / 'data' / 'bv03_nocomment.eeg').unlink()
    with NotebookSession(T2) as nb:
        nb.pick('chooser', study)
        nb.click('run_button')
        configs = nb.value('[sorted(c) for c in STATE.configs]')
        assert configs == [['C3', 'Cz', 'Fp1', 'IO', 'TP10', 'TP9']]     # EEG inputs + the reference, no auxiliary sensor
        assert nb.value('STATE.online_refs') == {'bv01': 'Cz', 'bv02': 'Cz'}
        text = nb.widget_text()
        assert 'online reference, not in the file (2 recording(s))' in text
        # Section 3: what the header says about the channels, ready in its menus
        menus = nb.value("{w.description: w.value for w in list(widgets.Widget.widgets.values()) "
                         "if isinstance(w, widgets.Dropdown) and w.description in ('EOG Left', 'EMG', 'ECG')}")
        assert menus == {'EOG Left': 'IO', 'EMG': 'EMG', 'ECG': 'ECG'}

        nb.click_labelled('Save context selection')
        nb.click_labelled('Save selection')
        nb.click_labelled('Save remapping')
        nb.click_labelled('Save re-reference')
        text = nb.widget_text()
        assert 'is the online reference, kept without a re-reference: it stays a flat channel' in text
        nb.click_labelled('Preview & Save')
        path = Path(nb.value('STATE.json_path'))
        saved = json.loads(path.read_text(encoding='utf-8'))
        assert sorted(saved) == ['bv01', 'bv02']
        assert 'Cz' in saved['bv01']['remap']
        # The extended 10-10 names keep their case: a montage does not know 'Tp9'
        assert saved['bv01']['remap']['TP9'] == 'TP9' and saved['bv01']['remap']['TP10'] == 'TP10'
        assert saved['bv01']['context_channels'] == {'eog_left': 'IO', 'emg': 'EMG', 'ecg': 'ECG'}

        # Section 7 reads the signal: the online reference is recreated, whatever the reference chosen
        saved = {pid: {**cfg, 'ref_channels': 'average'} for pid, cfg in saved.items()}
        path.write_text(json.dumps(saved), encoding='utf-8')
        nb.click('STATE.ui["run_testjson"]')
        text = nb.widget_text()
    assert 'The remapping worked' in text, text[-1500:]
    assert 'Cz' in text


@pytest.mark.tool2
def test_tool2_goes_on_after_a_recording_it_cannot_read(study):
    with NotebookSession(T2) as nb:
        nb.pick('chooser', study)
        nb.click('run_button')
        text = nb.widget_text()
        n_configs = nb.value('len(STATE.configs)')
    assert 'cannot be renamed file by file' in text
    assert n_configs == 2       # with and without the online reference
    failed = pd.read_csv(study / 'config_param' / 'failed_edf_read.tsv', sep='\t')
    assert [Path(p).name for p in failed['path']] == ['bv04_broken.vhdr']


# ----------------------------------------------------------------------------------------------- tool 3
@pytest.mark.tool3
def test_tool3_finds_the_hypnograms_of_brainvision_recordings(tmp_path):
    ids = ['bv01', 'bv02']
    stages = ['W', '1', '2', '3', '4', 'R', 'W', '2']
    for pid in ids:
        write_small(tmp_path, pid, duration_s=len(stages) * 30)
        (tmp_path / f'{pid}_Hypnogram_Export.txt').write_text('\n'.join(stages) + '\n', encoding='utf-8')
    with NotebookSession(T3) as nb:
        nb.pick('chooser', tmp_path)
        assert nb.value('suffix_box.value') == '_Hypnogram_Export.txt'
        nb.click('btn_scan')
        nb.click_labelled('Save remapping')
        nb.click_labelled('Save files')
        nb.click('btn_run_verify')
        text = nb.widget_text()
    assert 'Remapping successful' in text, text[-2000:]
    for pid in ids:
        remapped = np.loadtxt(tmp_path / f'{pid}_hypno_remapped.txt', dtype=str).tolist()
        assert remapped == ['W', 'N1', 'N2', 'N3', 'N3', 'R', 'W', 'N2']


# ----------------------------------------------------------------------------------------------- tool 6
@pytest.fixture(scope='module')
def same_signal(tmp_path_factory):
    """Two synthetic nights, once as EDF and once as BrainVision: the BrainVision files hold exactly what MNE reads from the
    EDF (float32 samples). Two study folders with the same channel configuration and hypnograms."""
    ids = minidb.PARTICIPANTS['synthetic'][:2]
    edf_study = minidb.build(tmp_path_factory.mktemp('edf'), 'synthetic')
    for pid in minidb.PARTICIPANTS['synthetic'][2:]:
        for path in edf_study.glob(f'{pid}*'):
            path.unlink()
    bv_study = tmp_path_factory.mktemp('bv') / 'data'
    (bv_study / 'config_param').mkdir(parents=True)
    config = json.loads((edf_study / 'config_param' / 'remap_reref_persubject.json').read_text(encoding='utf-8'))
    config = {pid: config[pid] for pid in ids}
    for study_folder in (edf_study, bv_study):
        (study_folder / 'config_param' / 'remap_reref_persubject.json').write_text(json.dumps(config), encoding='utf-8')
    for pid in ids:
        raw = mne.io.read_raw_edf(edf_study / f'{pid}.edf', preload=True, verbose=False)
        write_brainvision(bv_study, pid, raw.get_data() * 1e6, raw.ch_names, sfreq=raw.info['sfreq'],
                          resolutions=[1.0] * len(raw.ch_names))
        shutil.copy2(edf_study / f'{pid}_Hypnogram_remapped.txt', bv_study)
    return edf_study, bv_study, ids


def run_tool6(study):
    with NotebookSession(T6) as nb:
        nb.pick('fc_folder', study)
        acquisition = nb.value('acq_info.value')
        bounds_shown = nb.value('thresh_bounds_row.layout.display')
        nb.click('btn_run')
        return nb.widget_text(), acquisition, bounds_shown


@pytest.mark.tool6
def test_tool6_gives_the_same_metrics_in_edf_and_in_brainvision(same_signal):
    edf_study, bv_study, ids = same_signal
    run_tool6(edf_study)
    text, acquisition, bounds_shown = run_tool6(bv_study)
    assert 'DC-coupled' not in acquisition and 'not stated' in acquisition      # no [Comment]: it does not say
    assert bounds_shown == 'none'

    for pid in ids:
        edf = pd.read_csv(edf_study / 'reports_quality_overview' / f'{pid}_quality_metrics.tsv', sep='\t')
        bv = pd.read_csv(bv_study / 'reports_quality_overview' / f'{pid}_quality_metrics.tsv', sep='\t')
        assert list(bv.columns) == list(edf.columns), pid
        assert edf['channel'].tolist() == bv['channel'].tolist()
        for metric in ('mean_uV', 'std_uV', 'kurtosis', 'skewness', 'p99_abs_uV', 'p999_abs_uV', 'flat_pct',
                       'hist_extreme_pct'):
            assert bv[metric].to_numpy() == pytest.approx(edf[metric].to_numpy(), rel=1e-3, abs=1e-3), (pid, metric)
        assert bv['n_peaks'].tolist() == edf['n_peaks'].tolist(), pid
        # The physical range does not exist in BrainVision: blank, not 0 %, and the rest of the flags are the same
        assert bv['bounds_pct'].isna().all() and edf['bounds_pct'].notna().all()
        without_bounds = [';'.join(r for r in str(reason).split('; ') if not r.startswith('bounds_pct') and r != 'nan')
                          for reason in edf['suspect_reason']]
        found = ['' if pd.isna(reason) else reason for reason in bv['suspect_reason']]
        assert [r.replace(';', '; ') for r in without_bounds] == found, pid

        stage_bv = pd.read_csv(bv_study / 'reports_quality_overview' / f'{pid}_quality_by_stage.tsv', sep='\t')
        assert stage_bv['bounds_pct'].isna().all() and not stage_bv.empty
        report = (bv_study / 'reports_quality_overview' / f'{pid}_quality_overview.html').read_text(encoding='utf-8')
        assert 'At EDF bounds' not in report
    assert (bv_study / 'reports_quality_overview' / 'dataset_overview.html').exists()
    # A recording is flagged the same way whatever its format: the defects of the synthetic nights are found
    flagged = pd.read_csv(bv_study / 'reports_quality_overview' / 'quality_summary.tsv', sep='\t')
    flagged = flagged[flagged['exclude']]
    assert ['sim02_dead_channel', 'C3'] in flagged[['file_id', 'channel']].values.tolist()


@pytest.mark.tool6
@pytest.mark.parametrize('dc_coupled', [False, True])
def test_tool6_ticks_the_high_pass_of_a_dc_coupled_recording(tmp_path, dc_coupled):
    """DC in the hardware table and no software filter: the header says the recording has no high-pass. The online
    reference chosen in tool 2 is left out here, and said so: this tool does not re-reference."""
    write_small(tmp_path, 'bv01', software_low=None if dc_coupled else 15.9155)
    (tmp_path / 'bv01_hypno_remapped.txt').write_text('\n'.join(['W'] * 4) + '\n', encoding='utf-8')
    write_config(tmp_path, ['bv01'], 'average')
    with NotebookSession(T6) as nb:
        nb.pick('fc_folder', tmp_path)
        assert nb.value('hp_check.value') is dc_coupled
        assert nb.value('hp_freq.value') == (0.1 if dc_coupled else 0.5)
        assert nb.value('hp_freq.layout.display') == ('' if dc_coupled else 'none')
        acquisition = nb.value('acq_info.value')
        nb.click('btn_run')
        text = nb.widget_text()
    assert ('none/DC' in acquisition and 'starts ticked' in acquisition) if dc_coupled else '0.01 Hz' in acquisition
    assert 'Cz left out, not in the file' in text
    metrics = pd.read_csv(tmp_path / 'reports_quality_overview' / 'bv01_quality_metrics.tsv', sep='\t')
    assert sorted(metrics['channel']) == ['C3', 'Fp1', 'M1', 'M2']
    assert set(metrics['signal_preparation']) == ({'high-pass 0.1 Hz'} if dc_coupled else {'none'})
