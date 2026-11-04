# SomnoLens

**SomnoLens** is a sleep EEG toolkit that takes a database of overnight polysomnography (PSG) recordings from
the export to analysis-ready measures: it checks the files, harmonizes channels, hypnograms and scored events,
scores sleep automatically, computes the sleep macrostructure, reviews signal quality, preprocesses the EEG and
rejects artefacted epochs, then extracts spectral features.

It is written for **sleep researchers**: every tool is a point-and-click app in the web browser
([Voila](https://voila.readthedocs.io/)), no programming is needed. Recordings are read in **EDF**, the most common
format in clinical settings. Twins of some tools for **Curry 9** (`.cdt`) recordings are experimental.

SomnoLens is part of the [Dream-Toolkit](../../README.md) repository. It was called `Inspect_EDF` until October
2026, after its first tool.

## The tools

Each tool is one app in `tools/`. Run them in this order (tool 0 and the "bis" tools are optional):

| # | App | What it does |
|---|---|---|
| 0 | `0_live_explore_1file_voila.ipynb` | One participant through the whole pipeline, one tab per step: to diagnose a recording or to learn the tools |
| 1 | `1_inspect_edf_voila.ipynb` | Checks the EDF headers of the whole database, without loading the signal: montage consistency across participants, sampling rates, filters, units, inverted polarity, clipping risk, resolution, header anonymization. Batch versions: `1_inspect_edf_perdataset.py`, `1_inspect_edf_perparticipant.py` |
| 1bis | `1bis_anonymize_edf_voila.ipynb` | Writes anonymized copies of EDF files (header identity fields only, the signal stays byte-identical, originals untouched) |
| 2 | `2_select&remap_channels_edf_voila.ipynb` | Selects the channels to analyse, harmonizes their names and reference across participants, declares the EOG / EMG / ECG channels |
| 3 | `3_remap_hypno_voila.ipynb` | Harmonizes the sleep stage labels of the hypnograms to the AASM ones (W, N1, N2, N3, R), custom stages allowed |
| 3bis | `3bis_autoscoring_voila.ipynb` | Automatic sleep scoring with three published models (GSSC, YASA, SleepFM), and their agreement with the expert hypnogram |
| 4 | `4_remap_events_edf_voila.ipynb` | Harmonizes the labels of the scored events (apneas, arousals, leg movements...) to one vocabulary |
| 5 | `5_compute_sleep_macrostructure_voila.ipynb` | The classical PSG report per recording: TIB, SOL, TST, WASO, sleep efficiency, stage latencies and durations, AHI, PLM and arousal indices, optionally ODI and T90 |
| 5bis | `5bis_exclude_sleep_macrostructure_voila.ipynb` | Shows each metric's distribution across the database and excludes participants on criteria (e.g. AHI > 15) |
| 6 | `6_quality_overview_voila.ipynb` | Signal-quality overview per channel: amplitude distribution, flat or clipped signal, line noise, hypnospectrogram |
| 7 | `7_preprocessing_voila.ipynb` | Filtering, re-referencing, epoching, and flagging of artefacted epochs per channel (amplitude, gradient, flat signal, 1/f fit quality, scored events) |
| 8 | `8_reject_manually_voila.ipynb` | Reviews the flagged epochs one by one and decides which to reject (batch report: `8_reject_manually_batch.py`) |
| 8bis | `8bis_reject_automatically_voila.ipynb` | The automatic alternative to tool 8: drops the globally bad channels first, then rejects the epochs |
| 9 | `9_spectral_features_voila.ipynb` | Spectral features on the kept epochs: band powers, aperiodic (1/f) and periodic components |

Rejected data (excluded participants, rejected epochs, dropped channels) is never deleted: it is computed, flagged
in the tables, and left out of the group statistics and figures only.

Every tool writes **tables** (TSV, for analysis) and **HTML reports** (to read). A "Skip already processed" option
lets a tool run again on a growing database without redoing the participants already done.

### Curry 9 (experimental)

`tools_curry/` holds Curry twins of tools 1, 2, 3, 4, 6, 7, 8 and 8bis, most of them generated from the EDF tools.
No test runs them on Curry recordings yet: check their outputs.

## Installation

The tools run in a [conda](https://conda-forge.org/) environment named `somnolens`. The step-by-step guide for
Windows, with screenshots, is `tutorials/1_install_python.docx`. In short:

1. Install [Miniforge](https://conda-forge.org/download/).
2. From this folder, create the environment:
   ```bash
   conda env create -f environment.yml
   ```

Today this single environment holds every tool, automatic scoring included (about 4.7 GB).
**Planned**: a **light** installation (every tool except automatic scoring: no PyTorch, GSSC or SleepFM
packages) and a **full** one (light + the tool-3bis models).

SleepFM's model files (about 29 MB) are not part of the environment: tool 3bis downloads them from the official
repository the first time, and checks them.

## Quick start

```bash
conda activate somnolens
cd path/to/SomnoLens
voila tools/1_inspect_edf_voila.ipynb
```

The app opens in the web browser. Launch every tool from the `SomnoLens` folder, with
`voila tools/<app>.ipynb` (quote the names that contain a `&`:
`voila "tools/2_select&remap_channels_edf_voila.ipynb"`).

### The study folder

Every tool first asks for the **study folder**: the folder that holds your recordings (in any sub-folder tree)
and receives the outputs. Always pick this same folder, never `data/`. It is the only thing to pick: each tool finds
the files it needs in it, lists them with where each comes from and says which tool to run when one is missing
(*Choose another…* on a line picks another file by hand).

```
my_study/                      <- pick this folder in every tool
├── data/                      <- recordings + their companion files (hypnogram, scored events)
├── config_param/              <- choices saved by tools 2, 3, 4 + the participant exclusions
├── derivatives/               <- data written by the tools (epochs, decisions, features), one folder per step
├── reports_inspection/        <- tool 1's report
└── reports_<step>/            <- one HTML report folder per tool
```

Keep the study path short: Windows limits paths to 260 characters.

### Before the export: tutorials

The other guides in `tutorials/` cover the export from Compumedics Profusion: creating an EEG montage
(`2_create_EEG_montage_Profusion.docx`), exporting EDF files with it (`3_export_edf_withmontage_Profusion.docx`)
or by hand (`3bis_export_edf_manually_Profusion.docx`, not recommended).

## Planned

- **More feature tools** after tool 9, on the same skeleton: sleep microstructure such as spindles and slow waves.
- **Light / full installation** (see above), with deliberately refreshed package versions.
- **Easier distribution**: a dedicated repository, an installable package with a menu of the tools, a one-click
  Windows installation.
- **Tool 1** inferring the organisation of the database (groups, nights, missing companion files).
- **Tool 0**: an interactive epoch explorer (click, zoom, annotations drawn with the mouse), simulated recordings
  to try every step, and a Curry version.
- **Harmonized pages** across the tools (layout, file detection, English everywhere).

## Documentation

- `SPEC.md`: the full technical specification (every tool, its inputs, outputs and design choices).
- `TESTS.md`: what the test suite checks, in plain words.
- `CLAUDE.md`: the development rules.

For developers, the tests (`tests/`, pytest) drive the apps headlessly:
`python -m pytest -m quick` (about 2 minutes), see `SPEC.md`, *How to run → Tests*.

## Licence

To be confirmed. The automatic scoring models have their own terms: **GSSC** is under AGPL-3.0 (a separate
package, installed from PyPI by the environment), **SleepFM**'s code and weights under CC BY-NC 4.0
(non-commercial use; downloaded from its official repository, not redistributed here).
