# Plan: tool 7's band-pass phase + lighter conda environments (with the version refresh)

Status: **draft (2026-11-19), waiting for confirmation**. SPEC.md, *Planned modules → Order of the TODOs*, item 1:
*Tool 7: band-pass phase* and *Lighter conda environments*, done together between two studies (every output is
reprocessed once).

## 1. Decisions

Answered on 2026-11-19:
- **Band-pass phase**: tool 7 moves from `phase='zero-double'` to `phase='zero'` (MNE's default, the same as the notch),
  **without a choice on the page**. The params sidecar records the phase. A sidecar without it counts as `zero-double`
  (the only phase before), so *Skip* asks and tools 8bis / 9 see the fingerprint change.
- **Tool 6's display filter follows**: `quality_lib.compute_channel_spectrograms` (0.1-40 Hz, hypnospectrograms of tool 6
  and tool 0's step 5) moves to `zero` too. One filtering rule in the toolkit. Figures only, no metric.
- **Environments = light + full, two files, one env name `somnolens`**: `environment.yml` = light (every tool, tool 3bis
  with YASA only), `environment-full.yml` = light + `pytorch-cpu`, `gssc`, SleepFM's packages. A light user moves to
  full with `conda env update`. `environment-dev.yml` = full + the developer packages (tests).
- **YASA stays in the light install**: tool 3bis works there with YASA, GSSC and SleepFM greyed out with a message that
  points to the full install.

Already decided in SPEC (not reopened): `mne` → `mne-base`, `matplotlib` → `matplotlib-base`, developer packages out of
the user files, versions pinned and refreshed deliberately (this is the first refresh: every pin to its current stable
version, a few weeks old), `antropy<0.2` kept, `gssc` pinned to 0.0.9.

**Proposed defaults, to annotate** (my recommendation in each):
- **D1. Spyder**: in no file (VS Code is the IDE; it pulls PyQt5 + QtWebEngine). Whoever wants it installs it.
- **D2. Installation tutorial** (`tutorials/1_install_python.docx`): rewritten at the *Clean-up before distribution*
  step (it also needs the SSL fix listed there). This plan updates `README.md` and SPEC only.
- **D3. Major versions with breaking changes** (likely candidates: pandas 3, IPython 9, ipykernel 7, Voila 0.6, Python
  3.13 / 3.14; to confirm with the table of phase 2a): taken in this refresh when the tests pass or need small fixes. A
  major that needs real migration work (pandas 3 is the most likely one: copy-on-write and the new string dtype touch
  most tables) stays at the last minor of its current major, with a SPEC note and its own later task.
- **D4. A study in progress**: phase 1 changes tool 7's outputs as soon as it is committed. If a study is being
  processed in this environment right now, phase 1 waits for its end (or the study keeps the current commit).
- **D5. Package versions in tool 7's sidecar**: not in this plan (tool 3bis already records its models' versions).
  Possible later: an additive provenance block, never compared by *Skip*.

## 2. What the phase changes (measured on 2026-11-19, MNE 1.12.1, filter only, no recording)

Tool 7's defaults, 0.1-50 Hz at 256 Hz (512 Hz gives the same numbers): Hamming FIR, `firwin`, automatic length
8449 taps (33 s). Gain in dB:

| Hz          | 0.005 | 0.01 | 0.02 | 0.05 | 0.08 | 0.1 | 0.15-48 | 50 | 52 | 56.25 | 60 |
|-------------|------:|-----:|-----:|-----:|-----:|----:|--------:|---:|---:|------:|---:|
| zero        | −47 | −35 | −23 | −6.1 | −0.7 | −0.03 | within ±0.03 | −0.03 | −0.5 | −6.0 | −22 |
| zero-double | −94 | −70 | −45 | −12.1 | −1.4 | −0.06 | within ±0.03 | −0.06 | −0.9 | −12.1 | −44 |

- Within 0.15-48 Hz the two differ by less than 0.1 dB: the 1/f fit band (2-45 Hz), the slow waves (0.5-4 Hz) and every
  tool-9 band are unchanged to that precision.
- The transition bands move: a very slow drift is attenuated about half as much in dB (a DC-coupled recording with a large
  drift keeps more of it, the lever is `l_freq`), and 52-60 Hz keeps more power. The effective impulse response is half
  as long (33 s instead of 66 s), so the edge effect at the start and end of the night is shorter.
- Expected effect on the outputs: a few amplitude / gradient flags flip near their threshold, tool 9's values move
  slightly. Measured in phase 1 before anything is accepted.

## 3. Phases

Each phase: what is done, then its tests (the level proposed at that time, as usual), then a stop for your check.

### Phase 1: the band-pass phase (current environment, so its effect is measured alone)

1. Before any change: `refactor_check.py save before_phase` on the synthetic chain, and `outside_chain.py 6variant
   7variant` saved the same way.
2. `preprocessing_lib`: a constant `BANDPASS_PHASE = 'zero'` used by `preprocess_raw` ([E]), and by the sidecar writers
   (tool 7's `run_settings`, tool 0's `step6_sidecar_info`): the `filter` block gains `phase` (when applied).
3. `preprocessing_lib.settings_differences`: the band-pass label carries the phase (`0.1-50 Hz, zero-double` →
   `0.1-50 Hz, zero`), absent = `zero-double`. `recorded_settings` / *Use the recorded settings*: the phase cannot be put
   back (no widget), so it is named among what cannot, like the reference: *band-pass phase zero-double: no longer
   offered, reprocess*. The fingerprint of tools 8 / 8bis / 9 follows by itself (`filter` is in `PREPROC_SETTINGS`).
4. Tool 7's page: the help line under the band-pass says one zero-phase pass (same wording as the notch's), the report
   row reads the phase instead of the fixed text `FIR zero-double Hamming`.
5. `quality_lib.compute_channel_spectrograms`: `phase='zero'`.
6. Tests: `test_preprocessing_lib.py` (quick), an old sidecar against the current settings gives one *band-pass filter*
   row, the same phase none, the phase named by *Use the recorded settings*; a dedicated test that `preprocess_raw` equals
   MNE's `raw.filter(..., phase='zero')` on a generated signal and differs from `zero-double` (the intended change).
   Then the chain (markers `tool0`, `tool6`, `tool7`, `tool8bis`, `tool8`, `tool9`) and `refactor_check compare
   before_phase`: I report what changed (flags per method and stage, 8bis decisions, tool 9's largest differences).
7. Once you accept the differences: regenerate the synthetic golden (`make_golden.py --dataset synthetic`, the new code
   being the trusted one), and the real golden (local) if you allow a run on the real data.
8. Docs: SPEC §7 (step 4, Section 2.1, sidecar, *Skip and the settings*), §6 (spectrogram filter), §0 (step 6 sidecar),
   the TODO *Tool 7: band-pass phase* closed (removed, a line in the Order list), a reprocessing note (a study processed
   before: *Skip* asks, answer *Reprocess*). TESTS.md.

### Phase 2: the environment files and the version refresh

**2a. The table of versions** (nothing installed): for each package, its current pin, the newest version on conda-forge
(or PyPI for the pip ones) released at least 3 weeks ago, its date, whether it is a new major, its known breaking changes
(conda search, read only). You confirm the table (D3) before any install.

**2b. The three files** (no include in conda: each file repeats the list of the one below it):
- `environment.yml` (light, name `somnolens`): python, numpy, pandas, scipy, `mne-base`, `matplotlib-base`, yasa,
  `antropy` (<0.2), `lspopt`, statsmodels, scikit-learn, joblib, edfio, chardet, openpyxl, voila, ipywidgets,
  ipyfilechooser, ipyevents, ipykernel, ipython, tk, pip: specparam, curryreader.
- `environment-full.yml` (name `somnolens`): light + `pytorch-cpu`, h5py, einops, loguru, pyyaml, pip: `gssc==0.0.9`.
- `environment-dev.yml` (name `somnolens`): full + pytest, notebook.
- Every pin `x.y.*` (the pip ones `==` or `~=`), including the ones unpinned today (`pytorch-cpu`, specparam, still a
  release candidate `2.0.0rc7` here, curryreader, pytest).

**2c. Code adaptations of the refresh** (known today, completed by 2a):
- YASA 0.8: `score_night_yasa` takes `predict().proba` instead of the removed `predict_proba()` (the `W` / `R` renaming
  already exists). `_scoring_params.json` of YASA gains `antropy_version`.
- `quality_lib`: `from lspopt import spectrogram_lspopt` instead of `yasa.plotting` (the same function, re-exported).
  Tools 6 and 7 import `yasa` in their first cell: removed if unused, so neither depends on YASA.
- Messages: `autoscoring_lib.GSSC_MSG` and `sleepfm_lib.SLEEPFM_DEPS_MSG` point to the full install (`conda env update -n
  somnolens -f environment-full.yml`, from the SomnoLens folder, then restart the page). The `autoscore` environment is no
  longer named there. `YASA_MSG` points to `environment.yml`.
- Tests that need GSSC or SleepFM skip cleanly without them (`pytest.importorskip`), found by running the suite in the
  light environment.
- A new quick test, `tests/test_environment_files.py`: light ⊂ full ⊂ dev with the same pins, every third-party import of
  `tools/*.py` and `tools/*_voila.ipynb` declared in the light file, except the autoscoring ones (torch, gssc, h5py,
  einops, loguru, yaml), which must be in the full one (import name → package: `sklearn` → scikit-learn, `yaml` → pyyaml,
  `IPython` → ipython, `torch` → pytorch-cpu, ...). It replaces the audit by hand of 2026-10-19.
- `tests/chaincache.py`: the fingerprint of the harness includes the interpreter (`sys.prefix`) and the versions of the
  main packages, so a chain made in another environment is rerun instead of reused (otherwise `refactor_check` would
  compare stale outputs). A change to `BASE_DEPS`: the chain reruns once.

**2d. Build and compare** (each install announced and approved at the time; `CONDA_SSL_VERIFY=truststore`; disk space
checked first, each env 2-5 GB):
1. `somnolens_next` from `environment-dev.yml`: the chain from scratch, `refactor_check compare` against the outputs of
   phase 1. I explain each difference by its package (specparam final vs rc7 on the 1/f flags and tool 9, scipy / numpy
   on the PSD, YASA 0.8 on tool 3bis outside the chain).
2. `somnolens_light_check` from `environment.yml` + pytest: the quick level (every page executes, tool 3bis greys GSSC
   and SleepFM out with the new message, the tests skip cleanly), then the chain, compared with `somnolens_next`:
   identical expected (at most a float noise if the light env gets another BLAS than MKL, which `refactor_check` reports
   as within the tolerance).
3. YASA 0.8 on real data (with your permission, night 100): the stages against YASA 0.6.5, and the YASA part of
   `tests/score_edf_equivalence.py`, as SPEC asks.
4. Once you accept the differences: the goldens regenerated in `somnolens_next`, the full level there.

### Phase 3: the final environment and the docs

1. The environment: `somnolens` removed and recreated from `environment-dev.yml` (a fresh create, not a rename: a
   rename breaks torch and the pip launchers), so the paths of CLAUDE.md stay valid. Then `somnolens_next` and
   `somnolens_light_check` removed. Each step approved at the time (removing `somnolens` cannot be undone, only rebuilt).
2. SPEC: *Conda environment* rewritten (the three files and what each holds, the versions and the date of this refresh,
   the refresh rule and the Windows gotchas moved there from the TODO), §3bis (YASA 0.8, `antropy_version`, the
   messages), *How to run* (which file to install, the tests run in the dev env), the TODO *Lighter conda environments*
   closed, the Order list (item 1 done, the slow-wave and spindle extraction next).
3. `README.md`, *Installation*: light and full, the two commands, the sizes measured in 2d.
4. CLAUDE.md: the reminder "add any directly-imported package to `environment.yml`" becomes "to the right environment
   file (light, or full for the autoscoring): `tests/test_environment_files.py` checks it". Memory updated
   (environments, autoscoring env).
5. This plan deleted (its own commit).

## 4. Out of this plan

- The installation tutorial (D2) and a one-click Windows install: *Clean-up before distribution*, *Distribution and
  licensing*.
- The `autoscore` environment (python 3.10, Score_EDF's versions): untouched, still used by `score_edf_equivalence.py`.
- `python/Score_EDF/`: untouched.
- The untracked scripts that call `zero-double` (`tools/compare_flat_spectrogram_fix.py`,
  `tools/simple_hypnospectro_yasa_vs_fix.py`) and the archives (`tools/old/`, `tools/jupyter_notebook/`): untouched.
