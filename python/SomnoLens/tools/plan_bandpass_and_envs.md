# Plan: tool 7's band-pass phase + lighter conda environments (with the version refresh)

Status: **confirmed (2026-11-19)**. Phase 1 done (63a88e4). Phase 2: table confirmed, files and code adaptations
written, `somnolens_next` built (2.5 GB, 303 conda-forge packages + specparam, curryreader, gssc from pip, nothing else),
the standard level and the comparison with phase 1 running. SPEC.md, *Planned modules → Order of the TODOs*, item 1:
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

Annotated on 2026-11-19 (all accepted):
- **D1. Spyder**: in no file (VS Code is the IDE; it pulls PyQt5 + QtWebEngine). Whoever wants it installs it.
- **D2. Installation tutorial** (`tutorials/1_install_python.docx`): the full rewrite waits for the *Clean-up before
  distribution* step (it also needs the SSL fix listed there). This plan updates `README.md` and SPEC, and **gives the
  install lines of the light and full environments** (phase 3) so that the tutorial gets a small update by hand now.
- **D3. Major versions with breaking changes** (likely candidates: pandas 3, IPython 9, ipykernel 7, Voila 0.6, Python
  3.13 / 3.14; to confirm with the table of phase 2a): taken in this refresh when the tests pass or need small fixes. A
  major that needs real migration work (pandas 3 is the most likely one: copy-on-write and the new string dtype touch
  most tables) stays at the last minor of its current major, with a SPEC note and its own later task.
- **D4. A study in progress**: phase 1 changes tool 7's outputs as soon as it is in the working tree. If a study is being
  processed in this environment, phase 1 waits for its end (or the study keeps the current commit).
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
   **Measured on 2026-11-19 (synthetic, `refactor_check` against `before_phase`)**: 285 tests pass, 1 fails as expected
   (`test_tool9_reproduces_the_clean_epochs_route`: tool 9's values against the golden).
   - Tool 7's signal: RMS difference 1.0-1.2 µV (6.5-6.9 % of the signal's RMS), 97 % of its energy below 0.15 Hz (the
     transition band: the synthetic 1/f² background has much infra-slow power), 0.1 % in 4-45 Hz. Largest sample
     difference 6-16 µV (sim04, the movement bursts).
   - The chain (default settings): **no flag changes** (the `.fif` metadata and every flag table identical), tool 8bis's and
     tool 8's decisions identical (only `preproc_fingerprint` changes), tool 6's tables identical (its spectrogram figures
     change). Tool 9: PSD within 0.022 dB, band power within 0.028 dB per epoch, the stage means of the aperiodic exponent
     within 0.004 (per epoch up to 0.14: a fit that gains or loses a peak), 0.1-0.3 % fewer peaks detected.
   - `7variant` (128 Hz, 10 s epochs, no notch, LOWESS only): 3 (epoch, channel) pairs out of 17,280 change on the 1/f
     method (1 on sim01, 2 on sim02), nothing on amplitude / gradient / flat. `6variant`: tables identical.
   **Real data** (allowed on 2026-11-19, flag tables only read): flags do move, spread over the night, almost only
   added: per night amplitude +8 to +13 pairs, gradient +10 to +30, 1/f +6 to +8, out of 1,868-2,802 pairs (0.5-1.5 %),
   and 2-4 more epochs rejected by 8bis. As expected from the response: `zero` lets more of 50-64 Hz through (gradient)
   and more infra-slow drift (amplitude).
7. Goldens (accepted on 2026-11-19): `make_golden.py` turned out to replace the whole folder, and the goldens dated from
   before the exclusion refactor (three tests compared with that former content). Decided: **retire the pre-refactor
   reference**. Both goldens taken again with the current tools (the synthetic one restored from git first, then
   re-taken from make_golden's run), `test_tool9_reproduces_the_clean_epochs_route` and
   `test_tool6_matches_golden_but_the_flat_rule` removed, the clean-epo comparisons of `test_exclusion_tools.py` rewritten
   on the rule itself, `INTENDED_CHANGES` / `EXPECTED_MISSING` emptied, the `*_ap_removed_db` tolerance moved into
   `snapshot.compare_tables`.
8. Docs: SPEC §7 (step 4, Section 2.1, sidecar, *Skip and the settings*), §6 (spectrogram filter), §0 (step 6 sidecar),
   the TODO *Tool 7: band-pass phase* closed (removed, a line in the Order list), a reprocessing note (a study processed
   before: *Skip* asks, answer *Reprocess*). TESTS.md.

### Phase 2: the environment files and the version refresh

**2a. The table of versions** (nothing installed): for each package, its current pin, the newest version on conda-forge
(or PyPI for the pip ones) released at least 3 weeks ago, its date, whether it is a new major, its known breaking changes
(conda search, read only). You confirm the table (D3) before any install.

Read on 2026-11-19 (conda-forge, win-64; PyPI for the pip ones). Proposed = newest minor at least 3 weeks old, the
patch left free (`x.y.*`). **Bold** = to look at (a major, or kept below the newest).

| Package | Today | Proposed | Note |
|---|---|---|---|
| python | 3.12.10 | **3.14.\*** | 3.15 (Oct 2026) too recent: torch 2.14 has no 3.15 build. Every compiled package has 3.14 builds |
| numpy | 2.2.\* | 2.5.\* | |
| libblas | (MKL 2024.2 + intel-openmp, chosen by the solver) | **`libblas=*=*openblas`** | on Windows conda-forge picks MKL by default, today MKL 2026.1 + llvm-openmp, which crashes every numpy matrix product (torch row). Found 2026-11-20: the env rebuilt with torch from PyPI still got MKL 2026.1 and still crashed. OpenBLAS (pthreads, no OpenMP) removes MKL and tbb: lighter, and numpy, torch, YASA (lightgbm, `VCOMP140`) and GSSC run in one process, one Intel OpenMP loaded (torch's own). The results move by a float noise against MKL (2d) |
| pandas | 2.2.\* | **2.3.\*** (2.3.3) | D3: pandas 3 (Jan 2026; copy-on-write, new string type) = its own task. YASA 0.8 needs ≥ 2.3 |
| scipy | 1.16.\* | 1.18.\* | |
| mne → mne-base | 1.12.\* | 1.13.\* | |
| matplotlib → matplotlib-base | 3.10.\* | 3.11.\* | |
| yasa | 0.6.\* | 0.8.\* (0.8.1) | `predict_proba()` removed (2c) |
| antropy | 0.1.9 | 0.1.9 | kept below 0.2 on purpose (§3bis) |
| lspopt | (through yasa) | 1.4.\* | tool 6 imports it directly (2c) |
| statsmodels | 0.14.\* | 0.15.\* | its LOWESS smooths the PSD before the 1/f fit (tools 7, 8, 9) |
| scikit-learn | 1.7.\* | 1.9.\* | |
| joblib | 1.5.\* | 1.6.\* | |
| edfio | 0.4.\* | 0.4.\* (0.4.18) | |
| chardet | 5.2.\* | **5.2.\* kept** | 7.7 checked on 2026-11-19 (`edf_io_lib.detect_encoding` on the synthetic headers + headers written with accents): ASCII and UTF-8 the same (a UTF-8 header with one accent even better: 5.2 says MacRoman), but **latin-1 / cp1252 headers fall to a confidence of 0.01-0.04**, so tool 1 would refuse them (it asks 0.6): the Compumedics exports with accents. 5.2 is noarch, fine on Python 3.14 |
| openpyxl | 3.1.\* | 3.1.\* | nothing newer |
| voila | 0.5.\* | 0.5.\* (0.5.13) | |
| ipywidgets, ipyfilechooser, ipyevents | 8.1, 0.6, 2.0 | same | nothing newer but patches |
| ipykernel | 6.29.\* | **6.31.\*** | ipykernel 7 (Oct 2025) rewrote the kernel's event loop: `nbdriver`'s fix of the lost wake-up and the planned Stop button (`jupyter-ui-poll`, checked with ipykernel 6) depend on it. Its own check later |
| ipython | 8.36.\* | **9.\*** | major: the tools use `IPython.display` only |
| tk | 8.6.\* | 8.6.\* | python 3.14 still requires 8.6 |
| *full*: torch | conda `pytorch-cpu`, not pinned (2.8) | **pip `torch==2.14.*`** | conda-forge's torch (every Windows build is `cpu_mkl` / `cuda_mkl`) brings MKL 2026.1 + llvm-openmp 23, and llvm-openmp's `libiomp5md.dll` does not export `__atomic_compare_exchange` / `__atomic_load`, which `mkl_intel_thread.3.dll` delay-imports: **every numpy matrix product crashes Python** (`0xc06d007f`, found 2026-11-20 in tool 5's figures; llvm-openmp 23.1.4 / 23.1.5 / 23.1.6 alike; MKL sequential or TBB works). Decided: torch from PyPI (CPU wheel, its own MKL and OpenMP inside), numpy on OpenBLAS (libblas row); torch's pip requirements (sympy, networkx, filelock, fsspec, setuptools, typing_extensions, jinja2) from conda. GSSC's scoring loop is reproduced (§3bis): its equality is checked (test_tool0 step 3, night 100) |
| *full*: h5py, einops, pyyaml | 3.14, 0.8, 6.0 | 3.16.\*, 0.8.\*, 6.0.\* | |
| *full*: loguru | 0.7.\* | 0.7.\* | 0.8.0 is 7 days old |
| *full*: importlib_resources | (pip, through gssc) | added | GSSC's pip requirement, so pip installs nothing beside conda |
| *full*: gssc (pip) | ==0.0.9 | ==0.0.9 | |
| specparam (pip) | not pinned (2.0.0rc7) | ==2.0.0rc7 | still no final 2.0 |
| curryreader (pip) | not pinned (0.1.2) | ==0.1.2 | |
| *dev*: pytest | not pinned (9.1.1) | 9.2.\* | |
| *dev*: notebook | 7.4.\* | 7.6.\* | |

**2b. The three files** (no include in conda: each file repeats the list of the one below it):
- `environment.yml` (light, name `somnolens`): python, numpy, `libblas=*=*openblas`, pandas, scipy, `mne-base`,
  `matplotlib-base`, yasa, `antropy` (<0.2), `lspopt`, statsmodels, scikit-learn, joblib, edfio, chardet, openpyxl, voila,
  ipywidgets, ipyfilechooser, ipyevents, ipykernel, ipython, tk, pip: specparam, curryreader.
- `environment-full.yml` (name `somnolens`): light + torch's pip requirements from conda (sympy, networkx, filelock,
  fsspec, setuptools, typing_extensions, jinja2), importlib_resources, h5py, einops, loguru, pyyaml, pip: `torch==2.14.*`,
  `gssc==0.0.9`.
- `environment-dev.yml` (name `somnolens`): full + pytest, notebook.
- Every pin `x.y.*` (the pip ones `==` or `~=`), including the ones unpinned today (torch, specparam, still a
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
  `IPython` → ipython, ...). It replaces the audit by hand of 2026-10-19.
- numpy 2.5 removed `np.trapz` (renamed `np.trapezoid` in numpy 2.0, the same function): `spectral_lib` (tool 9, its
  former fallback evaluated `np.trapz` anyway) and `qc_rejected_epochs_lib` (tool 8's band powers) use `np.trapezoid`.
- `tests/chaincache.py`: the fingerprint of the harness includes the interpreter (`sys.prefix`), the versions of the
  main packages and numpy's BLAS (the `libblas-*.json` record of `conda-meta`), so a chain made in another environment
  is rerun instead of reused (otherwise `refactor_check` would compare stale outputs). A change to `BASE_DEPS`: the
  chain reruns once.
- matplotlib 3.11 removed `boxplot(labels=)` (renamed `tick_labels=` in 3.9): tool 6's stage inset and its dataset
  overview failed (a ⚠ per participant, no `dataset_overview.html`: 4 failures of the standard level on 2026-11-20).
  `quality_lib.py` and tool 6's page take `tick_labels=`. It also announces the end of `Colormap.set_bad`
  (`PendingDeprecationWarning`): `quality_lib.py` (the dead-epoch grey of the hypnospectrograms) and
  `autoscoring_compare_lib.py` take `with_extremes(bad=...)`, the same colours.

**2d. Build and compare** (each install announced and approved at the time; `CONDA_SSL_VERIFY=truststore`; disk space
checked first, each env 2-5 GB):
1. `somnolens_next` from `environment-dev.yml`: the chain from scratch, `refactor_check compare` against the outputs of
   phase 1. I explain each difference by its package (specparam final vs rc7 on the 1/f flags and tool 9, scipy / numpy
   on the PSD, YASA 0.8 on tool 3bis outside the chain).
2. `somnolens_light_check` from `environment.yml` + pytest: the quick level (every page executes, tool 3bis greys GSSC
   and SleepFM out with the new message, the tests skip cleanly), then the chain, compared with `somnolens_next`:
   identical expected (both on OpenBLAS).
3. YASA 0.8 on real data (with your permission, night 100): the stages against YASA 0.6.5, and the YASA part of
   `tests/score_edf_equivalence.py`, as SPEC asks.
4. Once you accept the differences: the goldens regenerated in `somnolens_next` (the real one copied aside first:
   `make_golden.py` replaces the folder), the full level there.

**Measured on 2026-11-20** (`somnolens_next`, OpenBLAS, synthetic dataset, against the outputs of phase 1):
- Tests: quick 331 passed; standard 498 passed, 4 failed (tool 6, matplotlib 3.11's `boxplot(labels=)`, 2c), then the
  tests of tools 0, 3bis, 5, 6 after the fix: 164 passed. The synthetic goldens of phase 1 still pass.
- The input EDF files regenerated under numpy 2.5: byte-identical.
- The chain (155 files): 82 identical, 38 within the tolerance, 35 different, of which no flag, no decision, no
  exclusion, no macrostructure metric. Different = the images (matplotlib 3.11), the footer of the MNE reports
  (`MNE-Python 1.13.2`), and a float noise: tool 9's `*_ap_removed_db` up to 2e-7 dB (the golden tolerance is 1e-5),
  its band powers and PSD up to 1e-10 relative; tool 6's metrics up to 1e-14; the `.fif` signals up to 5e-13 V.
- Outside the chain: tool 1 identical. Tool 3bis: the stages of GSSC and YASA identical (the 8 hypnograms byte-identical);
  YASA 0.8.1's probabilities move by at most 0.01 on 12 epochs, GSSC's by 0.0001 on one; `torch_version` 2.14.1+cpu,
  `yasa_version` 0.8.1 and the new `antropy_version` recorded. Variants of tools 6 and 7: float noise only.
- `somnolens_light_check` (`environment.yml` + pytest): 1.28 GB (the dev one 2.04 GB with torch, the former `somnolens`
  4.62 GB). Quick 331 passed; the chain rebuilt there is byte-identical to `somnolens_next`'s (155 files); the two
  GSSC tests of `test_autoscoring_formats` skip; tool 3bis greys GSSC and SleepFM out with the full-install message,
  YASA available.
- Night 100 (real, with your permission), tool 3bis with `score_edf_equivalence`'s settings (`run-3bis-real`), the
  former code in the former `somnolens` (YASA 0.6.5, torch 2.8, MKL) against the new code in `somnolens_next` (YASA
  0.8.1, torch 2.14 from PyPI, OpenBLAS): the three hypnograms byte-identical; YASA (C4) identical, probabilities
  included; GSSC the same 941 stages, probabilities within 1e-4 (the 4th stored decimal); SleepFM the same stages,
  its 5,652 5-s steps within 1e-6. YASA alone on C3 with the page's defaults: identical too. Against Score_EDF
  (`autoscore`, YASA 0.7 + antropy 0.2.3): GSSC identical, SleepFM 1 epoch (the tail, intended), YASA 5 / 941 epochs
  (antropy 0.2, as before: κ 0.110 against 0.113). Time: 86 s against 117 s.
- Full level on the real nights (`somnolens_next`): 70 passed, 2 failed against the real goldens of phase 1, both on the
  flat end of the nights (epochs 927-932 of 934, the signal at numerical zero: -273 dB against a night median of 7 dB,
  the median channel's std 4e-20 uV), where any measure is a rounding noise that moves with the libraries. Tool 9:
  the per-epoch tables differ there only (power, 1/f fit, peaks), every one of these epochs rejected, so the stage and
  global tables are unchanged. Tool 6: `noisy_vs_median_pct` of Fp1 on 731 goes from 0 to 0.54 % (epochs 928-932, a
  ratio of two rounding noises; a comparison column, not a flag). Tools 5, 7, 8bis, 8: identical to the goldens (no
  flag, decision or metric change on the real nights either). Its fix is a TODO of SPEC (*Flat epochs in the
  comparison with the median channel*).
- Goldens: the synthetic ones of phase 1 still pass (kept). The real ones retaken in `somnolens_next` (the former copied
  to `%TEMP%\dtk_golden_real_backup_phase1`): beyond the tolerance they differ on these 14 files only. Full level
  then: 72 passed.

### Phase 3: the final environment and the docs

1. The environment: `somnolens` removed and recreated from `environment-dev.yml` (a fresh create, not a rename: a
   rename breaks torch and the pip launchers), so the paths of CLAUDE.md stay valid. Then `somnolens_next` and
   `somnolens_light_check` removed. Each step approved at the time (removing `somnolens` cannot be undone, only rebuilt).
2. SPEC: *Conda environment* rewritten (the three files and what each holds, the versions and the date of this refresh,
   the refresh rule and the Windows gotchas moved there from the TODO), §3bis (YASA 0.8, `antropy_version`, the
   messages), *How to run* (which file to install, the tests run in the dev env), the TODO *Lighter conda environments*
   closed, the Order list (item 1 done, the slow-wave and spindle extraction next).
3. `README.md`, *Installation*: light and full, the two commands, the sizes measured in 2d. The same lines given in the
   chat for the tutorial's small update (D2): create the light env, create the full env, move a light env to full.
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
