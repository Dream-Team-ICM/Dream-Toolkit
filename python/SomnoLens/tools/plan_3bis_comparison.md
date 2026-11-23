# Plan: tool 3bis, more agreement metrics and Section 4 in the SomnoLens layout

Status: **confirmed (2026-11-23)**, D1 to D5 accepted as proposed. Phase 1 written: `refactor_check` on the synthetic
data gives step 1's former columns identical (new columns and `global_autoscoring_pooled.tsv` added, HTML reports
changed); `-m "quick or ((tool3bis or tool0) and not full)"`: 362 passed. Phase 2 next. Closes the SPEC item *Tool
3bis: follow-ups of the Score_EDF port → Comparison outputs in the SomnoLens layout*. Asked at the same time and put in
SPEC as a TODO (*Methods text for a paper*): a notebook or another tool that extracts the essential points to write
in a paper's methods section.

## 1. Decisions (conversation of 2026-11-23)

- **New metrics**, in step 1 and in Section 4, per night and pooled: **macro-F1**, **balanced accuracy**, **MCC**
  (multiclass, Gorodkin's formula = sklearn's `matthews_corrcoef`). No weighted F1. No probabilistic "soft recall", no
  sleep latency or SOREMP: the per-epoch stages and probabilities stay in the outputs, users compute those themselves.
- **Macro-F1 = mean of the per-stage F1 over the stages the expert scored on that night** (option A). The literature
  has no standard: YASA's and U-Sleep's evaluation code take the 5 stages with a stage absent from both sides counted
  as F1 = 1, or the union of expert and model stages (sklearn's default). The three give the same value on a night with
  the 5 stages. A is stable on naps, puts every model of a night on the same stages, and follows the same logic as the
  balanced accuracy. The per-stage F1 columns do not change (a stage the model gives but the expert never scored = 0,
  a stage absent from both = empty), so the other conventions stay computable.
- **Single-stage nights** (the expert scored one stage only; strict rule, `n_stages_expert == 1`): kappa is 0 (or
  undefined when the model is right everywhere) and MCC is 0 by construction, whatever the model. Accuracy, balanced
  accuracy and macro-F1 reduce to the share of epochs given the expert's stage and stay meaningful. So:
  - the per-night tables always hold the computed values, plus `n_stages_expert` and `single_stage_expert`;
  - a ⚠ line in the run log, a note in the night's report, an explanation in the page (intro and Section 4 help);
  - one checkbox, ticked by default: *Leave single-stage nights out of the kappa and MCC across nights*. It acts only
    on what summarises several nights (the median / mean figures, Section 4's per-night kappa panel). Pooled metrics
    keep their epochs. It changes no per-night file: no conflict with Skip, nothing to record.
- **Section 4 in the SomnoLens layout**: data in `derivatives/autoscoring_comparison/`, reports in
  `reports_autoscoring_comparison/`, TSV, stages W / N1 / N2 / N3 / R, per-night files and database tables globbed from
  them, HTML reports, the recalibration parameters saved (today only in the log), Score_EDF's empty `tet` column dropped.
- **Section 4 = one comparison of a set of nights, no Skip.** The recalibrations learn each night from the others, so a
  night's recalibrated result changes with the set. Each *Compare* recomputes every chosen night and **replaces the
  previous comparison** (both folders emptied once the inputs are read), so the globbed database tables are exactly that
  comparison.
- **`autoscore_outputs/`** of a former run stays as it is (not deleted, not migrated): one log line when it exists.
- **Step 1's nights already scored** (kept by Skip): Skip does not change. Their comparison table lacks the new columns
  (`–` in the database tables), and the database report counts them and points to Section 4 (every night gets the new
  metrics there) or to Skip unticked.
- **Section 4 keeps reproducing Score_EDF's computation** with its settings (CLAUDE.md): only the layout changes, and
  `tests/score_edf_equivalence.py` compares values instead of files.

## 2. Decided on 2026-11-23 (all accepted as proposed)

- **D1. Epochs a model did not predict, in Section 4** (no probability over the epoch: the incomplete tail a 30-s model
  drops, a gap). Score_EDF counted them as disagreements in the accuracy and as a sixth class `Unknown` in the kappa.
  Proposed: left out of every metric and counted (`n_not_predicted`), as step 1 compares on the epochs both scored.
  Section 4 then gives step 1's values for GSSC and YASA (a test checks it). The equivalence run says whether
  Score_EDF's data had any such epoch.
- **D2. The page of Section 4.** Proposed: the database view (pooled table, agreement figure, confusion matrices,
  hypnogram grid) and the paths of the reports. The per-night hypnodensities only in the per-night reports (today one
  figure per night in the page: heavy for a large study).
- **D3. Place of the checkbox.** Proposed: Section 1, under the expert hypnogram suffix. It concerns the comparison
  with the expert, and both step 1's database report and Section 4 read it.
- **D4. Recalibration B's refitted final layer.** Proposed: saved per held-out night (`.npz`, 5 × 128 weights and 5
  biases, a few kB) beside its penalty in the parameter table. Otherwise only the penalty.
- **D5. Model names in the tables.** Proposed: `gssc`, `yasa`, `sleepfm`, `sleepfm_recal_a`, `sleepfm_recal_b` (step 1's
  keys). The figures keep the display names (GSSC, SleepFM recal A, ...).

## 3. The metrics: one function for both steps, per night and pooled

`autoscoring_lib.agreement_metrics(cm)`, from a 5 × 5 confusion matrix (rows expert, columns model, W N1 N2 N3 R):
- rebuilds the (expert, model) pairs from the counts and calls sklearn with `labels=STAGES` for the kappa, the F1 per
  stage and the MCC: the same values as today's per-night kappa and F1, bit for bit;
- `macro_f1` = mean F1 over the stages with expert epochs, `balanced_accuracy` = mean recall over the same stages;
- `n_stages_expert`, `single_stage_expert`; sklearn's warnings on undefined values silenced, the value kept (kappa
  empty when undefined, MCC 0, sklearn's convention).

The same function gives a night's metrics (its confusion matrix) and the pooled ones (the summed matrix). Step 1's
`compare_scoring` calls it: its current columns keep their values, the new ones are added.

Known answer for the test, the nap discussed on 2026-11-23 (expert 30 W + 10 N1; the model: 28 W and 2 N1 on the W
epochs, 8 N1, 1 W and 1 N2 on the N1 epochs): accuracy 0.900, macro-F1 0.875 (the union of stages would give 0.583),
balanced accuracy 0.867.

## 4. Phases

Each phase: the work, its tests (level proposed at that time), then a stop for your check. Tool 3bis has no golden (it
is outside the chain): `tests/refactor_check.py` with `tests/outside_chain.py` is the before / after reference.

### Phase 1: the new metrics in step 1 (and tool 0's step 3)

1. Before any change: `outside_chain.py 3bis` saved with `refactor_check.py save` (synthetic).
2. `autoscoring_lib`: `agreement_metrics`; `compare_scoring` uses it; `metrics_html` gains macro-F1, balanced accuracy and
   MCC, and the single-stage note. Tool 0's step 3 shows them too (it calls `metrics_html`).
3. Notebook:
   - the checkbox (D3);
   - `score_one`: the new columns in `{id}_autoscoring_comparison.tsv` after `f1_R`, the ⚠ log line for a single-stage
     night;
   - `rebuild_database_summary`: `global_autoscoring_pooled.tsv` (one row: nights, epochs, the metrics of the pooled
     confusion matrix) shown as a table; the per-participant figure gains macro-F1, balanced accuracy and MCC
     (single-stage nights out of kappa and MCC when ticked, their number written under it); the participants table
     gains the columns; the line on nights compared before these metrics.
4. Page texts: the intro's step 1 item, the metric help.
5. Tests: `tests/test_autoscoring_metrics.py` (quick): the known answer above, a single-stage night (model right
   everywhere, model wrong on some epochs), a stage absent from both sides, a stage the model gives but the expert
   never scored, equality with sklearn on random labels. Then `refactor_check.py compare`: step 1's outputs identical
   except the new columns and the HTML reports. `test_tool0::test_step3_gssc_equals_tool3bis`.

### Phase 2: Section 4 in the SomnoLens layout

1. `autoscoring_compare_lib`:
   - the ported computation keeps Score_EDF's internal names (Wake, REM); one function turns its epoch table into the
     toolkit's: W / R, model keys (D5), recording-relative `epoch_idx` (the window starts on a whole expert epoch, so
     `epoch_idx = start_seconds / 30`, the index of step 1's `_hypnodensity.tsv`), no `tet`;
   - `compute_agreement` replaced by the per-night and pooled tables from `agreement_metrics` (D1);
   - figures: the toolkit's stage labels; the hypnogram grid split into figures of 20 nights (like step 1's stacked
     hypnodensities); the per-night kappa panel follows the checkbox.
2. Notebook, `run_compare`: reads as today; then empties both folders; writes the per-night data, the per-night tables,
   the per-night reports, the database tables, the database report (data before reports); `failed_comparison.tsv`;
   the log line on `autoscore_outputs/`; the page as D2 says.
3. Outputs:

   ```
   {study}/
     derivatives/autoscoring_comparison/
       recalibration_params.tsv               ← 1 row per method and held-out night: training nights,
                                                 A's temperature and biases, B's penalty and inner log-losses
       <subtree>/{id}_comparison_epochs.tsv   ← 1 row per epoch and model: file_id, model, epoch_idx, onset_s,
                                                 stage_expert, expert_coverage, stage_model, p_W … p_R
       <subtree>/{id}_recal_b_layer.npz       ← B's refitted final layer for that night (D4)
     reports_autoscoring_comparison/
       <subtree>/{id}_comparison.tsv          ← 1 row per model: n_compared, n_excluded_expert, n_not_predicted,
                                                 accuracy, kappa, macro_f1, balanced_accuracy, mcc, f1_W … f1_R,
                                                 n_stages_expert, single_stage_expert, the channels used
                                                 (today's channels_used.csv) and the EEG reference
       <subtree>/{id}_confusion.tsv           ← long: file_id, model, expert, predicted, n
       <subtree>/{id}_comparison_report.html  ← each model's hypnodensity over the expert hypnogram,
                                                 agreement table, confusion matrices, channels
       global_comparison_summary.tsv          ← every night × model (globbed)
       global_comparison_pooled.tsv           ← 1 row per model, from the pooled confusion matrix
       global_confusion.tsv                   ← pooled, per model
       global_comparison_report.html          ← pooled table, agreement figure, confusion matrices,
                                                 hypnogram grid, per-night table, recalibration parameters
       failed_comparison.tsv                  ← only when a night failed
   ```

4. Page texts: intro, the Details tree, Section 4's help (outputs, single-stage nights, no Skip, a *Compare* replaces
   the previous one).
5. Tests:
   - `tests/test_autoscoring_compare.py` (standard, needs gssc): on a copy of the synthetic study, GSSC on two
     participants, then *Compare*: the files of the tree, W / R, Section 4's per-night metrics equal to step 1's for GSSC
     (D1), a second *Compare* on one participant removes the other's files.
   - `outside_chain.run_tool3bis` also runs Section 4 (GSSC), so `refactor_check` covers it from now on.
   - `score_edf_equivalence.py`: `compare_tables` reads the new files (mapped back to Score_EDF's names) and compares
     the predicted stages, the probabilities, the pooled accuracy / kappa / F1 and the per-night kappa;
     `autoscore_outputs_strict/` becomes a copy of the two new folders. Run by hand in the `autoscore` environment, on
     the simulated nights and on night 100 (a real recording: your go-ahead before I run it, or you run it).

### Phase 3: documentation

- SPEC §3bis: *Comparison* (the new metrics, the macro-F1 rule and what the literature does, single-stage nights, the
  checkbox), *Section 4* rewritten (layout, no Skip, a *Compare* replaces the previous one, D1, the recalibration
  table), the output lists; the item closed in *Tool 3bis: follow-ups*; the last sentence of *Standalone comparison of
  hypnograms* (Section 4 now has F1 and MCC per night and pooled).
- TESTS.md: the two new test files. README if it lists 3bis's outputs (to check). CLAUDE.md: unchanged (no
  cross-cutting rule moves; "Section 4 must keep reproducing Score_EDF with its settings" still holds).
- This plan deleted (its own commit) once you confirm the task is done.
