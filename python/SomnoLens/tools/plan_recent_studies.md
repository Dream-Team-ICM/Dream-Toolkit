# Plan: remember the study folders (a *Recent studies* menu and a pasted path)

Status: **draft (2026-11-20)**, waiting for confirmation. Not started: phase 2 of `tools/plan_bandpass_and_envs.md` is
running its tests on the notebooks and the `somnolens` environment is being rebuilt, so this plan touches nothing before
that work is over (never edit files a running test suite executes).

## 1. The need

Each tool asks for the study folder first. Its picker (`FileChooser`, ipyfilechooser 0.6.0) opens on the launch folder,
the repo root, so a user whose data sit far from the toolkit climbs the parent menu or the drive letters in every tool,
for every session. Two facts make it worse:
- each Voila page runs in its own kernel: nothing picked in one tool reaches the next one, so the memory has to be a file;
- the picker has no field to type a path in: a network share `\\server\share\...` that is not mapped to a drive letter
  cannot be reached at all today.

## 2. Decisions

Answered on 2026-11-20:
- **A *Recent studies* menu, one click**: above the study folder's picker, on `(choose)` when the page opens. Choosing a
  study does exactly what picking it in the dialog does (the tool's callback runs, the *Inputs found* list fills). Nothing
  is chosen or scanned on its own.
- **A folder enters the list as soon as it is picked** (dialog, menu or pasted path). A wrong pick (`data/` instead of the
  study) enters too, and sinks as other studies are used.
- **A *Paste a path* field** beside the menu: a folder path copied from the Windows Explorer address bar (or *Copy as
  path*, quotes removed), then *Use*. The only way to reach an unmapped network share.
- **The file**: `%LOCALAPPDATA%\DreamToolkit\recent_studies.json`, one per Windows user, outside the repo and outside any
  study, beside SleepFM's download cache (`sleepfm_lib.cache_root`, same fallback `~/.cache/DreamToolkit` without
  `LOCALAPPDATA`).

Proposed here (to confirm or annotate):
- **R1. Which tools**: every page whose first picker is the study folder: 0, 1, 2, 3, 3bis, 4, 5, 5bis, 6, 7, 8, 8bis, 9.
  **Not 1bis**: its folder holds the identifying originals, outside the study. The archived Jupyter versions
  (`tools/jupyter_notebook/`) stay frozen.
- **R2. Contents**: at most **8 studies**, the most recent first, each `{"path", "last_used", "tool"}` (tool = `"7"`,
  `"8bis"`...). The same folder is kept once (compared with `os.path.normcase(os.path.abspath(...))`). Only folder paths:
  no participant data, nothing leaves the PC, nothing is written in a study.
- **R3. Menu label**: `D:\studies\my_study  (tool 7, 2026-11-19)`. The full path, since two studies may share a name.
- **R4. An unreachable study stays in the menu**: the list is **not** checked when the page opens, because asking
  Windows about a disconnected network share can block for tens of seconds per folder. The check happens when one is
  chosen: `❌ not reachable: <path> (a disconnected network drive?)`, the picker unchanged. The same message for a pasted
  path that is not a folder.
- **R5. The picker keeps opening where it opens today** (the launch folder). Opening it on the last study was considered
  and dropped: ipyfilechooser lists the folder as soon as it is built, so an unreachable last study would freeze or break
  the page at its opening. The menu already gives the last study in one click.
- **R6. Errors**: the list is a convenience, never a step. An unreadable or corrupt file → the menu is empty, a ⚠ line says
  so, and **the file is never overwritten** (the rule of the other registries). A failed save → a ⚠ line under the menu,
  the tool goes on with the folder picked. Written through a temporary file + `os.replace` (like `SF.write_names`).
- **R7. A page opened before** another tool remembered a study does not show it in its menu: the list is read when the
  page opens (reload the page).
- **R8. Tests never touch the user's list**: the environment variable `SOMNOLENS_RECENT_STUDIES` names another file
  (empty = remember nothing). `tests/conftest.py` sets it to a file of the test run for the whole session; the kernels
  started by the tests inherit it.

## 3. Where the code goes

`ui_lib` writes nothing to disk (its docstring), so the file goes to `study_files_lib` (already the module of the
study's registries, imported as `SF` next to `UI` by every page) and the widgets to `ui_lib`:

- `study_files_lib`:
  - `recent_studies_file()`: the path (the environment variable first, else `%LOCALAPPDATA%\DreamToolkit\...`).
  - `recent_studies()` → `(studies, warning)`: the list read from the file, `[]` + a warning when unreadable.
  - `remember_study(path, tool)` → `warning` (`''` when saved): moves the folder to the top, keeps 8.
- `ui_lib`:
  - `recent_studies_row(picker, studies)` → `(row, note)`: the menu + the paste field + its *Use* button (role `helper`)
    + a message line. Choosing or pasting a folder selects it in `picker` the way its dialog does (`reset(path)`, open,
    *Select*: the same three calls as the tests' `_dtk_pick`), so the tool's own callback runs unchanged. `note(text)`
    shows a ⚠ line (the warnings of `SF`).
- Each of the 13 notebooks, about three lines:
  ```python
  studies, warning = SF.recent_studies()
  recent_row, recent_note = UI.recent_studies_row(fc_data, studies)
  recent_note(warning)
  # ... recent_row placed right above fc_data in Section 1 ...

  def _on_data(_fc):                     # the study callback the tool already has
      recent_note(SF.remember_study(fc_data.selected_path, '9'))
      ...
  ```
  The picker variable and the callback keep their names (`fc_data`, `chooser`, `fc_folder`, `fc_edf`; `_on_data`,
  `_on_study`, `_detect_hypno_suffixes`, `_update_info`...): the tests drive them by name.

## 4. Phases

### Phase 1: the two libraries + their tests
- `study_files_lib`: the three functions above.
- `ui_lib`: `recent_studies_row`.
- `tests/conftest.py`: `SOMNOLENS_RECENT_STUDIES` for the session.
- Tests (in `tests/test_ui_lib.py`, quick): order and cap of 8, the same folder kept once (case and slashes), an
  unreadable file left as it is, the variable honoured; the row selects the folder in a picker and runs its callback
  (menu and pasted path, quotes removed), an unreachable path or a file refused with the picker unchanged.
- Check by hand that `reset(path=r'\\localhost\c$\...')` works in ipyfilechooser (a UNC path), if the admin share is
  open on this PC; otherwise noted as untested.

### Phase 2: the 13 notebooks
- The three lines in each page (rename-to-`.txt` method for the big notebooks, SPEC *Editing the larger notebooks*).
- `tests/test_ui_lib.py` page guard: every page with a study folder picker builds `recent_studies_row` and remembers
  the folder in its study callback.
- One page driven end to end (tool 9, quick): choosing a study in the menu fills *Inputs found* like a pick, and the
  folder is at the top of the test file.
- Outputs: none changes (the list is outside every output). The chain cache sees changed notebooks, so the chain reruns.

### Phase 3: docs
- SPEC: *Cross-cutting → Voila page conventions*, a *Recent studies* bullet (the file, R2 to R8, the two functions, the
  row); *How to run → Study folder layout*, one line (the menu, the paste field, where the list lives).
- CLAUDE.md: proposed, one reminder in the *Voila page conventions* line (`recent_studies_row` + `remember_study` on every
  study folder picker).
- README: one sentence for the users (the menu, the paste field for a network share).

## 5. Out of this plan
- A home page that holds the study and opens each tool with it: possible later on top of the same file.
- Remembering the other pickers (tool 5's and 7's output folder, tool 1bis's originals folder).
- A *Forget* button: the list renews itself (8 entries); the file can be deleted by hand.
