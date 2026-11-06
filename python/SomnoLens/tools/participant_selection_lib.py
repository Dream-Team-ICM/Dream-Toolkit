"""Shared participant handling: the exclusion registry and the participant selector widget.

A deliberate exception to the toolkit's "duplicate small helpers" rule (like qc_rejected_epochs_lib):
this code is used by tools 5, 5bis, 6, 7, 8, 8bis, 9 and their Curry twins, and seven copies would drift.

1. Exclusion registry: `<data_folder>/config_param/participant_exclusions.tsv`
   One row per decision, written by the tool that took it:
       file_id | source | decision | reason | comment | decided_at
   source   = 5bis_criteria / 5bis_manual / 7_manual / 8_manual / 8bis_auto
   decision = exclude, or include (a FORCED inclusion, written by 5bis's manual editor only)
   Each tool rewrites only its own rows. A decision of tool 5bis holds for the whole study; a decision taken from
   tool 7 on belongs to one run of tool 7, named in its source (8bis_auto for the first run, 8bis_auto_v2 for the run
   preproc_epo_v2), and a tool reading a run ignores the rows of the other runs. The effective status of a participant is:
       a forced inclusion exists         -> included (it overrides every exclusion)
       else at least one exclusion       -> excluded, reasons joined with their source
       else                              -> included
   Excluded participants are still processed by every tool; they are only flagged in the tables
   (`participant_excluded`, `participant_exclude_reason`) and left out of group statistics / figures.

2. ParticipantSelector: the "All participants / Subset only" block shared by the processing tools. With Skip
   ticked, it also asks what to do with the participants made with other settings than the run's.
"""
import datetime
import html
import os
import re
from pathlib import Path

import pandas as pd

REGISTRY_NAME = 'participant_exclusions.tsv'
REGISTRY_COLUMNS = ['file_id', 'source', 'decision', 'reason', 'comment', 'decided_at']
STUDY_SOURCES = ('5bis_criteria', '5bis_manual')       # decided on the macrostructure, before the preprocessing
RUN_SOURCES = ('7_manual', '8_manual', '8bis_auto')    # decided from tool 7 on: + the suffix of the run
SOURCES = STUDY_SOURCES + RUN_SOURCES
EXCLUDE, INCLUDE = 'exclude', 'include'
COL_EXCLUDED = 'participant_excluded'
COL_REASON = 'participant_exclude_reason'


def id_key(file_id):
    """Comparison key for a participant id (normcase: Windows ids differ only by case / slashes)."""
    return os.path.normcase(str(file_id).strip())


# ---------------------------------------------------------------------------
# Locating the data folder
# ---------------------------------------------------------------------------
def registry_path(data_folder):
    return Path(data_folder) / 'config_param' / REGISTRY_NAME


def find_data_folder(path):
    """The data folder a path belongs to: the first ancestor (or the path itself) that holds a
    `config_param/` folder, else the parent of the `derivatives/` or `reports_*` folder the path lies
    in, else None. Lets a tool that only knows e.g. `<data>/derivatives/preproc_epo` find the registry."""
    if path is None:
        return None
    p = Path(path)
    if p.is_file():
        p = p.parent
    for anc in [p] + list(p.parents):
        if (anc / 'config_param').is_dir():
            return anc
    for anc in [p] + list(p.parents):
        if anc.name == 'derivatives' or anc.name.startswith('reports_'):
            return anc.parent
    return None


# ---------------------------------------------------------------------------
# Reading / writing the registry
# ---------------------------------------------------------------------------
def empty_registry():
    return pd.DataFrame(columns=REGISTRY_COLUMNS)


def load_registry(data_folder):
    """(registry DataFrame, warning or None). Absent file -> empty registry, no warning (nobody
    excluded yet). Unreadable file -> empty registry + a warning: every participant is then treated
    as included, which the calling tool must show to the user."""
    if data_folder is None:
        return empty_registry(), None
    path = registry_path(data_folder)
    if not path.is_file() or path.stat().st_size == 0:        # absent, or an empty file: no decision yet
        return empty_registry(), None
    try:
        df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    except Exception as e:
        return empty_registry(), f'exclusion registry unreadable ({path}): {e}. Everyone treated as included.'
    missing = [c for c in ('file_id', 'source', 'decision') if c not in df.columns]
    if missing:
        return empty_registry(), (f'exclusion registry {path} has no {", ".join(missing)} column: '
                                  f'everyone treated as included.')
    for c in REGISTRY_COLUMNS:
        if c not in df.columns:
            df[c] = ''
    df = df[REGISTRY_COLUMNS + [c for c in df.columns if c not in REGISTRY_COLUMNS]]
    df['decision'] = df['decision'].str.strip().str.lower()
    return df.reset_index(drop=True), None


def _write_atomic(df, path):
    """Write to a temporary file, then rename it over the registry: an interruption never leaves a
    half-written registry behind (os.replace is atomic on the same volume)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    df.to_csv(tmp, sep='\t', index=False)
    os.replace(tmp, path)


def write_registry_rows(data_folder, source, rows, replace_ids=None):
    """Replace this tool's rows in the registry and return the registry written.

    source      : who decides (one of SOURCES, with the suffix of the run for a decision taken on a later run)
    rows        : list of dicts with file_id, decision ('exclude' / 'include'), reason, comment
    replace_ids : the participants whose `source` rows are replaced by `rows`; None = ALL the rows
                  of `source` (e.g. a new 5bis validation replaces the previous one entirely).
    Rows of the other sources are never touched. Raises when the existing registry is unreadable
    (refusing to overwrite decisions we could not read)."""
    if source not in STUDY_SOURCES and not source.startswith(RUN_SOURCES):
        raise ValueError(f'unknown registry source {source!r}')
    path = registry_path(data_folder)
    reg, warn = load_registry(data_folder)
    if warn:
        raise RuntimeError(warn.replace('Everyone treated as included.', 'Not overwritten.'))
    mine = reg['source'] == source
    if replace_ids is not None:
        ids = {id_key(i) for i in replace_ids}
        mine &= reg['file_id'].map(id_key).isin(ids)
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new = pd.DataFrame([{'file_id': str(r['file_id']), 'source': source,
                         'decision': r.get('decision', EXCLUDE), 'reason': r.get('reason', ''),
                         'comment': r.get('comment', ''), 'decided_at': r.get('decided_at') or now}
                        for r in rows], columns=REGISTRY_COLUMNS)
    for r in new.itertuples():
        if r.decision not in (EXCLUDE, INCLUDE):
            raise ValueError(f'decision must be exclude or include, got {r.decision!r}')
        if r.decision == INCLUDE and source != '5bis_manual':
            raise ValueError('a forced inclusion can only be written by 5bis_manual')
    # Same decisions as already recorded (e.g. tool 8bis re-run with the same thresholds): leave the file
    # untouched, so the original decided_at survives.
    content = ['file_id', 'decision', 'reason', 'comment']
    old_rows = sorted(map(tuple, reg.loc[mine, content].astype(str).values.tolist()))
    new_rows = sorted(map(tuple, new[content].astype(str).values.tolist()))
    if path.is_file() and old_rows == new_rows:
        return reg
    out = pd.concat([reg[~mine], new], ignore_index=True)
    out = out.sort_values(['file_id', 'source', 'decided_at'], kind='stable').reset_index(drop=True)
    _write_atomic(out, path)
    return out


def remove_registry_rows(data_folder, source, file_ids):
    """Delete this source's rows for these participants (e.g. 'Remove exclusion' in tool 8)."""
    return write_registry_rows(data_folder, source, [], replace_ids=file_ids)


# ---------------------------------------------------------------------------
# Effective status
# ---------------------------------------------------------------------------
def rows_of_run(registry, run=''):
    """The rows that hold for one run of tool 7 (its suffix, '' = the first run): the study's and the run's own. The
    decisions taken on the other runs are left out."""
    own = {source + run for source in RUN_SOURCES}
    return registry[[not (s.startswith(RUN_SOURCES) and s not in own) for s in registry['source']]]


def runs_in_registry(registry):
    """The suffixes of the later runs the registry holds decisions for: 8bis_auto_v2 -> _v2."""
    runs = set()
    for s in registry['source']:
        for source in RUN_SOURCES:
            if s.startswith(source) and s != source:
                runs.add(s[len(source):])
    return runs


def manual_exclusions(registry, run=''):
    """The participants excluded by hand on one run of tool 7 (tools 7 and 8)."""
    rows = registry[registry['source'].isin(['7_manual' + run, '8_manual' + run])]
    return sorted(set(rows['file_id']))


def effective_status(registry, run=''):
    """{normcase(file_id): {'file_id', 'excluded', 'reason', 'forced_include'}} for every participant
    that has at least one row holding for this run of tool 7 ('' = the first run, the one a tool without a choice of
    run reads). A participant with no row is included (use `status_of`)."""
    out = {}
    if registry is None or len(registry) == 0:
        return out
    registry = rows_of_run(registry, run)
    for key, grp in registry.groupby(registry['file_id'].map(id_key), sort=False):
        forced = grp[grp['decision'] == INCLUDE]
        excl = grp[grp['decision'] == EXCLUDE]
        reasons = ' | '.join(f'{r.source}: {r.reason}' if r.reason else r.source for r in excl.itertuples())
        out[key] = {'file_id': grp['file_id'].iloc[0],
                    'excluded': len(excl) > 0 and len(forced) == 0,
                    'reason': reasons if len(forced) == 0 else '',
                    'forced_include': len(forced) > 0,
                    'overridden_reason': reasons if len(forced) > 0 else ''}
    return out


def status_of(status, file_id):
    """(excluded, reason) of one participant; (False, '') when the registry says nothing about it."""
    s = status.get(id_key(file_id))
    return (s['excluded'], s['reason']) if s else (False, '')


def excluded_ids(status):
    return sorted(s['file_id'] for s in status.values() if s['excluded'])


def add_exclusion_columns(df, status, id_col='file_id'):
    """Copy of `df` with participant_excluded (bool) and participant_exclude_reason (str) appended
    (replaced when already present, so a rebuilt table always reflects the CURRENT registry)."""
    out = df.drop(columns=[c for c in (COL_EXCLUDED, COL_REASON) if c in df.columns]).copy()
    pairs = [status_of(status, fid) for fid in out[id_col]]
    out[COL_EXCLUDED] = [p[0] for p in pairs]
    out[COL_REASON] = [p[1] for p in pairs]
    return out


def tables_out_of_date(status, *tables):
    """True when a database table does not say what the registry says about one of its participants (`status`: the
    effective status for the run of these tables): a decision taken or lifted since it was built is not in it yet. The
    contents are compared, not the dates: a decision on another run of tool 7 does not concern these tables. A table
    absent, unreadable or without the exclusion columns: nothing to compare."""
    for table in tables:
        if table is None or not Path(table).is_file():
            continue
        try:
            df = pd.read_csv(table, sep='\t', dtype=str, keep_default_na=False,
                             usecols=['file_id', COL_EXCLUDED, COL_REASON]).drop_duplicates()
        except Exception:
            continue
        for fid, excluded, reason in zip(df['file_id'], df[COL_EXCLUDED], df[COL_REASON]):
            if status_of(status, fid) != (excluded == 'True', reason):
                return True
    return False


def stale_tables_html(how_to_refresh):
    """Orange box shown at scan / load when `tables_out_of_date` is True."""
    import ui_lib as UI      # the look shared by the Voila pages (imported on use: screen only)
    return UI.box('warn', '<b>The exclusion registry changed after this tool\'s '
                  'database tables were built</b>: a participant excluded (or re-included) since then is not '
                  f'flagged in them yet. {how_to_refresh}')


STALE_RERUN = ('Click <b>Run</b> with <i>Skip already processed</i> ticked: nobody is reprocessed, the '
               'database tables and reports are rebuilt with the current registry.')


def excluded_banner_html(reason):
    """Red banner for the top of an excluded participant's own report."""
    import html as _html
    return ('<div style="background:#fde8e8;border:2px solid #d03b3b;color:#8b1a1a;padding:8px 12px;'
            'margin:6px 0;border-radius:4px;font-weight:bold">EXCLUDED participant: '
            f'{_html.escape(reason)}<br><span style="font-weight:normal;font-size:90%">Still computed so '
            'the decision can be revisited; left out of the group statistics and figures.</span></div>')


# ---------------------------------------------------------------------------
# Participant selector (All / Subset + skip), shared by the processing tools
# ---------------------------------------------------------------------------
def parse_id_list(text):
    """'165, 176;180\\n181' -> ['165', '176', '180', '181'] (comma / semicolon / space / newline)."""
    return [t for t in re.split(r'[,;\s]+', text or '') if t]


# ---------------------------------------------------------------------------
# Participants made with other settings (Skip ticked)
# ---------------------------------------------------------------------------
# differences = {file_id: [(setting, recorded, this run), ...]}, found by each tool before any analysis. The run asks
# what to do with them instead of reprocessing on its own: a page starts from its defaults at every launch, so a
# difference may only mean that it was just opened, and reprocessing could overwrite a whole study.
DIFFERENCES_SHOWN = 20       # lines of the box, then '… (+N)'


def group_differences(differences):
    """{file_id: [(setting, recorded, this run), ...]} -> [(setting, recorded, this run, [file_ids])], one row per
    distinct difference, the most frequent first."""
    groups = {}
    for fid, rows in differences.items():
        for row in rows:
            groups.setdefault(tuple(str(value) for value in row), []).append(fid)
    ordered = sorted(groups.items(), key=lambda item: -len(item[1]))     # a stable sort: first seen first on a tie
    return [(*key, fids) for key, fids in ordered]


def other_settings_html(differences, n_processed):
    """The box listing the differences, with the participants concerned, before the two answers."""
    import ui_lib as UI      # the look shared by the Voila pages (imported on use: screen only)
    groups = group_differences(differences)
    lines = []
    for setting, recorded, current, fids in groups[:DIFFERENCES_SHOWN]:
        names = ', '.join(fids[:UI.MISSING_SHOWN])
        if len(fids) > UI.MISSING_SHOWN:
            names += f' (+{len(fids) - UI.MISSING_SHOWN})'
        lines.append(f'<li><b>{html.escape(setting)}</b>: {html.escape(recorded)} → {html.escape(current)} '
                     f'<small>({len(fids)} participant(s): {html.escape(names)})</small></li>')
    if len(groups) > DIFFERENCES_SHOWN:
        lines.append(f'<li>… (+{len(groups) - DIFFERENCES_SHOWN} other differences)</li>')
    return UI.box('warn', f'<b>{len(differences)} of the {n_processed} processed participant(s) were made with other '
                  f'settings</b> (recorded → this run). Nothing has run yet.<ul style="margin:4px 0;">'
                  f'{"".join(lines)}</ul>Keeping them mixes settings in the database tables.')


def other_settings_list(differences):
    """'165 (signal preparation), 176 (flags of C3, flags of O1)', for the log at the end of a run."""
    return ', '.join(f'{fid} ({", ".join(dict.fromkeys(row[0] for row in rows))})'
                     for fid, rows in differences.items())


class ParticipantSelector:
    """'All participants / Subset only' + 'Skip already processed', with a summary line and a
    prominent warning box. Usage in a tool:

        sel = ParticipantSelector()
        display(sel.widget)
        sel.set_participants(ids, done={...}, excluded={id: reason})   # after the scan
        to_run = sel.selected_ids()                                      # at Run
    """

    def __init__(self, skip_default=True):
        import ipywidgets as w
        import ui_lib            # the look shared by the Voila pages
        self._w = w
        self._ui = ui_lib
        self._ids = []                     # all participant ids, display order
        self._done = set()                 # normcase keys of the already-processed ones
        self._excluded = {}                # normcase key -> reason
        self.dd_mode = w.ToggleButtons(options=['All participants', 'Subset only'],
                                       value='All participants', style={'button_width': '150px'})
        # Two widgets: the Combobox ADDS a participant (its autocompletion offers only the ones not
        # selected yet, validated on Enter), the TagsInput SHOWS the subset (x on a tag removes it).
        # TagsInput's own autocompletion (allowed_tags) cannot shrink as ids are picked: its validator
        # rejects any value outside allowed_tags. So it gets none, and unknown typed ids are removed.
        self.cmb_add = w.Combobox(placeholder='type an id to add it to the subset...', options=[],
                                  ensure_option=True, continuous_update=False,
                                  layout=w.Layout(width='300px'))
        self.tags = w.TagsInput(value=[], allow_duplicates=False, layout=w.Layout(width='620px'))
        self._cleaning = False
        self.txt_paste = w.Text(placeholder='or paste a list: 165, 176 ...',
                                layout=w.Layout(width='300px'))
        self.btn_paste = w.Button(description='Add list', icon='plus', layout=w.Layout(width='100px'))
        self.btn_clear = w.Button(description='Clear', icon='times', layout=w.Layout(width='80px'))
        self.tgl_list = w.ToggleButton(value=False, description='Show participant list', icon='list',
                                       layout=w.Layout(width='190px'))
        self.lst = w.Select(options=[], rows=10, layout=w.Layout(width='620px', display='none'))
        self.cb_skip = w.Checkbox(value=skip_default, indent=False,
                                  description='Skip already processed participants',
                                  layout=w.Layout(width='420px'))
        self.lbl_summary = w.HTML('<i>Scan first.</i>')
        self.html_warn = w.HTML('')
        self.lbl_paste = w.HTML('')
        self.box_subset = w.VBox([w.HBox([self.cmb_add, self.btn_clear]), w.HBox([self.tags]),
                                  w.HBox([self.txt_paste, self.btn_paste, self.lbl_paste])],
                                 layout=w.Layout(display='none'))
        self.widget = w.VBox([w.HBox([self.dd_mode, self.tgl_list]), self.box_subset, self.lst,
                              self.cb_skip, self.lbl_summary, self.html_warn])
        # The answers when participants were made with other settings (ask_other_settings): shown in the run's log.
        self.btn_redo_other = ui_lib.button('Reprocess them', 'caution')
        self.btn_keep_other = ui_lib.button('Keep them', 'run')
        self._box_other = w.HBox([self.btn_redo_other, self.btn_keep_other])
        self._on_other_answer = None
        # Beside the page's Run button: the question sits under the progress bars, possibly below the window.
        self.run_note = w.HTML('')
        self.btn_redo_other.on_click(lambda _: self._answer_other('reprocess'))
        self.btn_keep_other.on_click(lambda _: self._answer_other('keep'))
        self.dd_mode.observe(self._on_mode, names='value')
        self.tags.observe(self._on_tags, names='value')
        self.cmb_add.observe(self._on_add, names='value')
        self.cb_skip.observe(self._refresh, names='value')
        self.tgl_list.observe(self._on_toggle_list, names='value')
        self.lst.observe(self._on_list_pick, names='value')
        self.btn_paste.on_click(self._on_paste)
        self.btn_clear.on_click(lambda _: setattr(self.tags, 'value', []))

    # -- data -----------------------------------------------------------------
    def set_participants(self, ids, done=(), excluded=None):
        """ids: every participant found by the scan; done: the already-processed ones;
        excluded: {file_id: reason} from the registry (they stay selectable: still processed)."""
        self._ids = [str(i) for i in ids]
        self._done = {id_key(i) for i in done}
        self._excluded = {id_key(k): v for k, v in (excluded or {}).items()}
        self._on_tags()               # drops subset ids that vanished from the new scan
        self._fill_list()
        self._refresh()

    def _status_text(self, fid):
        k = id_key(fid)
        parts = ['processed' if k in self._done else 'not processed']
        if k in self._excluded:
            parts.append(f'EXCLUDED ({self._excluded[k]})')
        return ' - '.join(parts)

    def _fill_list(self):
        width = max([len(i) for i in self._ids] + [8])
        self.lst.options = [(f'{fid.ljust(width)}   {self._status_text(fid)}', fid) for fid in self._ids]
        self.lst.value = None

    def add_under_skip(self, widget):
        """A widget of the tool right under the 'Skip' tick (tool 7's New version)."""
        children = list(self.widget.children)
        children.insert(children.index(self.cb_skip) + 1, widget)
        self.widget.children = children

    # -- selection ------------------------------------------------------------
    @property
    def subset_mode(self):
        return self.dd_mode.value == 'Subset only'

    def chosen_ids(self):
        """The participants chosen, before the skip rule (all of them, or the subset in id order)."""
        if not self.subset_mode:
            return list(self._ids)
        chosen = {id_key(t) for t in self.tags.value}
        return [fid for fid in self._ids if id_key(fid) in chosen]

    def skipped_ids(self):
        """Chosen participants that the skip rule leaves out (already processed, skip ticked)."""
        if not self.cb_skip.value:
            return []
        return [fid for fid in self.chosen_ids() if id_key(fid) in self._done]

    def selected_ids(self):
        """The participants to run now."""
        skipped = {id_key(f) for f in self.skipped_ids()}
        return [fid for fid in self.chosen_ids() if id_key(fid) not in skipped]

    def warning_html(self):
        """The prominent box shown when a subset contains already-processed participants that the
        skip rule will leave out ('' otherwise). Also printed at the top of the run log."""
        if not self.subset_mode:
            return ''
        skipped = self.skipped_ids()
        if not skipped:
            return ''
        names = ', '.join(skipped[:20]) + (f' (+{len(skipped) - 20} more)' if len(skipped) > 20 else '')
        import ui_lib as UI      # the look shared by the Voila pages (imported on use: screen only)
        return UI.box('warn', f'<b>{len(skipped)} of the {len(self.chosen_ids())} selected participant(s) will '
                      f'NOT be processed: already processed</b> ({names}).<br>Untick <b>Skip already processed '
                      'participants</b> to reprocess them.')

    def ask_other_settings(self, differences, n_processed, n_new, on_answer):
        """Skip met participants made with other settings: show the differences and the two answers in the output
        area this is called from (the tool's run log), then return. on_answer('reprocess' or 'keep') runs when a
        button is clicked. The tool calls it before any analysis, and finds the differences again on the answer. It
        shows run_note beside its Run button and empties it at the start of each Run."""
        from IPython.display import HTML, display
        import ui_lib as UI      # the look shared by the Voila pages (imported on use: screen only)
        self.run_note.value = UI.msg('warn', 'Your answer is needed, just below')
        n_redo = len(differences)
        new_ones = f'run the {n_new} new one(s)'
        self.btn_redo_other.description = f'Reprocess {n_redo} participant(s)' + (f' + {new_ones}' if n_new else '')
        self.btn_keep_other.description = f'Keep them, {new_ones}' if n_new else 'Keep them, only rebuild the tables'
        self._on_other_answer = on_answer
        display(HTML(other_settings_html(differences, n_processed)))
        display(self._box_other)

    def _answer_other(self, answer):
        on_answer, self._on_other_answer = self._on_other_answer, None     # one answer per question
        if on_answer is not None:
            on_answer(answer)

    # -- callbacks ----------------------------------------------------------------
    def _on_mode(self, change=None):
        self.box_subset.layout.display = '' if self.subset_mode else 'none'
        self._refresh()

    def _on_toggle_list(self, change):
        self.lst.layout.display = '' if change['new'] else 'none'
        self.tgl_list.description = 'Hide participant list' if change['new'] else 'Show participant list'

    def _on_list_pick(self, change):
        fid = change['new']
        if fid is None:
            return
        if fid not in self.tags.value:
            self.tags.value = list(self.tags.value) + [fid]
        if not self.subset_mode:
            self.dd_mode.value = 'Subset only'
        self.lst.value = None

    def _on_add(self, change=None):
        """An id picked (or typed + Enter) in the add box: into the subset, then the box is emptied."""
        known = {id_key(i): i for i in self._ids}
        fid = known.get(id_key(self.cmb_add.value or ''))
        if fid is None:
            return
        if fid not in self.tags.value:
            self.tags.value = list(self.tags.value) + [fid]
        if not self.subset_mode:
            self.dd_mode.value = 'Subset only'
        self.cmb_add.value = ''

    def _on_tags(self, change=None):
        """Keep the subset clean (known ids only, in their scan spelling, no duplicate), then refresh
        the add box so it no longer offers the participants already selected."""
        if self._cleaning:
            return
        known = {id_key(i): i for i in self._ids}
        clean = []
        for t in self.tags.value:
            fid = known.get(id_key(t))
            if fid is not None and fid not in clean:
                clean.append(fid)
        unknown = [t for t in self.tags.value if id_key(t) not in known]
        if clean != list(self.tags.value):
            self._cleaning = True
            try:
                self.tags.value = clean
            finally:
                self._cleaning = False
        if unknown:
            self.lbl_paste.value = f'<span style="color:{self._ui.ERROR_COLOR}">unknown id(s): {", ".join(unknown)}</span>'
        chosen = {id_key(t) for t in clean}
        self.cmb_add.options = [fid for fid in self._ids if id_key(fid) not in chosen]
        self._refresh()

    def _on_paste(self, _=None):
        known = {id_key(i): i for i in self._ids}
        wanted = parse_id_list(self.txt_paste.value)
        found = [known[id_key(t)] for t in wanted if id_key(t) in known]
        unknown = [t for t in wanted if id_key(t) not in known]
        self.tags.value = list(self.tags.value) + [f for f in found if f not in self.tags.value]
        if found and not self.subset_mode:
            self.dd_mode.value = 'Subset only'
        self.txt_paste.value = ''
        self.lbl_paste.value = (f'<span style="color:{self._ui.ERROR_COLOR}">unknown id(s): {", ".join(unknown)}</span>'
                                if unknown else '')

    def _refresh(self, change=None):
        n = len(self._ids)
        if not n:
            self.lbl_summary.value = '<i>Scan first.</i>'
            self.html_warn.value = ''
            return
        n_done = sum(1 for i in self._ids if id_key(i) in self._done)
        n_excl = sum(1 for i in self._ids if id_key(i) in self._excluded)
        chosen = self.chosen_ids()
        run = self.selected_ids()
        excl_txt = (f', <b>{n_excl}</b> excluded (still computed, flagged in the tables)' if n_excl else '')
        if self.subset_mode:
            head = f'Subset: <b>{len(chosen)}</b> of {n} participants'
        else:
            head = f'<b>{n}</b> participants, <b>{n_done}</b> already processed{excl_txt}'
        self.lbl_summary.value = f'{head} &rarr; <b>{len(run)}</b> will run.'
        self.html_warn.value = self.warning_html()
