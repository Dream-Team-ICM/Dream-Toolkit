"""Where the files of a study are and what they are called.

Format-agnostic (no EDF or Curry reading, no MNE), so every Voila page and Curry twin can import it next to
ui_lib. Four things live here:

- the **names registry** `config_param/file_names.json`: each tool notes the names it wrote (or read), e.g. the
  suffix of the hypnograms tool 3 remapped, so the later tools follow a name the user changed;
- the **choice of a companion suffix**: a companion file sits beside its recording, named
  `<recording name><suffix>`, and the tools pre-choose the suffix to read among the ones found;
- the **inputs of a tool** (tool 2's channel configuration, tool 4's event mapping...): where each is looked for
  in the study folder, and a light read of the file found, for the 'Inputs found' list of the pages;
- the **runs of tool 7** and the decisions made on them, found from the study folder (tools 8, 8bis, 9).
"""
import datetime
import json
import os
import re
from pathlib import Path

NAMES_FILE = 'file_names.json'

# Hypnogram names. New studies get the short defaults; a study already processed keeps its names (the former
# defaults), recognised everywhere.
HYPNO_REMAPPED_DEFAULT = '_hypno_remapped.txt'          # tool 3's output
HYPNO_REMAPPED_FORMER = '_Hypnogram_remapped.txt'
AUTO_ALGOS = ('gssc', 'yasa', 'sleepfm')                # tool 3bis's models


def hypno_auto_default(algo):
    """Tool 3bis's default output suffix for one model, e.g. '_hypno_gssc.txt'."""
    return f'_hypno_{algo}.txt'


def hypno_auto_former(algo):
    """The former default, e.g. '_hypnogram_gssc.txt'."""
    return f'_hypnogram_{algo}.txt'


RESCORED_END = '_rescored.txt'                          # tool 0 writes <hypnogram name>_rescored.txt
# .txt companions that are not hypnograms: scored-event, summary and study-log exports (Profusion / Curry)
NOT_HYPNO_TOKENS = ('event', 'summary', 'studylog')


def _norm(s):
    return os.path.normcase(str(s))


# ---------------------------------------------------------------------------------------------------------
# Names registry: config_param/file_names.json
# ---------------------------------------------------------------------------------------------------------
def names_path(study):
    return Path(study) / 'config_param' / NAMES_FILE


def load_names(study):
    """The names registry of the study, as (dict, warning). An absent or empty file gives ({}, None); an unreadable
    one gives ({}, warning) and the tools fall back on their detection rules."""
    path = names_path(study)
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return {}, None
        names = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(names, dict):
            raise ValueError('not a JSON object')
        return names, None
    except Exception as e:
        return {}, f'config_param/{NAMES_FILE} unreadable ({e}): the names are detected instead'


def recorded(names, key):
    """The value recorded under `key` (a suffix, a path or a list), or None."""
    entry = (names or {}).get(key)
    return entry.get('value') if isinstance(entry, dict) else None


def write_names(study, tool, values):
    """Record `values` ({key: value}) for `tool` in the names registry. Only these keys are rewritten; a key whose
    value is unchanged keeps its date, and the file is left untouched when nothing changes. An unreadable registry
    is never overwritten. Returns None, or a warning to show."""
    path = names_path(study)
    names, warn = load_names(study)
    if warn:
        return f'{warn}; it was not updated'
    changed = False
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for key, value in values.items():
        if recorded(names, key) == value:
            continue
        names[key] = {'value': value, 'tool': str(tool), 'saved_at': now}
        changed = True
    if not changed:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + '.tmp')
        tmp.write_text(json.dumps(names, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
        os.replace(tmp, path)
    except Exception as e:
        return f'could not write config_param/{NAMES_FILE}: {e}'
    return None


# ---------------------------------------------------------------------------------------------------------
# Companion suffixes
# ---------------------------------------------------------------------------------------------------------
def companion_suffix_owners(recordings, files):
    """{suffix: [names of the recordings having a file <recording name><suffix>]} over `files`.

    A file counts for the recording with the LONGEST name it starts with: 10_hypno.txt belongs to recording 10,
    never to recording 1 (which would add a false suffix '0_hypno.txt'). The folder is not compared: tools 5, 6
    and 7 find a hypnogram anywhere in the study by its file name."""
    stems = [Path(r).stem for r in recordings]
    owners_by_suffix = {}
    for f in files:
        name = Path(f).name
        owners = [s for s in stems if _norm(name).startswith(_norm(s)) and len(name) > len(s)]
        if not owners:
            continue
        stem = max(owners, key=len)
        owners_by_suffix.setdefault(name[len(stem):], []).append(stem)
    return owners_by_suffix


def companion_suffix_counts(recordings, files):
    """{suffix: number of files named <recording name><suffix>} (see companion_suffix_owners)."""
    return {s: len(r) for s, r in companion_suffix_owners(recordings, files).items()}


def toolkit_hypno_suffixes(names, own_outputs=()):
    """The hypnogram suffixes the toolkit writes: the names recorded by tools 3bis and 0, the default names of
    tool 3bis (current and former) and the tool's own output suffixes. (Tool 0's '..._rescored.txt' is recognised
    by its ending, see is_toolkit_hypno.)"""
    out = [hypno_auto_default(a) for a in AUTO_ALGOS] + [hypno_auto_former(a) for a in AUTO_ALGOS]
    out += [recorded(names, f'hypnogram_{a}') for a in AUTO_ALGOS]
    out += list(recorded(names, 'hypnogram_rescored') or [])
    out += list(own_outputs)
    return [s for s in out if s]


def is_toolkit_hypno(suffix, names, own_outputs=()):
    """True for a hypnogram written by the toolkit (tool 3bis, tool 0's rescoring, or the tool itself)."""
    return (_norm(suffix).endswith(RESCORED_END)
            or _norm(suffix) in {_norm(s) for s in toolkit_hypno_suffixes(names, own_outputs)})


def pick_hypno_suffix(counts, intent, names=None, own_outputs=()):
    """Pre-choice of the hypnogram suffix among `counts` ({suffix: number of recordings}).

    intent 'expert': the expert hypnogram remapped by tool 3 (tools 0, 3bis, 5, 6, 7); 'scored': the raw export of
    the scoring software, before tool 3 (tool 3). Order:
      1. the suffix recorded for this intent in config_param/file_names.json, when files with it exist;
      2. else the detection rule, after setting aside the event / summary / study-log exports, the hypnograms the
         toolkit writes, the tool's own outputs and, for 'scored', the remapped hypnograms (for 'expert', the raw
         export recorded by tool 3). 'expert' = the longest among the suffixes found for >= 50 % of the most
         frequent one (the more specific, remapped version); 'scored' = the most frequent, the shortest on ties;
      3. nothing left: no pre-choice (the user picks one, e.g. an automatic hypnogram, knowingly).
    Returns (suffix, count, reason), reason = 'recorded', 'detected' or 'none' (suffix None, count 0)."""
    names = names or {}
    key = 'hypnogram_remapped' if intent == 'expert' else 'hypnogram_scored'
    rec = recorded(names, key)
    if rec:
        for s, c in counts.items():
            if _norm(s) == _norm(rec) and c > 0:
                return s, c, 'recorded'
    if intent == 'expert':
        set_aside = [recorded(names, 'hypnogram_scored')]
    else:
        set_aside = [recorded(names, 'hypnogram_remapped'), HYPNO_REMAPPED_DEFAULT, HYPNO_REMAPPED_FORMER]
    set_aside = {_norm(s) for s in set_aside if s}
    sel = {s: c for s, c in counts.items()
           if not any(tok in s.lower() for tok in NOT_HYPNO_TOKENS)
           and not is_toolkit_hypno(s, names, own_outputs)
           and _norm(s) not in set_aside}
    if not sel:
        return None, 0, 'none'
    if intent == 'expert':
        max_count = max(sel.values())
        candidates = {s: c for s, c in sel.items() if c >= max_count * 0.5}
        suffix, count = max(candidates.items(), key=lambda x: (len(x[0]), x[1]))
    else:
        suffix, count = max(sel.items(), key=lambda x: (x[1], -len(x[0])))
    return suffix, count, 'detected'


def choice_notes(reason, intent, names_warn=None):
    """The lines a tool shows under its hypnogram suffix after pick_hypno_suffix, as [(kind, text)] for UI.msg
    (kind = muted / warn)."""
    notes = []
    if reason == 'recorded':
        notes.append(('muted', f'Pre-chosen: the name recorded in config_param/{NAMES_FILE}.'))
    elif reason == 'none':
        what = 'expert hypnogram' if intent == 'expert' else 'raw hypnogram export'
        notes.append(('warn', f'No {what} among the files found (only exports or hypnograms written by the '
                              'toolkit): type the suffix to use.'))
    if names_warn:
        notes.append(('warn', names_warn))
    return notes


def select_event_suffix(suffix_counts):
    """Tool 7's event TXT / CSV rule: the most frequent suffix, the shortest on ties. (None, 0) if none."""
    if not suffix_counts:
        return None, 0
    return min(suffix_counts.items(), key=lambda x: (-x[1], len(x[0])))


def output_suffix(names, key, counts, new_default, former_default):
    """The output suffix a tool proposes, so that a study keeps its names: the one recorded under `key`, else the
    former default when the study already has files with it, else the new default."""
    rec = recorded(names, key)
    if rec:
        return rec
    if any(_norm(s) == _norm(former_default) and c > 0 for s, c in counts.items()):
        return former_default
    return new_default


# ---------------------------------------------------------------------------------------------------------
# The inputs of a tool: where each is looked for in the study folder, and a light read of the file found
# (the 'Inputs found' list of the Voila pages, ui_lib.input_row)
# ---------------------------------------------------------------------------------------------------------
# Tool 2's channel configuration: new studies get the new name, a study already processed keeps the former one
CHANNEL_CONFIG_DEFAULT = 'channel_config.json'
CHANNEL_CONFIG_FORMER = 'remap_reref_persubject.json'
EVENT_REMAP_DEFAULT = 'event_remap.json'

# key -> the places looked for, relative to the study folder, after the path recorded under the same key in the
# names registry
INPUT_PLACES = {
    'channel_config': (f'config_param/{CHANNEL_CONFIG_DEFAULT}', f'config_param/{CHANNEL_CONFIG_FORMER}'),  # tool 2
    'event_remap': (f'config_param/{EVENT_REMAP_DEFAULT}',),                  # tool 4
    'quality_summary': ('reports_quality_overview/quality_summary.tsv',),     # tool 6
    'sleep_metrics_table': ('reports_features_macrostructure/global_sleep_metrics.tsv',),   # tool 5
    'participant_info': ('participants.tsv', 'participants.csv', 'participants.xlsx'),       # the user's table
}


def find_input(study, names, key):
    """The file of one input of a tool, as (path, how, folder):
      - the path recorded under `key` in the names registry, when the file exists (how = 'recorded by tool N');
      - else the first of INPUT_PLACES[key] found in the study (how = 'default location');
      - else (None, '', folder).
    `folder` is where 'Choose another…' opens: the file's folder, else the first place's folder that exists, else
    the study folder. A recorded path is relative to the study folder, or absolute when the file is elsewhere."""
    study = Path(study)
    rec = recorded(names, key)
    if rec:
        path = Path(rec) if Path(rec).is_absolute() else study / rec
        if path.is_file():
            return path, f"recorded by tool {names[key].get('tool', '?')}", path.parent
    for place in INPUT_PLACES[key]:
        if (study / place).is_file():
            return study / place, 'default location', (study / place).parent
    folders = [(study / place).parent for place in INPUT_PLACES[key]]
    return None, '', next((f for f in folders if f.is_dir()), study)


def path_to_record(study, path):
    """How a path is written in the names registry: relative to the study folder (with '/') when inside it, else
    absolute. find_input reads both."""
    try:
        return Path(path).resolve().relative_to(Path(study).resolve()).as_posix()
    except ValueError:
        return str(Path(path).resolve())


def config_file_name(study, names, key, default, former=None):
    """The name of the file a tool writes in config_param/ (tool 2's channel configuration, tool 4's event mapping),
    so that a study keeps its name: the name recorded under `key` when it is in config_param/, else `former` when
    the study already has that file, else `default`."""
    rec = recorded(names, key)
    if rec and Path(rec).parent.as_posix() == 'config_param':
        return Path(rec).name
    config_param = Path(study) / 'config_param'
    if former and (config_param / former).is_file() and not (config_param / default).is_file():
        return former
    return default


def check_channel_config(path):
    """A light read of tool 2's channel configuration: ('ok', '11 participants') or ('warn', what is wrong)."""
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(cfg, dict) or not cfg:
        return 'warn', 'not a channel configuration: empty, or not a JSON object'
    n_bad = sum(1 for v in cfg.values() if not (isinstance(v, dict) and 'remap' in v))
    if n_bad:
        return 'warn', f"not a channel configuration: {n_bad} of {len(cfg)} entries have no 'remap' key"
    return 'ok', f'{len(cfg)} participants'


def check_event_remap(path):
    """A light read of tool 4's event mapping: ('ok', '14 labels → 5 event types') or ('warn', what is wrong)."""
    text = Path(path).read_text(encoding='utf-8')
    try:
        remap = json.loads(text)
    except json.JSONDecodeError:   # the lenient read of the tools: one trailing comma repaired (load_event_remap)
        remap = json.loads(re.sub(r',(\s*[}\]])', r'\1', text))
    if not isinstance(remap, dict) or not remap:
        return 'warn', 'not an event mapping: empty, or not a JSON object'
    if not all(v is None or isinstance(v, str) for v in remap.values()):
        return 'warn', 'not an event mapping: the values must be event types (text)'
    n_types = len({v for v in remap.values() if v})
    return 'ok', f'{len(remap)} labels → {n_types} event types'


def _tsv_ids(path):
    """The distinct values of the 'file_id' column of a TSV table, or None when it has no such column."""
    with open(path, encoding='utf-8') as f:
        header = f.readline().rstrip('\r\n').split('\t')
        if 'file_id' not in header:
            return None
        col = header.index('file_id')
        return {line.rstrip('\r\n').split('\t')[col] for line in f if line.strip()}


def check_quality_summary(path):
    """A light read of tool 6's quality summary: ('ok', '11 recordings') or ('warn', what is wrong)."""
    ids = _tsv_ids(path)
    if ids is None:
        return 'warn', "not a quality summary: no 'file_id' column"
    return 'ok', f'{len(ids)} recordings'


def check_sleep_metrics_table(path):
    """A light read of tool 5's global_sleep_metrics.tsv: ('ok', '11 participants') or ('warn', what is wrong)."""
    ids = _tsv_ids(path)
    if ids is None:
        return 'warn', "not a tool-5 sleep metrics table: no 'file_id' column"
    return 'ok', f'{len(ids)} participants'


# ---------------------------------------------------------------------------------------------------------
# The runs of tool 7 and the decisions made on them (tools 8, 8bis, 9)
# ---------------------------------------------------------------------------------------------------------
NO_DECISION = ''                 # the value of the 'none' option of a decision menu


def epoch_folders(study, names=None):
    """The folders of preprocessed epochs of the study, as [(folder, number of participants)]: the
    derivatives/preproc_epo* holding *_all-epo.fif, in the study folder then in the output folder tool 7 noted.
    Tool 7 always writes preproc_epo: there are several only when a folder was renamed by hand to keep a former run
    (preproc_epo_v2), or with tool 7's output folder."""
    study = Path(study)
    roots = [study]
    rec = recorded(names, 'preproc_output')
    if rec:
        other = Path(rec) if Path(rec).is_absolute() else study / rec
        if _norm(other.resolve()) != _norm(study.resolve()):
            roots.append(other)
    found = []
    for root in roots:
        derivatives = root / 'derivatives'
        if not derivatives.is_dir():
            continue
        for folder in sorted(derivatives.iterdir()):
            if folder.is_dir() and folder.name.lower().startswith('preproc_epo'):
                n = sum(1 for _ in folder.rglob('*_all-epo.fif'))
                if n:
                    found.append((folder, n))
    return found


def epoch_folder_options(study, folders):
    """The options of a menu of epoch_folders: [(label, folder as text)]."""
    return [(f'{path_to_record(study, folder)}  ({n} participants)', str(folder)) for folder, n in folders]


def paired_reports_folder(epoch_folder):
    """The reports folder of a run of tool 7, by its name beside derivatives/: preproc_epo_v2 -> reports_preproc_v2.
    None when there is none."""
    epoch_folder = Path(epoch_folder)
    if not epoch_folder.name.lower().startswith('preproc_epo'):
        return None
    candidate = epoch_folder.parent.parent / ('reports_preproc' + epoch_folder.name[len('preproc_epo'):])
    return candidate if candidate.is_dir() else None


def decision_folders(epoch_folder):
    """The decision folders beside an epochs folder (rejection_auto* of tool 8bis, rejection_manual* of tool 8), the
    most recent first, as [(folder, date of its latest *_epoch_decision.tsv, number of participants)]."""
    found = []
    for folder in sorted(Path(epoch_folder).parent.iterdir()):
        if folder.is_dir() and folder.name.lower().startswith(('rejection_auto', 'rejection_manual')):
            tables = list(folder.rglob('*_epoch_decision.tsv'))
            if tables:
                found.append((folder, max(t.stat().st_mtime for t in tables), len(tables)))
    return sorted(found, key=lambda x: -x[1])


def decision_folder_options(study, folders):
    """The options of a menu of decision_folders, then 'none': [(label, folder as text or NO_DECISION)]."""
    options = []
    for folder, mtime, n in folders:
        tool = 'tool 8bis' if folder.name.lower().startswith('rejection_auto') else 'tool 8'
        saved = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
        options.append((f'{path_to_record(study, folder)}  ({tool}, {n} participants, {saved})', str(folder)))
    return options + [('none: no decision', NO_DECISION)]


def _count_files(folder, pattern):
    return sum(1 for _ in Path(folder).rglob(pattern))


def check_epoch_folder(path):
    """A light read of a folder of preprocessed epochs: ('ok', '11 participants') or ('warn', what is wrong)."""
    n = _count_files(path, '*_all-epo.fif')
    return ('ok', f'{n} participants') if n else ('warn', 'no tool-7 *_all-epo.fif in this folder')


def check_reports_folder(path):
    """A light read of tool 7's reports folder: ('ok', '11 participants') or ('warn', what is wrong)."""
    n = _count_files(path, '*_epoch_channel_rejection.tsv')
    return ('ok', f'{n} participants') if n else ('warn', 'no tool-7 *_epoch_channel_rejection.tsv in this folder')


def check_decision_folder(path):
    """A light read of a decision folder: ('ok', '11 participants') or ('warn', what is wrong)."""
    n = _count_files(path, '*_epoch_decision.tsv')
    return ('ok', f'{n} participants') if n else ('warn', 'no *_epoch_decision.tsv in this folder')
