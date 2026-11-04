"""Where the files of a study are and what they are called.

Format-agnostic (no EDF or Curry reading, no MNE), so every Voila page and Curry twin can import it next to
ui_lib. Two things live here:

- the **names registry** `config_param/file_names.json`: each tool notes the names it wrote (or read), e.g. the
  suffix of the hypnograms tool 3 remapped, so the later tools follow a name the user changed;
- the **choice of a companion suffix**: a companion file sits beside its recording, named
  `<recording name><suffix>`, and the tools pre-choose the suffix to read among the ones found.
"""
import datetime
import json
import os
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
