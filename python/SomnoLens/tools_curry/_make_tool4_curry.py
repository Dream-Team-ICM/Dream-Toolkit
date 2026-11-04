"""
_make_tool4_curry.py — Generate 4_remap_events_curry_voila.ipynb from the EDF original.

Adaptations from the EDF version:
- File discovery: .edf -> .cdt (bare suffix)
- Event source: the EDF version reads three companions (TXT-first, then CSV, then *.edf.XML);
  the Curry version has a single source, the Curry French text export
  (*_ScoredEvents_Export.txt), parsed via curry_io.load_events_curry using the .cdt.dpo
  recording-start datetime. load_events() keeps a (path, suffix) -> (events_list, source)
  signature so the scan/harmonize/verify logic is untouched.
- Section 1bis (text/CSV vs XML consistency) is REMOVED: Curry has a single event source, so the
  cross-source check is meaningless. That drops the EDF's XML/CSV/TXT parser helpers, the EDF
  header reader and _events_multiset.
- Curry has a single suffix field (its csv_suffix widget, relabelled/defaulted to the .txt export);
  the EDF's extra txt_suffix widget + layout entry are removed.
- Curry shared-module imports (curry_header, curry_io); the now-unused xml.etree import is dropped.

The French-label suggestions (FRENCH_EVENT_RULES / suggest_canonical) and the canonical vocabulary
are format-agnostic and pass through unchanged — Curry benefits from them directly.

Run from SomnoLens root:
    & "$env:LOCALAPPDATA\\miniforge3\\envs\\somnolens\\python.exe" tools_curry/_make_tool4_curry.py
"""

import json, os, sys, ast

SRC = "tools/4_remap_events_edf_voila.ipynb"
DST = "tools_curry/4_remap_events_curry_voila.ipynb"

with open(SRC, encoding="utf-8") as f:
    nb = json.load(f)

errors = []


def get_src(cell):
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def set_src(cell, text):
    orig = cell["source"]
    if isinstance(orig, list):
        cell["source"] = text.splitlines(keepends=True)
    else:
        cell["source"] = text


def replace_in_cell(cell, old, new, label):
    src = get_src(cell)
    if old not in src:
        errors.append(label)
        print(f"  X NOT FOUND: {label}")
        return False
    if src.count(old) > 1:
        print(f"  ! MULTIPLE matches: {label} — replacing all")
    set_src(cell, src.replace(old, new))
    print(f"  ok {label}")
    return True


def remove_between(cell, start_marker, end_marker, label):
    """Delete text from start_marker up to (but not including) end_marker."""
    src = get_src(cell)
    i = src.find(start_marker)
    j = src.find(end_marker)
    if i == -1 or j == -1 or j <= i:
        errors.append(label)
        print(f"  X NOT FOUND / bad range: {label}")
        return False
    set_src(cell, src[:i] + src[j:])
    print(f"  ok {label}")
    return True


md0  = nb["cells"][0]
code = nb["cells"][1]

# ===========================================================================
print("=== Markdown cell 0 ===")
NEW_MD = (
    "# Tool 4: Harmonize the scored-event labels (Curry 9 .cdt)\n"
    "\n"
    "This tool lets you **visualize** the scored-event configurations present in a Curry 9 database (events\n"
    "annotated during sleep scoring and exported by Curry as a text file: arousals, apnea, hypopnea, limb movements,\n"
    "desaturations…) and **harmonize** their raw labels to a single canonical vocabulary.\n"
    "\n"
    "- **Inputs**: a study folder holding Curry 9 `.cdt` recordings, each with its Curry event text export\n"
    "  `*_ScoredEvents_Export.txt` next to it.\n"
    "- **Outputs**: `config_param/event_remap.json` in the study folder, a flat Python dict\n"
    "  `{raw_label: canonical_label}` (`null` for the labels you choose to ignore) that the later tools use (epoch\n"
    "  flagging in `7_preprocessing_curry`).\n"
    "- **Sections**: 1. Study folder (scan the events), 2. Event configurations found, 3. Harmonize the labels,\n"
    "  4. Preview & save the JSON, 5. Verify.\n"
    "\n"
    "<details><summary>Details</summary>\n"
    "\n"
    "The raw labels are the export strings (often in French, e.g. `Micro-éveil 1 ARO SPONT`). The canonical\n"
    "suggestions recognize the common French Compumedics/Curry labels (arousals, apnea, hypopnea, desaturation,\n"
    "snoring…), so most are pre-filled. Unusual labels still need a manual choice.\n"
    "\n"
    "</details>"
)
set_src(md0, NEW_MD)
print("  ok markdown rewritten")

# ===========================================================================
print("\n=== Cell 1: imports ===")
# Drop the now-unused XML parser import (Curry has no .edf.XML companion)
replace_in_cell(code, "    import xml.etree.ElementTree as ET\n", "", "remove ET import")
# Add curry shared-module imports inside the try block
replace_in_cell(code,
    "    from IPython.display import display, HTML, clear_output\n"
    "except ImportError as e:",
    "    from IPython.display import display, HTML, clear_output\n"
    "    import sys as _sys\n"
    "    # curry shared modules — found whether Voila is launched from the repo root or tools_curry/\n"
    "    _here = os.getcwd()\n"
    "    for _cand in (_here, os.path.join(_here, 'tools_curry'),\n"
    "                  os.path.join(os.path.dirname(_here), 'tools_curry')):\n"
    "        if os.path.isfile(os.path.join(_cand, 'curry_header.py')):\n"
    "            _cand = os.path.abspath(_cand)\n"
    "            if _cand not in _sys.path:\n"
    "                _sys.path.insert(0, _cand)\n"
    "            break\n"
    "    from curry_header import read_curry_header\n"
    "    from curry_io import load_events_curry, rec_start_from_header\n"
    "except ImportError as e:",
    "curry imports")

# ===========================================================================
print("\n=== Cell 1: event loader ===")
# The EDF event-loader block (EDF-header reader + 3-source companion loaders) -> single Curry loader.
OLD_EVENTS = (
    '# ---- EDF recording start (only needed to convert text-export clock times to seconds) ----\n'
    'def read_edf_start_datetime(edf_path):\n'
    '    """Read the EDF recording-start datetime from the fixed header (offset 168 = date\n'
    '    \'dd.mm.yy\', 176 = time \'hh.mm.ss\'), applying the EDF 2-digit-year clipping\n'
    '    (00-84 -> 20xx, 85-99 -> 19xx). Header-only read; returns a datetime or None on failure."""\n'
    '    try:\n'
    '        with open(edf_path, "rb") as f:\n'
    '            f.seek(168)\n'
    '            date_str = f.read(8).decode("ascii", "replace").strip()   # dd.mm.yy\n'
    '            time_str = f.read(8).decode("ascii", "replace").strip()   # hh.mm.ss\n'
    '        dd, mm, yy = (int(x) for x in date_str.split("."))\n'
    '        hh, mi, ss = (int(x) for x in time_str.split("."))\n'
    '        year = 2000 + yy if yy <= 84 else 1900 + yy\n'
    '        return datetime.datetime(year, mm, dd, hh, mi, ss)\n'
    '    except Exception:\n'
    '        return None\n'
    '\n\n'
    '# ---- event companion loading: TXT first, then CSV, then XML (<ScoredEvents>) fallback ----\n'
    'def event_companion_paths(edf_path, txt_suffix="_ScoredEvents_Export.txt",\n'
    '                          csv_suffix="_event_xml.csv"):\n'
    '    """Return (txt_path_or_None, csv_path_or_None, xml_path_or_None) for an EDF stem.\n'
    '    txt_suffix / csv_suffix are the configurable Compumedics event-export suffixes."""\n'
    '    edf_path = Path(edf_path)\n'
    '    txt = edf_path.with_name(f"{edf_path.stem}{txt_suffix}")\n'
    '    txt = txt if txt.exists() else None\n'
    '    csv = edf_path.with_name(f"{edf_path.stem}{csv_suffix}")\n'
    '    csv = csv if csv.exists() else None\n'
    '    xml = None\n'
    '    for cand in (f"{edf_path.name}.XML", f"{edf_path.name}.xml"):\n'
    '        p = edf_path.with_name(cand)\n'
    '        if p.exists():\n'
    '            xml = p\n'
    '            break\n'
    '    return txt, csv, xml\n'
    '\n\n'
    'def load_events_from_csv(csv_path):\n'
    '    """Parse a Compumedics *_event_xml.csv -> list of (name, start, duration)."""\n'
    '    df = pd.read_csv(csv_path)\n'
    '    for col in ("Name", "Start", "Duration"):\n'
    '        if col not in df.columns:\n'
    '            raise ValueError(f"missing column \'{col}\' in {Path(csv_path).name}")\n'
    '    events = []\n'
    '    for _, row in df.iterrows():\n'
    '        events.append((str(row["Name"]).strip(), float(row["Start"]), float(row["Duration"])))\n'
    '    return events\n'
    '\n\n'
    'def load_events_from_xml(xml_path):\n'
    '    """Parse the <ScoredEvents> of a Profusion CMPStudyConfig .edf.XML\n'
    '    -> list of (name, start, duration). <Input> is ignored (absent from the CSV)."""\n'
    '    root = ET.parse(xml_path).getroot()\n'
    '    events = []\n'
    '    for se in root.iter("ScoredEvent"):\n'
    '        name_el = se.find("Name")\n'
    '        if name_el is None or name_el.text is None:\n'
    '            continue\n'
    '        start_el = se.find("Start")\n'
    '        dur_el = se.find("Duration")\n'
    '        start = float(start_el.text) if (start_el is not None and start_el.text) else float("nan")\n'
    '        dur = float(dur_el.text) if (dur_el is not None and dur_el.text) else float("nan")\n'
    '        events.append((name_el.text.strip(), start, dur))\n'
    '    return events\n'
    '\n\n'
    '_TXT_DUR_RE = re.compile(r"^(\\d+):(\\d+(?:\\.\\d+)?)$")   # "M:SS" or "M:SS.s"\n'
    '\n\n'
    'def load_events_from_txt(txt_path, rec_start=None):\n'
    '    """Parse a Compumedics/Curry French text export (*_ScoredEvents_Export.txt)\n'
    '    -> list of (name, start, duration). Comma-separated, no header, columns:\n'
    '        HH:MM:SS , epoch# , stage_FR , event_label_FR , M:SS[.s] , - , - , position\n'
    '    Encoding varies (UTF-16 with BOM on some exports, UTF-8/ANSI on others) -> the BOM is\n'
    '    sniffed. Clock times are converted to seconds-from-recording-start (with midnight\n'
    '    rollover) when rec_start is given; without it (harmonization only needs the names)\n'
    '    Start and Duration are returned as NaN, so no EDF-header read is required for the scan."""\n'
    '    raw = open(txt_path, "rb").read()\n'
    '    if raw[:2] in (b"\\xff\\xfe", b"\\xfe\\xff"):\n'
    '        text = raw.decode("utf-16")\n'
    '    else:\n'
    '        text = raw.decode("utf-8", errors="replace")\n'
    '    rec_date = rec_start.date() if rec_start is not None else None\n'
    '    events = []\n'
    '    for line in text.splitlines():\n'
    '        line = line.strip()\n'
    '        if not line:\n'
    '            continue\n'
    '        parts = [p.strip() for p in line.split(",")]\n'
    '        if len(parts) < 5:\n'
    '            continue\n'
    '        name = parts[3]\n'
    '        if rec_start is None:\n'
    '            events.append((name, float("nan"), float("nan")))\n'
    '            continue\n'
    '        # clock time -> seconds from recording start (hours may be single-digit)\n'
    '        try:\n'
    '            hh, mm, ss = parts[0].split(":")\n'
    '            clock_t = datetime.time(int(hh), int(mm), int(ss))\n'
    '        except (ValueError, TypeError):\n'
    '            events.append((name, float("nan"), float("nan")))\n'
    '            continue\n'
    '        event_dt = datetime.datetime.combine(rec_date, clock_t)\n'
    '        # events recorded after midnight fall on the next calendar day\n'
    '        if (rec_start - event_dt).total_seconds() > 3600:\n'
    '            event_dt += datetime.timedelta(days=1)\n'
    '        start_sec = (event_dt - rec_start).total_seconds()\n'
    '        m = _TXT_DUR_RE.match(parts[4])\n'
    '        dur_sec = int(m.group(1)) * 60 + float(m.group(2)) if m else 0.0\n'
    '        events.append((name, start_sec, dur_sec))\n'
    '    return events\n'
    '\n\n'
    'def load_events(edf_path, txt_suffix="_ScoredEvents_Export.txt", csv_suffix="_event_xml.csv"):\n'
    '    """TXT-first, then CSV, then XML-fallback event loader.\n'
    '    Returns (events_list, source) with source in {\'txt\', \'csv\', \'xml\'} or (None, None) when\n'
    '    no companion is usable. events_list = list of (name, start, duration). For the TXT source\n'
    '    only the names are needed here (harmonization), so Start/Duration are left NaN (no EDF\n'
    '    header read); the 1bis check reads them with the recording-start datetime."""\n'
    '    txt, csv, xml = event_companion_paths(edf_path, txt_suffix, csv_suffix)\n'
    '    if txt is not None:\n'
    '        try:\n'
    '            return load_events_from_txt(txt), "txt"\n'
    '        except Exception:\n'
    '            pass  # fall through to the CSV\n'
    '    if csv is not None:\n'
    '        try:\n'
    '            return load_events_from_csv(csv), "csv"\n'
    '        except Exception:\n'
    '            pass  # fall through to the XML\n'
    '    if xml is not None:\n'
    '        try:\n'
    '            return load_events_from_xml(xml), "xml"\n'
    '        except Exception:\n'
    '            pass\n'
    '    return None, None\n'
)
NEW_EVENTS = (
    '# ---- event loading: Curry French text export (*_ScoredEvents_Export.txt) ----\n'
    'def load_events(cdt_path, event_suffix="_ScoredEvents_Export.txt"):\n'
    '    """Load Curry scored events next to the .cdt file (French text export).\n'
    '    Returns (events_list, \'txt\') with events_list = list of (name, start, duration) in\n'
    '    seconds, or (None, None) when the export is absent/unreadable. The .cdt.dpo header gives\n'
    '    the recording-start datetime used to convert the export clock times (load_events_curry)."""\n'
    '    cdt_path = Path(cdt_path)\n'
    '    ev_path = cdt_path.with_name(f"{cdt_path.stem}{event_suffix}")\n'
    '    if not ev_path.exists():\n'
    '        return None, None\n'
    '    try:\n'
    '        hdr = read_curry_header(str(cdt_path))\n'
    '        rec_start = rec_start_from_header(hdr)\n'
    '        if rec_start is None:\n'
    '            return None, None\n'
    '        df = load_events_curry(str(ev_path), rec_start)\n'
    '        if df is None or df.empty:\n'
    '            return None, None\n'
    '        events = [(str(n).strip(), float(s), float(d))\n'
    '                  for n, s, d in zip(df["Name"], df["Start"], df["Duration"])]\n'
    '        return events, "txt"\n'
    '    except Exception:\n'
    '        return None, None\n'
)
replace_in_cell(code, OLD_EVENTS, NEW_EVENTS, "event loader")

# Remove _canon_events + _match_events (only used by the text/CSV-vs-XML check)
remove_between(code,
    "def _canon_events(events):",
    "# ---- shared state filled by the scan ----",
    "remove _canon_events + _match_events")

# ===========================================================================
print("\n=== Cell 1: Section 1 banner + widgets ===")
# section1 banner (3-source wording -> Curry single text export)
replace_in_cell(code,
    'section1 = widgets.HTML(UI.section_html(1, \'Study folder\', """\n'
    '<p>Pick the folder of your .edf database. Each .edf is expected to have a Compumedics/Profusion\n'
    'event companion next to it. Three sources are supported, in priority order:\n'
    '<br>&#x2022; the <b>text export</b> <code>*_ScoredEvents_Export.txt</code> (read first; suffix in\n'
    'the <b>TXT suffix</b> field: default <code>_ScoredEvents_Export.txt</code>);\n'
    '<br>&#x2022; then the <b>CSV</b> <code>*_event_xml.csv</code> (suffix in the <b>CSV suffix</b> field);\n'
    '<br>&#x2022; then, as a fallback, the <code>&lt;ScoredEvents&gt;</code> of the <code>*.edf.XML</code>.\n'
    '<br>&#x2022; Selecting the folder auto-detects both suffixes, finds the existing mapping (<b>Inputs found</b>;\n'
    '<i>Choose another…</i> starts from another file, e.g. another cohort\'s) and refreshes the info lines below. The\n'
    '<b>Advanced</b> box holds the name of the file Section 4 saves.\n'
    '<br>&#x2022; Click <b>Scan</b> to read the events and list the configurations.\n'
    '<br>&#x2022; "Skip labels already mapped" hides labels already present in an existing\n'
    '<code>event_remap.json</code> (incremental harmonization when you add a new cohort).</p>\n'
    '"""))',
    'section1 = widgets.HTML(UI.section_html(1, \'Study folder\', """\n'
    '<p>Pick the folder of your Curry (.cdt) database. Each .cdt is expected to have a Curry event\n'
    'text export next to it, whose suffix is set in the <b>Event export suffix</b> field below\n'
    '(default <code>_ScoredEvents_Export.txt</code>).\n'
    '<br>&#x2022; Selecting the folder auto-detects the suffix, finds the existing mapping (<b>Inputs found</b>;\n'
    '<i>Choose another…</i> starts from another file, e.g. another cohort\'s) and refreshes the info line below. The\n'
    '<b>Advanced</b> box holds the name of the file Section 4 saves.\n'
    '<br>&#x2022; Click <b>Scan</b> to read the events and list the configurations.\n'
    '<br>&#x2022; "Skip labels already mapped" hides labels already present in an existing\n'
    '<code>event_remap.json</code> (incremental harmonization when you add a new cohort).</p>\n'
    '"""))',
    "section1 banner")

# Remove section1bis banner
replace_in_cell(code,
    'section1bis = widgets.HTML(UI.subsection_html(None, \'Optional: check text/CSV vs XML consistency\', """\n'
    '<p>For every file having <b>both</b> a primary source (the <code>.txt</code> text export if present,\n'
    'else the <code>*_event_xml.csv</code>) <b>and</b> the <code>&lt;ScoredEvents&gt;</code> of the\n'
    '<code>*.edf.XML</code>, checks that the two describe the same events. Labels are normalized to the\n'
    '<b>canonical vocabulary</b> (so English XML and French <code>.txt</code> compare equal) and events are\n'
    'matched by <b>type + start time within ±(Match&nbsp;tolerance) seconds</b> (default 1&nbsp;s: the\n'
    '<code>.txt</code> truncates clock times to the second and the two exports can round a start\n'
    'differently; set 0 for strict same-second matching). Events paired within the tolerance but carrying\n'
    '<b>different labels</b> are listed in <code>cooccur_label_pairs</code> as candidate same-events whose\n'
    'names are not yet harmonized (e.g. a cross-language pair): <b>inspect those pairs to decide whether\n'
    'they are truly one event or two distinct events that merely fall within the tolerance</b>. Events with\n'
    'no counterpart show up as only-in-one-source (e.g. an export that omits snoring). Writes\n'
    '<code>config_param/event_source_mismatch.tsv</code>. Opt-in because it reads both files per EDF.</p>\n'
    '"""))\n\n',
    "",
    "remove section1bis banner")

# Remove the EDF's extra txt_suffix widget (Curry keeps a single suffix field = csv_suffix)
replace_in_cell(code,
    'txt_suffix = widgets.Text(value="_ScoredEvents_Export.txt", description="TXT suffix:",\n'
    '                          style=UI.LABEL_STYLE,\n'
    '                          layout=widgets.Layout(width=UI.FIELD_WIDTH))\n'
    'txt_suffix_info = widgets.HTML(value="")\n',
    "",
    "remove txt_suffix widget")

# csv_suffix widget default + label (Curry's single field carries the .txt export)
replace_in_cell(code,
    'csv_suffix = widgets.Text(value="_event_xml.csv", description="CSV suffix:",',
    'csv_suffix = widgets.Text(value="_ScoredEvents_Export.txt", description="Event export suffix:",',
    "csv_suffix widget")

# Remove section1bis widgets (run_check_button + tol_seconds + out_check)
replace_in_cell(code,
    '# Section 1bis\n'
    'run_check_button = UI.button("Check text/CSV vs XML", "run", icon="check")\n'
    'tol_seconds = widgets.BoundedIntText(value=1, min=0, max=10, description="Match tolerance (s):",\n'
    '                                     style={"description_width": "initial"},\n'
    '                                     layout=widgets.Layout(width="170px"))\n'
    'out_check = widgets.Output()\n\n',
    "",
    "remove section1bis widgets")

# ===========================================================================
print("\n=== Cell 1: _update_info + run_scan ===")
# _update_info discovery + event detection (EDF: TXT + CSV detection -> Curry: .cdt + TXT detection)
replace_in_cell(code,
    '        folder = Path(chooser.selected_path)\n'
    '        edfs = [f for f in folder.rglob("*") if f.suffix.lower() == ".edf" and not f.name.startswith("._")]\n'
    '        if not edfs:\n'
    "            existing_info.value = UI.msg('warn', 'No EDF files found in selected folder.')\n"
    '            csv_suffix_info.value = ""\n'
    '            txt_suffix_info.value = ""\n'
    '            return\n'
    '        existing = load_existing_mapping()\n'
    '        msg = f"<small>{len(edfs)} EDF file(s) found. "\n'
    '        msg += (f"{len(existing)} label(s) already mapped."\n'
    '                if existing else "No existing mapping yet.")\n'
    '        existing_info.value = msg + "</small>" + mapping_note\n'
    '        stems = [f.stem for f in edfs]\n'
    '        # --- Event TXT-export suffix auto-detection (Compumedics/Curry *_ScoredEvents_Export.txt) ---\n'
    "        # Scan .txt files whose name contains 'event' so hypnogram .txt files are not counted.\n"
    '        all_txt = [f for f in folder.rglob("*")\n'
    '                   if f.suffix.lower() == ".txt" and "event" in f.name.lower()]\n'
    '        txt_owners = SF.companion_suffix_owners(edfs, all_txt)\n'
    '        if not txt_owners:\n'
    '            txt_suffix_info.value = (\n'
    '                f\'<small style="color:{UI.WARN_COLOR};">No event .txt detected next to the EDFs \'\n'
    "                '— set the suffix manually or rely on CSV/XML.</small>')\n"
    '        else:\n'
    '            # the most frequent suffix, shortest on ties\n'
    '            txt_suffix.value, _ = SF.select_event_suffix({s: len(r) for s, r in txt_owners.items()})\n'
    '            fill_txt_menu(txt_owners, stems)\n'
    '        # --- Event-CSV suffix auto-detection ---\n'
    '        all_csv = [f for f in folder.rglob("*") if f.suffix.lower() == ".csv"]\n'
    '        csv_owners = SF.companion_suffix_owners(edfs, all_csv)\n'
    '        if not csv_owners:\n'
    '            csv_suffix_info.value = (\n'
    '                f\'<small style="color:{UI.WARN_COLOR};">No event CSV detected next to the EDFs \'\n'
    "                '— set the suffix manually or rely on the XML fallback.</small>')\n"
    '        else:\n'
    '            # Events have no "more specific remapped" variant (unlike hypnograms), so prefer the\n'
    '            # MOST FREQUENT suffix (shortest on ties).\n'
    '            csv_suffix.value, _ = SF.select_event_suffix({s: len(r) for s, r in csv_owners.items()})\n'
    '            fill_csv_menu(csv_owners, stems)\n',
    '        folder = Path(chooser.selected_path)\n'
    '        cdts = [f for f in folder.rglob("*") if f.suffix == ".cdt" and not f.name.startswith("._")]\n'
    '        if not cdts:\n'
    "            existing_info.value = UI.msg('warn', 'No .cdt files found in selected folder.')\n"
    '            csv_suffix_info.value = ""\n'
    '            return\n'
    '        existing = load_existing_mapping()\n'
    '        msg = f"<small>{len(cdts)} .cdt file(s) found. "\n'
    '        msg += (f"{len(existing)} label(s) already mapped."\n'
    '                if existing else "No existing mapping yet.")\n'
    '        existing_info.value = msg + "</small>" + mapping_note\n'
    '        # --- Event-export suffix auto-detection (Curry: *_ScoredEvents_Export.txt) ---\n'
    "        # Scan .txt files whose name contains 'event' so hypnogram .txt files are not counted.\n"
    '        all_evt = [f for f in folder.rglob("*")\n'
    '                   if f.suffix.lower() == ".txt" and "event" in f.name.lower()]\n'
    '        evt_owners = SF.companion_suffix_owners(cdts, all_evt)\n'
    '        if not evt_owners:\n'
    '            csv_suffix_info.value = (\n'
    '                f\'<small style="color:{UI.WARN_COLOR};">No event export detected next to the .cdt files \'\n'
    "                '— set the suffix manually.</small>')\n"
    '        else:\n'
    '            # Prefer the MOST FREQUENT suffix (shortest on ties).\n'
    '            csv_suffix.value, _ = SF.select_event_suffix({s: len(r) for s, r in evt_owners.items()})\n'
    '            fill_csv_menu(evt_owners, [f.stem for f in cdts])\n',
    "_update_info block")
# the twin has a single suffix field: no TXT menu
replace_in_cell(code,
    'txt_menu, fill_txt_menu = UI.suffix_menu(txt_suffix, txt_suffix_info)\n',
    '',
    "remove the TXT menu")
replace_in_cell(code,
    '    fill_txt_menu({}, [])   # emptied until the detection below\n    fill_csv_menu({}, [])\n',
    '    fill_csv_menu({}, [])   # emptied until the detection below\n',
    "reset of the TXT menu")

# run_scan discovery
replace_in_cell(code,
    '        folder = Path(chooser.selected_path)\n'
    '        edfs = sorted(f for f in folder.rglob("*")\n'
    '                      if f.suffix.lower() == ".edf" and not f.name.startswith("._"))\n'
    '        if not edfs:\n'
    '            print("⚠️ No EDF files found in the selected folder.")\n'
    '            return\n',
    '        folder = Path(chooser.selected_path)\n'
    '        cdts = sorted(f for f in folder.rglob("*")\n'
    '                      if f.suffix == ".cdt" and not f.name.startswith("._"))\n'
    '        if not cdts:\n'
    '            print("⚠️ No .cdt files found in the selected folder.")\n'
    '            return\n',
    "run_scan discovery")

# run_scan loop head + failed reason
replace_in_cell(code,
    '        scan_progress.max = max(1, len(edfs))\n'
    '        scan_progress.value = 0\n'
    "        scan_progress.bar_style = 'info'\n"
    '        for edf in edfs:\n'
    '            scan_progress.value += 1\n'
    '            fid = edf.stem\n'
    '            try:\n'
    '                events, source = load_events(edf, txt_suffix.value, csv_suffix.value)\n'
    '            except Exception as e:\n'
    '                failed.append((fid, f"{type(e).__name__}: {e}"))\n'
    '                continue\n'
    '            if events is None:\n'
    '                failed.append((fid, "no readable event companion (.txt, .csv or .edf.XML)"))\n'
    '                continue\n',
    '        scan_progress.max = max(1, len(cdts))\n'
    '        scan_progress.value = 0\n'
    "        scan_progress.bar_style = 'info'\n"
    '        for cdt in cdts:\n'
    '            scan_progress.value += 1\n'
    '            fid = cdt.stem\n'
    '            try:\n'
    '                events, source = load_events(cdt, csv_suffix.value)\n'
    '            except Exception as e:\n'
    '                failed.append((fid, f"{type(e).__name__}: {e}"))\n'
    '                continue\n'
    '            if events is None:\n'
    '                failed.append((fid, "no readable event export (*_ScoredEvents_Export.txt)"))\n'
    '                continue\n',
    "run_scan loop")

# run_scan "Scanned" summary line
replace_in_cell(code,
    'print(f"✅ Scanned {len(edfs)} EDF file(s): {n_with} with events, {len(failed)} failed."',
    'print(f"✅ Scanned {len(cdts)} .cdt file(s): {n_with} with events, {len(failed)} failed."',
    "run_scan summary")

# Remove the TXT/CSV/XML source-count print (single source in Curry)
replace_in_cell(code,
    '        n_txt = sum(1 for s in source_by_file.values() if s == "txt")\n'
    '        n_csv = sum(1 for s in source_by_file.values() if s == "csv")\n'
    '        n_xml = sum(1 for s in source_by_file.values() if s == "xml")\n'
    '        print(f"   event source: {n_txt} from TXT, {n_csv} from CSV, {n_xml} from XML fallback.")\n',
    "",
    "remove source-count print")

# ===========================================================================
print("\n=== Cell 1: remove consistency check + wiring + layout ===")
remove_between(code,
    "def run_events_consistency_check(_=None):",
    "# ========================= Wiring & layout =========================",
    "remove run_events_consistency_check fn")

replace_in_cell(code,
    "run_scan_button.on_click(run_scan)\n"
    "run_check_button.on_click(run_events_consistency_check)\n"
    "preview_save_button.on_click(on_preview_save)",
    "run_scan_button.on_click(run_scan)\n"
    "preview_save_button.on_click(on_preview_save)",
    "remove run_check wiring")

# Remove the EDF's txt_suffix + txt_suffix_info from the layout row
replace_in_cell(code,
    '    section1, chooser, inputs_list, txt_menu, txt_suffix, txt_suffix_info, csv_menu, csv_suffix, csv_suffix_info,\n    existing_info, skip_existing, advanced, run_scan_button, scan_progress_row, out_scan,',
    '    section1, chooser, inputs_list, csv_menu, csv_suffix, csv_suffix_info,\n    existing_info, skip_existing, advanced, run_scan_button, scan_progress_row, out_scan,',
    "remove txt_suffix from layout")

replace_in_cell(code,
    "    section1bis, widgets.HBox([run_check_button, tol_seconds]), out_check,\n",
    "",
    "remove section1bis from layout")

# ===========================================================================
print("\n=== Validation ===")
try:
    out_str = json.dumps(nb, ensure_ascii=False, indent=1)
    json.loads(out_str)
    print(f"JSON valid. n_cells={len(nb['cells'])}")
except Exception as e:
    print(f"X JSON invalid: {e}")
    sys.exit(1)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        try:
            ast.parse(get_src(cell))
        except SyntaxError as e:
            print(f"X SyntaxError in code cell {i}: {e}")
            errors.append(f"syntax cell {i}")

if errors:
    print(f"\n! {len(errors)} issue(s): {errors}")

with open(DST, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")
print(f"Written: {DST}")
