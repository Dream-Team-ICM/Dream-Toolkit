"""The scored-event files found but not read are said, never skipped silently (code review of 2026-10-24, #10), and the
per-channel fields of tool 5's EDF header reader (#20).

load_events (edf_io_lib for tools 0 and 5, tool 7's copy kept identical by test_edf_io_sync, tool 4's own for the
event names) reads the text export, then the CSV, then the XML. A file found but not used gives one line per recording:
which file, why, and what was read instead. The value returned does not change."""
import ast
import datetime
import json
import logging
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import edfio
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
import edf_io_lib as EIO             # noqa: E402
import macrostructure_lib as ML      # noqa: E402

pytestmark = [pytest.mark.quick, pytest.mark.tool0, pytest.mark.tool4, pytest.mark.tool5, pytest.mark.tool7]

T4 = ROOT / 'tools' / '4_remap_events_edf_voila.ipynb'
GOOD_TXT = '23:10:00,21,N2,Hypopnée,0:12.0,-,-,Dos\n'                 # 600 s after the start
GOOD_CSV = 'Name,Start,Duration\nHypopnea,30.0,10.0\n'
GOOD_XML = ('<CMPStudyConfig><ScoredEvents><ScoredEvent><Name>Snore</Name><Start>1</Start><Duration>1</Duration>'
            '</ScoredEvent></ScoredEvents></CMPStudyConfig>')
BROKEN_TXT = b'\xff\xfea'            # a UTF-16 byte order mark, then an odd number of bytes
NO_SCORED_EVENT_XML = '<CMPStudyConfig><ScoredEvents></ScoredEvents></CMPStudyConfig>'
BROKEN_XML = '<CMPStudyConfig><ScoredEvents>'


def write_recording(folder, start=b'02.01.2623.00.00'):
    """A 30 s EDF of three channels (two sampling rates, three units) starting 2026-01-02 at 23:00:00, or with the
    16 bytes of start date + time given."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'rec.edf'
    signals = [edfio.EdfSignal(np.zeros(256 * 30), 256, label='C3', physical_dimension='uV', physical_range=(-500, 500)),
               edfio.EdfSignal(np.zeros(32 * 30), 32, label='SpO2', physical_dimension='%', physical_range=(0, 100)),
               edfio.EdfSignal(np.zeros(256 * 30), 256, label='EMG chin', physical_dimension='mV',
                               physical_range=(-1, 1))]
    edfio.Edf(signals).write(path)
    with open(path, 'r+b') as f:
        f.seek(168)
        f.write(start)
    return path


def companions(folder, txt=None, csv=None, xml=None):
    """The event files beside rec.edf: text or bytes, None = no file."""
    for name, content in (('rec_ScoredEvents_Export.txt', txt), ('rec_event_xml.csv', csv), ('rec.edf.XML', xml)):
        if content is None:
            continue
        if isinstance(content, bytes):
            (folder / name).write_bytes(content)
        else:
            (folder / name).write_text(content, encoding='utf-8')


def load(path, **kw):
    lines = []
    result = EIO.load_events(path, warn=lines.append, **kw)
    return result, lines


# ---------------------------------------------------------------------------------------------------------
# edf_io_lib.load_events (tools 0, 5, and 7 through its copy)
# ---------------------------------------------------------------------------------------------------------
def test_nothing_is_said_when_every_file_found_is_read(tmp_path):
    path = write_recording(tmp_path)
    companions(tmp_path, txt=GOOD_TXT, csv=GOOD_CSV, xml=GOOD_XML)
    (events, source), lines = load(path)
    assert source == 'txt' and events['Start'].tolist() == [600.0]
    assert lines == []


def test_an_unreadable_text_export_is_named_and_the_csv_read_instead(tmp_path):
    path = write_recording(tmp_path)
    companions(tmp_path, txt=BROKEN_TXT, csv=GOOD_CSV)
    (events, source), lines = load(path)
    assert source == 'csv' and events['Name'].tolist() == ['Hypopnea']
    assert len(lines) == 1
    assert lines[0].startswith('rec_ScoredEvents_Export.txt unreadable (UnicodeDecodeError: ')
    assert lines[0].endswith(': events read from rec_event_xml.csv instead')


def test_files_without_events_are_named_in_one_line(tmp_path):
    path = write_recording(tmp_path)
    companions(tmp_path, txt='not an export\n', csv='a,b\n1,2\n', xml=GOOD_XML)
    (events, source), lines = load(path)
    assert source == 'xml' and events['Name'].tolist() == ['Snore']
    assert lines == ['rec_ScoredEvents_Export.txt: no event line in the expected format; '
                     'rec_event_xml.csv: no Name / Start / Duration columns: events read from rec.edf.XML instead']


def test_nothing_read_says_so(tmp_path):
    path = write_recording(tmp_path)
    companions(tmp_path, txt=BROKEN_TXT, csv='', xml=BROKEN_XML)
    result, lines = load(path)
    assert result == (None, None)
    assert len(lines) == 1
    assert re.fullmatch(r'rec_ScoredEvents_Export\.txt unreadable \(UnicodeDecodeError: .*\); '
                        r'rec_event_xml\.csv unreadable \(EmptyDataError: .*\); '
                        r'rec\.edf\.XML unreadable \(ParseError: .*\): no scored event read', lines[0])


def test_an_xml_without_scored_event(tmp_path):
    path = write_recording(tmp_path)
    companions(tmp_path, xml=NO_SCORED_EVENT_XML)
    result, lines = load(path)
    assert result == (None, None)
    assert lines == ['rec.edf.XML: no <ScoredEvent>: no scored event read']


def test_a_text_export_without_recording_start(tmp_path):
    path = write_recording(tmp_path, start=b'xx.xx.xxxx.xx.xx')
    companions(tmp_path, txt=GOOD_TXT, csv=GOOD_CSV)
    (events, source), lines = load(path)
    assert source == 'csv'
    assert lines == ['rec_ScoredEvents_Export.txt not read: the recording start is unknown (or the recording was '
                     'paused): events read from rec_event_xml.csv instead']


def test_an_unreadable_annotation_file_leaves_the_scored_events(tmp_path):
    path = write_recording(tmp_path)
    companions(tmp_path, csv=GOOD_CSV)
    (tmp_path / 'rec_manual_events.tsv').write_text('', encoding='utf-8')
    (events, source), lines = load(path, include_manual=True)
    assert source == 'csv' and events['Name'].tolist() == ['Hypopnea']
    assert len(lines) == 1
    assert re.fullmatch(r'rec_manual_events\.tsv unreadable \(EmptyDataError: .*\): the manual annotations are left out',
                        lines[0])


def test_an_annotation_file_emptied_in_tool_8_says_nothing(tmp_path):
    path = write_recording(tmp_path)
    companions(tmp_path, csv=GOOD_CSV)
    (tmp_path / 'rec_manual_events.tsv').write_text('type\tonset_s\tduration_s\n', encoding='utf-8')
    (events, source), lines = load(path, include_manual=True)
    assert source == 'csv' and lines == []


def test_without_warn_the_line_goes_to_logging(tmp_path, caplog):
    path = write_recording(tmp_path)
    companions(tmp_path, txt=BROKEN_TXT, csv=GOOD_CSV)
    with caplog.at_level(logging.WARNING, logger='edf_io_lib'):
        events, source = EIO.load_events(path)
    assert source == 'csv'
    assert any('rec_ScoredEvents_Export.txt unreadable' in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------------------------------------
# Tool 4's load_events (the event names only): same fallback as edf_io_lib, same lines
# ---------------------------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def tool4():
    """Tool 4's event loaders, taken from its notebook (the page itself is not built)."""
    names = {'load_events', 'load_events_from_txt', 'load_events_from_csv', 'load_events_from_xml'}
    cells = json.loads(T4.read_text(encoding='utf-8'))['cells']
    source = ''.join(cells[1]['source'])
    tree = ast.parse(source)
    defs = [ast.get_source_segment(source, n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(defs) == len(names)
    namespace = {'Path': Path, 'pd': pd, 'ET': ET, 'datetime': datetime, '_TXT_DUR_RE': EIO._TXT_DUR_RE,
                 'event_companion_paths': EIO.event_companion_paths}
    exec('\n\n'.join(defs), namespace)
    return namespace['load_events']


def test_tool4_reads_the_names_of_the_text_export(tmp_path, tool4):
    path = write_recording(tmp_path)
    companions(tmp_path, txt=GOOD_TXT, csv=GOOD_CSV)
    lines = []
    events, source = tool4(path, warn=lines.append)
    assert source == 'txt' and [e[0] for e in events] == ['Hypopnée'] and lines == []


def test_tool4_passes_an_export_without_event_to_the_csv(tmp_path, tool4):
    """Before, an export without any event line gave 0 label from the TXT while tools 5 and 7 read the CSV: their
    labels could miss from event_remap.json."""
    path = write_recording(tmp_path)
    companions(tmp_path, txt='not an export\n', csv=GOOD_CSV)
    lines = []
    events, source = tool4(path, warn=lines.append)
    assert source == 'csv' and [e[0] for e in events] == ['Hypopnea']
    assert lines == ['rec_ScoredEvents_Export.txt: no event line in the expected format: '
                     'events read from rec_event_xml.csv instead']


def test_tool4_nothing_read_says_why(tmp_path, tool4):
    path = write_recording(tmp_path)
    companions(tmp_path, txt=BROKEN_TXT, csv='a,b\n1,2\n', xml=NO_SCORED_EVENT_XML)
    lines = []
    assert tool4(path, warn=lines.append) == (None, None)
    assert len(lines) == 1
    assert re.fullmatch(r"rec_ScoredEvents_Export\.txt unreadable \(UnicodeDecodeError: .*\); "
                        r"rec_event_xml\.csv unreadable \(ValueError: missing column 'Name' in rec_event_xml\.csv\); "
                        r"rec\.edf\.XML: no <ScoredEvent>: no scored event read", lines[0])


# ---------------------------------------------------------------------------------------------------------
# #20: tool 5's header reader finds the same per-channel fields as tool 1's
# ---------------------------------------------------------------------------------------------------------
def test_tool5_header_fields_match_the_custom_parser(tmp_path):
    path = write_recording(tmp_path)
    mine = ML.read_edf_header_info(path)
    ref = EIO.read_edf_header_custom(path)
    assert mine['ch_names'] == ref['channel'] == ['C3', 'SpO2', 'EMG chin']
    assert mine['dims'] == ref['dimension'] == ['uV', '%', 'mV']
    assert mine['samples_per_record'] == [int(x) for x in ref['samples_per_record']]
    assert mine['sfreq_per_ch'] == {'C3': 256.0, 'SpO2': 32.0, 'EMG chin': 256.0}
    assert mine['start_dt'] == datetime.datetime(2026, 1, 2, 23, 0, 0)
    assert mine['duration_s'] == 30.0
