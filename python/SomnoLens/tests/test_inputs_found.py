"""Section 1 of the tools (phase 3 of tools/plan_file_selection.md): the study folder is the only thing to pick. Each tool
finds its input files in it (the 'Inputs found' list), each line says where its file comes from, and 'Choose another…'
picks another one by hand. The Advanced box (output folder, custom stages, output names) says when a default was changed.

Picking the study folder only reads it: only the test that runs tool 5 writes, on a copy."""
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

import pipeline
from nbdriver import NotebookSession

T0 = 'tools/0_live_explore_1file_voila.ipynb'
T2 = 'tools/2_select&remap_channels_edf_voila.ipynb'
T3 = 'tools/3_remap_hypno_voila.ipynb'
T3BIS = 'tools/3bis_autoscoring_voila.ipynb'
T4 = 'tools/4_remap_events_edf_voila.ipynb'
IDS = ('s1', 's2')


def line(nb, row):
    """The text of an input's line in the list (the first child of its row)."""
    return nb.value(f'{row}.children[0].children[0].value')


def shown(nb):
    """Is the list shown? (its box, after the grey line that waits for the study folder)"""
    return nb.value('inputs_list.children[1].layout.display') != 'none'


def make_study(folder, config='remap_reref_persubject.json', events=True):
    folder.mkdir()
    for fid in IDS:
        (folder / f'{fid}.edf').write_bytes(b'')
        (folder / f'{fid}_Hypnogram_remapped.txt').write_text('W\n', encoding='utf-8')
    cp = folder / 'config_param'
    cp.mkdir()
    if config:
        cfg = {fid: {'config': 'config. 1', 'remap': {'C3': 'C3'}, 'ref_channels': ['M2']} for fid in IDS}
        (cp / config).write_text(json.dumps(cfg), encoding='utf-8')
    if events:
        (cp / 'event_remap.json').write_text('{"Arousal": "arousal", "Eveil": "arousal"}', encoding='utf-8')
    (cp / 'custom_stages.json').write_text('{"custom_stages": ["N4"]}', encoding='utf-8')
    return folder


def write_names(study, **keys):
    """Names as the tools record them in config_param/file_names.json: key=(value, tool)."""
    names = {k: {'value': v, 'tool': t, 'saved_at': '2026-11-04 10:00:00'} for k, (v, t) in keys.items()}
    (study / 'config_param' / 'file_names.json').write_text(json.dumps(names), encoding='utf-8')


# ----------------------------------------------------------------------------------------------- tool 7
@pytest.mark.tool7
def test_tool7_finds_its_inputs_and_follows_a_file_chosen_by_hand(tmp_path):
    study = make_study(tmp_path / 'study')
    with NotebookSession(pipeline.T7) as nb:
        assert not shown(nb)                                            # nothing to see before the study folder
        assert nb.value('fc_edf.layout.display') != 'none'              # but its picker
        nb.pick('fc_edf', study)
        assert shown(nb)
        assert nb.value('fc_edf.layout.display') != 'none' and nb.value('fc_out.layout.display') != 'none'
        assert Path(nb.value('fc_config.selected')) == study / 'config_param' / 'remap_reref_persubject.json'
        assert Path(nb.value('fc_events.selected')) == study / 'config_param' / 'event_remap.json'
        assert nb.value('fc_quality.selected') is None
        assert '(default location)' in line(nb, 'config_row') and '2 participants' in line(nb, 'config_row')
        assert '2 labels → 1 event types' in line(nb, 'events_row')
        assert 'not found: no channel is pre-deselected (run tool 6 first)' in line(nb, 'quality_row')
        assert 'N4' in nb.value('custom_stages_line.value')
        assert nb.value('advanced.get_title(0)') == 'Advanced'

        # a file chosen by hand wins, and the line says what it holds
        wrong = study / 'not_a_config.json'
        wrong.write_text('{"Arousal": "arousal"}', encoding='utf-8')
        nb.pick('fc_config', wrong)
        assert '(chosen by hand)' in line(nb, 'config_row') and "no 'remap' key" in line(nb, 'config_row')

        # the Advanced box says when a default was changed
        nb.set('custom_stages_box', 'N4, N5')
        assert 'default changed: custom stages' in nb.value('advanced.get_title(0)')
        assert 'changed in Advanced' in nb.value('custom_stages_line.value')
        elsewhere = tmp_path / 'elsewhere'
        elsewhere.mkdir()
        nb.pick('fc_out', elsewhere)
        assert 'default changed: output folder, custom stages' in nb.value('advanced.get_title(0)')


@pytest.mark.tool7
def test_tool7_refuses_to_scan_without_a_channel_configuration(tmp_path):
    study = make_study(tmp_path / 'study', config=None)
    with NotebookSession(pipeline.T7) as nb:
        nb.pick('fc_edf', study)
        assert 'not found: run tool 2 first' in line(nb, 'config_row')
        nb.click('btn_load')
        assert 'run tool 2 first' in nb.value('load_info.value')


@pytest.mark.tool7
def test_tool7_finds_the_new_name_and_the_recorded_name(tmp_path):
    study = make_study(tmp_path / 'study', config='channel_config.json')
    with NotebookSession(pipeline.T7) as nb:
        nb.pick('fc_edf', study)
        assert Path(nb.value('fc_config.selected')) == study / 'config_param' / 'channel_config.json'
    mine = study / 'config_param' / 'mine.json'
    mine.write_text('{"s1": {"remap": {}}}', encoding='utf-8')
    write_names(study, channel_config=('config_param/mine.json', '2'))
    with NotebookSession(pipeline.T7) as nb:
        nb.pick('fc_edf', study)
        assert Path(nb.value('fc_config.selected')) == mine
        assert '(recorded by tool 2)' in line(nb, 'config_row') and '1 participants' in line(nb, 'config_row')


# ----------------------------------------------------------------------------------------------- tool 6
@pytest.mark.tool6
def test_tool6_finds_the_channel_configuration(tmp_path):
    study = make_study(tmp_path / 'study')
    with NotebookSession(pipeline.T6) as nb:
        assert not shown(nb)
        nb.pick('fc_folder', study)
        assert shown(nb) and nb.value('fc_folder.layout.display') != 'none'
        assert Path(nb.value('fc_config.selected')) == study / 'config_param' / 'remap_reref_persubject.json'
        assert '2 participants' in line(nb, 'config_row')
        assert 'Selected channels' in nb.value('acq_info.value')        # the line that needs the configuration
        assert 'N4' in nb.value('custom_stages_line.value')
        nb.set('custom_stages_box', '')
        assert 'default changed: custom stages' in nb.value('advanced.get_title(0)')
    study2 = make_study(tmp_path / 'study2', config=None)
    with NotebookSession(pipeline.T6) as nb:
        nb.pick('fc_folder', study2)
        assert 'not found: run tool 2 first' in line(nb, 'config_row')
        nb.click('btn_run')
        assert 'run tool 2 first' in nb.widget_text()


# ----------------------------------------------------------------------------------------------- tool 5
@pytest.mark.tool5
def test_tool5_finds_the_event_remap_and_the_participant_table(tmp_path):
    study = make_study(tmp_path / 'study')
    (study / 'participants.tsv').write_text('participant_id\tage\ns1\t30\ns2\t40\n', encoding='utf-8')
    with NotebookSession(pipeline.T5) as nb:
        assert not shown(nb)
        nb.pick('fc_data', study)
        assert shown(nb) and nb.value('fc_out.layout.display') != 'none'
        assert Path(nb.value('fc_remap.selected')) == study / 'config_param' / 'event_remap.json'
        assert Path(nb.value('fc_subj.selected')) == study / 'participants.tsv'
        assert '2 rows' in line(nb, 'subj_row')
        assert nb.value('dd_join_col.value') == 'participant_id' and nb.value('dd_join_col.layout.display') == ''
        assert 'N4' in nb.value('custom_stages_line.value')
        elsewhere = tmp_path / 'elsewhere'
        elsewhere.mkdir()
        nb.pick('fc_out', elsewhere)
        assert 'default changed: output folder' in nb.value('advanced.get_title(0)')
    study2 = make_study(tmp_path / 'study2', events=False)
    with NotebookSession(pipeline.T5) as nb:
        nb.pick('fc_data', study2)
        assert nb.value('fc_remap.selected') is None and nb.value('fc_subj.selected') is None
        assert 'not found: the event indices' in line(nb, 'remap_row')
        assert nb.value('dd_join_col.layout.display') == 'none'


@pytest.mark.tool5
@pytest.mark.tool5bis
def test_tool5_notes_its_table_outside_the_study_and_tool5bis_finds_it(chain, tmp_path):
    """An output folder other than the study folder: tool 5 notes where the table is, and tool 5bis finds it from the
    study folder alone. Without an output folder nothing is noted (the default place)."""
    study, out = tmp_path / 'study', tmp_path / 'out'
    shutil.copytree(chain['data'], study)
    out.mkdir()
    names_file = study / 'config_param' / 'file_names.json'
    assert not names_file.exists()
    with NotebookSession(pipeline.T5) as nb:
        nb.pick('fc_data', study)
        nb.pick('fc_out', out)
        nb.set('dd_join_col', 'participant_id')
        nb.click('btn_scan')
        nb.click('btn_run')
    table = out / 'reports_features_macrostructure' / 'global_sleep_metrics.tsv'
    assert table.is_file()
    assert json.loads(names_file.read_text(encoding='utf-8'))['sleep_metrics_table']['value'] == str(table.resolve())
    with NotebookSession(pipeline.T5BIS) as nb:
        nb.pick('fc_data', study)
        assert Path(nb.value('fc_table.selected')) == table.resolve()
        assert '(recorded by tool 5)' in line(nb, 'table_row')


# --------------------------------------------------------------------------------------------- tool 5bis
@pytest.mark.tool5bis
def test_tool5bis_finds_the_table_or_says_to_run_tool_5(tmp_path):
    study = make_study(tmp_path / 'study')
    with NotebookSession(pipeline.T5BIS) as nb:
        assert not shown(nb)
        nb.pick('fc_data', study)
        assert 'not found: run tool 5 first' in line(nb, 'table_row')
        nb.click('btn_load')
        assert 'Run tool 5 first' in nb.value('lbl_load.value')
    reports = study / 'reports_features_macrostructure'
    reports.mkdir()
    table = reports / 'global_sleep_metrics.tsv'
    table.write_text('file_id\tTST\ns1\t400\ns2\t420\n', encoding='utf-8')
    with NotebookSession(pipeline.T5BIS) as nb:
        nb.pick('fc_data', study)
        assert Path(nb.value('fc_table.selected')) == table
        assert '(default location)' in line(nb, 'table_row') and '2 participants' in line(nb, 'table_row')


# --------------------------------------------------------------------------------------------- tool 3bis
@pytest.mark.tool3bis
def test_tool3bis_finds_the_channel_configuration_and_keeps_the_outputs_in_advanced(tmp_path):
    study = make_study(tmp_path / 'study', config='channel_config.json')
    with NotebookSession(T3BIS) as nb:
        assert not shown(nb)
        nb.pick('fc_data', study)
        assert Path(nb.value('fc_config.selected')) == study / 'config_param' / 'channel_config.json'
        assert nb.value('advanced.get_title(0)') == 'Advanced'
        nb.set('out_suffix', '_mine.txt')
        assert 'default changed: GSSC output suffix' in nb.value('advanced.get_title(0)')
        nb.click('btn_scan')
        assert 'configured in tool 2' in nb.value('scan_info.value')
    study2 = make_study(tmp_path / 'study2', config=None)
    with NotebookSession(T3BIS) as nb:
        nb.pick('fc_data', study2)
        nb.click('btn_scan')
        assert 'run tool 2' in nb.value('scan_info.value')


# ----------------------------------------------------------------------------------------------- tool 3
@pytest.mark.tool3
def test_tool3_keeps_the_output_suffix_in_advanced(tmp_path):
    study = make_study(tmp_path / 'study')
    with NotebookSession(T3) as nb:
        nb.pick('chooser', study)
        assert nb.value('output_suffix_box.value') == '_Hypnogram_remapped.txt'     # a study keeps its name
        assert nb.value('advanced.get_title(0)') == 'Advanced'
        nb.set('output_suffix_box', '_mine.txt')
        assert 'default changed: output suffix' in nb.value('advanced.get_title(0)')


# ----------------------------------------------------------------------------------------------- tool 4
@pytest.mark.tool4
def test_tool4_finds_the_mapping_and_keeps_its_name(tmp_path):
    study = make_study(tmp_path / 'study')
    with NotebookSession(T4) as nb:
        assert not shown(nb)
        nb.pick('chooser', study)
        assert Path(nb.value('fc_mapping.selected')) == study / 'config_param' / 'event_remap.json'
        assert '2 labels → 1 event types' in line(nb, 'mapping_row')
        assert nb.value('fname_text.value') == 'event_remap.json'
        assert '2 label(s) already mapped' in nb.value('existing_info.value')
    mine = study / 'config_param' / 'my_events.json'
    mine.write_text('{"Arousal": "arousal"}', encoding='utf-8')
    write_names(study, event_remap=('config_param/my_events.json', '4'))
    with NotebookSession(T4) as nb:
        nb.pick('chooser', study)
        assert nb.value('fname_text.value') == 'my_events.json'
        assert Path(nb.value('fc_mapping.selected')) == mine and '(recorded by tool 4)' in line(nb, 'mapping_row')
        nb.set('fname_text', 'other.json')
        assert 'default changed: file name' in nb.value('advanced.get_title(0)')
    study2 = make_study(tmp_path / 'study2', events=False)
    with NotebookSession(T4) as nb:
        nb.pick('chooser', study2)
        assert nb.value('fc_mapping.selected') is None and 'a new mapping is started' in line(nb, 'mapping_row')


# ----------------------------------------------------------------------------------------------- tool 2
@pytest.mark.tool2
def test_tool2_names_its_file_channel_config_and_a_study_keeps_its_name(tmp_path):
    new, old = make_study(tmp_path / 'new', config=None), make_study(tmp_path / 'old')
    with NotebookSession(T2) as nb:
        nb.pick('chooser', new)
        assert nb.value('fname_text.value') == 'channel_config.json'
        assert Path(nb.value('str(config_json_path(chooser.selected_path))')) == new / 'config_param' / 'channel_config.json'
        assert '0 / 2 participant(s) already configured' in nb.value('existing_config_info.value')
        nb.pick('chooser', old)
        assert nb.value('fname_text.value') == 'remap_reref_persubject.json'
        assert '2 / 2 participant(s) already configured' in nb.value('existing_config_info.value')
        assert nb.value('advanced.get_title(0)') == 'Advanced'
        nb.set('fname_text', 'other.json')
        assert 'default changed: file name' in nb.value('advanced.get_title(0)')
        assert '0 / 2 participant(s) already configured' in nb.value('existing_config_info.value')
    write_names(old, channel_config=('config_param/mine.json', '2'))
    with NotebookSession(T2) as nb:
        nb.pick('chooser', old)
        assert nb.value('fname_text.value') == 'mine.json'


# ----------------------------------------------------------------------------------------------- tool 0
@pytest.mark.tool0
def test_tool0_finds_the_new_name_like_the_other_tools(tmp_path):
    study = make_study(tmp_path / 'study', config='channel_config.json')
    with NotebookSession(T0) as nb:
        nb.pick('fc_data', study)
        assert 'channel configuration of tool 2 read (2 participant(s))' in nb.value('html_study.value')
        assert 'custom stages: N4' in nb.value('html_study.value')
        assert nb.value('advanced.get_title(0)') == 'Advanced'


# ------------------------------------------------------------------------- the chain's study folder
@pytest.mark.tool7
def test_tool7_finds_in_the_chain_study_the_files_the_chain_picks(chain):
    """With the study folder only, tool 7 selects exactly the files the chain picks by hand (pipeline.run_tool7):
    its run reads only these selections, so its outputs are the chain's."""
    data = Path(chain['data'])
    with NotebookSession(pipeline.T7) as nb:
        nb.pick('fc_edf', data)
        assert Path(nb.value('fc_config.selected')) == data / 'config_param' / 'remap_reref_persubject.json'
        assert Path(nb.value('fc_quality.selected')) == data / 'reports_quality_overview' / 'quality_summary.tsv'
        assert Path(nb.value('fc_events.selected')) == data / 'config_param' / 'event_remap.json'
        assert Path(nb.value('str(out_root_path())')) == data
        for row in ('config_row', 'quality_row', 'events_row'):
            assert '(default location)' in line(nb, row)


@pytest.mark.tool6
def test_tool6_finds_in_the_chain_study_the_file_the_chain_picks(chain):
    data = Path(chain['data'])
    with NotebookSession(pipeline.T6) as nb:
        nb.pick('fc_folder', data)
        assert Path(nb.value('fc_config.selected')) == data / 'config_param' / 'remap_reref_persubject.json'


@pytest.mark.tool5
def test_tool5_finds_in_the_chain_study_the_files_the_chain_picks(chain):
    data = Path(chain['data'])
    with NotebookSession(pipeline.T5) as nb:
        nb.pick('fc_data', data)
        assert Path(nb.value('fc_subj.selected')) == data / 'participants.tsv'
        assert Path(nb.value('fc_remap.selected')) == data / 'config_param' / 'event_remap.json'


@pytest.mark.tool5bis
def test_tool5bis_finds_in_the_chain_study_the_table_it_reads(chain):
    data = Path(chain['data'])
    table = data / 'reports_features_macrostructure' / 'global_sleep_metrics.tsv'
    assert len(pd.read_csv(table, sep='\t', dtype={'file_id': str})) > 0
    with NotebookSession(pipeline.T5BIS) as nb:
        nb.pick('fc_data', data)
        assert Path(nb.value('fc_table.selected')) == table
