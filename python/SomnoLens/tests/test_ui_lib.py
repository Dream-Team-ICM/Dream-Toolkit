"""tools/ui_lib.py: the look shared by every Voila page (SPEC *Cross-cutting -> Voila page conventions*)."""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
import ui_lib as UI  # noqa: E402

pytestmark = pytest.mark.quick

# Every Voila page: the EDF tools and their Curry twins
PAGES = sorted(ROOT.glob('tools/*_voila.ipynb')) + sorted(ROOT.glob('tools_curry/*_voila.ipynb'))


def cells(page):
    """(cell type, text) of every cell of a notebook."""
    nb = json.loads(page.read_text(encoding='utf-8'))
    return [(c['cell_type'], c['source'] if isinstance(c['source'], str) else ''.join(c['source']))
            for c in nb['cells']]


def test_messages_carry_their_colour_and_symbol():
    assert UI.msg('warn', '2 recordings have no hypnogram.') == \
        f'<span style="color:{UI.WARN_COLOR}">{UI.WARN} 2 recordings have no hypnogram.</span>'
    assert UI.ERROR not in UI.msg('error', 'failed', symbol=False)
    assert UI.msg('muted', 'details') == f'<span style="color:{UI.MUTED_COLOR}">details</span>'
    box = UI.box('ok', '<b>Done</b>')
    assert f'border-left:4px solid {UI.OK_COLOR}' in box and f'{UI.OK} <b>Done</b>' in box


def test_section_headers():
    html = UI.section_html(2, 'Parameters', '<p>Set them.</p>')
    assert html.startswith(f'<div class="{UI.MARKDOWN_CLASS}">{UI.SECTION_RULE}')
    assert '<h2>2. Parameters</h2>' in html and '<p>Set them.</p>' in html
    assert '<h2>References</h2>' in UI.section_html(None, 'References')
    assert '<h3>2.1 Epoching</h3>' in UI.subsection_html('2.1', 'Epoching')
    assert UI.section_markdown(1, 'Study folder') == f'{UI.SECTION_RULE}\n\n## 1. Study folder'


def test_picker_title_puts_the_detail_on_its_own_line():
    # the widget HTML's line height is a one-line field's (28 px): the title sets its own
    title = UI.picker_title('Study folder', '(holds <code>derivatives/</code>)')
    assert title.startswith('<div style="line-height:1.4;"><b>Study folder</b><br>')
    assert UI.MUTED_COLOR in title and '(holds <code>derivatives/</code>)' in title
    assert '<br>' not in UI.picker_title('Study folder')


def test_long_texts_get_a_normal_line_height():
    assert UI.text_block('<small>a long help text</small>') == '<div style="line-height:1.4;"><small>a long help text</small></div>'
    assert 'line-height:1.4' in UI.help_html('help') and 'line-height:1.4' in UI.box_style('warn')


@pytest.mark.parametrize('role', list(UI.BUTTON_ROLES))
def test_buttons_follow_their_role(role):
    btn = UI.button('Do it', role, tooltip='a tooltip')
    assert (btn.button_style, btn.icon) == UI.BUTTON_ROLES[role]
    assert btn.layout.width == 'max-content' and btn.tooltip == 'a tooltip'
    assert UI.button('Preview', 'read', icon='eye').icon == 'eye'


def test_progress_bar_text_follows_the_bar():
    bar, row = UI.progress_bar('Participants:')
    assert row.children[0] is bar and row.children[1].value == ''
    bar.max, bar.value = 10, 3
    assert row.children[1].value == '3 / 10'
    UI.finish_progress(bar)
    assert bar.bar_style == 'success'
    UI.finish_progress(bar, n_failed=1)
    assert bar.bar_style == 'warning'


def test_suffix_menu_lists_the_suffixes_and_follows_its_field():
    import ipywidgets as widgets
    field = widgets.Text(value='_x.txt', description='Hypno suffix:', layout=widgets.Layout(width='420px'))
    info = widgets.HTML('')
    menu, fill = UI.suffix_menu(field, info)
    assert menu.layout.display == 'none' and field.layout.display != 'none'      # nothing detected yet
    fill({'_a.txt': ['s1', 's2'], '_b.txt': ['s1']}, ['s1', 's2'])
    assert menu.options[0] == ('_a.txt  (2/2)', '_a.txt') and menu.options[-1][1] == UI.OTHER_SUFFIX
    assert menu.value == UI.OTHER_SUFFIX and field.description == 'Other suffix:'   # '_x.txt': typed by hand
    field.value = '_a.txt'           # set by a test or by the tool's own detection: the menu follows
    assert menu.value == '_a.txt' and field.layout.display == 'none' and '2/2 recordings' in info.value
    menu.value = '_b.txt'            # chosen in the menu: the field, the value the tool reads, follows
    assert field.value == '_b.txt' and 'missing: s2' in info.value
    menu.value = UI.OTHER_SUFFIX
    assert field.layout.display == '' and field.value == '_b.txt'
    fill({}, [])                     # another folder, nothing found: back to the plain field
    assert menu.layout.display == 'none' and field.description == 'Hypno suffix:'


def test_input_row_says_what_was_found_and_follows_a_file_chosen_by_hand(tmp_path):
    from ipyfilechooser import FileChooser
    study = tmp_path / 'study'
    (study / 'config_param').mkdir(parents=True)
    found = study / 'config_param' / 'channels.json'
    found.write_text('{}', encoding='utf-8')
    mine = tmp_path / 'mine.json'
    mine.write_text('{}', encoding='utf-8')
    study_picker, picker, changes = FileChooser(), FileChooser(), []
    row, locate = UI.input_row(picker, 'Channel configuration', 'required', 'run tool 2 first',
                               check=lambda p: ('ok', '11 participants') if Path(p) == found else ('warn', 'empty'),
                               on_change=lambda fc: changes.append(fc.selected))
    line = row.children[0].children[0]
    assert row.layout.display == 'none'                                    # no study folder yet
    assert study_picker.layout.display != 'none'      # the pickers share a default Layout: only this one is hidden
    locate(study, None, '', study / 'config_param')
    assert row.layout.display == '' and UI.ERROR in line.value and 'not found: run tool 2 first' in line.value
    assert picker.selected is None and changes == [None]
    locate(study, found, 'default location', found.parent)
    assert picker.selected == str(found) and picker.layout.display == 'none'   # selected without a click
    assert 'config_param/channels.json' in line.value and '(default location)' in line.value
    assert '11 participants' in line.value and UI.OK in line.value
    picker.reset(path=str(tmp_path), filename='mine.json')                 # chosen by hand, as with the dialog
    picker._show_dialog()
    picker._on_select_click(None)
    assert '(chosen by hand)' in line.value and str(mine) in line.value    # outside the study: shown in full
    assert UI.WARN in line.value and changes[-1] == str(mine)
    locate(None)
    assert row.layout.display == 'none' and picker.selected is None
    for need, symbol in [('recommended', UI.WARN), ('optional', UI.INFO)]:
        row, locate = UI.input_row(FileChooser(), 'Event remap', need, 'no event flagging')
        locate(study, None, '', study)
        assert symbol in row.children[0].children[0].value


def test_input_row_of_a_folder_and_its_note(tmp_path):
    from ipyfilechooser import FileChooser
    folder = tmp_path / 'derivatives' / 'preproc_epo'
    folder.mkdir(parents=True)
    picker = FileChooser()
    picker.show_only_dirs = True
    row, locate = UI.input_row(picker, 'Preprocessed epochs', 'optional', 'run tool 7 first')
    line = row.children[0].children[0]
    locate(tmp_path, folder, 'the only run found', tmp_path)
    assert Path(picker.selected_path) == folder                            # the folder itself, not its parent
    assert 'derivatives/preproc_epo' in line.value and '(the only run found)' in line.value
    locate(tmp_path, None, '', tmp_path, note='none chosen: no rejection is applied')
    assert picker.selected_path is None and 'none chosen' in line.value and 'not found' not in line.value
    locate(tmp_path, None, '', tmp_path)
    assert 'not found: run tool 7 first' in line.value


def test_choice_menu_shows_only_with_several_candidates():
    chosen = []
    menu, fill = UI.choice_menu('Run:', chosen.append)
    assert menu.layout.display == 'none'
    fill([('one', 'a')])
    assert menu.layout.display == 'none' and menu.value == 'a' and chosen == []     # filling chooses nothing
    fill([('one', 'a'), ('two', 'b')])
    assert menu.layout.display == '' and menu.value == 'a' and chosen == []
    menu.value = 'b'
    assert chosen == ['b']
    fill([])
    assert menu.layout.display == 'none' and chosen == ['b']


def test_inputs_list_waits_for_the_study_folder():
    import ipywidgets as widgets
    field = widgets.Text()
    widget, show = UI.inputs_list([field])
    wait, box = widget.children
    assert 'Pick the study folder' in wait.value and box.layout.display == 'none' and field in box.children
    show(True)
    assert box.layout.display == '' and wait.layout.display == 'none'
    show(False)
    assert box.layout.display == 'none' and wait.layout.display == ''


def test_custom_stages_line_says_where_they_come_from():
    assert 'custom_stages.json' in UI.custom_stages_html(['N4'], ['N4']) and UI.OK in UI.custom_stages_html(['N4'], ['N4'])
    assert 'changed in Advanced' in UI.custom_stages_html(['N4', 'N5'], ['N4'])
    assert 'none' in UI.custom_stages_html([], ['N4']) and 'changed in Advanced' in UI.custom_stages_html([], ['N4'])
    assert 'none registered' in UI.custom_stages_html([], [])


def test_advanced_box_is_closed_and_says_when_a_default_was_changed():
    import ipywidgets as widgets
    box, mark = UI.advanced_box([widgets.Text()])
    assert box.selected_index is None and box.get_title(0) == 'Advanced'
    mark(['output folder'])
    assert box.get_title(0) == f'Advanced  {UI.WARN} default changed: output folder'
    mark([])
    assert box.get_title(0) == 'Advanced'


# ---------------------------------------------------------------------------
# Guards: every page keeps to the conventions (no drift back to a page's own style)
# ---------------------------------------------------------------------------
def test_there_are_pages():
    assert len(PAGES) >= 20, [p.name for p in PAGES]


@pytest.mark.parametrize('page', PAGES, ids=lambda p: p.name)
def test_page_follows_the_conventions(page):
    found = cells(page)
    code = '\n'.join(text for kind, text in found if kind == 'code')
    assert found[0][0] == 'markdown' and found[0][1].startswith('# Tool '), 'the page must open on its "# Tool N: ..." title'
    assert re.search(r'^\s*import ui_lib as UI\b', code, re.M), 'the page must import ui_lib as UI'
    assert not re.search(r"button_style\s*=\s*['\"]info['\"]", code), "no 'info' button style: use UI.button and a role"
    for kind, text in found:
        if kind == 'markdown' and re.search(r'^## ', text, re.M):
            assert text.startswith(UI.SECTION_RULE + '\n\n## '), f'section cell without the thick rule: {text[:60]!r}'
