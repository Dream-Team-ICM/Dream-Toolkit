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
