"""The look of the SomnoLens Voila pages, in one place (SPEC *Cross-cutting -> Voila page conventions*).

Colours of the messages, section headers, buttons, field widths and progress bars, so that every tool looks
and reads the same, and the small widgets every page builds the same way (the suffix menus, the 'Inputs found'
list, the 'Advanced' box). No computation, nothing written to disk. The HTML reports keep their own style:
they are outputs, and an output must not change with the look of the page.

Imported by every Voila tool (and its Curry twin), like the other shared modules:
    import ui_lib as UI
    section1 = widgets.HTML(UI.section_html(1, 'Study folder', '<p>Pick the study folder.</p>'))
    btn_run = UI.button('Run', 'run')
    info.value = UI.msg('warn', '2 recordings have no hypnogram.')
    progress, progress_row = UI.progress_bar('Participants:')   # put progress_row in the page
"""
import os
from pathlib import Path

import ipywidgets as widgets

# ---------------------------------------------------------------------------
# Message colours: one colour per meaning
# ---------------------------------------------------------------------------
OK_COLOR = '#2e7d32'       # done, nothing to fix
WARN_COLOR = '#e65100'     # something to check, the run goes on
ERROR_COLOR = '#c62828'    # failed (kept apart from the REM red #c0392b of the figures)
INFO_COLOR = '#1565c0'     # neutral information
MUTED_COLOR = '#666'       # help text, details

# The same symbols in the HTML messages and in the print() logs
OK, WARN, ERROR, INFO = '✅', '⚠️', '❌', 'ℹ️'

# kind -> (text colour, pale background of a box, symbol)
MESSAGE_STYLES = {
    'ok': (OK_COLOR, '#e8f5e9', OK),
    'warn': (WARN_COLOR, '#fff3e0', WARN),
    'error': (ERROR_COLOR, '#ffebee', ERROR),
    'info': (INFO_COLOR, '#e3f2fd', INFO),
    'muted': (MUTED_COLOR, '#f5f5f5', ''),
}


def msg(kind, text, symbol=True):
    """One coloured line, e.g. msg('warn', '2 recordings have no hypnogram.'). `text` is HTML.
    kind = ok / warn / error / info / muted; symbol=False leaves the symbol out."""
    color, _, sym = MESSAGE_STYLES[kind]
    prefix = f'{sym} ' if symbol and sym else ''
    return f'<span style="color:{color}">{prefix}{text}</span>'


def box_style(kind):
    """The CSS of a boxed message, for a <div> built by hand (a box holding a table...)."""
    color, background, _ = MESSAGE_STYLES[kind]
    return (f'border-left:4px solid {color}; background:{background}; color:#222; '
            f'padding:6px 10px; margin:6px 0; border-radius:3px; line-height:1.4;')


def box(kind, text, symbol=True):
    """A boxed message (pale background, coloured left border, dark text), for long warnings and the
    summary at the end of a run. `text` is HTML."""
    sym = MESSAGE_STYLES[kind][2]
    prefix = f'{sym} ' if symbol and sym else ''
    return f'<div style="{box_style(kind)}">{prefix}{text}</div>'


def help_html(text):
    """A help line in small grey text (under a folder picker, a field...). `text` is HTML. A block with its own
    line height: in a widget HTML, a wrapped line would otherwise take a one-line field's height (28 px)."""
    return f'<div style="color:{MUTED_COLOR}; font-size:90%; line-height:1.4;">{text}</div>'


def text_block(html):
    """A long text of a widget HTML, as a block with a normal line height. The widget's own line height is a
    one-line field's (28 px, so that a short label lines up with the fields beside it): a text that wraps would
    otherwise leave a gap between its lines. Its colour and size stay the ones `html` sets."""
    return f'<div style="line-height:1.4;">{html}</div>'


def waiting_html(section):
    """What a section shows until the step it needs is done."""
    return msg('muted', f'Complete section {section} first.', symbol=False)


# ---------------------------------------------------------------------------
# Section headers: a thick black line above every section, a thin one above a sub-section
# ---------------------------------------------------------------------------
SECTION_RULE = '<hr style="height:4px; background-color:black; border:none;">'
SUBSECTION_RULE = '<hr style="height:1px; background-color:black; border:none;">'
# The CSS class of a rendered Markdown cell: a header built as a widget then looks exactly like one written in
# a Markdown cell (same title sizes, same text and `code` style)
MARKDOWN_CLASS = 'jp-RenderedHTMLCommon'


def markdown_html(html):
    """Any HTML shown by a widget, rendered like the text of a Markdown cell (titles, paragraphs, `code`)."""
    return f'<div class="{MARKDOWN_CLASS}">{html}</div>'


def section_html(number, title, intro=''):
    """Header of a page section: thick black line, 'N. Title', then the section's explanation (HTML).
    number=None for the unnumbered sections at the end of a page ('Outputs: where to look', 'References').
    The Markdown cells of the other tools start with the same lines (section_markdown)."""
    heading = f'{number}. {title}' if number is not None else title
    return markdown_html(f'{SECTION_RULE}\n<h2>{heading}</h2>\n{intro}')


def subsection_html(number, title, intro=''):
    """Header of a sub-section: thin black line, 'N.M Title' (number='2.1'), then its explanation (HTML).
    number=None for an unnumbered sub-section (an optional step)."""
    heading = f'{number} {title}' if number is not None else title
    return markdown_html(f'{SUBSECTION_RULE}\n<h3>{heading}</h3>\n{intro}')


def section_markdown(number, title):
    """The first lines of a section's Markdown cell (tools whose sections are Markdown cells)."""
    heading = f'{number}. {title}' if number is not None else title
    return f'{SECTION_RULE}\n\n## {heading}'


# ---------------------------------------------------------------------------
# Buttons: one style per role
# ---------------------------------------------------------------------------
# role -> (button_style, default icon)
BUTTON_ROLES = {
    'run': ('success', 'play'),       # the main action of a section: computes or writes
    'save': ('success', 'save'),      # saves a choice or a file
    'read': ('primary', 'search'),    # reads or lists, writes nothing (Scan, Load, Preview)
    'caution': ('warning', ''),       # changes many items at once, or excludes
    'helper': ('', ''),               # small buttons: navigation, tick all, add, delete
}


def button(description, role, icon=None, **kwargs):
    """A button styled by its role (see BUTTON_ROLES); its width follows its text. Other arguments
    (tooltip, disabled, layout...) go to widgets.Button."""
    style, default_icon = BUTTON_ROLES[role]
    layout = kwargs.pop('layout', widgets.Layout(width='max-content'))
    return widgets.Button(description=description, button_style=style,
                          icon=default_icon if icon is None else icon, layout=layout, **kwargs)


# ---------------------------------------------------------------------------
# Fields: a fixed label width, so that the fields of a section line up
# ---------------------------------------------------------------------------
LABEL_WIDTH = '170px'
LABEL_STYLE = {'description_width': LABEL_WIDTH}
FIELD_WIDTH = '460px'      # suffixes, file names (label included)
SHORT_WIDTH = '260px'      # numbers, thresholds (label included)
WIDE_WIDTH = '620px'       # long lists (label included)


def picker_title(name, detail=''):
    """Title of a folder / file picker: the name in bold and, on its own line under it, an optional detail in
    small grey text. The explicit line height matters: the title is a widget HTML whose line height is the
    height of a one-line field (28 px), which spread a wrapped detail far below the name. The picker is
    500 px wide: keep a detail under ~85 characters so it stays on one line."""
    extra = f'<br><span style="color:{MUTED_COLOR}; font-size:90%;">{detail}</span>' if detail else ''
    return f'<div style="line-height:1.4;"><b>{name}</b>{extra}</div>'


def log_output(max_height='400px'):
    """The frame of a run log: thin grey border, scrolls past max_height."""
    return widgets.Output(layout=widgets.Layout(border='1px solid #ccc', padding='2px',
                                                max_height=max_height, overflow='auto'))


# ---------------------------------------------------------------------------
# Progress bar: blue while running, green when done, orange when some items failed
# ---------------------------------------------------------------------------
def progress_bar(description='Participants:', width=FIELD_WIDTH):
    """A progress bar followed by its 'i / N' text. Returns (bar, row): set bar.max and bar.value as usual
    and put `row` in the page; the text follows the bar by itself. `width` (label included) lines it up with
    a second bar under it (the 'pipeline' bar of tools 5, 7, 9)."""
    bar = widgets.IntProgress(value=0, min=0, max=1, description=description, bar_style='info',
                              style=LABEL_STYLE, layout=widgets.Layout(width=width))
    text = widgets.Label('')

    def _update_text(_=None):
        text.value = f'{bar.value} / {bar.max}' if bar.value else ''

    bar.observe(_update_text, names=['value', 'max'])
    return bar, widgets.HBox([bar, text])


def finish_progress(bar, n_failed=0):
    """Colour of a finished bar: green, or orange when some items failed."""
    bar.bar_style = 'warning' if n_failed else 'success'


# ---------------------------------------------------------------------------
# Suffix menus: the companion suffixes found beside the recordings, with their number of files
# ---------------------------------------------------------------------------
OTHER_SUFFIX = 'Other suffix…'
MISSING_SHOWN = 10            # recordings named in the 'missing' line, then '… (+N)'


def suffix_menu(field, info):
    """A menu of the detected suffixes in front of a suffix text field. `field` stays the value the tool reads (a
    test, or the tool's own detection, may still set it: the menu follows); it shows only with 'Other suffix…',
    for a suffix typed by hand, and the menu stays hidden until a detection fills it. `info` (a widget HTML under
    the field) says how many recordings have the selected suffix and names the ones that miss it.

    Returns (menu, fill): put `menu` just before `field` in the page and, after each detection, call
    fill(owners, recordings, notes='') with owners = {suffix: [recording names]} (study_files_lib), the names of
    every recording, and optional HTML lines shown under the count (e.g. the notes of the hypnogram pre-choice).
    fill({}, []) empties it (no folder, nothing found)."""
    label = field.description
    menu = widgets.Dropdown(options=[(OTHER_SUFFIX, OTHER_SUFFIX)], value=OTHER_SUFFIX, description=label,
                            style={'description_width': field.style.description_width},
                            layout=widgets.Layout(width=field.layout.width, display='none'))
    state = {'owners': {}, 'recordings': [], 'notes': '', 'busy': False}

    def _match(text):
        key = os.path.normcase(text.strip())
        return next((s for s in state['owners'] if os.path.normcase(s) == key), None) if key else None

    def _show():
        menu_on = bool(state['owners'])
        menu.layout.display = '' if menu_on else 'none'
        field.description = 'Other suffix:' if menu_on else label
        field.layout.display = 'none' if (menu_on and menu.value != OTHER_SUFFIX) else ''
        n = len(state['recordings'])
        if not n or not field.value.strip():
            info.value = state['notes'].removeprefix('<br>')
            return
        have = {os.path.normcase(r) for r in state['owners'].get(_match(field.value), [])}
        missing = [r for r in state['recordings'] if os.path.normcase(r) not in have]
        if missing:
            shown = ', '.join(missing[:MISSING_SHOWN])
            if len(missing) > MISSING_SHOWN:
                shown += f', … (+{len(missing) - MISSING_SHOWN})'
            line = msg('warn', f'{n - len(missing)}/{n} recordings have it; missing: {shown}.')
        else:
            line = msg('ok', f'{n}/{n} recordings have it.')
        info.value = f'<small>{line}</small>' + state['notes']

    def _on_menu(change):
        if state['busy']:
            return
        if change['new'] != OTHER_SUFFIX:
            state['busy'] = True
            try:
                field.value = change['new']
            finally:
                state['busy'] = False
        _show()

    def _on_field(change):
        if state['busy']:
            return
        state['busy'] = True
        try:
            menu.value = _match(change['new']) or OTHER_SUFFIX
        finally:
            state['busy'] = False
        _show()

    def fill(owners, recordings, notes=''):
        state.update(owners=dict(owners), recordings=list(recordings), notes=notes)
        n = len(state['recordings'])
        state['busy'] = True
        try:
            menu.options = ([(f'{s}  ({len(r)}/{n})', s)
                             for s, r in sorted(state['owners'].items(), key=lambda x: (-len(x[1]), x[0]))]
                            + [(OTHER_SUFFIX, OTHER_SUFFIX)])
            menu.value = _match(field.value) or OTHER_SUFFIX
        finally:
            state['busy'] = False
        _show()

    menu.observe(_on_menu, names='value')
    field.observe(_on_field, names='value')
    return menu, fill


# ---------------------------------------------------------------------------
# The 'Inputs found' list: one line per input of a tool, found in the study folder or chosen by hand
# ---------------------------------------------------------------------------
CHOOSE_ANOTHER = 'Choose another…'
# the kind of the line of an input that is not found: required = the run refuses, recommended, optional
NEED_KIND = {'required': 'error', 'recommended': 'warn', 'optional': 'info'}


def input_line_html(kind, name, detail):
    """One line of the 'Inputs found' list: the symbol of `kind`, the input's name in bold (a fixed width, so the
    lines align) and `detail` (HTML)."""
    sym = MESSAGE_STYLES[kind][2]
    return (f'<div style="line-height:1.4;">{sym} <b style="display:inline-block; min-width:{LABEL_WIDTH};">'
            f'{name}</b> {detail}</div>')


def _shown_path(path, study):
    """A path as the list shows it: relative to the study folder when inside it, else in full."""
    try:
        rel = os.path.relpath(path, study) if study else str(path)
    except ValueError:            # another drive (Windows)
        rel = str(path)
    return str(path) if rel.startswith('..') else rel.replace(os.sep, '/')


def input_row(picker, name, need, absent, check=None, on_change=None):
    """One line of the 'Inputs found' list, for an input file the tool finds in the study folder: the file found
    (and how), or what is missing, then a 'Choose another…' button that shows the tool's own `picker` (a
    FileChooser, kept under its name: the run, and the tests, read its selection) for a file chosen by hand.

    need = 'required' (❌ when not found: the run refuses), 'recommended' (⚠️) or 'optional' (ℹ️); absent = what the
    line says after 'not found: ' (e.g. 'run tool 2 first'); check(path) -> (kind, text) = a light read of the file
    (e.g. ('ok', '11 participants') or ('warn', 'not a channel configuration')); on_change(picker) is called after
    every change of the selection, found or chosen by hand (the picker's own callback: do not register another).

    Returns (row, locate): put `row` in the page and, after the study folder is picked, call
    locate(study, path, how, folder) with what study_files_lib.find_input returned (path None when not found; a
    folder for a picker of folders). locate(None) empties it (no study folder). `note` replaces 'not found: …' when no
    file is the user's choice (e.g. 'none': every epoch is kept)."""
    line = widgets.HTML('')
    another = button(CHOOSE_ANOTHER, 'helper', tooltip=f'Pick the {name.lower()} by hand')
    # A FileChooser built without a layout shares ONE Layout object with every other one (ipyfilechooser's default
    # argument): this picker gets its own before it is hidden, or the study folder's picker would hide with it.
    picker.layout = widgets.Layout(width=picker.layout.width)
    picker.layout.display = 'none'
    row = widgets.VBox([widgets.HBox([line, another], layout=widgets.Layout(align_items='center')), picker],
                       layout=widgets.Layout(display='none'))
    state = {'study': None, 'auto': None, 'how': '', 'note': None}

    def _show():
        row.layout.display = '' if (state['study'] or picker.selected) else 'none'
        path = picker.selected
        if not path:
            kind = 'info' if state['note'] else NEED_KIND[need]
            line.value = input_line_html(kind, name, msg(kind, state['note'] or f'not found: {absent}', symbol=False))
            return
        same = state['auto'] is not None and (os.path.normcase(os.path.abspath(str(path)))
                                              == os.path.normcase(os.path.abspath(str(state['auto']))))
        how = state['how'] if same else 'chosen by hand'
        kind, read = 'ok', ''
        if check is not None:
            try:
                kind, read = check(path)
            except Exception as e:
                kind, read = 'warn', f'unreadable: {e}'
        detail = f'<code>{_shown_path(path, state["study"])}</code> {msg("muted", f"({how})", symbol=False)}'
        if read:
            detail += ' · ' + msg('muted' if kind == 'ok' else kind, read, symbol=False)
        line.value = input_line_html(kind, name, detail)

    def _changed():
        _show()
        if on_change is not None:
            on_change(picker)

    def _on_another(_):
        picker.layout.display = ''
        try:
            picker._show_dialog()      # open its file list at once (ipyfilechooser 0.6)
        except Exception:
            pass

    def locate(study, path=None, how='', folder=None, note=None):
        state.update(study=study, auto=path, how=how, note=note)
        picker.layout.display = 'none'
        if path:
            if picker.show_only_dirs:  # a picker of folders: the folder itself, no file name
                picker.reset(path=str(path), filename='')
            else:
                picker.reset(path=str(Path(path).parent), filename=Path(path).name)
            picker._apply_selection()  # selected without a click, without the callback (called below)
        elif folder or study:
            picker.reset(path=str(folder or study), filename='')
        else:
            picker.reset()
        _changed()

    another.on_click(_on_another)
    picker.register_callback(lambda _fc: _changed())
    return row, locate


def choice_menu(description, on_choose, min_shown=2):
    """A menu in front of an input row, when the study holds several candidates (several runs of tool 7, several
    decision folders). Hidden with fewer than `min_shown` options. Returns (menu, fill): fill(options) sets
    [(label, value)] and selects the first; on_choose(value) is called when the menu is changed, not by fill."""
    menu = widgets.Dropdown(description=description, options=[('', '')], value='', style=LABEL_STYLE,
                            layout=widgets.Layout(width=WIDE_WIDTH, display='none'))
    state = {'busy': False}

    def _on_change(change):
        if not state['busy']:
            on_choose(change['new'])

    def fill(options):
        state['busy'] = True
        try:
            menu.options = list(options) or [('', '')]
            menu.value = menu.options[0][1]
        finally:
            state['busy'] = False
        menu.layout.display = '' if len(options) >= min_shown else 'none'

    menu.observe(_on_change, names='value')
    return menu, fill


def inputs_list(children):
    """The 'Inputs found' list of Section 1: its title, then `children` (input rows, suffix menus, lines...). Returns
    (widget, show): put `widget` under the study folder's picker and call show(True) once the study folder is
    picked, show(False) when there is none. Until then a grey line says to pick it."""
    wait = widgets.HTML(msg('muted', 'Pick the study folder: the tool finds its inputs in it.', symbol=False))
    box_ = widgets.VBox([widgets.HTML(text_block('<b>Inputs found</b>'))] + list(children),
                        layout=widgets.Layout(display='none'))

    def show(on):
        box_.layout.display = '' if on else 'none'
        wait.layout.display = 'none' if on else ''

    return widgets.VBox([wait, box_]), show


def custom_stages_html(stages, loaded):
    """The custom stages line of the 'Inputs found' list: `stages` = the non-AASM stages the run will use (the field
    of the Advanced box), `loaded` = the ones read from config_param/custom_stages.json (tool 3)."""
    if list(stages) != list(loaded):
        return input_line_html('info', 'Custom stages', f'{", ".join(stages) or "none"} '
                               + msg('muted', '(changed in Advanced, for this run)', symbol=False))
    if stages:
        return input_line_html('ok', 'Custom stages', f'{", ".join(stages)} '
                               + msg('muted', '(<code>config_param/custom_stages.json</code>, tool 3)', symbol=False))
    return input_line_html('info', 'Custom stages', msg('muted', 'none registered: the AASM stages only', symbol=False))


# ---------------------------------------------------------------------------
# The 'Advanced' box: the settings a beginner leaves alone, closed by default
# ---------------------------------------------------------------------------
def advanced_box(children, title='Advanced'):
    """A closed box holding `children` (widgets). Returns (box, mark): call mark(changes) with the names of the
    settings that differ from their default ([] = none), so that a changed box says so in its title and is not
    forgotten: 'Advanced  ⚠️ default changed: output folder'."""
    inner = widgets.VBox(list(children))
    box_ = widgets.Accordion(children=[inner], selected_index=None, layout=widgets.Layout(width=WIDE_WIDTH))
    box_.set_title(0, title)

    def mark(changes):
        box_.set_title(0, f'{title}  {WARN} default changed: {", ".join(changes)}' if changes else title)

    return box_, mark
