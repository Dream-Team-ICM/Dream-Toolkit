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


# A line of a run log (print) in the colour of its symbol: ANSI codes, which Voila shows in colour. No tool writes its log
# to a file, so the codes never reach an output.
LOG_SYMBOLS = [('❌', 'error'), ('⚠', 'warn'), ('✅', 'ok')]     # the first one found in the line decides


def log(text, kind=None):
    """print(text) in the colour of `kind` (ok / warn / error), by default the kind of the first symbol of the line:
    ❌ red, ⚠ orange, ✅ green, anything else as print. The text itself is not changed."""
    text = str(text)
    if kind is None:
        found = [(text.find(sym), k) for sym, k in LOG_SYMBOLS if sym in text]
        kind = min(found)[1] if found else None
    if kind not in ('ok', 'warn', 'error'):
        print(text)
        return
    color = MESSAGE_STYLES[kind][0].lstrip('#')
    r, g, b = (int(color[i:i + 2], 16) for i in (0, 2, 4))
    print(f'\x1b[38;2;{r};{g};{b}m{text}\x1b[0m')


def log_html(html):
    """The same for a run log written as HTML lines (tools 5 and 9): the line in the colour of its first symbol."""
    found = [(html.find(sym), k) for sym, k in LOG_SYMBOLS if sym in html]
    if not found:
        return html
    return f'<span style="color:{MESSAGE_STYLES[min(found)[1]][0]}">{html}</span>'


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


# A Details block of a Markdown cell: a box with a bold title, so that it shows it can be opened. A Markdown cell cannot
# call Python: its opening line is written out, the one details_open returns (tests/test_ui_lib.py checks it).
DETAILS_BOX_STYLE = 'border:1px solid #bbb; border-radius:4px; background:#f7f7f7; padding:8px 12px; margin:10px 0;'
DETAILS_TITLE_STYLE = 'cursor:pointer; font-weight:bold;'
DETAILS_HINT_STYLE = f'font-size:85%; font-weight:normal; color:{MUTED_COLOR};'


def details_open(title, hint='click to open'):
    """The opening line of a Details block: <details> + its <summary>. Leave a blank line after it in a Markdown cell,
    and close the block with </details>."""
    return (f'<details style="{DETAILS_BOX_STYLE}"><summary style="{DETAILS_TITLE_STYLE}">{title} '
            f'<span style="{DETAILS_HINT_STYLE}">({hint})</span></summary>')


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
# Progress bars: blue while running, green when done, orange when some items failed. Every bar of every page has the
# same width, its label in a column of LABEL_WIDTH and its text on the right: two bars one under the other, and the
# legend of steps glued under a pipeline bar, line up by construction (the bar itself carries no label, whose inner
# margins shifted a legend placed by calculation).
# ---------------------------------------------------------------------------
PROGRESS_WIDTH = WIDE_WIDTH                                                     # label included
PROGRESS_TRACK = f'{int(PROGRESS_WIDTH[:-2]) - int(LABEL_WIDTH[:-2])}px'        # the bar itself
PROGRESS_TEXT_MIN = '180px'     # narrower than this beside the bar, the text goes under it (and wraps) instead of being cut
PHASE_COLORS = ['#e7ecff', '#eef7e7', '#fff4dc', '#ffeaea', '#f1e8fb', '#e6f4f6']   # the steps of a pipeline bar, in turn
LEGEND_CHAR_PX = 5.5            # width of a letter of the legend under a pipeline bar (9 px font), to keep each name whole
SHOWN_STEPS = 1000              # the resolution of the bar shown under a pipeline bar's legend

# A bar never stays still while a step works (SPEC *Voila page conventions -> Progress*): a step that loops moves its bar
# at each item, and the step being done shows this small turning sign beside its name. The browser animates it itself
# (SVG), so it keeps turning while Python is busy in one long call (an MNE read, a model's prediction).
SPINNER = ('<svg width="12" height="12" viewBox="0 0 12 12" style="vertical-align:-1px; margin-right:5px;">'
           '<circle cx="6" cy="6" r="4.5" fill="none" stroke="#c5cae9" stroke-width="2"/>'
           f'<path d="M6 1.5 A4.5 4.5 0 0 1 10.5 6" fill="none" stroke="{INFO_COLOR}" stroke-width="2">'
           '<animateTransform attributeName="transform" type="rotate" from="0 6 6" to="360 6 6" dur="0.9s" '
           'repeatCount="indefinite"/></path></svg>')
_STEP_BLOCK = '<div class="dtk-step" style="line-height:1.4;">'
# The text beside a bar: one size whatever the tool writes (a <small> in it is dropped), its first line centred on the bar
# (28 px high), so that the texts of two bars one under the other line up.
_BAR_TEXT_BLOCK = '<div class="dtk-step" style="line-height:1.4; padding-top:5px; font-size:12px;">'
_STEP_TEXTS = []                # the step texts of the page's phase bars: finish_progress stops their sign


def _progress_parts(description):
    """The label column, the bar and the text on its right, shared by both bars."""
    label = widgets.HTML(f'<div style="text-align:right; padding-right:8px;">{description}</div>',
                         layout=widgets.Layout(width=LABEL_WIDTH, min_width=LABEL_WIDTH, margin='0'))
    bar = widgets.IntProgress(value=0, min=0, max=1, bar_style='info',
                              layout=widgets.Layout(width=PROGRESS_TRACK, margin='0'))
    text = widgets.HTML('', layout=widgets.Layout(margin='0 0 0 8px', flex=f'1 1 {PROGRESS_TEXT_MIN}',
                                                  min_width=PROGRESS_TEXT_MIN))
    return label, bar, text


def _progress_row(children):
    """The row of a bar: when the page is too narrow for the text beside the bar, the text goes under it."""
    return widgets.HBox(children, layout=widgets.Layout(align_items='flex-start', flex_flow='row wrap'))


def progress_bar(description='Participants:', width=None):
    """A progress bar followed by its 'i / N' text. Returns (bar, row): set bar.max and bar.value as usual
    and put `row` in the page; the text follows the bar by itself. `width` is ignored (every bar has PROGRESS_WIDTH):
    kept for the frozen Curry twins, which still pass one, until their removal (tools/plan_brainvision.md, phase 6)."""
    label, bar, text = _progress_parts(description)

    def _update_text(_=None):
        text.value = f'{_BAR_TEXT_BLOCK}{bar.value} / {bar.max}</div>' if bar.value else ''

    bar.observe(_update_text, names=['value', 'max'])
    return bar, _progress_row([label, bar, text])


def stop_working():
    """Stops the turning sign of every step text (the run is over, or failed): the text stays."""
    for step_text in _STEP_TEXTS:
        if SPINNER in step_text.value:
            step_text.value = step_text.value.replace(SPINNER, '')


def working(text):
    """The HTML of a step at work, for a line replaced or cleared when the step ends: the turning sign, then `text`
    (HTML) in italics."""
    return f'{_STEP_BLOCK}{SPINNER}<i>{text}</i></div>'


def working_text(widget, block=_STEP_BLOCK):
    """Makes `widget` (a widget HTML saying what a step is doing) put the turning sign before any text written in it, as
    a block that wraps; a text ending with ✅ (done) gets no sign, '' empties it, finish_progress / stop_working stop it.
    `block` = the opening tag of that block (phase_bar gives _BAR_TEXT_BLOCK, whose text drops its <small>). Returns the
    widget."""
    _STEP_TEXTS.append(widget)

    def _on_step(change):
        new = change['new']
        if not new or new.startswith(block):
            return
        if block == _BAR_TEXT_BLOCK:
            new = new.replace('<small>', '').replace('</small>', '')
        done = new.rstrip().removesuffix('</small>').removesuffix('</i>').rstrip().endswith('✅')
        widget.value = f'{block}{"" if done else SPINNER}{new}</div>'

    widget.observe(_on_step, names='value')
    return widget


def _legend_widths(phases):
    """The share of a pipeline bar's legend each step takes: its cost, but at least the width of its name (short last
    steps lost theirs: 'Figur', 'HTM'). Fractions that sum to 1."""
    track_px = int(PROGRESS_TRACK[:-2])
    need = [(LEGEND_CHAR_PX * len(name) + 8) / track_px for name, _, _ in phases]
    costs = [max(float(cost), 0.0) for _, cost, _ in phases]
    fixed = {}                     # the steps given the width of their name
    while len(fixed) < len(phases):
        free = 1.0 - sum(fixed.values())
        cost_free = sum(c for k, c in enumerate(costs) if k not in fixed)
        narrow = {k: need[k] for k, c in enumerate(costs)
                  if k not in fixed and (cost_free <= 0 or free * c / cost_free < need[k])}
        if not narrow:
            break
        fixed.update(narrow)
    free = 1.0 - sum(fixed.values())
    cost_free = sum(c for k, c in enumerate(costs) if k not in fixed)
    widths = [fixed[k] if k in fixed else (free * c / cost_free if cost_free > 0 else 0.0) for k, c in enumerate(costs)]
    total = sum(widths)
    return [w / total for w in widths] if total > 0 else widths


def phase_bar(description='Pipeline:'):
    """The second bar of a run, under the progress_bar of the participants: one participant's steps, in units of time
    cost, with the legend of its steps glued under the bar (a segment per step, its width following its cost, at least
    the width of its name). Returns (bar, row, set_phases, step_text): set bar.max and bar.value as usual, call
    set_phases([(label, cost, tooltip), ...]) when the steps are known ([] hides the legend), and write the current step
    into step_text.value (HTML, on the right of the bar). The text written there gets the turning sign of a step at work
    (SPINNER) and wraps as a block; a text ending with ✅ (done) gets no sign, '' empties it, and finish_progress /
    stop_working stop it.

    `bar` counts in the tool's units and is not shown: the bar shown follows it on the widths of the legend, so that it
    reaches the end of a step's segment when the tool reaches the end of that step, even for a step widened to its name.
    Its bar_style follows too."""
    label, shown, step_text = _progress_parts(description)
    shown.max = SHOWN_STEPS
    bar = widgets.IntProgress(value=0, min=0, max=1, bar_style='info', layout=widgets.Layout(display='none'))
    legend = widgets.HTML('', layout=widgets.Layout(width=PROGRESS_TRACK, margin='0'))
    working_text(step_text, block=_BAR_TEXT_BLOCK)
    edges = {'cost': [0.0, 1.0], 'shown': [0.0, 1.0]}      # where each step starts and ends, in cost and on the legend

    def _follow(_=None):
        done = min(max(bar.value / bar.max, 0.0), 1.0) if bar.max else 0.0
        position = 1.0
        for k in range(1, len(edges['cost'])):
            if done <= edges['cost'][k]:
                c0, c1 = edges['cost'][k - 1], edges['cost'][k]
                s0, s1 = edges['shown'][k - 1], edges['shown'][k]
                position = s0 + (s1 - s0) * ((done - c0) / (c1 - c0) if c1 > c0 else 1.0)
                break
        shown.value = int(round(SHOWN_STEPS * position))
        shown.bar_style = bar.bar_style

    bar.observe(_follow, names=['value', 'max', 'bar_style'])

    def set_phases(phases):
        total = sum(max(float(cost), 0.0) for _, cost, _ in phases)
        widths = _legend_widths(phases) if phases and total > 0 else []
        cost_edges, shown_edges = [0.0], [0.0]
        for (_, cost, _), width in zip(phases, widths):
            cost_edges.append(cost_edges[-1] + max(float(cost), 0.0) / total)
            shown_edges.append(shown_edges[-1] + width)
        edges['cost'], edges['shown'] = (cost_edges, shown_edges) if widths else ([0.0, 1.0], [0.0, 1.0])
        segments = []
        for k, ((name, _, tip), width) in enumerate(zip(phases, widths)):
            border = '' if k == len(phases) - 1 else 'border-right:1px solid #bbb;'
            segments.append(f'<div title="{tip}" style="width:{100 * width:.1f}%;{border}text-align:center;'
                            f'background:{PHASE_COLORS[k % len(PHASE_COLORS)]};overflow:hidden;white-space:nowrap;">'
                            f'{name}</div>')
        legend.value = ('<div style="display:flex;width:100%;box-sizing:border-box;height:14px;font-size:9px;'
                        'line-height:14px;color:#444;border:1px solid #bbb;border-top:none;border-radius:0 0 3px 3px;'
                        'overflow:hidden;">' + ''.join(segments) + '</div>') if segments else ''
        _follow()

    row = _progress_row([label, widgets.VBox([shown, legend], layout=widgets.Layout(margin='0')), step_text])
    return bar, row, set_phases, step_text


def finish_progress(bar, n_failed=0):
    """Colour of a finished bar: green, or orange when some items failed. The turning signs of the page stop."""
    bar.bar_style = 'warning' if n_failed else 'success'
    stop_working()


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


# ---------------------------------------------------------------------------
# Recent studies: under the study folder's picker, the study folders picked lately and a field to paste a path
# ---------------------------------------------------------------------------
PASTE_PLACEHOLDER = r'e.g. D:\studies\my_study or \\server\share\my_study'


def recent_studies_row(picker, studies, warning=None):
    """The 'Recent studies' menu and the 'Or paste a path' field, put right under the study folder's `picker`. studies,
    warning = what study_files_lib.recent_studies() returns (the menu hidden when there is no study, the warning shown
    under it):  recent_row, recent_note = UI.recent_studies_row(fc_data, *SF.recent_studies())

    Choosing a study, or pasting a folder's path then 'Use', selects it in `picker` the way its dialog does: the tool's
    own callback runs (and remembers the folder). A folder that cannot be reached is refused with a message, the picker
    unchanged; nothing is checked before, a disconnected network share can take tens of seconds to answer.

    Returns (row, note): put `row` in the page; note(text) shows a warning under it (the warnings of study_files_lib),
    note(None) clears it."""
    options = [('(choose)', '')] + [(f'{s["path"]}  (tool {s.get("tool", "?")}, {str(s.get("last_used", ""))[:10]})',
                                     str(s['path'])) for s in studies]
    menu = widgets.Dropdown(description='Recent studies:', options=options, value='', style=LABEL_STYLE,
                            layout=widgets.Layout(width=WIDE_WIDTH, display='' if studies else 'none'))
    field = widgets.Text(description='Or paste a path:', placeholder=PASTE_PLACEHOLDER, style=LABEL_STYLE,
                         layout=widgets.Layout(width=FIELD_WIDTH))
    use = button('Use', 'helper', tooltip='Select this folder as the study folder')
    line = widgets.HTML('')
    state = {'busy': False}

    def note(text):
        line.value = msg('warn', text) if text else ''

    def _select(folder):
        if not os.path.isabs(folder):
            line.value = msg('error', f'paste the full path of the study folder: <code>{folder}</code>')
            return False
        if not os.path.isdir(folder):
            line.value = msg('error', f'not reachable: <code>{folder}</code> (a disconnected network drive? a file '
                                      'instead of a folder?)')
            return False
        line.value = ''
        try:
            picker.reset(path=folder, filename='')
            picker._show_dialog()          # then its own Select: the selection, then the tool's callback (ipyfilechooser 0.6)
            picker._on_select_click(None)
        except Exception as e:
            line.value = msg('error', f'could not select <code>{folder}</code>: {e}')
            return False
        return True

    def _on_menu(change):
        if state['busy'] or not change['new']:
            return
        _select(change['new'])
        state['busy'] = True               # back on '(choose)': the picker shows the folder, the same study can be chosen again
        try:
            menu.value = ''
        finally:
            state['busy'] = False

    def _on_use(_):
        # 'Copy as path' in the Windows Explorer puts the path between quotes
        if _select(field.value.strip().strip('"').strip()):
            field.value = ''

    menu.observe(_on_menu, names='value')
    use.on_click(_on_use)
    row = widgets.VBox([menu, widgets.HBox([field, use], layout=widgets.Layout(align_items='center')), line])
    note(warning)
    return row, note


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
