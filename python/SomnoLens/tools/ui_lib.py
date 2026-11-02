"""The look of the SomnoLens Voila pages, in one place (SPEC *Cross-cutting -> Voila page conventions*).

Colours of the messages, section headers, buttons, field widths and progress bars, so that every tool looks
and reads the same. Only the look: no computation, no state, nothing written to disk. The HTML reports keep
their own style: they are outputs, and an output must not change with the look of the page.

Imported by every Voila tool (and its Curry twin), like the other shared modules:
    import ui_lib as UI
    section1 = widgets.HTML(UI.section_html(1, 'Study folder', '<p>Pick the study folder.</p>'))
    btn_run = UI.button('Run', 'run')
    info.value = UI.msg('warn', '2 recordings have no hypnogram.')
    progress, progress_row = UI.progress_bar('Participants:')   # put progress_row in the page
"""
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
