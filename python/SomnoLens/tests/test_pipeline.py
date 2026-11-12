"""The whole chain still produces the golden numbers (invariant 1 of the plan).

Each tool's outputs are compared to the golden snapshot taken before the change. Added columns and
added files are allowed (additive changes); a removed column/file or a changed value fails, with
the first differing value named in the message.
"""
import pytest

import snapshot

# Which output folders belong to which tool (relative to the data folder).
TOOL_OUTPUTS = {
    '5': ['derivatives/features_macrostructure/',
          'reports_features_macrostructure/global_', 'reports_features_macrostructure/sleep_'],
    '5bis': ['reports_features_macrostructure/participant_selection',
             'config_param/participant_exclusions.tsv'],     # the registry: in the chain, 5bis rows only
    '6': ['reports_quality_overview/'],
    '7': ['derivatives/preproc_epo/', 'reports_preproc/'],
    '8bis': ['derivatives/clean_epo_auto/', 'derivatives/rejection_auto/', 'reports_rejection_auto/'],
    '8': ['derivatives/clean_epo_manual/', 'derivatives/rejection_manual/', 'reports_rejection_manual/'],
    '9': ['derivatives/features_spectral/', 'reports_features_spectral/'],
}


def test_every_golden_file_is_owned_by_a_tool(golden):
    """Guard for this test file: a golden output not mapped above would never be checked."""
    prefixes = [p for ps in TOOL_OUTPUTS.values() for p in ps]
    orphans = [f.relative_to(golden).as_posix() for f in golden.rglob('*') if f.is_file()
               and not any(f.relative_to(golden).as_posix().startswith(p) for p in prefixes)]
    assert not orphans, orphans


# Outputs whose values are MEANT to differ from the pre-refactor golden, each with its own test:
INTENDED_CHANGES = {
    # pooled over the NON-excluded participants only (test_exclusion_tools.py)
    '7': ['reports_preproc/global_rejection_by_stage.tsv'],
    # the decision covers every epoch: n_epochs / pct_epochs_rejected now count the whole night (the
    # former in-scope counts moved to the *_ref columns), checked by
    # test_exclusion_tools.py::test_tool8bis_decision_tables_match_the_clean_epochs
    '8bis': ['reports_rejection_auto/'],
    # every epoch / channel now computed + decision columns: compared through the kept epochs of the
    # kept channels by test_tool9.py::test_tool9_reproduces_the_clean_epochs_route
    '9': ['derivatives/features_spectral/', 'reports_features_spectral/'],
    # flat_pct counts only what holds still for 50 ms (2026-11-12): every other value, and the exclude decision tool 7
    # reads, checked by test_tool6_matches_golden_but_the_flat_rule below
    '6': ['reports_quality_overview/'],
}
# Outputs the refactor REMOVES on purpose: the clean-epo copies (tool 9 now reads tool 7's epochs +
# the decision tables, checked against the golden clean-epo metadata in test_exclusion_tools.py)
EXPECTED_MISSING = {'8bis': ['derivatives/clean_epo_auto/'], '8': ['derivatives/clean_epo_manual/']}


# one test per tool, carrying that tool's marker (-m tool7 ...)
TOOL_PARAMS = [pytest.param(t, marks=getattr(pytest.mark, f'tool{t}')) for t in TOOL_OUTPUTS]


@pytest.mark.parametrize('tool', TOOL_PARAMS)
def test_outputs_match_golden(chain, golden, tool):
    report = snapshot.compare(chain['snapshot'], golden, prefixes=TOOL_OUTPUTS[tool],
                              ignore=INTENDED_CHANGES.get(tool, ()),
                              expect_missing=EXPECTED_MISSING.get(tool, ()))
    assert not report, f'tool {tool} outputs differ from golden:\n' + snapshot.format_report(report)


def _without_flat(reasons):
    """'bounds_pct=…; flat_pct=4.9% > 3.5%; hist_extreme…' without its flat_pct reason."""
    if not isinstance(reasons, str):
        return ''
    return '; '.join(r for r in reasons.split('; ') if not r.startswith('flat_pct'))


@pytest.mark.tool6
def test_tool6_matches_golden_but_the_flat_rule(chain, golden):
    """The 50-ms flat rule moves flat_pct (never up: it keeps a part of the still samples) and, where flat_pct crossed the
    threshold, that reason only. Every other value matches the golden, the exclude decision included, so tool 7 reads
    the same channels."""
    report = {}
    for old_path in sorted((golden / 'reports_quality_overview').rglob('*.tsv.gz')):
        rel = old_path.relative_to(golden).as_posix()
        new_path = chain['snapshot'] / rel
        if not new_path.exists():
            report[rel] = ['file missing in the new run']
            continue
        old, new = snapshot._read_table(old_path), snapshot._read_table(new_path)
        diffs = snapshot.compare_tables(new, old, ignore=('flat_pct', 'suspect_reason'))
        if len(new) == len(old):
            if (new['flat_pct'] > old['flat_pct'] + 1e-6).any():
                diffs.append('flat_pct went up')
            if 'suspect_reason' in old and (new['suspect_reason'].map(_without_flat)
                                            != old['suspect_reason'].map(_without_flat)).any():
                diffs.append('a reason other than flat_pct changed')
        if diffs:
            report[rel] = diffs
    assert not report, snapshot.format_report(report)


@pytest.mark.parametrize('tool', TOOL_PARAMS)
def test_no_traceback_in_logs(chain, tool):
    log = chain['logs'][tool]
    assert 'Traceback (most recent call last)' not in log, log[-3000:]
