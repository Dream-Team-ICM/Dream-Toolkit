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
    '8bis': ['derivatives/rejection_auto/', 'reports_rejection_auto/'],
    '8': ['derivatives/rejection_manual/', 'reports_rejection_manual/'],
    '9': ['derivatives/features_spectral/', 'reports_features_spectral/'],
}


def test_every_golden_file_is_owned_by_a_tool(golden):
    """Guard for this test file: a golden output not mapped above would never be checked."""
    prefixes = [p for ps in TOOL_OUTPUTS.values() for p in ps]
    orphans = [f.relative_to(golden).as_posix() for f in golden.rglob('*') if f.is_file()
               and not any(f.relative_to(golden).as_posix().startswith(p) for p in prefixes)]
    assert not orphans, orphans


# Outputs whose values are MEANT to differ from the golden, each with its own test: {tool: [path prefixes]}.
# Empty since the goldens were taken again on 2026-11-19 (tool 7's band-pass phase): the former pre-refactor golden and
# the checks against it (the clean-epo copies, the 50-ms flat rule, tool 9's former route) are in the git history.
INTENDED_CHANGES = {}
# Outputs a change REMOVES on purpose: {tool: [path prefixes]}. Empty since 2026-11-19 (see above).
EXPECTED_MISSING = {}


# one test per tool, carrying that tool's marker (-m tool7 ...)
TOOL_PARAMS = [pytest.param(t, marks=getattr(pytest.mark, f'tool{t}')) for t in TOOL_OUTPUTS]


@pytest.mark.parametrize('tool', TOOL_PARAMS)
def test_outputs_match_golden(chain, golden, tool):
    report = snapshot.compare(chain['snapshot'], golden, prefixes=TOOL_OUTPUTS[tool],
                              ignore=INTENDED_CHANGES.get(tool, ()),
                              expect_missing=EXPECTED_MISSING.get(tool, ()))
    assert not report, f'tool {tool} outputs differ from golden:\n' + snapshot.format_report(report)


@pytest.mark.parametrize('tool', TOOL_PARAMS)
def test_no_traceback_in_logs(chain, tool):
    log = chain['logs'][tool]
    assert 'Traceback (most recent call last)' not in log, log[-3000:]
