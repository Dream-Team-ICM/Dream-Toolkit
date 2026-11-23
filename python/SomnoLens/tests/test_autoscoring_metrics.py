"""Tool 3bis's agreement metrics (tools/plan_3bis_comparison.md): one function, autoscoring_lib.agreement_metrics, from a
confusion matrix, for a night and for nights pooled. Macro-F1 and balanced accuracy average over the stages the expert
scored; a night where the expert scored one stage only is flagged (its kappa and MCC are 0 or undefined whatever the
model). Known answers worked by hand, then equality with sklearn where every stage is present."""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             matthews_corrcoef)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
import autoscoring_lib as A     # noqa: E402

pytestmark = [pytest.mark.quick, pytest.mark.tool3bis, pytest.mark.tool0]
STAGES = A.STAGES


def labels(*runs):
    """('W', 3), ('N1', 2) -> ['W', 'W', 'W', 'N1', 'N1']"""
    return np.array([s for s, n in runs for _ in range(n)])


def test_a_nap_without_n3_and_rem():
    """The nap of 2026-11-23: the expert scored 30 W and 10 N1; the model gives 28 W and 2 N1 on the W epochs,
    8 N1, 1 W and 1 N2 on the N1 epochs."""
    expert = labels(('W', 30), ('N1', 10))
    model = labels(('W', 28), ('N1', 2), ('N1', 8), ('W', 1), ('N2', 1))
    res, cm, _ = A.compare_scoring(expert, model)
    assert res['n_compared'] == 40 and res['n_excluded_expert'] == 0
    assert res['accuracy'] == pytest.approx(0.9)
    assert res['f1_W'] == pytest.approx(56 / 59) and res['f1_N1'] == pytest.approx(0.8)
    assert res['f1_N2'] == 0                                   # given by the model, never scored by the expert
    assert np.isnan(res['f1_N3']) and np.isnan(res['f1_R'])    # absent from both sides
    # over the expert's two stages; the union of stages (sklearn's default) would give 0.583
    assert res['macro_f1'] == pytest.approx((56 / 59 + 0.8) / 2)
    assert res['balanced_accuracy'] == pytest.approx((28 / 30 + 8 / 10) / 2)
    assert res['n_stages_expert'] == 2 and res['single_stage_expert'] is False
    np.testing.assert_equal(res, {**A.agreement_metrics(cm), 'n_excluded_expert': 0})   # a night = its matrix


def test_a_single_stage_night_is_flagged():
    """The expert scored W only: kappa and MCC say nothing of the model, the other metrics are its share of W."""
    expert = labels(('W', 40))
    right = A.compare_scoring(expert, labels(('W', 40)))[0]
    assert right['single_stage_expert'] is True and right['n_stages_expert'] == 1
    assert np.isnan(right['kappa']) and right['mcc'] == 0      # undefined, and sklearn's 0
    assert right['accuracy'] == right['balanced_accuracy'] == right['macro_f1'] == 1
    wrong = A.compare_scoring(expert, labels(('W', 38), ('N1', 2)))[0]
    assert wrong['kappa'] == pytest.approx(0, abs=1e-12) and wrong['mcc'] == 0
    assert wrong['accuracy'] == pytest.approx(0.95) and wrong['balanced_accuracy'] == pytest.approx(0.95)
    assert wrong['macro_f1'] == pytest.approx(76 / 78)          # the F1 of W alone
    assert 'single stage here (W)' in A.metrics_html(wrong)
    assert 'single stage' not in A.metrics_html(A.compare_scoring(labels(('W', 5), ('N2', 5)),
                                                                  labels(('W', 5), ('N2', 5)))[0])


def test_epochs_outside_the_aasm_stages_and_a_longer_scoring():
    """MT, custom stages and '?' left out and counted; the comparison stops at the shorter hypnogram."""
    expert = np.array(['W', 'MT', 'N1', '?', 'N2', 'N2'])
    model = np.array(['W', 'W', 'N1', 'N1', 'N2', 'N3', 'R', 'R'])
    res = A.compare_scoring(expert, model)[0]
    assert res['n_compared'] == 4 and res['n_excluded_expert'] == 2
    assert res['accuracy'] == pytest.approx(0.75)


def test_no_comparable_epoch():
    res, cm, _ = A.compare_scoring(np.array(['MT', '?']), np.array(['W', 'W']))
    assert res['n_compared'] == 0 and cm.sum() == 0
    assert all(np.isnan(res[k]) for k in ('accuracy', 'kappa', 'mcc', 'macro_f1', 'balanced_accuracy'))
    assert res['single_stage_expert'] is False


def test_expert_labels_tool3_does_not_write_are_listed():
    """'Wake' / 'REM' (a hypnogram that skipped tool 3) are left out of the comparison like MT: listed for the
    warning, while MT, '?' and the declared custom stages are not."""
    expert = labels(('W', 3), ('Wake', 2), ('N2', 4), ('REM', 1), ('MT', 1), ('?', 1), ('N2a', 1))
    found = A.unexpected_stage_labels(expert, ['N2a'])
    assert found == {'Wake': 2, 'REM': 1}
    assert A.compare_scoring(expert, labels(('W', 13)))[0]['n_excluded_expert'] == 6
    note = A.unexpected_labels_note(found, len(expert))
    assert note.startswith("3 of the 13 expert epochs carry a label tool 3 does not write ('Wake' x2, 'REM' x1)")
    assert A.unexpected_stage_labels(labels(('W', 2), ('MT', 1), ('?', 1)), []) == {}
    assert A.unexpected_labels_note({}, 4) == ''


def test_equal_to_sklearn_when_every_stage_is_present():
    rng = np.random.default_rng(0)
    expert = rng.choice(STAGES, 500)
    model = np.where(rng.random(500) < 0.7, expert, rng.choice(STAGES, 500))
    res = A.compare_scoring(expert, model)[0]
    assert res['kappa'] == cohen_kappa_score(expert, model, labels=STAGES)
    assert res['mcc'] == pytest.approx(matthews_corrcoef(expert, model))
    assert res['macro_f1'] == pytest.approx(f1_score(expert, model, labels=STAGES, average='macro'))
    assert res['balanced_accuracy'] == pytest.approx(balanced_accuracy_score(expert, model))
    f1 = f1_score(expert, model, labels=STAGES, average=None)
    assert [res[f'f1_{s}'] for s in STAGES] == pytest.approx(list(f1))


def test_pooled_nights_are_their_summed_confusion_matrix():
    """Two nights pooled: the metrics of the summed matrix are those of the concatenated epochs."""
    rng = np.random.default_rng(1)
    nights = [(rng.choice(STAGES, n), rng.choice(STAGES, n)) for n in (120, 80)]
    cm = sum(confusion_matrix(e, g, labels=STAGES) for e, g in nights)
    pooled = A.agreement_metrics(cm)
    expert = np.concatenate([e for e, _ in nights])
    model = np.concatenate([g for _, g in nights])
    assert pooled['n_compared'] == 200
    assert pooled['kappa'] == pytest.approx(cohen_kappa_score(expert, model, labels=STAGES))
    assert pooled['mcc'] == pytest.approx(matthews_corrcoef(expert, model))
