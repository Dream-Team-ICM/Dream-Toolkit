"""
Tool 3bis step 2: compare the automatic scorings (step 1) with the expert hypnogram, the way python/Score_EDF did.
Its alignment (sleephd/alignment.py), recalibrations (sleephd/recalibration.py), agreement metrics and figures
(scoredf/plots.py) are ported here (SPEC §3bis). Reads only step 1's outputs and the
expert hypnogram (tool 3 format): no EDF.

The ported computation keeps Score_EDF's names inside (Wake, REM, the models' display names); the outputs follow the
toolkit (tools/plan_3bis_comparison.md, 2026-11-23): W / R, the models by step 1's keys, one epoch table and one
agreement table per night, the metrics of step 1 (autoscoring_lib.agreement_metrics) per night and pooled.
tests/score_edf_equivalence.py maps them back to compare them with Score_EDF. Changes from Score_EDF, all intended:
- an epoch that no 5-s step overlaps is left out of the recalibration fit instead of turning it to NaN (plan 2.2);
- the recalibrations fit N3 too when the training nights have N3 (Score_EDF fixed it: its naps had none; 2.7);
- the thread count of torch is restored after recalibration B (Score_EDF left it at 1; 2.6);
- for GSSC and YASA (30-s models) the predicted stage is the model's own stage, read from step 1 (Score_EDF
  took the argmax of the same probabilities: identical, except that step 1 stores them rounded to 4 decimals);
- the stage colours and the hypnogram layout of the rest of the toolkit (W on top, REM in red);
- an epoch a model gave no probability for is left out of the metrics and counted (n_not_predicted), as in step 1:
  Score_EDF counted it as a disagreement (accuracy) and as a sixth class (kappa);
- the recalibration parameters are returned, to be saved (Score_EDF only logged them).
"""
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.metrics import confusion_matrix

from edf_io_lib import draw_hypnogram     # the toolkit's hypnogram, one look everywhere (same folder)
# step 1's metrics, the same for a night of step 1 and for the comparison of the models (same folder)
from autoscoring_lib import STAGES, STAGE_COLORS as TOOLKIT_COLORS, agreement_metrics

# Score_EDF's stage names, kept in the output tables (one definition, plan 2.4)
PROBABILITY_COLUMNS = ['Wake', 'N1', 'N2', 'N3', 'REM']
STAGE_NAMES = dict(enumerate(PROBABILITY_COLUMNS))
# Expert hypnogram written by tool 3 (canonical labels). MT, custom stages and '?' are not scored stages.
EXPERT_TO_NUMBER = {'W': 0, 'WAKE': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 4, 'REM': 4}
# Step 1 writes the probabilities as p_W ... p_R
STEP1_TO_COLUMN = {'p_W': 'Wake', 'p_N1': 'N1', 'p_N2': 'N2', 'p_N3': 'N3', 'p_R': 'REM'}
STAGERS = ['YASA', 'GSSC', 'SleepFM', 'SleepFM recal A', 'SleepFM recal B']   # Score_EDF's order
NATIVE_30S = {'YASA', 'GSSC'}                                                   # models scoring 30-s epochs
# Stage colours of the toolkit (autoscoring_lib.STAGE_COLORS), under Score_EDF's stage names
STAGE_COLORS = {'Wake': '#969696', 'N1': '#9e9ac8', 'N2': '#807dba', 'N3': '#6a51a3', 'REM': '#c994c7'}
STAGER_COLORS = {'SleepFM': '#2a78d6', 'SleepFM recal A': '#008300', 'SleepFM recal B': '#7b3fb0',
                 'GSSC': '#eb6834', 'YASA': '#1baf7a'}
# The outputs use the toolkit's names: stages W / N1 / N2 / N3 / R, models by step 1's keys (Score_EDF's names stay
# inside the ported computation)
TOOLKIT_STAGE = {'Wake': 'W', 'N1': 'N1', 'N2': 'N2', 'N3': 'N3', 'REM': 'R'}
MODEL_KEY = {'YASA': 'yasa', 'GSSC': 'gssc', 'SleepFM': 'sleepfm', 'SleepFM recal A': 'sleepfm_recal_a',
             'SleepFM recal B': 'sleepfm_recal_b'}
MODEL_NAME = {key: name for name, key in MODEL_KEY.items()}


# ---------------------------------------------------------------------------------------------------------
# Expert hypnogram and analysis window
# ---------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Window:
    """Analysis window in seconds from the recording start: first to last scored expert epoch."""
    lights_off_seconds: float
    reveil_seconds: float
    lights_off_clock: str
    reveil_clock: str

    @property
    def duration_seconds(self):
        return self.reveil_seconds - self.lights_off_seconds


def read_expert(path):
    """The expert hypnogram (tool 3 format, one label per line) as Score_EDF's table: source_epoch,
    stage_token, stage_number (0-4, NaN when not a scored stage), stage."""
    tokens = [line.strip().upper() for line in open(path, encoding='utf-8-sig').read().splitlines()]
    numbers = [EXPERT_TO_NUMBER.get(t, np.nan) for t in tokens]
    return pd.DataFrame({
        'source_epoch': np.arange(len(tokens), dtype=int),
        'stage_token': tokens,
        'stage_number': pd.array(numbers, dtype='Float64'),
        'stage': [STAGE_NAMES.get(int(x)) if pd.notna(x) else 'Unknown' for x in numbers],
    })


def window_from_expert(hypnogram, start_dt=None, epoch_seconds=30.0):
    """From the first to the last scored expert epoch (Score_EDF's window_from_scoring). start_dt = the
    recording start (datetime or None), only used for the clock times shown in the log."""
    scored = hypnogram.loc[hypnogram['stage_number'].notna(), 'source_epoch']
    if scored.empty:
        raise ValueError('no scored epoch in the expert hypnogram')
    lights_off = int(scored.min()) * epoch_seconds
    reveil = (int(scored.max()) + 1) * epoch_seconds
    clock = (lambda s: (start_dt + pd.Timedelta(seconds=s)).strftime('%H:%M:%S')) if start_dt else (lambda s: '')
    return Window(lights_off, reveil, clock(lights_off), clock(reveil))


def make_epochs(window, epoch_seconds=30.0):
    """Epochs anchored on the window start, a final partial epoch kept."""
    count = math.ceil(window.duration_seconds / epoch_seconds)
    starts = window.lights_off_seconds + np.arange(count) * epoch_seconds
    stops = np.minimum(starts + epoch_seconds, window.reveil_seconds)
    return pd.DataFrame({'epoch': np.arange(count, dtype=int), 'start_seconds': starts, 'stop_seconds': stops,
                         'duration_seconds': stops - starts,
                         'minutes_from_lights_off': (starts - window.lights_off_seconds) / 60})


def align_manual_hypnogram(hypnogram, epochs, source_epoch_seconds=30.0):
    """The expert stage with the largest overlap in each epoch (Score_EDF, verbatim)."""
    source_starts = hypnogram['source_epoch'].to_numpy(dtype=float) * source_epoch_seconds
    source_stops = source_starts + source_epoch_seconds
    source_values = hypnogram['stage_number'].to_numpy(dtype=float, na_value=np.nan)
    aligned, coverage = [], []
    for row in epochs.itertuples():
        overlap = np.maximum(0.0, np.minimum(source_stops, row.stop_seconds)
                             - np.maximum(source_starts, row.start_seconds))
        valid = (~np.isnan(source_values)) & (overlap > 0)
        coverage.append(float(overlap[valid].sum() / row.duration_seconds))
        if not valid.any():
            aligned.append(np.nan)
            continue
        votes = {stage: overlap[valid & (source_values == stage)].sum() for stage in range(5)}
        aligned.append(float(max(votes, key=votes.get)))
    result = epochs.copy()
    result['manual_stage_number'] = aligned
    result['manual_stage'] = [STAGE_NAMES.get(int(v), 'Unknown') if pd.notna(v) else 'Unknown' for v in aligned]
    result['manual_coverage'] = coverage
    return result


def aggregate_hypnodensity(five_second, epochs):
    """Overlap-weighted mean of the model's probability rows (5 s or 30 s) in each epoch (Score_EDF, verbatim)."""
    source_starts = five_second['start_seconds'].to_numpy(float)
    source_stops = five_second['stop_seconds'].to_numpy(float)
    probabilities = five_second[PROBABILITY_COLUMNS].to_numpy(float)
    rows = []
    for epoch in epochs.itertuples():
        overlap = np.maximum(0.0, np.minimum(source_stops, epoch.stop_seconds)
                             - np.maximum(source_starts, epoch.start_seconds))
        total = overlap.sum()
        rows.append([np.nan] * 5 if total == 0 else (probabilities * overlap[:, None]).sum(axis=0) / total)
    result = epochs.copy()
    result[PROBABILITY_COLUMNS] = rows
    covered = result[PROBABILITY_COLUMNS].notna().all(axis=1)
    stage_number = pd.Series(pd.NA, index=result.index, dtype='Int64')
    stage_number.loc[covered] = result.loc[covered, PROBABILITY_COLUMNS].to_numpy().argmax(axis=1)
    result['predicted_stage_number'] = stage_number
    result['predicted_stage'] = stage_number.map(STAGE_NAMES).fillna('Unknown')
    return result


def build_epoch_table(window, hypnogram, hypnodensity, epoch_seconds=30.0, native_stages=None):
    """One row per epoch of the window: the expert stage, the model's probabilities and predicted stage.
    Score_EDF's columns (its 'tet' column, a nap-study measure, stays empty). native_stages = {onset_s: stage
    number} of a 30-s model: its own stage is the predicted one."""
    epochs = make_epochs(window, epoch_seconds)
    table = epochs.copy()
    table['tet'] = np.nan
    manual = align_manual_hypnogram(hypnogram, epochs, epoch_seconds)
    table[['manual_stage_number', 'manual_stage', 'manual_coverage']] = manual[
        ['manual_stage_number', 'manual_stage', 'manual_coverage']]
    predicted = aggregate_hypnodensity(hypnodensity, epochs)
    if native_stages is not None:
        own = [native_stages.get(round(float(s), 3)) for s in epochs['start_seconds']]
        mask = [o is not None for o in own]
        number = predicted['predicted_stage_number'].copy()
        number.loc[mask] = [o for o in own if o is not None]
        predicted['predicted_stage_number'] = number
        predicted['predicted_stage'] = number.map(STAGE_NAMES).fillna('Unknown')
    columns = PROBABILITY_COLUMNS + ['predicted_stage_number', 'predicted_stage']
    table[columns] = predicted[columns]
    return table


# ---------------------------------------------------------------------------------------------------------
# Step 1's outputs in Score_EDF's format
# ---------------------------------------------------------------------------------------------------------
def step1_frame(density_tsv, five_second_tsv=None):
    """A model's probability rows as Score_EDF used them (start_seconds, stop_seconds, Wake..REM) + its own
    30-s stages {onset_s: stage number} (None for SleepFM, whose stage is the argmax of the aggregated
    5-s probabilities, as in Score_EDF)."""
    if five_second_tsv is not None:
        d = pd.read_csv(five_second_tsv, sep='\t')
        frame = pd.DataFrame({'start_seconds': d['start_s'], 'stop_seconds': d['stop_s']})
        for src, dst in STEP1_TO_COLUMN.items():
            frame[dst] = d[src]
        return frame, None
    d = pd.read_csv(density_tsv, sep='\t', dtype={'file_id': str}, keep_default_na=False)
    frame = pd.DataFrame({'start_seconds': d['onset_s'].astype(float), 'stop_seconds': d['onset_s'].astype(float) + 30.0})
    for src, dst in STEP1_TO_COLUMN.items():
        frame[dst] = d[src].astype(float)
    stage_col = next(c for c in d.columns if c.startswith('stage_') and c != 'stage_expert')
    to_number = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 4}
    native = {round(float(o), 3): to_number[s] for o, s in zip(d['onset_s'], d[stage_col]) if s in to_number}
    return frame, native


# ---------------------------------------------------------------------------------------------------------
# Recalibration of SleepFM (Score_EDF's sleephd/recalibration.py), leave-one-recording-out
# ---------------------------------------------------------------------------------------------------------
@dataclass
class Recording:
    """What the recalibration needs from one recording."""
    subject: str
    five_second: pd.DataFrame   # start_seconds, stop_seconds, Wake..REM (SleepFM)
    log_p: np.ndarray           # (steps, 5) log of the 5-s probabilities
    weights: np.ndarray         # (epochs, steps) overlap weights, rows sum to 1 (0 when no step overlaps)
    labels: np.ndarray          # (epochs,) expert stage 0-4, -1 when unscored or not covered


def prepare_recording(subject, five_second, window, hypnogram, epoch_seconds=30.0):
    epochs = make_epochs(window, epoch_seconds)
    starts = five_second['start_seconds'].to_numpy(float)
    stops = five_second['stop_seconds'].to_numpy(float)
    overlap = np.maximum(0.0, np.minimum(stops[None], epochs['stop_seconds'].to_numpy()[:, None])
                         - np.maximum(starts[None], epochs['start_seconds'].to_numpy()[:, None]))
    total = overlap.sum(axis=1, keepdims=True)
    # An epoch no 5-s step overlaps has nothing to recalibrate: weight 0 and left out of the fit (Score_EDF
    # divided by 0 there, and the NaN broke the optimisation; plan 2.2).
    weights = np.where(total > 0, overlap / np.where(total > 0, total, 1.0), 0.0)
    manual = align_manual_hypnogram(hypnogram, epochs, epoch_seconds)['manual_stage_number']
    labels = manual.fillna(-1).to_numpy(int)
    labels[total[:, 0] == 0] = -1
    probabilities = five_second[PROBABILITY_COLUMNS].to_numpy(float)
    log_p = np.log(np.clip(probabilities, np.finfo(float).tiny, 1.0))
    return Recording(subject, five_second, log_p, weights, labels)


def free_stages(recordings, free_n3='auto'):
    """Stages whose bias / output row the recalibration fits. Wake is the reference. N3 is fitted only when
    the training nights contain N3 (free_n3='auto'): Score_EDF fixed it (free_n3=False) because its naps had
    no N3, which left nights' N3 uncalibrated (plan 2.7)."""
    if free_n3 == 'auto':
        free_n3 = any((r.labels == 3).any() for r in recordings)
    return [1, 2, 3, 4] if free_n3 else [1, 2, 4]


def _unpack(theta, free):
    bias = np.zeros(len(PROBABILITY_COLUMNS))
    bias[free] = theta[1:]
    return float(np.exp(theta[0])), bias


def apply_temperature(log_p, temperature, bias):
    return softmax(log_p / temperature + bias, axis=1)


def _negative_log_likelihood(theta, recordings, free):
    temperature, bias = _unpack(theta, free)
    total, count = 0.0, 0
    for rec in recordings:
        scored = rec.labels >= 0
        p30 = rec.weights[scored] @ apply_temperature(rec.log_p, temperature, bias)
        total -= np.log(np.clip(p30[np.arange(scored.sum()), rec.labels[scored]], 1e-12, 1)).sum()
        count += scored.sum()
    return total / count


def fit_temperature(recordings, free_n3='auto'):
    """Recalibration A: maximum-likelihood temperature and biases (starting from the identity). The
    recalibrated 5-s probability is softmax(log p / T + b): no SleepFM rerun needed."""
    free = free_stages(recordings, free_n3)
    result = minimize(_negative_log_likelihood, np.zeros(1 + len(free)), args=(recordings, free),
                      method='L-BFGS-B', bounds=[(-3, 3)] + [(-15, 15)] * len(free))
    temperature, bias = _unpack(result.x, free)
    return {'temperature': temperature, **{f'bias_{PROBABILITY_COLUMNS[i]}': bias[i] for i in free},
            'train_nll': float(result.fun), 'converged': bool(result.success), 'iterations': int(result.nit)}


def recalibrated_five_second(recording, parameters):
    bias = np.zeros(len(PROBABILITY_COLUMNS))
    for i, name in enumerate(PROBABILITY_COLUMNS):
        bias[i] = parameters.get(f'bias_{name}', 0.0)
    out = recording.five_second[['start_seconds', 'stop_seconds']].copy()
    out[PROBABILITY_COLUMNS] = apply_temperature(recording.log_p, parameters['temperature'], bias)
    return out


# --- Level B: re-fit SleepFM's final linear layer on its frozen BiLSTM features ----------------------------
LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


@dataclass
class HeadRecording:
    """Frozen BiLSTM output (input of the final layer) aligned to the epochs of one recording."""
    subject: str
    five_second: pd.DataFrame   # start_seconds, stop_seconds of each 5-s step
    features: np.ndarray        # (steps, 128)
    weights: np.ndarray
    labels: np.ndarray


def prepare_head_recording(base, features):
    return HeadRecording(base.subject, base.five_second[['start_seconds', 'stop_seconds']].copy(),
                         features, base.weights, base.labels)


def _stack(recordings):
    import torch
    features = torch.from_numpy(np.concatenate([r.features for r in recordings])).double()
    rows, cols, values, labels, offset, row = [], [], [], [], 0, 0
    for r in recordings:
        scored = np.flatnonzero(r.labels >= 0)
        block = r.weights[scored]
        i, j = np.nonzero(block)
        rows.append(i + row)
        cols.append(j + offset)
        values.append(block[i, j])
        labels.append(r.labels[scored])
        row += len(scored)
        offset += len(r.features)
    weights = torch.sparse_coo_tensor(np.vstack([np.concatenate(rows), np.concatenate(cols)]),
                                      np.concatenate(values), (row, offset)).double()
    return features, weights, torch.from_numpy(np.concatenate(labels)).long()


def _fc_nll(weight, bias, features, weights, labels):
    import torch
    p5 = torch.softmax(features @ weight.T + bias, dim=1)
    p30 = torch.sparse.mm(weights, p5)
    return -torch.log(p30[torch.arange(len(labels)), labels].clamp_min(1e-12)).mean()


def fit_fc(recordings, weight0, bias0, penalty, free_n3='auto'):
    """Penalised maximum likelihood of the final layer, starting from and shrunk toward the published one."""
    import torch
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        features, weights, labels = _stack(recordings)
        w0, b0 = torch.from_numpy(weight0).double(), torch.from_numpy(bias0).double()
        free = torch.zeros(5, dtype=torch.bool)
        free[[0] + free_stages(recordings, free_n3)] = True          # Wake's row is fitted too (Score_EDF)
        dw = torch.zeros_like(w0, requires_grad=True)
        db = torch.zeros_like(b0, requires_grad=True)
        optimizer = torch.optim.LBFGS([dw, db], max_iter=500, tolerance_grad=1e-7, line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            weight = w0 + dw * free[:, None]
            bias = b0 + db * free
            loss = _fc_nll(weight, bias, features, weights, labels) + penalty * (
                (dw[free] ** 2).sum() + (db[free] ** 2).sum())
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            return (w0 + dw * free[:, None]).numpy(), (b0 + db * free).numpy()
    finally:
        torch.set_num_threads(previous_threads)    # Score_EDF left torch on 1 thread (plan 2.6)


def heldout_nll(recordings, weight, bias):
    import torch
    features, weights, labels = _stack(recordings)
    with torch.no_grad():
        nll = _fc_nll(torch.from_numpy(weight).double(), torch.from_numpy(bias).double(), features, weights, labels)
    return float(nll) * len(labels), len(labels)


def fit_fc_nested(recordings, weight0, bias0, grid=LAMBDA_GRID, inner_folds=5, free_n3='auto'):
    """Choose the penalty by grouped inner cross-validation (log-loss), then refit on all."""
    groups = np.arange(len(recordings)) % inner_folds
    inner = {}
    for penalty in grid:
        total, count = 0.0, 0
        for g in range(inner_folds):
            train = [r for r, k in zip(recordings, groups) if k != g]
            test = [r for r, k in zip(recordings, groups) if k == g]
            weight, bias = fit_fc(train, weight0, bias0, penalty, free_n3)
            nll, n = heldout_nll(test, weight, bias)
            total, count = total + nll, count + n
        inner[penalty] = total / count
    best = min(inner, key=inner.get)
    weight, bias = fit_fc(recordings, weight0, bias0, best, free_n3)
    return {'penalty': best, 'inner_nll': inner, 'weight': weight, 'bias': bias}


def refit_five_second(recording, weight, bias):
    out = recording.five_second.copy()
    out[PROBABILITY_COLUMNS] = softmax(recording.features @ weight.T + bias, axis=1)
    return out


def recalibrate_a(base_recs, free_n3='auto', log=print):
    """Leave one recording out: each one's recalibrated SleepFM comes from the others' fit.
    Returns ({recording: recalibrated 5-s table}, [one parameter row per held-out recording])."""
    out, rows = {}, []
    for held_out in base_recs:
        train = [r for r in base_recs if r.subject != held_out.subject]
        params = fit_temperature(train, free_n3)
        out[held_out.subject] = recalibrated_five_second(held_out, params)
        log(f'  {held_out.subject}: T={params["temperature"]:.3f}, converged={params["converged"]}')
        # bias_N1 ... bias_R under the toolkit's stage names (Wake is the reference, no bias)
        named = {(f'bias_{TOOLKIT_STAGE[k[5:]]}' if k.startswith('bias_') else k): v for k, v in params.items()}
        rows.append({'method': 'A', 'file_id': held_out.subject,
                     'training_nights': ', '.join(r.subject for r in train), **named})
    return out, rows


def recalibrate_b(base_recs, features, fc_weight, fc_bias, free_n3='auto', log=print):
    """features = {subject: (steps, 128)}. Leave one recording out, penalty chosen by inner cross-validation.
    Returns ({recording: recalibrated 5-s table}, [one parameter row per held-out recording],
    {recording: (refitted final-layer weight (5, 128), bias (5,))})."""
    heads = [prepare_head_recording(r, features[r.subject]) for r in base_recs]
    inner_folds = min(5, len(heads) - 1) if len(heads) > 2 else len(heads) - 1
    out, rows, layers = {}, [], {}
    for held_out in heads:
        train = [h for h in heads if h.subject != held_out.subject]
        result = fit_fc_nested(train, fc_weight, fc_bias, inner_folds=max(2, inner_folds), free_n3=free_n3)
        out[held_out.subject] = refit_five_second(held_out, result['weight'], result['bias'])
        log(f'  {held_out.subject}: lambda={result["penalty"]}')
        rows.append({'method': 'B', 'file_id': held_out.subject,
                     'training_nights': ', '.join(h.subject for h in train), 'penalty': result['penalty'],
                     **{f'inner_nll_{p:g}': v for p, v in result['inner_nll'].items()}})
        layers[held_out.subject] = (result['weight'], result['bias'])
    return out, rows, layers


# ---------------------------------------------------------------------------------------------------------
# Outputs in the toolkit's layout: one epoch table per night, the agreement per night and pooled
# ---------------------------------------------------------------------------------------------------------
METRIC_COLUMNS = ['n_compared', 'accuracy', 'kappa', 'mcc', 'macro_f1', 'balanced_accuracy',
                  *[f'f1_{s}' for s in STAGES], 'n_stages_expert', 'single_stage_expert']


def toolkit_epochs(table, file_id, stager, epoch_seconds=30.0):
    """One model's epoch table (build_epoch_table) in the toolkit's layout: file_id, model, epoch_idx (counted from
    the recording start, like step 1's: the window starts on a whole expert epoch), onset_s, stage_expert,
    expert_coverage, stage_model, p_W ... p_R. A stage is '' where the expert scored no W/N1/N2/N3/R or where the
    model gave no probability."""
    out = pd.DataFrame({'file_id': file_id, 'model': MODEL_KEY[stager],
                        'epoch_idx': np.rint(table['start_seconds'] / epoch_seconds).astype(int),
                        'onset_s': table['start_seconds'],
                        'stage_expert': table['manual_stage'].map(TOOLKIT_STAGE).fillna(''),
                        'expert_coverage': table['manual_coverage'],
                        'stage_model': table['predicted_stage'].map(TOOLKIT_STAGE).fillna('')})
    for src, dst in TOOLKIT_STAGE.items():
        out[f'p_{dst}'] = table[src]
    return out


def _models_in(frame):
    """The models of a table, in Score_EDF's order."""
    present = set(frame['model'])
    return [m for m in MODEL_KEY.values() if m in present]


def is_true(column):
    """A True / False column read back from a TSV (bool or text)."""
    return column.astype(str).str.lower() == 'true'


def night_agreement(epochs):
    """One night's agreement with the expert, model by model, on the epochs where the expert scored W/N1/N2/N3/R and
    the model gave a stage. An epoch the model gave no probability for (the incomplete tail a 30-s model drops, or the
    last epoch gssc 0.0.9 drops, see SPEC §3bis) is left out and counted in n_not_predicted: Score_EDF counted it as
    a disagreement, step 1 compares on the epochs both scored. Returns (one row per model, the long confusion matrix
    model, expert, predicted, n)."""
    rows, confusion = [], []
    for model in _models_in(epochs):
        g = epochs[epochs['model'] == model]
        scored = g['stage_expert'].isin(STAGES)
        predicted = g['stage_model'].isin(STAGES)
        both = scored & predicted
        cm = (confusion_matrix(g.loc[both, 'stage_expert'], g.loc[both, 'stage_model'], labels=STAGES)
              if both.any() else np.zeros((5, 5), dtype=int))
        res = agreement_metrics(cm)
        rows.append({'model': model, 'n_compared': res['n_compared'], 'n_excluded_expert': int((~scored).sum()),
                     'n_not_predicted': int((scored & ~predicted).sum()), **{c: res[c] for c in METRIC_COLUMNS}})
        confusion += [{'model': model, 'expert': e, 'predicted': p, 'n': int(cm[i, j])}
                      for i, e in enumerate(STAGES) for j, p in enumerate(STAGES)]
    return pd.DataFrame(rows), pd.DataFrame(confusion)


def summed_confusion(confusion, model):
    """One model's 5 x 5 confusion matrix (rows expert, columns model), summed over the nights of `confusion`."""
    g = confusion[confusion['model'] == model].groupby(['expert', 'predicted'])['n'].sum()
    return g.unstack().reindex(index=STAGES, columns=STAGES).fillna(0).astype(int).to_numpy()


def pooled_agreement(confusion, summary):
    """Every night pooled, model by model: the metrics of the summed confusion matrix (the same function as a
    night's), the number of nights and the epochs left out. `summary` = the per-night rows."""
    rows = []
    for model in _models_in(confusion):
        nights = summary[summary['model'] == model]
        res = agreement_metrics(summed_confusion(confusion, model))
        rows.append({'model': model, 'n_nights': len(nights), 'n_compared': res['n_compared'],
                     'n_excluded_expert': int(pd.to_numeric(nights['n_excluded_expert']).sum()),
                     'n_not_predicted': int(pd.to_numeric(nights['n_not_predicted']).sum()),
                     **{c: res[c] for c in METRIC_COLUMNS if c != 'single_stage_expert'}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------
# Figures (Score_EDF's scoredf/plots.py, the toolkit's names and colours)
# ---------------------------------------------------------------------------------------------------------
def _manual_hypnogram(ax, minutes, stage_names):
    """Expert hypnogram: the toolkit's hypnogram (edf_io_lib.draw_hypnogram), Score_EDF's 'Wake' / 'REM' read as W / R.
    `minutes` = the start of each epoch."""
    x = np.asarray(minutes, float)
    step = np.median(np.diff(x)) if len(x) > 1 else 0.5
    edges = np.append(x, x[-1] + step) if len(x) else np.array([0.0])
    labels = [{'Wake': 'W', 'REM': 'R'}.get(str(s), str(s)) for s in stage_names]
    draw_hypnogram(ax, edges, labels, fontsize=8)


def _stage_legend():
    return [Patch(color=TOOLKIT_COLORS[s], label=s) for s in STAGES]


def plot_hypnodensities(epochs, file_id):
    """One night: each model's hypnodensity (stacked probabilities) above the expert hypnogram."""
    night = epochs[epochs['file_id'] == file_id]
    models = _models_in(night)
    n_panels = len(models) + 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(13, 1.4 * n_panels), sharex=True,
                             gridspec_kw={'height_ratios': [1] * len(models) + [1.1]})
    start = night['onset_s'].min()
    for ax, model in zip(axes, models):
        g = night[night['model'] == model]
        minutes = (g['onset_s'] - start) / 60
        ax.set_ylabel(MODEL_NAME[model], rotation=0, ha='right', va='center', fontsize=9)
        ax.stackplot(minutes, *[g[f'p_{s}'] for s in STAGES], colors=[TOOLKIT_COLORS[s] for s in STAGES],
                     step='post', alpha=0.9)
        ax.set(ylim=(0, 1), yticks=[0, 1], xlim=(0, minutes.max()))
    first = night[night['model'] == models[0]]
    _manual_hypnogram(axes[-1], (first['onset_s'] - start) / 60, first['stage_expert'])
    axes[-1].set_ylabel('Expert', rotation=0, ha='right', va='center', fontsize=9)
    axes[-1].set_xlabel('Minutes from the first scored epoch')
    axes[0].legend(handles=_stage_legend(), ncol=5, loc='lower center', bbox_to_anchor=(0.5, 1.02), frameon=False,
                   fontsize=8)
    fig.suptitle(f'{file_id}: hypnodensity comparison', y=1.0, fontsize=11)
    fig.tight_layout()
    return fig


def plot_confusion_matrices(confusion, title):
    """One confusion matrix per model (`confusion`: model, expert, predicted, n; one night, or several summed):
    % of each expert stage (row)."""
    models = _models_in(confusion)
    cols = min(len(models), 4)
    rows = (len(models) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)
    greens = plt.cm.Greens
    for idx, model in enumerate(models):
        ax = axes.flat[idx]
        ax.set_title(MODEL_NAME[model])
        cm = summed_confusion(confusion, model)
        present = cm.sum(axis=1) > 0
        if not present.any():
            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center', color='grey')
            continue
        labels = [s for s, p in zip(STAGES, present) if p]
        pct = cm[present] / cm[present].sum(axis=1, keepdims=True) * 100
        ax.imshow(pct, cmap=greens, vmin=0, vmax=100, aspect='auto')
        for i in range(pct.shape[0]):
            for j in range(pct.shape[1]):
                r, gg, b, _ = greens(pct[i, j] / 100)
                color = 'white' if 0.2126 * r + 0.7152 * gg + 0.0722 * b < 0.5 else '#222222'
                ax.text(j, i, f'{pct[i, j]:.1f}', ha='center', va='center', fontsize=8, color=color)
        ax.set_xticks(range(len(STAGES)))
        ax.set_xticklabels(STAGES, fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set(xlabel='Predicted', ylabel='Expert')
    for idx in range(len(models), axes.size):
        axes.flat[idx].axis('off')
    fig.suptitle(title, y=1.01, fontsize=11)
    fig.tight_layout()
    return fig


def plot_agreement_summary(pooled, summary, single_stage_out=True):
    """F1 per stage with every night pooled (left) and the kappa of each night (right). single_stage_out: a night
    where the expert scored a single stage is left out of the right panel (its kappa is 0 whatever the model)."""
    models = _models_in(pooled)
    names = [MODEL_NAME[m] for m in models]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={'width_ratios': [1.3, 1]})
    f1 = pooled.set_index('model').loc[models, [f'f1_{s}' for s in STAGES]].astype(float).dropna(axis=1, how='all')
    f1.index = names
    f1.columns = [c.removeprefix('f1_') for c in f1.columns]
    f1.T.plot.bar(ax=axes[0], color=[STAGER_COLORS.get(n, '#888888') for n in names], width=0.8, rot=0)
    axes[0].set(ylim=(0, 1), ylabel='F1 (all nights pooled)', title='Agreement per stage')
    axes[0].legend(title='', ncol=min(4, len(names)), fontsize=7, loc='lower center',
                   bbox_to_anchor=(0.5, 1.05), frameon=False)
    nights = summary[~is_true(summary['single_stage_expert'])] if single_stage_out else summary
    file_ids = sorted(nights['file_id'].astype(str).unique())
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '+']
    low = 0.0
    for i, fid in enumerate(file_ids):
        k = (nights[nights['file_id'].astype(str) == fid].set_index('model')['kappa'].reindex(models)
             .astype(float))
        low = min(low, np.nanmin(k.values)) if np.isfinite(k.values).any() else low
        axes[1].plot(range(len(models)), k.values, color='#8a8a8a', lw=0.8, marker=markers[i % len(markers)], ms=7,
                     label=fid)
    axes[1].scatter(range(len(models)), pooled.set_index('model').loc[models, 'kappa'].astype(float), marker='_',
                    s=700, color='black', zorder=3, label='pooled')
    axes[1].set(xticks=range(len(models)), ylim=(low - 0.05 if low < 0 else 0, 1), ylabel="Cohen's kappa",
                title='Agreement per night')
    axes[1].set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    if len(file_ids) <= 12:                       # beyond, a legend of every night hides the figure
        axes[1].legend(title='', fontsize=7)
    fig.tight_layout()
    return fig


def plot_hypnogram_grid(epochs, file_ids):
    """Colour-coded epochs: the expert then each model, one panel per night (grey = no stage)."""
    stage_code = {s: i for i, s in enumerate(STAGES)}
    cmap = ListedColormap([TOOLKIT_COLORS[s] for s in STAGES]).with_extremes(bad='#e0e0e0')
    fig, axes = plt.subplots(len(file_ids), 1, figsize=(13, 2.6 * len(file_ids)))
    axes_arr = np.atleast_1d(axes)
    for ax, fid in zip(axes_arr, file_ids):
        g = epochs[epochs['file_id'].astype(str) == fid]
        models = _models_in(g)
        if not models:
            ax.text(0.5, 0.5, f'{fid}: no model output', transform=ax.transAxes, ha='center')
            continue
        manual = g[g['model'] == models[0]].set_index('epoch_idx')['stage_expert'].map(stage_code)
        predicted = (g.assign(code=g['stage_model'].map(stage_code))
                     .pivot(index='model', columns='epoch_idx', values='code').reindex(models))
        image = np.vstack([manual.to_numpy(float), predicted.to_numpy(float)])
        ax.imshow(np.ma.masked_invalid(image), aspect='auto', cmap=cmap, vmin=-0.5, vmax=4.5,
                  interpolation='nearest', extent=(0, image.shape[1] / 2, image.shape[0], 0))
        ax.set(yticks=np.arange(image.shape[0]) + 0.5, yticklabels=['Expert', *[MODEL_NAME[m] for m in models]],
               title=fid)
        ax.axhline(1, color='white', lw=3)
        ax.grid(False)
    axes_arr[-1].set_xlabel('Minutes from the first scored epoch')
    axes_arr[0].legend(handles=_stage_legend(), ncol=5, loc='lower right', bbox_to_anchor=(1, 1.02), frameon=False,
                       fontsize=8)
    fig.tight_layout()
    return fig
