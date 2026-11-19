"""Tool 9 after step 4: every epoch and channel computed from tool 7's *_all-epo.fif + the decision tables.

1. Every epoch / channel is in the tables, with the decision columns; excluded participants flagged.
2. 'Re-aggregate only' after a decision change equals a full recomputation with that decision.

Its values are compared to the golden by test_pipeline.py. The former check against the route before the exclusion
refactor (tool 9 on the clean-epo written by 8bis, equal on the kept epochs of the kept channels) retired with that
golden on 2026-11-19: see the git history.
"""
import json
import shutil
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

import minidb
import pipeline
import snapshot

pytestmark = pytest.mark.tool9

EPOCH_TABLES = ['bandpower_epoch', 'aperiodic_epoch', 'periodic_peaks']
STAGE_TABLES = ['spectral_stage', 'aperiodic_stage', 'psd_stage']


def tsv(path):
    return pd.read_csv(path, sep='\t', dtype={'file_id': str})


def same(new, old, ignore=('source',), atol=1e-9):
    # the power minus the aperiodic fit gets its own tolerance there (snapshot.AP_REMOVED_ATOL_DB)
    return snapshot.compare_tables(new.reset_index(drop=True), old.reset_index(drop=True), atol=atol, ignore=ignore)


def test_tool9_computes_every_epoch_and_channel(chain):
    data = chain['data']
    p = minidb.PARTICIPANTS[chain['dataset']]
    feat, dec = data / 'derivatives' / 'features_spectral', data / 'derivatives' / 'rejection_auto'
    for fid in p:
        ep = mne.read_epochs(str(data / 'derivatives' / 'preproc_epo' / f'{fid}_all-epo.fif'),
                             preload=False, verbose=False)
        ed = tsv(dec / f'{fid}_epoch_decision.tsv').set_index('epoch_idx')
        cd = tsv(dec / f'{fid}_channel_decision.tsv')
        dropped = set(cd.loc[cd['dropped'].astype(bool), 'channel']) | set(ep.info['bads'])
        ap = tsv(feat / f'{fid}_aperiodic_epoch.tsv')
        assert len(ap) == len(ep) * len(ep.ch_names)                      # every epoch x channel
        assert set(ap['channel']) == set(ep.ch_names)
        by_epoch = ap.drop_duplicates('epoch_idx').set_index('epoch_idx')
        assert (by_epoch['in_scope'] == ed['in_scope'].reindex(by_epoch.index)).all()
        assert (by_epoch['rejected'] == ed['rejected'].reindex(by_epoch.index)).all()
        assert set(ap.loc[ap['channel_dropped'], 'channel']) == dropped
        if dropped:                                                        # dropped: still computed
            assert ap.loc[ap['channel'].isin(dropped), 'exponent'].notna().any()
        z = np.load(feat / f'{fid}_psd_epoch.npz')
        assert z['psd'].shape[:2] == (len(ep), len(ep.ch_names))
        st = tsv(feat / f'{fid}_spectral_stage.tsv')
        assert {'channel_dropped', 'n_epochs_rejected', 'n_epochs_out_of_scope'} <= set(st.columns)
    g = tsv(data / 'reports_features_spectral' / 'global_spectral_stage.tsv')
    assert g.groupby('file_id')['participant_excluded'].first().to_dict() == {
        p[0]: False, p[1]: True, p[2]: False, p[3]: False}


@pytest.fixture
def data_copy(chain, tmp_path):
    dst = tmp_path / 'data'
    shutil.copytree(chain['data'], dst)
    return dst


def test_tool9_reaggregate_equals_a_full_run(chain, data_copy):
    """Reject 25 more epochs in the 8bis decision of one participant: 'Re-aggregate only' (no PSD, no
    fit) must give the same tables as recomputing everything with that decision."""
    fid = minidb.PARTICIPANTS[chain['dataset']][0]
    feat = data_copy / 'derivatives' / 'features_spectral'
    dec_path = data_copy / 'derivatives' / 'rejection_auto' / f'{fid}_epoch_decision.tsv'
    ed = tsv(dec_path)
    newly = ed.index[ed['in_scope'] & ~ed['rejected']][:25]
    ed.loc[newly, 'rejected'] = True
    ed.loc[newly, 'reject_reason'] = 'test: rejected by hand'
    ed.to_csv(dec_path, sep='\t', index=False)
    before = tsv(feat / f'{fid}_spectral_stage.tsv')

    raw_dir, dec_dir = data_copy / 'derivatives' / 'preproc_epo', data_copy / 'derivatives' / 'rejection_auto'
    log = pipeline.run_tool9(data_copy, raw_dir, dec_dir, subset=[fid], reaggregate=True)
    assert 'Re-aggregate only' in log and '1 processed' in log and 'failed:' not in log, log[-2000:]
    reagg = {n: tsv(feat / f'{fid}_{n}.tsv') for n in EPOCH_TABLES + STAGE_TABLES + ['spectral_stage_third']}
    assert reagg['spectral_stage']['n_epochs'].sum() < before['n_epochs'].sum()   # the change took effect
    assert reagg['spectral_stage']['n_epochs_rejected'].sum() > before['n_epochs_rejected'].sum()

    log = pipeline.run_tool9(data_copy, raw_dir, dec_dir, subset=[fid], skip=False)  # full recomputation
    assert '1 processed' in log and 'failed:' not in log, log[-2000:]
    for name, a in reagg.items():
        b = tsv(feat / f'{fid}_{name}.tsv')
        # the per-epoch spectrum is stored in float32: psd_stage agrees to ~1e-7 relative
        diffs = same(a, b, ignore=(), atol=1e-5)
        assert not diffs, f'{name}: ' + '; '.join(diffs)


def test_tool9_night_thirds_count_their_own_epochs(chain):
    """n_epochs_rejected / n_epochs_out_of_scope / pct_rejected of a stage x third cell are THAT cell's,
    recounted here from the per-epoch table (which carries `third`, `in_scope`, `rejected`)."""
    feat = chain['data'] / 'derivatives' / 'features_spectral'
    for fid in minidb.PARTICIPANTS[chain['dataset']]:
        th = tsv(feat / f'{fid}_spectral_stage_third.tsv')
        ep = tsv(feat / f'{fid}_aperiodic_epoch.tsv').drop_duplicates('epoch_idx')
        for (stage, third), row in th.drop_duplicates(['stage', 'third']).set_index(['stage', 'third']).iterrows():
            cell = ep[(ep['stage'] == stage) & (ep['third'] == third)]
            n_rej = int((cell['in_scope'] & cell['rejected']).sum())
            n_kept = int((cell['in_scope'] & ~cell['rejected']).sum())
            assert row['n_epochs_rejected'] == n_rej, (fid, stage, third)
            assert row['n_epochs_out_of_scope'] == int((~cell['in_scope']).sum())
            if n_kept + n_rej == 0:          # a stage with no epoch at all in this third: padded row
                assert pd.isna(row['pct_rejected']), (fid, stage, third)
            else:
                assert row['pct_rejected'] == pytest.approx(100.0 * n_rej / (n_kept + n_rej), abs=0.01)
