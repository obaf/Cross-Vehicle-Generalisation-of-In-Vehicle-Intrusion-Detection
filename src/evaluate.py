"""
evaluate.py -- imbalance- and false-alarm-aware metrics.

Reported for every (detector, fold):

  * PR-AUC (average precision) and ROC-AUC: threshold-free ranking quality.
  * Recall at a false-alarm budget of 1, 10 and 100 alarms per driving hour,
    with the threshold set on held-out BENIGN windows of the target vehicle
    only. Each budget is tagged with whether the calibration set can resolve
    it at all: a budget that allows fewer than one calibration window to fire
    cannot be estimated, and recall reported there is decided by the single
    most extreme calibration window. The prior run used 1/h as its primary
    operating point although its own check marked it unresolvable on every
    fold; here the primary is 10/h and 1/h is reported with its flag.
  * Recall at a fixed benign false-positive rate (1e-3 and 1e-2 of windows),
    which is invariant to window duration and class ratio.
  * Per-attack-family recall and per-capture EVENT recall at every operating
    point. Event recall asks whether at least one window of an attack capture
    fires; it is the operationally relevant statistic for attacks such as
    suspension, where the label covers an interval but the evidence may be
    concentrated in a few windows of it.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

FA_BUDGETS = (1.0, 10.0, 100.0)
FPR_TARGETS = (1e-3, 1e-2)
PRIMARY = "fa10"


def budget_resolvable(n_calib, win_seconds, fa_per_hour):
    if n_calib <= 0:
        return False
    allowed = min(max(fa_per_hour * win_seconds / 3600.0, 0.0), 1.0)
    return allowed * n_calib >= 1.0


def _tie_aware_threshold(s_sorted, k):
    """Lowest threshold whose firing count on the sorted calibration scores is
    <= k, stepping through tied values."""
    n = s_sorted.size
    if k <= 0:
        return float(np.nextafter(s_sorted[-1], np.inf))
    thr = float(s_sorted[n - k])
    if int(np.count_nonzero(s_sorted >= thr)) <= k:
        return thr
    uniq = np.unique(s_sorted)
    idx = int(np.searchsorted(uniq, thr, side="right"))
    while idx < uniq.size and int(np.count_nonzero(s_sorted >= uniq[idx])) > k:
        idx += 1
    if idx >= uniq.size:
        return float(np.nextafter(s_sorted[-1], np.inf))
    return float(uniq[idx])


def threshold_for_fa_rate(calib, win_seconds, fa_per_hour):
    s = np.asarray(calib, dtype=np.float64)
    s = np.sort(s[np.isfinite(s)])
    if s.size == 0:
        return np.inf
    allowed = min(max(fa_per_hour * win_seconds / 3600.0, 0.0), 1.0)
    return _tie_aware_threshold(s, int(np.floor(allowed * s.size)))


def threshold_for_fpr(calib, fpr):
    s = np.asarray(calib, dtype=np.float64)
    s = np.sort(s[np.isfinite(s)])
    if s.size == 0:
        return np.inf
    return _tie_aware_threshold(s, int(np.floor(fpr * s.size)))


def _at(scores, y, fam, cap, thr, tag, win_seconds, out):
    pred = scores >= thr
    pos, neg = y == 1, y == 0
    tp = int((pred & pos).sum()); fp = int((pred & neg).sum())
    out[f"recall@{tag}"] = tp / max(int(pos.sum()), 1)
    out[f"precision@{tag}"] = tp / max(tp + fp, 1) if tp + fp else 0.0
    out[f"fpr@{tag}"] = fp / max(int(neg.sum()), 1)
    out[f"thr@{tag}"] = float(thr)
    out[f"fa_per_hour@{tag}"] = out[f"fpr@{tag}"] * 3600.0 / max(win_seconds, 1e-9)
    if pos.any():
        for f in np.unique(fam[pos]):
            sel = pos & (fam == f)
            out[f"recall[{f}]@{tag}"] = float(pred[sel].mean())
        # event level: a capture with any positive window counts as one event
        ev_hit, ev_tot, ev_fam = {}, {}, {}
        for c in np.unique(cap[pos]):
            sel = pos & (cap == c)
            f = fam[sel][0]
            ev_tot[f] = ev_tot.get(f, 0) + 1
            ev_hit[f] = ev_hit.get(f, 0) + int(pred[sel].any())
        out[f"event_recall@{tag}"] = sum(ev_hit.values()) / max(sum(ev_tot.values()), 1)
        for f in ev_tot:
            out[f"event_recall[{f}]@{tag}"] = ev_hit[f] / ev_tot[f]
    return out


def evaluate(scores, y, fam, cap, win_seconds, calib_scores,
             budgets=FA_BUDGETS, fprs=FPR_TARGETS) -> dict:
    y = np.asarray(y).astype(np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    scores = np.where(np.isfinite(scores), scores, -1e30)
    fam = np.asarray(fam); cap = np.asarray(cap)
    out = {"n_windows": int(len(y)), "n_pos": int(y.sum()),
           "pos_rate": float(y.mean()) if len(y) else np.nan}
    if 0 < y.sum() < len(y):
        out["roc_auc"] = float(roc_auc_score(y, scores))
        out["pr_auc"] = float(average_precision_score(y, scores))
    else:
        out["roc_auc"] = out["pr_auc"] = np.nan
    calib = np.asarray(calib_scores, dtype=np.float64)
    calib = calib[np.isfinite(calib)]
    out["n_calib"] = int(len(calib))
    out["win_seconds"] = float(win_seconds)
    for b in budgets:
        tag = f"fa{int(b)}"
        out[f"resolvable@{tag}"] = bool(budget_resolvable(len(calib), win_seconds, b))
        _at(scores, y, fam, cap, threshold_for_fa_rate(calib, win_seconds, b), tag,
            win_seconds, out)
    for f in fprs:
        tag = f"fpr{f:g}"
        out[f"resolvable@{tag}"] = bool(f * len(calib) >= 1.0)
        _at(scores, y, fam, cap, threshold_for_fpr(calib, f), tag, win_seconds, out)
    return out


def cluster_bootstrap(scores, y, cap, n_boot=300, seed=0, metric="pr_auc"):
    """Bootstrap over captures (the natural sampling unit) for a threshold-free
    metric. Returns (lo, hi) 2.5/97.5 percentiles."""
    rng = np.random.default_rng(seed)
    caps = np.unique(cap)
    idx_by_cap = {c: np.flatnonzero(cap == c) for c in caps}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(caps, len(caps), replace=True)
        idx = np.concatenate([idx_by_cap[c] for c in pick])
        yy = y[idx]
        if 0 < yy.sum() < len(yy):
            if metric == "pr_auc":
                vals.append(average_precision_score(yy, scores[idx]))
            else:
                vals.append(roc_auc_score(yy, scores[idx]))
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
