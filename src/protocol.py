"""
protocol.py -- how windows are assigned to training, calibration and
evaluation, for both protocols.

Every capture's windows are divided by time into four quarters Q1..Q4,
separately for its attack windows and its benign windows (see quarters()).
Windows of a capture are contiguous and in temporal order in every domain
array.

cross_domain (leave-one-vehicle-out), target vehicle v:
    train       every window of every other vehicle
    calibration benign windows of v in Q1 u Q2
    evaluation  every window of v in Q3 u Q4, plus the attack windows of Q1 u Q2
    => every attack window of v is evaluated; calibration and evaluation
       benign windows are disjoint; calibration precedes evaluation in time.

in_domain, two folds:
    fold 1  train Q1 u Q2 | calibration benign(Q3) | evaluation Q4 + attack(Q3)
    fold 2  train Q3 u Q4 | calibration benign(Q2) | evaluation Q1 + attack(Q2)
    => every attack family of the vehicle is present on both sides of every
       fold (the prior run held out whole captures, and because CIDv2 has one
       capture per family it was in fact testing on unseen families and
       reporting the result as in-domain); every attack window is evaluated
       exactly once across the two folds; no window is ever in both train and
       evaluation of the same fold.

Training-set subsampling is stratified: up to cap/2 positives, the remainder
benign. The prior run kept every positive and let the benign count fall to
max(cap - n_pos, 1000), which for the cross-vehicle protocol left exactly
1,000 benign windows against ~100,000 attack windows in every fold.
"""

from __future__ import annotations

import numpy as np


def quarters(cap_codes: np.ndarray, y: np.ndarray = None) -> np.ndarray:
    """Temporal quarter (0..3) of each window within its capture, assigned
    separately to the capture's attack windows and to its benign windows.

    Most attack captures contain a single attack event (one DoS flood, one
    10-second suspension, one burst of injected frames). Quartering the
    capture as a whole would put such an event entirely on one side of an
    in-domain split, so a fold would train on a family it never evaluates or
    evaluate a family it never saw. Quartering the attack windows by their own
    temporal rank puts the first half of every event in Q1 u Q2 and the second
    half in Q3 u Q4, while benign windows keep a clean temporal split."""
    q = np.zeros(len(cap_codes), dtype=np.int8)
    if len(cap_codes) == 0:
        return q
    if y is None:
        y = np.zeros(len(cap_codes), dtype=np.int8)
    bounds = np.flatnonzero(np.r_[True, cap_codes[1:] != cap_codes[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        for cls in (0, 1):
            idx = a + np.flatnonzero(y[a:b] == cls)
            n = len(idx)
            if n:
                q[idx] = np.minimum((np.arange(n) * 4) // n, 3)
    return q


def cross_split(y, q):
    ben = y == 0
    calib = np.flatnonzero(ben & (q <= 1))
    ev = np.flatnonzero((q >= 2) | (y == 1))
    return dict(calib=calib, eval=ev)


def in_domain_folds(y, q):
    ben = y == 0
    f1 = dict(train=np.flatnonzero(q <= 1),
              calib=np.flatnonzero(ben & (q == 2)),
              eval=np.flatnonzero((q == 3) | ((y == 1) & (q == 2))))
    f2 = dict(train=np.flatnonzero(q >= 2),
              calib=np.flatnonzero(ben & (q == 1)),
              eval=np.flatnonzero((q == 0) | ((y == 1) & (q == 1))))
    return [f1, f2]


def subsample_train(y, cap, rng, pos_frac_max=0.5):
    """Stratified cap: at most cap*pos_frac_max positives, benign fills the rest."""
    idx_pos = np.flatnonzero(y == 1)
    idx_neg = np.flatnonzero(y == 0)
    if cap is None or len(y) <= cap:
        return np.arange(len(y))
    n_pos = min(len(idx_pos), int(cap * pos_frac_max))
    n_neg = min(len(idx_neg), cap - n_pos)
    if n_pos < len(idx_pos):
        idx_pos = rng.choice(idx_pos, n_pos, replace=False)
    if n_neg < len(idx_neg):
        idx_neg = rng.choice(idx_neg, n_neg, replace=False)
    return np.sort(np.concatenate([idx_pos, idx_neg]))


def match_volume(y, n_total, n_pos, rng):
    """Subsample to exactly n_total windows with n_pos positives (as available)."""
    idx_pos = np.flatnonzero(y == 1)
    idx_neg = np.flatnonzero(y == 0)
    n_pos = min(n_pos, len(idx_pos))
    n_neg = min(n_total - n_pos, len(idx_neg))
    return np.sort(np.concatenate([rng.choice(idx_pos, n_pos, replace=False),
                                   rng.choice(idx_neg, n_neg, replace=False)]))


def seq_index(cap_codes: np.ndarray, L: int) -> np.ndarray:
    """(n, L) indices of the L consecutive windows ending at each window,
    clipped at the start of the window's own capture."""
    n = len(cap_codes)
    start = np.zeros(n, dtype=np.int64)
    if n:
        b = np.flatnonzero(np.r_[True, cap_codes[1:] != cap_codes[:-1]])
        lens = np.diff(np.r_[b, n])
        start = np.repeat(b, lens)
    i = np.arange(n)[:, None] - (L - 1) + np.arange(L)[None, :]
    return np.maximum(i, start[:, None]).astype(np.int64)
