"""
features.py -- window-level feature extraction for the cross-vehicle CAN IDS
benchmark (experiment1-by-fable).

Two contrasting feature spaces are computed from identical windows.

FS-A  identity-coupled (the conventional encoding, implemented strictly)
      A frequency histogram over the *training* traffic's most common
      arbitration identifiers plus an overflow bucket, raw identifier moments,
      raw timing and length statistics and raw payload-byte moments. The
      histogram vocabulary is fixed at fit time from the training vehicles, so
      on an unseen vehicle whose identifiers do not occur the histogram really
      does collapse into the overflow bucket. (The prior run computed each
      vehicle's histogram over its *own* top identifiers, which is a
      rank-indexed encoding and not the conventional one; that variant is kept
      as an ablation under the name FS-A-rank.)

FS-B  identity-free. The 34 statistics of the prior run are retained verbatim
      (so the two runs are directly comparable) and six are added:
        stale_max, stale_frac   staleness of periodic identifiers relative to
                                their benign period -- the only frame-level
                                evidence a suspension attack leaves while it is
                                in progress
        dt_mean_norm, dt_p95_norm, dt_max_norm
                                inter-arrival statistics normalised by the
                                vehicle's benign mean inter-arrival time, so
                                they are dimensionless
        id_unique_norm          unique identifiers in the window as a fraction
                                of the vehicle's benign vocabulary
      FS-B(34) versus FS-B(40) is reported as an ablation.

Everything a window feature needs from a specific vehicle comes from a
BenignProfile estimated on attack-free traffic only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EPS = 1e-9
PAY = [f"b{i}" for i in range(8)]


# ==========================================================================
# per-vehicle benign profile
# ==========================================================================
@dataclass
class BenignProfile:
    ref_period: dict            # aid -> median inter-arrival (s)
    ref_period_iqr: dict        # aid -> IQR of inter-arrival (s)
    ref_count: dict             # aid -> messages seen in benign traffic
    ref_ids: set
    ref_dlc: dict               # aid -> modal DLC
    ref_bus_rate: float         # messages / s
    periodic_ids: list = field(default_factory=list)
    periodic_periods: np.ndarray = None

    @property
    def ref_dt_mean(self) -> float:
        return 1.0 / max(self.ref_bus_rate, EPS)

    @staticmethod
    def fit(dfs, max_period=0.25, min_count=50, max_rel_iqr=0.25) -> "BenignProfile":
        if isinstance(dfs, pd.DataFrame):
            dfs = [dfs]
        period, iqr, cnt, dlcm = {}, {}, {}, {}
        n_total, dur_total = 0, 0.0
        # accumulate per capture so that inter-capture gaps never enter dt
        per_id_dt = {}
        for df in dfs:
            df = df.sort_values("t", kind="mergesort")
            n_total += len(df)
            dur_total += float(df["t"].iloc[-1] - df["t"].iloc[0]) if len(df) > 1 else 0.0
            for aid, g in df.groupby("aid", sort=False):
                aid = int(aid)
                tt = g["t"].values
                d = np.diff(tt)
                d = d[d > 0]
                per_id_dt.setdefault(aid, []).append(d)
                cnt[aid] = cnt.get(aid, 0) + len(g)
                dl = g["dlc"].values
                dlcm.setdefault(aid, {})
                for v, c in zip(*np.unique(dl, return_counts=True)):
                    dlcm[aid][int(v)] = dlcm[aid].get(int(v), 0) + int(c)
        for aid, parts in per_id_dt.items():
            d = np.concatenate(parts) if parts else np.zeros(0)
            if len(d):
                q1, q2, q3 = np.percentile(d, [25, 50, 75])
                period[aid], iqr[aid] = float(q2), float(q3 - q1)
            else:
                period[aid], iqr[aid] = np.nan, np.nan
        ref_dlc = {a: max(m.items(), key=lambda kv: kv[1])[0] for a, m in dlcm.items()}
        p = BenignProfile(ref_period=period, ref_period_iqr=iqr, ref_count=cnt,
                          ref_ids=set(period), ref_dlc=ref_dlc,
                          ref_bus_rate=n_total / max(dur_total, 1e-6))
        per = [a for a in period
               if cnt[a] >= min_count and np.isfinite(period[a])
               and 0 < period[a] <= max_period
               and iqr[a] / max(period[a], EPS) <= max_rel_iqr]
        p.periodic_ids = sorted(per)
        p.periodic_periods = np.array([period[a] for a in p.periodic_ids], dtype=np.float64)
        return p


# ==========================================================================
# per-message quantities
# ==========================================================================
def _run_entropy(sorted_rows: np.ndarray) -> np.ndarray:
    """Shannon entropy (bits) of each row of an already row-sorted int matrix."""
    nw, N = sorted_rows.shape
    if nw == 0:
        return np.zeros(0)
    isnew = np.ones_like(sorted_rows, dtype=bool)
    isnew[:, 1:] = sorted_rows[:, 1:] != sorted_rows[:, :-1]
    idx = np.flatnonzero(isnew.ravel())
    ends = np.r_[idx[1:], nw * N]
    lens = (ends - idx).astype(np.float64)
    rows = idx // N
    p = lens / N
    return np.bincount(rows, weights=-p * np.log2(p + EPS), minlength=nw)


def per_message(df: pd.DataFrame, prof: BenignProfile) -> dict:
    t = df["t"].values.astype(np.float64)
    aid = df["aid"].values.astype(np.int64)
    dlc = df["dlc"].values.astype(np.int16)
    pay = df[PAY].values.astype(np.int16)
    n = len(t)

    dt = np.zeros(n, dtype=np.float64)
    if n > 1:
        dt[1:] = t[1:] - t[:-1]

    order = np.lexsort((t, aid))
    aid_s, t_s = aid[order], t[order]
    first = np.ones(n, dtype=bool)
    first[1:] = aid_s[1:] != aid_s[:-1]
    dt_id_s = np.full(n, np.nan)
    dt_id_s[1:] = t_s[1:] - t_s[:-1]
    dt_id_s[first] = np.nan
    pay_s = pay[order]
    ham_s = np.full(n, np.nan, dtype=np.float32)
    ham_s[1:] = (pay_s[1:] != pay_s[:-1]).sum(axis=1)
    ham_s[first] = np.nan
    dt_id = np.empty(n); dt_id[order] = dt_id_s
    ham = np.empty(n, dtype=np.float32); ham[order] = ham_s

    # profile lookups through tables rather than per-message dict access
    hi = int(max(4096, aid.max() if n else 0, max(prof.ref_ids) if prof.ref_ids else 0)) + 1
    lut_p = np.full(hi, np.nan); lut_seen = np.zeros(hi, dtype=bool)
    lut_dlc = np.full(hi, -1, dtype=np.int16)
    for a, p in prof.ref_period.items():
        if 0 <= a < hi:
            lut_p[a] = p; lut_seen[a] = True; lut_dlc[a] = prof.ref_dlc.get(a, -1)
    aidc = np.clip(aid, 0, hi - 1)
    ref = lut_p[aidc]
    with np.errstate(all="ignore"):
        period_ratio = dt_id / (ref + EPS)
    unseen = (~lut_seen[aidc]).astype(np.float32)
    refdlc = lut_dlc[aidc]
    dlc_mismatch = ((refdlc >= 0) & (refdlc != dlc)).astype(np.float32)

    # payload byte entropy, fully vectorised by byte count
    valid = pay >= 0
    nb = valid.sum(axis=1)
    ent = np.zeros(n, dtype=np.float32)
    for k in range(1, 9):
        rows = np.flatnonzero(nb == k)
        if len(rows):
            ent[rows] = _run_entropy(np.sort(pay[rows, :k].astype(np.int32), axis=1))
    allzero = ((np.where(valid, pay, 0).sum(axis=1) == 0) & (nb > 0)).astype(np.float32)
    allff = (((pay == 255) | ~valid).all(axis=1) & (nb > 0)).astype(np.float32)

    return dict(t=t, aid=aid, dlc=dlc, dt=dt, dt_id=dt_id, ham=ham,
                period_ratio=period_ratio, unseen=unseen, dlc_mismatch=dlc_mismatch,
                ent=ent, allzero=allzero, allff=allff, pay=pay, n=n)


# ==========================================================================
# FS-B
# ==========================================================================
FS_B_ORIG = [
    "dt_mean", "dt_std", "dt_cv", "dt_p95", "dt_max", "bus_rate_norm",
    "idle_frac", "burstiness",
    "id_entropy", "id_entropy_norm", "id_unique_frac", "id_max_share",
    "unseen_rate", "dlc_mismatch_rate",
    "pr_mean", "pr_std", "pr_p05", "pr_p95", "pr_lt_half", "pr_gt_two",
    "pr_absdev_mean", "pr_absdev_max", "dtid_nan_frac",
    "ham_mean", "ham_std", "ham_max", "ham_zero_frac",
    "ent_mean", "ent_std", "ent_min", "allzero_frac", "allff_frac",
    "dlc_mean", "dlc_std",
]
FS_B_EXTRA = ["stale_max", "stale_frac", "dt_mean_norm", "dt_p95_norm",
              "dt_max_norm", "id_unique_norm"]
FS_B_NAMES = FS_B_ORIG + FS_B_EXTRA
IDX_B = {n: i for i, n in enumerate(FS_B_NAMES)}


def _reshape(a, nw, N):
    return a[: nw * N].reshape(nw, N)


def _staleness(t: np.ndarray, aid: np.ndarray, nw: int, N: int, prof: BenignProfile):
    """Per window: max and fraction-over-3 of (time since last arrival / period)
    over the vehicle's periodic identifiers. NaN before an identifier's first
    arrival in the capture, so a slow start never looks like a suspension."""
    if not prof.periodic_ids or nw == 0:
        return np.zeros(nw), np.zeros(nw)
    t_end = t[np.arange(1, nw + 1) * N - 1]
    stale = np.full((len(prof.periodic_ids), nw), np.nan)
    for k, (a, p) in enumerate(zip(prof.periodic_ids, prof.periodic_periods)):
        ta = t[aid == a]
        if len(ta) == 0:
            continue
        j = np.searchsorted(ta, t_end, side="right") - 1
        ok = j >= 0
        stale[k, ok] = (t_end[ok] - ta[j[ok]]) / p
    with np.errstate(all="ignore"):
        smax = np.nanmax(np.where(np.isfinite(stale), stale, -np.inf), axis=0)
        smax[~np.isfinite(smax)] = 0.0
        valid = np.isfinite(stale)
        frac = (stale > 3.0).sum(axis=0) / np.maximum(valid.sum(axis=0), 1)
    return smax, frac


def window_features_B(pm: dict, prof: BenignProfile, N: int = 64) -> np.ndarray:
    n = pm["n"]
    nw = n // N
    if nw == 0:
        return np.zeros((0, len(FS_B_NAMES)), dtype=np.float32)
    dt = _reshape(pm["dt"], nw, N)
    aidw = _reshape(pm["aid"], nw, N)
    pr = _reshape(pm["period_ratio"], nw, N)
    ham = _reshape(pm["ham"], nw, N)
    ent = _reshape(pm["ent"], nw, N)
    dlc = _reshape(pm["dlc"].astype(np.float64), nw, N)
    unseen = _reshape(pm["unseen"], nw, N)
    dmm = _reshape(pm["dlc_mismatch"], nw, N)
    az = _reshape(pm["allzero"], nw, N)
    af = _reshape(pm["allff"], nw, N)

    with np.errstate(all="ignore"):
        dt_mean = dt.mean(axis=1); dt_std = dt.std(axis=1)
        dt_cv = dt_std / (dt_mean + EPS)
        dt_p95 = np.percentile(dt, 95, axis=1); dt_max = dt.max(axis=1)
        bus_rate = N / (dt.sum(axis=1) + EPS)
        bus_rate_norm = bus_rate / (prof.ref_bus_rate + EPS)
        idle_frac = (dt > dt_mean[:, None] * 3).mean(axis=1)
        burstiness = (dt_std - dt_mean) / (dt_std + dt_mean + EPS)

        srt = np.sort(aidw, axis=1)
        id_ent = _run_entropy(srt)
        isnew = np.ones_like(srt, dtype=bool)
        isnew[:, 1:] = srt[:, 1:] != srt[:, :-1]
        id_uniq = isnew.sum(axis=1).astype(np.float64)
        id_ent_norm = id_ent / (np.log2(id_uniq + EPS) + EPS)
        idx = np.flatnonzero(isnew.ravel()); ends = np.r_[idx[1:], nw * N]
        lens = (ends - idx).astype(np.float64); rows = idx // N
        id_max_share = np.zeros(nw); np.maximum.at(id_max_share, rows, lens / N)

        prm = np.nanmean(pr, axis=1); prs = np.nanstd(pr, axis=1)
        pr05 = np.nanpercentile(pr, 5, axis=1); pr95 = np.nanpercentile(pr, 95, axis=1)
        pr_lt = np.nanmean((pr < 0.5).astype(np.float64), axis=1)
        pr_gt = np.nanmean((pr > 2.0).astype(np.float64), axis=1)
        absdev = np.abs(pr - 1.0)
        pr_ad_mean = np.nanmean(absdev, axis=1)
        pr_ad_max = np.nanmax(np.where(np.isfinite(absdev), absdev, -np.inf), axis=1)
        dtid_nan = np.isnan(_reshape(pm["dt_id"], nw, N)).mean(axis=1)
        ham_mean = np.nanmean(ham, axis=1); ham_std = np.nanstd(ham, axis=1)
        ham_max = np.nanmax(np.where(np.isfinite(ham), ham, -np.inf), axis=1)
        ham_zero = np.nanmean((ham == 0).astype(np.float64), axis=1)

        smax, sfrac = _staleness(pm["t"], pm["aid"], nw, N, prof)
        rdt = prof.ref_dt_mean
        F = np.column_stack([
            dt_mean, dt_std, dt_cv, dt_p95, dt_max, bus_rate_norm, idle_frac, burstiness,
            id_ent, id_ent_norm, id_uniq / N, id_max_share,
            unseen.mean(axis=1), dmm.mean(axis=1),
            prm, prs, pr05, pr95, pr_lt, pr_gt, pr_ad_mean, pr_ad_max, dtid_nan,
            ham_mean, ham_std, ham_max, ham_zero,
            ent.mean(axis=1), ent.std(axis=1), ent.min(axis=1), az.mean(axis=1), af.mean(axis=1),
            dlc.mean(axis=1), dlc.std(axis=1),
            smax, sfrac, dt_mean / rdt, dt_p95 / rdt, dt_max / rdt,
            id_uniq / max(len(prof.ref_ids), 1),
        ]).astype(np.float32)
    F[~np.isfinite(F)] = 0.0
    return F


# ==========================================================================
# FS-A (strict, identity-coupled)
# ==========================================================================
TOP_K = 48
FS_A_RAW_NAMES = (["aid_mean", "aid_std", "aid_min", "aid_max", "aid_p25", "aid_p75",
                   "dt_mean", "dt_std", "dt_max", "dlc_mean", "dlc_std"]
                  + [f"b{i}_mean" for i in range(8)] + [f"b{i}_std" for i in range(8)])


def window_raw_A(pm: dict, N: int = 64):
    """Vehicle-specific raw statistics of FS-A, plus the identifier window
    matrix from which any histogram vocabulary can be applied later."""
    n = pm["n"]; nw = n // N
    if nw == 0:
        return (np.zeros((0, N), dtype=np.int16),
                np.zeros((0, len(FS_A_RAW_NAMES)), dtype=np.float32))
    aidw = _reshape(pm["aid"], nw, N)
    dt = _reshape(pm["dt"], nw, N)
    dlc = _reshape(pm["dlc"].astype(np.float64), nw, N)
    pay = pm["pay"][: nw * N].reshape(nw, N, 8).astype(np.float32)
    pay[pay < 0] = 0.0                      # absent byte -> 0, as raw encodings do
    with np.errstate(all="ignore"):
        raw = np.column_stack([
            aidw.mean(axis=1), aidw.std(axis=1), aidw.min(axis=1), aidw.max(axis=1),
            np.percentile(aidw, 25, axis=1), np.percentile(aidw, 75, axis=1),
            dt.mean(axis=1), dt.std(axis=1), dt.max(axis=1),
            dlc.mean(axis=1), dlc.std(axis=1),
            pay.mean(axis=1), pay.std(axis=1),
        ]).astype(np.float32)
    raw[~np.isfinite(raw)] = 0.0
    return aidw.astype(np.int16), raw


def vocabulary(id_counts: dict, top_k: int = TOP_K) -> list:
    """The top_k identifiers by benign frequency, padded with -1 (a value no
    frame can take) so that FS-A has identical width on every vehicle."""
    ids = [int(a) for a, _ in sorted(id_counts.items(), key=lambda kv: -kv[1])[:top_k]]
    return ids + [-1] * (top_k - len(ids))


def fsa_matrix(aidw: np.ndarray, raw: np.ndarray, vocab: list) -> np.ndarray:
    """Histogram of the window's identifiers over `vocab` (+ overflow bucket),
    concatenated with the raw statistics."""
    nw, N = aidw.shape
    K = len(vocab)
    hi = int(max(4096, int(aidw.max()) if aidw.size else 0, max(vocab))) + 1
    lut = np.full(hi, K, dtype=np.int32)                  # K = overflow bucket
    for i, a in enumerate(vocab):
        if a >= 0:
            lut[a] = i
    codes = lut[np.clip(aidw.astype(np.int64), 0, hi - 1)]
    hist = np.zeros((nw, K + 1), dtype=np.float32)
    np.add.at(hist, (np.repeat(np.arange(nw), N), codes.ravel()), 1.0)
    hist /= N
    return np.hstack([hist, raw]).astype(np.float32)


def fsa_names(vocab):
    return ([f"hist_{a:#05x}" if a >= 0 else "hist_pad" for a in vocab]
            + ["hist_other"] + FS_A_RAW_NAMES)


# ==========================================================================
# labels
# ==========================================================================
def window_labels(df: pd.DataFrame, N: int, susp_interval=None) -> np.ndarray:
    y = df["y"].values.astype(np.int8)
    nw = len(y) // N
    if nw == 0:
        return np.zeros(0, dtype=np.int8)
    yw = _reshape(y, nw, N).max(axis=1)
    if susp_interval is not None:
        tw = _reshape(df["t"].values, nw, N)
        t0, t1 = susp_interval
        yw = np.maximum(yw, ((tw[:, -1] >= t0) & (tw[:, 0] <= t1)).astype(np.int8))
    return yw.astype(np.int8)
