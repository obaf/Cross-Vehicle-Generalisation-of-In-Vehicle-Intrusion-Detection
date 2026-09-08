"""
bench_lib.py -- shared machinery: featurisation to a local feature store,
feature-space assembly, training-pool construction, sequence gathering,
detector lists and the per-job fit/score/evaluate loop.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines as B      # noqa: E402
import evaluate as EV      # noqa: E402
import features as FT      # noqa: E402
import models as MD        # noqa: E402
import protocol as PR      # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

WINDOW = 64
EXCLUDE_FAMILIES = {"advanced"}          # ROAD accelerator: no per-message truth
FAMILIES = ["benign", "dos", "fuzzing", "spoofing", "replay", "diagnostic",
            "suspension", "masquerade", "timing"]
FAM_CODE = {f: i for i, f in enumerate(FAMILIES)}
LOAD_COLS = ["t", "aid", "dlc"] + [f"b{i}" for i in range(8)] + ["y"]
FEATURE_SETS = ["FS-A", "FS-B"]
CAP_TRAIN = 150_000


# ==========================================================================
# featurisation
# ==========================================================================
def _sanitise_cidv2_labels(path, df, name):
    """A frame whose payload text is not valid hex is a corrupted log line, not
    an injected frame; the count-aware anti-join of the prior run labelled one
    such frame (0x0F1, payload ending in a backslash) as fuzzing."""
    hx = pd.read_parquet(path, columns=["data_hex"])["data_hex"].fillna("")
    bad = ~hx.str.fullmatch(r"[0-9A-Fa-f]*") | (hx.str.len() % 2 != 0)
    n_fix = int((bad.values & (df["y"].values == 1)).sum())
    if n_fix:
        print(f"    label fix: {name}: {n_fix} malformed frame(s) relabelled benign", flush=True)
        df.loc[bad.values & (df["y"].values == 1), "y"] = 0
    return df, n_fix


def featurise_domain(cache_dir, vehicle, out_dir, N=WINDOW):
    man = json.load(open(os.path.join(cache_dir, "manifest.json")))
    entries = [m for m in man if m["vehicle"] == vehicle and m["family"] not in EXCLUDE_FAMILIES]
    entries.sort(key=lambda m: (m["family"] != "benign", m["name"]))
    dfs, fixes = {}, {}
    for m in entries:
        p = os.path.join(cache_dir, m["file"])
        df = pd.read_parquet(p, columns=LOAD_COLS)
        if m["dataset"] == "CIDv2" and m["family"] != "benign":
            df, nf = _sanitise_cidv2_labels(p, df, m["name"])
            if nf:
                fixes[m["name"]] = nf
        dfs[m["name"]] = df
    benign = [m for m in entries if m["family"] == "benign"]
    prof = FT.BenignProfile.fit([dfs[m["name"]] for m in benign])
    id_counts = {}
    for m in benign:
        vc = dfs[m["name"]]["aid"].value_counts()
        for a, c in vc.items():
            id_counts[int(a)] = id_counts.get(int(a), 0) + int(c)

    XB, RAW, AIDW, Y, FAM, CAP, capnames, durs = [], [], [], [], [], [], [], []
    for m in entries:
        df = dfs[m["name"]]
        nw = len(df) // N
        if nw < 2:
            continue
        pm = FT.per_message(df, prof)
        fb = FT.window_features_B(pm, prof, N)
        aidw, raw = FT.window_raw_A(pm, N)
        susp = tuple(m["susp_interval"]) if m.get("susp_interval") else None
        yw = FT.window_labels(df, N, susp)
        k = min(len(fb), len(aidw), len(yw))
        XB.append(fb[:k]); RAW.append(raw[:k]); AIDW.append(aidw[:k]); Y.append(yw[:k])
        FAM.append(np.full(k, FAM_CODE[m["family"]], dtype=np.int8))
        CAP.append(np.full(k, len(capnames), dtype=np.int32))
        capnames.append(m["name"])
        dur = float(df["t"].iloc[-1] - df["t"].iloc[0])
        durs.append((m["family"] == "benign", dur / k, k))
        print(f"    {vehicle:<18s} {m['name']:<58s} win={k:7d} pos={int(yw[:k].sum()):6d}", flush=True)
        del pm
    os.makedirs(out_dir, exist_ok=True)
    arrs = dict(XB=np.vstack(XB), RAW=np.vstack(RAW), AIDW=np.vstack(AIDW),
                y=np.concatenate(Y), fam=np.concatenate(FAM), cap=np.concatenate(CAP))
    arrs["q"] = PR.quarters(arrs["cap"], arrs["y"])
    for k, v in arrs.items():
        np.save(os.path.join(out_dir, f"{k}.npy"), v)
    ben_w = [(d, k) for isb, d, k in durs if isb]
    win_seconds = float(sum(d * k for d, k in ben_w) / max(sum(k for _, k in ben_w), 1))
    meta = dict(vehicle=vehicle, window=N, captures=capnames,
                families=[m["family"] for m in entries if m["name"] in capnames],
                win_seconds_benign=win_seconds,
                win_seconds_all=float(sum(d * k for _, d, k in durs) / max(sum(k for *_, k in durs), 1)),
                n_windows=int(len(arrs["y"])), n_pos=int(arrs["y"].sum()),
                id_counts={str(a): c for a, c in id_counts.items()},
                n_benign_ids=len(prof.ref_ids), n_periodic_ids=len(prof.periodic_ids),
                ref_bus_rate=prof.ref_bus_rate, label_fixes=fixes,
                fs_b_names=FT.FS_B_NAMES, fs_a_raw_names=FT.FS_A_RAW_NAMES)
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    return meta


def featurise_all(cache_dir, work_dir, vehicles=None, workers=4):
    man = json.load(open(os.path.join(cache_dir, "manifest.json")))
    vs = sorted({m["vehicle"] for m in man})
    if vehicles:
        vs = [v for v in vs if v in vehicles]
    todo = [v for v in vs if not os.path.exists(os.path.join(work_dir, "features", v, "meta.json"))]
    print(f"featurising {len(todo)} of {len(vs)} domains -> {work_dir}", flush=True)
    if not todo:
        return
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=min(workers, len(todo))) as ex:
        futs = {ex.submit(featurise_domain, cache_dir, v,
                          os.path.join(work_dir, "features", v)): v for v in todo}
        for f in futs:
            m = f.result()
            print(f"  -> {m['vehicle']}: {m['n_windows']:,} windows, {m['n_pos']:,} positive, "
                  f"win={m['win_seconds_benign']*1000:.1f} ms", flush=True)


# ==========================================================================
# feature store access
# ==========================================================================
class Domain:
    def __init__(self, work_dir, name):
        d = os.path.join(work_dir, "features", name)
        self.name = name
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        ld = lambda k: np.load(os.path.join(d, f"{k}.npy"), mmap_mode="r")  # noqa: E731
        self.XB, self.RAW, self.AIDW = ld("XB"), ld("RAW"), ld("AIDW")
        self.y = np.asarray(ld("y")); self.fam = np.asarray(ld("fam"))
        self.cap = np.asarray(ld("cap")); self.q = np.asarray(ld("q"))
        self.win_seconds = self.meta["win_seconds_benign"]
        self.id_share = {int(a): c for a, c in self.meta["id_counts"].items()}
        tot = sum(self.id_share.values()) or 1
        self.id_share = {a: c / tot for a, c in self.id_share.items()}
        self.fam_names = np.array(FAMILIES)[self.fam]

    def own_vocab(self):
        return FT.vocabulary(self.id_share)

    def X(self, fs, vocab=None):
        if fs == "FS-B":
            return np.asarray(self.XB)
        if fs == "FS-B34":
            return np.asarray(self.XB[:, : len(FT.FS_B_ORIG)])
        if fs in ("FS-A", "FS-A-rank"):
            v = vocab if (fs == "FS-A" and vocab is not None) else self.own_vocab()
            return FT.fsa_matrix(np.asarray(self.AIDW), np.asarray(self.RAW), v)
        raise ValueError(fs)


def pooled_vocab(domains):
    """Top identifiers by *per-vehicle normalised* share, so that a vehicle with
    a long recording does not dictate the vocabulary."""
    share = {}
    for d in domains:
        for a, s in d.id_share.items():
            share[a] = share.get(a, 0.0) + s
    return FT.vocabulary(share)


def load_domains(work_dir, names=None):
    root = os.path.join(work_dir, "features")
    names = names or sorted(os.listdir(root))
    return {n: Domain(work_dir, n) for n in names}


# ==========================================================================
# detectors
# ==========================================================================
def detector_list(fs, threads, cross, include_nn=True, include_ens=True, fast=False):
    """Yields (scope, model). scope 'fleet' fits on the training pool,
    'target' on the target's benign calibration windows."""
    det = []
    if fs.startswith("FS-B") and fs != "FS-B34":
        for r in B.rule_baselines():
            det.append(("fleet", r))
        if cross:
            for r in B.rule_baselines():
                r.name = r.name + "@target"
                det.append(("target", r))
    if not fast:
        det.append(("fleet", B.IsolationForestDetector(threads=threads)))
        if cross:
            m = B.IsolationForestDetector(threads=threads); m.name = "IForest@target"
            det.append(("target", m))
        det += [("fleet", m) for m in MD.model_zoo(threads, include_nn, include_ens)]
    else:
        det += [("fleet", m) for m in MD.fast_zoo(threads)]
    return det


# ==========================================================================
# fit / score
# ==========================================================================
def gather_seq(Xsrc, seq, rows, L=MD.SEQ_LEN):
    return Xsrc[seq[rows]]                     # (n, L, d)


def score_chunked(model, Xsrc, seq, rows, chunk=65536):
    out = np.empty(len(rows), dtype=np.float64)
    for i in range(0, len(rows), chunk):
        r = rows[i:i + chunk]
        S = gather_seq(Xsrc, seq, r) if model.needs_seq else None
        out[i:i + chunk] = model.score(Xsrc[r], S)
    return out


def fit_score(model, scope, Xpool, ypool, seqpool, sel, Xt, seqt, calib, ev):
    """Returns (eval_scores, calib_scores, fit_seconds, score_seconds)."""
    t0 = time.time()
    if scope == "target":
        model.fit(Xt[calib], None, None)
    elif getattr(model, "benign_only", False):
        neg = sel[ypool[sel] == 0]
        model.fit(Xpool[neg], None, None)
    else:
        S = gather_seq(Xpool, seqpool, sel) if model.needs_seq else None
        model.fit(Xpool[sel], ypool[sel], S)
    t1 = time.time()
    s_ev = score_chunked(model, Xt, seqt, ev)
    s_cal = score_chunked(model, Xt, seqt, calib)
    return s_ev, s_cal, t1 - t0, time.time() - t1


def build_pool(domains, fs, vocab, cap_train, rng, exclude=None, volume=None):
    """Concatenate the full window matrices of the training vehicles, then
    choose a stratified training subset. seq indices are global to the pool."""
    Xs, ys, seqs, off, fam, dom = [], [], [], 0, [], []
    for name, d in domains.items():
        if exclude and name in exclude:
            continue
        X = d.X(fs, vocab)
        Xs.append(X); ys.append(d.y); fam.append(d.fam)
        seqs.append(PR.seq_index(d.cap, MD.SEQ_LEN) + off)
        dom.append(np.full(len(X), name, dtype=object))
        off += len(X)
    Xpool = np.vstack(Xs); ypool = np.concatenate(ys); seqpool = np.vstack(seqs)
    if volume is not None:
        sel = PR.match_volume(ypool, volume[0], volume[1], rng)
    else:
        sel = PR.subsample_train(ypool, cap_train, rng)
    return Xpool, ypool, seqpool, sel, np.concatenate(fam), np.concatenate(dom)


def job_seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) % (2**31)


def run_job(spec, domains, threads, out_rows, out_scores, include_nn=True,
            include_ens=True, fast=False, cap_train=CAP_TRAIN, log=print):
    """spec: dict(protocol, fold, target, fs). Writes a CSV of metric rows and
    an npz of scores; returns the rows."""
    protocol, fold, target, fs = spec["protocol"], spec["fold"], spec["target"], spec["fs"]
    tag = f"{protocol}_f{fold}_{target}_{fs}"
    rng = np.random.default_rng(job_seed(tag))
    dt = domains[target]
    cross = protocol == "cross_domain"
    if cross:
        others = {k: v for k, v in domains.items() if k != target}
        vocab = pooled_vocab(list(others.values())) if fs == "FS-A" else None
        Xpool, ypool, seqpool, sel, _, _ = build_pool(others, fs, vocab, cap_train, rng)
        Xt = dt.X(fs, vocab)
        sp = PR.cross_split(dt.y, dt.q)
        calib, ev = sp["calib"], sp["eval"]
    else:
        vocab = dt.own_vocab() if fs == "FS-A" else None
        Xt = dt.X(fs, vocab)
        f = PR.in_domain_folds(dt.y, dt.q)[fold]
        Xpool, ypool = Xt, dt.y
        seqpool = PR.seq_index(dt.cap, MD.SEQ_LEN)
        sel = f["train"][PR.subsample_train(dt.y[f["train"]], cap_train, rng)]
        calib, ev = f["calib"], f["eval"]
    seqt = PR.seq_index(dt.cap, MD.SEQ_LEN)
    ytr = ypool[sel]
    fam_ev = dt.fam_names[ev]; cap_ev = dt.cap[ev]
    log(f"[{tag}] train={len(sel):,} (pos {int(ytr.sum()):,}) calib={len(calib):,} "
        f"eval={len(ev):,} (pos {int(dt.y[ev].sum()):,}) dim={Xt.shape[1]}")

    rows, scores = [], {"eval_idx": ev, "calib_idx": calib, "y": dt.y[ev],
                        "fam": dt.fam[ev], "cap": cap_ev}
    for scope, m in detector_list(fs, threads, cross, include_nn, include_ens, fast):
        name = m.name
        try:
            s_ev, s_cal, tf, ts = fit_score(m, scope, Xpool, ypool, seqpool, sel,
                                            Xt, seqt, calib, ev)
            r = EV.evaluate(s_ev, dt.y[ev], fam_ev, cap_ev, dt.win_seconds, s_cal)
        except Exception as e:     # noqa: BLE001
            log(f"   !! {tag} {name}: {type(e).__name__}: {e}")
            continue
        r.update(protocol=protocol, fold=fold, test_domain=target, feature_set=fs,
                 model=name, scope=scope, fit_s=round(tf, 2), score_s=round(ts, 2),
                 n_train=int(len(sel)), n_train_pos=int(ytr.sum()), dim=int(Xt.shape[1]))
        rows.append(r)
        scores[f"ev::{name}"] = s_ev.astype(np.float32)
        scores[f"cal::{name}"] = s_cal.astype(np.float32)
        log(f"   [{tag}] {name:<24s} PR-AUC={r['pr_auc']:.4f} ROC={r['roc_auc']:.4f} "
            f"R@10FA={r['recall@fa10']:.3f} R@fpr1e-3={r['recall@fpr0.001']:.3f} "
            f"({tf:.1f}s+{ts:.1f}s)")
    os.makedirs(os.path.dirname(out_rows), exist_ok=True)
    os.makedirs(os.path.dirname(out_scores), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_rows, index=False)
    np.savez_compressed(out_scores, **scores)
    return rows
