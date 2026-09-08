"""
run_controls.py -- the controlled experiments that the prior run's discussion
said the public corpora could not support. They can.

A. Volume-matched control. The prior run compared in-domain training on part of
   one vehicle with cross-domain training on seven vehicles and could not say
   whether the difference was the vehicle change or the training volume. Here
   the cross-domain training pool is subsampled to exactly the in-domain
   fold-0 training size and positive count, and evaluated on the in-domain
   fold-0 evaluation set as well as on the cross-domain set. Same volume, same
   class ratio, same evaluation windows: the only difference is whether the
   training vehicle is the test vehicle.

B. Fleet-size sweep. Fixed training budget (up to 20k attack + 20k benign
   windows), k in 1..7 training vehicles drawn from the other seven, held-out
   vehicle evaluated cross-domain. Separates diversity from volume.

C. Feature-space ablation, cross-domain: FS-B34 (prior run's 34 features),
   FS-B (34 + 6), FS-A-rank (what the prior run actually computed), FS-A
   (strict identity-coupled).

All with the fast model subset so that the sweep is tractable.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_THREADS = 2


def _init(threads):
    global _THREADS
    _THREADS = threads
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(threads)
    import warnings
    warnings.filterwarnings("ignore")


def _fit_eval(m, scope, Xpool, ypool, sel, Xt, calib, ev, dt, extra):
    import evaluate as EV
    t0 = time.time()
    if scope == "target":
        m.fit(Xt[calib], None, None)
    elif getattr(m, "benign_only", False):
        m.fit(Xpool[sel][ypool[sel] == 0], None, None)
    else:
        m.fit(Xpool[sel], ypool[sel], None)
    r = EV.evaluate(m.score(Xt[ev]), dt.y[ev], dt.fam_names[ev], dt.cap[ev],
                    dt.win_seconds, m.score(Xt[calib]))
    r.update(model=m.name, scope=scope, n_train=int(len(sel)),
             n_train_pos=int(ypool[sel].sum()), fit_s=round(time.time() - t0, 2), **extra)
    return r


def _models(fs, threads, rule34=False):
    import baselines as B
    import models as MD
    det = [("fleet", m) for m in MD.fast_zoo(threads)]
    if fs == "FS-B":
        det.append(("fleet", B.CombinedRuleIDS()))
        det.append(("fleet", B.StalenessRule()))
    elif fs == "FS-B34":
        high = ["pr_absdev_max", "pr_gt_two", "pr_lt_half", "unseen_rate", "bus_rate_norm",
                "dlc_mismatch_rate", "ham_mean", "dt_cv", "id_max_share", "allff_frac"]
        two = ["id_entropy", "id_unique_frac", "dt_mean", "ent_mean"]
        det.append(("fleet", B.CombinedRuleIDS(name="Rule:Combined", high=high, two_sided=two)))
    return det


def control_A(target, work, out, cap_train):
    """Volume-matched cross-domain training."""
    import bench_lib as BL
    import protocol as PR
    D = BL.load_domains(work)
    dt = D[target]
    others = {k: v for k, v in D.items() if k != target}
    rows = []
    for fs in ("FS-A", "FS-B"):
        f0 = PR.in_domain_folds(dt.y, dt.q)[0]
        rng0 = np.random.default_rng(BL.job_seed("in_domain_f0", target, fs))
        sel_in = f0["train"][PR.subsample_train(dt.y[f0["train"]], cap_train, rng0)]
        n_total, n_pos = int(len(sel_in)), int(dt.y[sel_in].sum())
        cs = PR.cross_split(dt.y, dt.q)
        vocab = BL.pooled_vocab(list(others.values())) if fs == "FS-A" else None
        Xt = dt.X(fs, vocab)
        for seed in range(3):
            rng = np.random.default_rng(BL.job_seed("ctrlA", target, fs, seed))
            Xpool, ypool, _, sel, _, _ = BL.build_pool(others, fs, vocab, cap_train, rng,
                                                       volume=(n_total, n_pos))
            for scope, m in _models(fs, _THREADS):
                for ev_name, calib, ev in (("cross", cs["calib"], cs["eval"]),
                                           ("in_f0", f0["calib"], f0["eval"])):
                    try:
                        rows.append(_fit_eval(m, scope, Xpool, ypool, sel, Xt, calib, ev, dt,
                                              dict(control="A_volume_matched", test_domain=target,
                                                   feature_set=fs, seed=seed, eval_set=ev_name,
                                                   matched_n=n_total, matched_pos=n_pos)))
                    except Exception as e:      # noqa: BLE001
                        print(f"!! A {target} {fs} {m.name}: {e}", flush=True)
        print(f"[A] {target} {fs} done ({len(rows)} rows)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, f"ctrlA_{target}.csv"), index=False)
    return len(rows)


def control_B(target, work, out, budget=40_000, max_subsets=5):
    """Fleet-size sweep at fixed training volume."""
    import bench_lib as BL
    import protocol as PR
    D = BL.load_domains(work)
    dt = D[target]
    others = sorted(k for k in D if k != target)
    cs = PR.cross_split(dt.y, dt.q)
    rng_sub = np.random.default_rng(BL.job_seed("ctrlB-subsets", target))
    subsets = []
    for k in range(1, len(others) + 1):
        combos = list(itertools.combinations(others, k))
        if len(combos) > max_subsets:
            pick = rng_sub.choice(len(combos), max_subsets, replace=False)
            combos = [combos[i] for i in sorted(pick)]
        subsets += [(k, c) for c in combos]
    rows = []
    for fs in ("FS-B", "FS-A"):
        for k, combo in subsets:
            src = {c: D[c] for c in combo}
            vocab = BL.pooled_vocab(list(src.values())) if fs == "FS-A" else None
            Xt = dt.X(fs, vocab)
            for seed in range(2):
                rng = np.random.default_rng(BL.job_seed("ctrlB", target, fs, k, "|".join(combo), seed))
                Xpool, ypool, _, sel, _, _ = BL.build_pool(src, fs, vocab, None, rng,
                                                           volume=(budget, budget // 2))
                models = _models(fs, _THREADS)
                models = [(s, m) for s, m in models if m.name in ("XGBoost", "HistGB", "Rule:Combined")]
                for scope, m in models:
                    try:
                        rows.append(_fit_eval(m, scope, Xpool, ypool, sel, Xt, cs["calib"], cs["eval"], dt,
                                              dict(control="B_fleet_size", test_domain=target,
                                                   feature_set=fs, seed=seed, k=k,
                                                   sources="|".join(combo))))
                    except Exception as e:      # noqa: BLE001
                        print(f"!! B {target} {fs} k={k} {m.name}: {e}", flush=True)
        print(f"[B] {target} {fs} done ({len(rows)} rows)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, f"ctrlB_{target}.csv"), index=False)
    return len(rows)


def control_C(target, work, out, cap_train):
    """Feature-space ablation, cross-domain."""
    import bench_lib as BL
    import protocol as PR
    D = BL.load_domains(work)
    dt = D[target]
    others = {k: v for k, v in D.items() if k != target}
    cs = PR.cross_split(dt.y, dt.q)
    rows = []
    for fs in ("FS-B34", "FS-B", "FS-A-rank", "FS-A"):
        vocab = BL.pooled_vocab(list(others.values())) if fs == "FS-A" else None
        Xt = dt.X(fs, vocab)
        rng = np.random.default_rng(BL.job_seed("cross_domain_f0", target, fs if fs in ("FS-A", "FS-B") else "FS-B"))
        Xpool, ypool, _, sel, _, _ = BL.build_pool(others, fs, vocab, cap_train, rng)
        for scope, m in _models(fs, _THREADS):
            try:
                rows.append(_fit_eval(m, scope, Xpool, ypool, sel, Xt, cs["calib"], cs["eval"], dt,
                                      dict(control="C_feature_ablation", test_domain=target,
                                           feature_set=fs, seed=0)))
            except Exception as e:      # noqa: BLE001
                print(f"!! C {target} {fs} {m.name}: {e}", flush=True)
        print(f"[C] {target} {fs} done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, f"ctrlC_{target}.csv"), index=False)
    return len(rows)


def _run(job):
    kind, target, work, out, cap = job
    return {"A": control_A, "B": control_B, "C": control_C}[kind](
        target, work, out, cap) if kind != "B" else control_B(target, work, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cap-train", type=int, default=150_000)
    ap.add_argument("--controls", nargs="+", default=["A", "B", "C"])
    args = ap.parse_args()
    outd = os.path.join(args.out, "controls")
    os.makedirs(outd, exist_ok=True)
    domains = sorted(os.listdir(os.path.join(args.work, "features")))
    jobs = [(k, t, args.work, outd, args.cap_train) for k in args.controls for t in domains
            if not os.path.exists(os.path.join(outd, f"ctrl{k}_{t}.csv"))]
    print(f"{len(jobs)} control jobs", flush=True)
    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.threads,)) as ex:
        futs = {ex.submit(_run, j): j for j in jobs}
        for f in as_completed(futs):
            j = futs[f]
            try:
                print(f"== {j[0]} {j[1]}: {f.result()} rows ({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:      # noqa: BLE001
                print(f"== FAILED {j[0]} {j[1]}: {type(e).__name__}: {e}", flush=True)
    for k in "ABC":
        fs = sorted(f for f in os.listdir(outd) if f.startswith(f"ctrl{k}_"))
        if fs:
            pd.concat([pd.read_csv(os.path.join(outd, f)) for f in fs], ignore_index=True).to_csv(
                os.path.join(args.out, f"controls_{k}.csv"), index=False)
    print(f"controls done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
