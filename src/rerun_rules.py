"""
rerun_rules.py -- recompute only the deterministic rule detectors for every
job (both fitting scopes), replacing their rows in results/rows/*.csv and their
scores in the archives. Used to add Rule:CombinedQ after the main run without
refitting any learned model. Learned-model rows and scores are untouched.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines as B      # noqa: E402
import bench_lib as BL     # noqa: E402
import evaluate as EV      # noqa: E402
import protocol as PR      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--cap-train", type=int, default=150_000)
    a = ap.parse_args()
    D = BL.load_domains(a.work)
    for f in sorted(glob.glob(os.path.join(a.out, "rows", "*_FS-B.csv"))):
        tag = os.path.basename(f)[:-4]
        protocol, rest = tag.split("_f", 1)
        fold, rest = int(rest[0]), rest[2:]
        target = rest[: -len("_FS-B")]
        spec = dict(protocol=protocol, fold=fold, target=target, fs="FS-B")
        rng = np.random.default_rng(BL.job_seed(tag))
        dt = D[target]; cross = protocol == "cross_domain"
        if cross:
            others = {k: v for k, v in D.items() if k != target}
            Xpool, ypool, _, sel, _, _ = BL.build_pool(others, "FS-B", None, a.cap_train, rng)
            Xt = dt.X("FS-B"); sp = PR.cross_split(dt.y, dt.q); calib, ev = sp["calib"], sp["eval"]
        else:
            Xt = dt.X("FS-B"); fo = PR.in_domain_folds(dt.y, dt.q)[fold]
            Xpool, ypool = Xt, dt.y
            sel = fo["train"][PR.subsample_train(dt.y[fo["train"]], a.cap_train, rng)]
            calib, ev = fo["calib"], fo["eval"]
        neg = sel[ypool[sel] == 0]
        rows = pd.read_csv(f)
        rows = rows[~rows.model.str.startswith("Rule:")]
        zpath = os.path.join(a.work, "scores", tag + ".npz")
        z = dict(np.load(zpath))
        for k in [k for k in z if k.split("::", 1)[-1].startswith("Rule:")]:
            del z[k]
        new = []
        dets = [("fleet", r) for r in B.rule_baselines()]
        if cross:
            for r in B.rule_baselines():
                r.name += "@target"; dets.append(("target", r))
        for scope, m in dets:
            m.fit(Xt[calib] if scope == "target" else Xpool[neg])
            s_ev, s_cal = m.score(Xt[ev]), m.score(Xt[calib])
            r = EV.evaluate(s_ev, dt.y[ev], dt.fam_names[ev], dt.cap[ev], dt.win_seconds, s_cal)
            r.update(protocol=protocol, fold=fold, test_domain=target, feature_set="FS-B", model=m.name,
                     scope=scope, fit_s=0.0, score_s=0.0, n_train=int(len(sel)),
                     n_train_pos=int(ypool[sel].sum()), dim=int(Xt.shape[1]))
            new.append(r)
            z[f"ev::{m.name}"] = s_ev.astype(np.float32); z[f"cal::{m.name}"] = s_cal.astype(np.float32)
        out = pd.concat([pd.DataFrame(new), rows], ignore_index=True)
        out.to_csv(f, index=False)
        np.savez_compressed(zpath, **z)
        cq = [r for r in new if r["model"].startswith("Rule:CombinedQ")]
        print(f"{tag}: {len(new)} rule rows; CombinedQ PR-AUC="
              + "/".join(f"{r['pr_auc']:.3f}" for r in cq)
              + " susp@10FA=" + "/".join(f"{r.get('recall[suspension]@fa10', float('nan')):.2f}" for r in cq), flush=True)
    allrows = [pd.read_csv(x) for x in sorted(glob.glob(os.path.join(a.out, "rows", "*.csv")))]
    df = pd.concat(allrows, ignore_index=True)
    df.to_csv(os.path.join(a.out, "results_raw.csv"), index=False)
    print(len(df), "rows ->", os.path.join(a.out, "results_raw.csv"))


if __name__ == "__main__":
    main()
