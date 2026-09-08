"""
audit_experiment1.py -- reproduce, from experiment1's own artefacts, the
evidence for every fault listed in AUDIT_of_experiment1.md.

    python src/audit_experiment1.py --exp1 ../experiment1

Nothing here is re-run; every statement is checked against the files the
prior experiment left behind (results_raw.csv, run_full.log, the label
validation CSV, the Parquet cache and the source code).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd


def check(title, ok, detail):
    print(f"[{'FAULT' if ok else 'ok   '}] {title}\n        {detail}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp1", default=os.path.join(os.path.dirname(__file__), "..", "..", "experiment1"))
    a = ap.parse_args()
    E = os.path.abspath(a.exp1)
    df = pd.read_csv(os.path.join(E, "results", "results_raw.csv"))
    src = open(os.path.join(E, "src", "run_benchmark.py")).read()
    fsrc = open(os.path.join(E, "src", "features.py")).read()

    # 1. cross-domain training used 1,000 benign windows
    crd = df[(df.protocol == "cross_domain") & (df.model == "XGBoost") & (df.feature_set == "FS-B")]
    ind = df[(df.protocol == "in_domain") & (df.model == "XGBoost") & (df.feature_set == "FS-B")]
    pos_by_dom = crd.set_index("test_domain").n_pos
    total_pos = pos_by_dom.sum()
    n_neg = (crd.set_index("test_domain").n_train - (total_pos - pos_by_dom))
    check("cross-domain training sets contained exactly 1,000 benign windows",
          bool((n_neg == 1000).all()),
          "n_train minus pooled positives of the other seven vehicles = "
          f"{sorted(set(n_neg.astype(int)))} for every held-out vehicle; "
          "source: subsample() uses n_neg = max(cap - len(pos), 1000) with --cap-train 50000")
    assert "max(cap - len(pos), 1000)" in src

    # 2. in-domain split held out whole attack families (CIDv2)
    famcols = [c for c in df.columns if c.startswith("recall[") and c.endswith("@fa1")]
    lines = []
    for _, r in ind.iterrows():
        fams = sorted(c[7:-5] for c in famcols if pd.notna(r[c]))
        allf = sorted(c[7:-5] for c in famcols if pd.notna(crd[crd.test_domain == r.test_domain].iloc[0][c]))
        missing = sorted(set(allf) - set(fams))
        if missing:
            lines.append(f"{r.test_domain}: tested on {fams}; families absent from the test half: {missing}")
    check("in-domain protocol held out whole attack families", bool(lines), "\n        ".join(lines))

    # 3. in-domain seed not reproducible
    out = [subprocess.run([sys.executable, "-c", "print(hash('CIDv2_OpelAstra') % 1000)"],
                          capture_output=True, text=True).stdout.strip() for _ in range(3)]
    check("in-domain split seed hash(v) % 1000 differs between processes",
          len(set(out)) > 1, f"three fresh interpreters gave seeds {out}; source: split_by_capture(d, seed=hash(v) % 1000)")

    # 4. primary operating point unresolvable
    check("primary metric recall@fa1 flagged unresolvable on every fold by the code's own check",
          bool((~df["resolvable@fa1"]).all()),
          f"resolvable@fa1 is False in {int((~df['resolvable@fa1']).sum())} of {len(df)} rows; "
          f"resolvable@fa10 True in {int(df['resolvable@fa10'].sum())}")

    # 5. CIDv2 label counts
    v = pd.read_csv(os.path.join(E, "data_cache", "cidv2_label_validation.csv"))
    check("paper states 11 of 14 CIDv2 counts match the README; the validation file says otherwise",
          int(v.ok.sum()) != 11, f"{int(v.ok.sum())} of {len(v)} match; mismatches: "
          + ", ".join(f"{r.capture} ({r.expected} vs {r.labelled})" for r in v[~v.ok].itertuples()))
    fz = pd.read_parquet(os.path.join(E, "data_cache", "CIDv2__CIDv2_OpelAstra__OpelAstra_fuzzing_canid.parquet"),
                         columns=["aid", "data_hex", "y"])
    bad = fz[(fz.y == 1) & ~fz.data_hex.str.fullmatch(r"[0-9A-F]*")]
    check("a corrupted log line was labelled as an injected fuzzing frame",
          len(bad) > 0, f"{len(bad)} labelled frame(s) with non-hex payload: "
          + ", ".join(f"id={hex(r.aid)} payload={r.data_hex!r}" for r in bad.itertuples()))

    # 6. FS-A as implemented is rank-indexed per vehicle, not identity-coupled
    check("FS-A histogram vocabulary is each vehicle's OWN top identifiers (rank-indexed), not the training vehicles'",
          "counts = pd.concat([c.df[\"aid\"] for c in benign]).value_counts()" in src
          and "top_ids = FT.pad_top_ids" in src,
          "build_domain() derives top_ids per domain from that domain's benign traffic and the cross-domain loop "
          "only truncates to a common width; the paper's claim that the histogram 'collapses into its overflow "
          "bucket' on another vehicle does not describe this code")
    check("FS-A contains no payload-byte features although the docstring and paper say it does",
          "b0" not in fsrc.split("def window_features_A")[1].split("def pad_top_ids")[0],
          "window_features_A() uses identifier, inter-arrival and DLC statistics only")

    # 7. 'consistently across 20 detectors'
    h = pd.read_csv(os.path.join(E, "results", "table_headline.csv"))
    p = h.pivot(index="model", columns="feature_set", values="cross_pr").dropna()
    worse = p[p["FS-B"] < p["FS-A"]]
    check("FS-B did not beat FS-A for every detector, contrary to the paper",
          len(worse) > 0, "FS-A > FS-B cross-vehicle for: "
          + ", ".join(f"{m} ({r['FS-A']:.3f} vs {r['FS-B']:.3f})" for m, r in worse.iterrows()))

    # 8. sequence models trained on shuffled, subsampled rows
    msrc = open(os.path.join(E, "src", "models.py")).read()
    check("sequence models stacked adjacent rows of a shuffled, subsampled matrix",
          "def _seq(self, X)" in msrc and "np.vstack([pad, X])" in msrc,
          "TorchNet._seq() builds sequences from row adjacency after subsample() and after vstack over domains")

    # 9. in-domain calibration on test benign vs cross-domain held-out
    check("in-domain thresholds calibrated on the evaluation set's own benign windows; cross-domain on a held-out half",
          "calib_scores=s[yte == 0]" in src, "run_models(): in_domain passes calib_scores=s[yte == 0]")

    # 10. suspension recall zero everywhere with the 34-feature space
    sus = df[(df.protocol == "cross_domain")]["recall[suspension]@fa10"].dropna()
    check("suspension essentially undetected by every detector (no feature could see an absent frame)",
          float(sus.max()) < 0.1, f"max recall[suspension]@fa10 over all cross-domain rows = {sus.max():.3f}; "
          f"mean = {sus.mean():.4f}")

    # 11. window duration for FA/hour conversion averaged over attack captures
    check("win_seconds averaged over all captures including DoS captures, not benign traffic",
          "DUR.append(c.duration / max(nw, 1))" in src and "win_seconds=float(np.mean(DUR))" in src,
          "build_domain(): win_seconds = mean over captures of duration/windows; DoS captures compress it")


if __name__ == "__main__":
    main()
