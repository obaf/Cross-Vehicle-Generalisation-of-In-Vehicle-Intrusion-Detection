"""
analyse.py -- tables, figures, uncertainty and tests from results_raw.csv, the
saved score archives and the control experiments.

    python src/analyse.py --results results --work <local work dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as EV  # noqa: E402

PRIMARY = "recall@fa10"
FPRKEY = "recall@fpr0.001"
PALETTE = {"FS-A": "#C44E52", "FS-B": "#4C72B0", "FS-B34": "#8FA9D3", "FS-A-rank": "#E4A0A4"}
FAM_ORDER = ["dos", "fuzzing", "spoofing", "replay", "diagnostic", "timing", "suspension", "masquerade"]


def is_rule(m):
    return m.startswith("Rule:")


def is_target(m):
    return m.endswith("@target")


def fold_mean(df, protocol):
    """Mean over folds within a domain, then the per-domain table."""
    sub = df[df.protocol == protocol]
    return sub.groupby(["feature_set", "model", "test_domain"]).mean(numeric_only=True).reset_index()


def table_headline(df):
    ind = fold_mean(df, "in_domain"); crd = fold_mean(df, "cross_domain")
    rows = []
    for (fs, model), g in crd.groupby(["feature_set", "model"]):
        gi = ind[(ind.feature_set == fs) & (ind.model == model.replace("@target", ""))]
        r = dict(feature_set=fs, model=model,
                 cross_pr=g.pr_auc.mean(), cross_roc=g.roc_auc.mean(),
                 cross_rec=g[PRIMARY].mean(), cross_rec_fpr=g[FPRKEY].mean(),
                 cross_event=g["event_recall@fa10"].mean(),
                 cross_pr_min=g.pr_auc.min(), cross_pr_sd=g.pr_auc.std(),
                 n_domains=len(g))
        if len(gi):
            r.update(in_pr=gi.pr_auc.mean(), in_roc=gi.roc_auc.mean(),
                     in_rec=gi[PRIMARY].mean(), in_rec_fpr=gi[FPRKEY].mean())
            r["gap_pr"] = r["in_pr"] - r["cross_pr"]
            r["retention_pr"] = 100 * r["cross_pr"] / max(r["in_pr"], 1e-9)
            r["gap_rec"] = r["in_rec"] - r["cross_rec"]
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["feature_set", "cross_pr"], ascending=[True, False])


def table_per_domain(df, protocol="cross_domain", metric="pr_auc"):
    return fold_mean(df, protocol).pivot_table(index=["feature_set", "model"],
                                               columns="test_domain", values=metric)


def table_per_family(df, budget="fa10", protocol="cross_domain", prefix="recall"):
    cols = [c for c in df.columns if c.startswith(f"{prefix}[") and c.endswith(f"@{budget}")]
    sub = df[df.protocol == protocol]
    out = sub.groupby(["feature_set", "model"])[cols].mean()
    out.columns = [c.replace(f"{prefix}[", "").replace(f"]@{budget}", "") for c in out.columns]
    return out[[c for c in FAM_ORDER if c in out.columns]]


def rank_instability(df, fs="FS-B", metric="pr_auc"):
    sub = fold_mean(df, "cross_domain")
    sub = sub[sub.feature_set == fs]
    piv = sub.pivot_table(index="model", columns="test_domain", values=metric)
    ranks = piv.rank(ascending=False, axis=0)
    return pd.DataFrame(dict(mean_rank=ranks.mean(axis=1), best_rank=ranks.min(axis=1),
                             worst_rank=ranks.max(axis=1),
                             rank_range=ranks.max(axis=1) - ranks.min(axis=1),
                             mean_metric=piv.mean(axis=1), min_metric=piv.min(axis=1),
                             max_metric=piv.max(axis=1))).sort_values("mean_rank")


def table_resolvability(df):
    sub = df[(df.protocol == "cross_domain") & (df.model == "XGBoost") & (df.feature_set == "FS-B")]
    cols = ["test_domain", "n_calib", "win_seconds", "n_windows", "n_pos",
            "resolvable@fa1", "resolvable@fa10", "resolvable@fa100", "resolvable@fpr0.001",
            "fa_per_hour@fa1", "fa_per_hour@fa10", "fa_per_hour@fa100"]
    t = sub[cols].copy()
    t["windows_per_hour"] = 3600 / t.win_seconds
    t["min_calib_for_fa1"] = np.ceil(t.windows_per_hour).astype(int)
    t["min_calib_for_fa10"] = np.ceil(t.windows_per_hour / 10).astype(int)
    return t.sort_values("test_domain")


def paired_tests(df):
    """Paired comparisons across (model, domain) cells."""
    out = []
    crd = fold_mean(df, "cross_domain"); ind = fold_mean(df, "in_domain")
    shared = sorted(set(crd[crd.feature_set == "FS-A"].model) & set(crd[crd.feature_set == "FS-B"].model))
    for metric in ("pr_auc", "roc_auc", PRIMARY, FPRKEY):
        a = crd[(crd.feature_set == "FS-A") & crd.model.isin(shared)].set_index(["model", "test_domain"])[metric]
        b = crd[(crd.feature_set == "FS-B") & crd.model.isin(shared)].set_index(["model", "test_domain"])[metric]
        j = pd.concat([a.rename("A"), b.rename("B")], axis=1).dropna()
        w = stats.wilcoxon(j.B, j.A, alternative="greater") if len(j) > 5 else None
        out.append(dict(comparison="FS-B > FS-A (cross-vehicle)", metric=metric, n_pairs=len(j),
                        mean_diff=float((j.B - j.A).mean()), median_diff=float((j.B - j.A).median()),
                        frac_B_better=float((j.B > j.A).mean()),
                        wilcoxon_p=float(w.pvalue) if w else np.nan))
        # in-domain vs cross for FS-B (learned fleet-fitted models only)
        for fs in ("FS-A", "FS-B"):
            ci = crd[(crd.feature_set == fs) & ~crd.model.str.contains("@target")].set_index(["model", "test_domain"])[metric]
            ii = ind[(ind.feature_set == fs)].set_index(["model", "test_domain"])[metric]
            j2 = pd.concat([ii.rename("in"), ci.rename("cross")], axis=1).dropna()
            w2 = stats.wilcoxon(j2["in"], j2["cross"]) if len(j2) > 5 else None
            out.append(dict(comparison=f"in-domain vs cross-vehicle ({fs})", metric=metric, n_pairs=len(j2),
                            mean_diff=float((j2["in"] - j2["cross"]).mean()),
                            median_diff=float((j2["in"] - j2["cross"]).median()),
                            frac_B_better=float((j2["in"] > j2["cross"]).mean()),
                            wilcoxon_p=float(w2.pvalue) if w2 else np.nan))
    return pd.DataFrame(out)


def domain_bootstrap(df, n_boot=2000, seed=0):
    """95% CI of the across-domain mean, bootstrapping over held-out vehicles."""
    rng = np.random.default_rng(seed)
    crd = fold_mean(df, "cross_domain")
    rows = []
    for (fs, model), g in crd.groupby(["feature_set", "model"]):
        for metric in ("pr_auc", PRIMARY):
            v = g[metric].values
            if len(v) < 2:
                continue
            b = rng.choice(v, (n_boot, len(v)), replace=True).mean(axis=1)
            rows.append(dict(feature_set=fs, model=model, metric=metric, mean=v.mean(),
                             ci_lo=np.percentile(b, 2.5), ci_hi=np.percentile(b, 97.5)))
    return pd.DataFrame(rows)


def capture_bootstrap(work, df, models, n_boot=200):
    """Cluster bootstrap over captures per fold for selected models (cross-domain)."""
    rows = []
    for f in sorted(glob.glob(os.path.join(work, "scores", "cross_domain_f0_*.npz"))):
        tag = os.path.basename(f)[:-4]
        _, _, rest = tag.partition("f0_")
        target, _, fs = rest.rpartition("_")
        z = np.load(f)
        y, cap = z["y"], z["cap"]
        for m in models:
            k = f"ev::{m}"
            if k not in z:
                continue
            lo, hi = EV.cluster_bootstrap(z[k].astype(np.float64), y, cap, n_boot=n_boot, seed=1)
            rows.append(dict(feature_set=fs, model=m, test_domain=target,
                             pr_auc=float(average_precision_score(y, z[k])), ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


def road_without_masquerade(work, df):
    """ROAD re-scored with masquerade windows removed from the evaluation set."""
    rows = []
    fam_code = {"masquerade": 7}
    for fs in ("FS-A", "FS-B"):
        f = os.path.join(work, "scores", f"cross_domain_f0_ROAD_ORNL_{fs}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        y, fam = z["y"], z["fam"]
        keep = fam != fam_code["masquerade"]
        for k in z.files:
            if not k.startswith("ev::"):
                continue
            m = k[4:]; s = z[k].astype(np.float64)
            rows.append(dict(feature_set=fs, model=m,
                             pr_auc_all=float(average_precision_score(y, s)),
                             pr_auc_no_masq=float(average_precision_score(y[keep], s[keep])),
                             roc_auc_all=float(roc_auc_score(y, s)),
                             roc_auc_no_masq=float(roc_auc_score(y[keep], s[keep]))))
    return pd.DataFrame(rows)


def cross_full_on_infold(work, df):
    """The full-pool cross-domain detector re-scored on exactly the in-domain
    fold-0 evaluation windows (a subset of the cross-domain evaluation set), so
    that control A can compare in-domain, volume-matched cross-domain and
    full-pool cross-domain training on identical windows."""
    rows = []
    for fs in ("FS-A", "FS-B"):
        for target in sorted(df.test_domain.unique()):
            fc = os.path.join(work, "scores", f"cross_domain_f0_{target}_{fs}.npz")
            fi = os.path.join(work, "scores", f"in_domain_f0_{target}_{fs}.npz")
            if not (os.path.exists(fc) and os.path.exists(fi)):
                continue
            zc, zi = np.load(fc), np.load(fi)
            pos = {int(i): k for k, i in enumerate(zc["eval_idx"])}
            sel = np.array([pos[int(i)] for i in zi["eval_idx"] if int(i) in pos])
            y = zc["y"][sel]
            if y.sum() == 0 or y.sum() == len(y):
                continue
            for k in zc.files:
                if k.startswith("ev::"):
                    s = zc[k][sel].astype(np.float64)
                    rows.append(dict(feature_set=fs, model=k[4:], test_domain=target,
                                     eval_set="in_f0", pr_auc=float(average_precision_score(y, s)),
                                     roc_auc=float(roc_auc_score(y, s)), n_eval=int(len(y)),
                                     covered=float(len(sel) / len(zi["eval_idx"]))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def fig_gap(head, path):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.5), sharey=False)
    for ax, fs in zip(axes, ["FS-A", "FS-B"]):
        h = head[(head.feature_set == fs) & head.in_pr.notna()].sort_values("cross_pr")
        yy = np.arange(len(h))
        ax.barh(yy + .2, h.in_pr, .38, label="in-domain (2-fold, all families both sides)",
                color="#BBBBBB", edgecolor="#555555")
        ax.barh(yy - .2, h.cross_pr, .38, label="cross-vehicle (leave-one-vehicle-out)",
                color=PALETTE[fs], edgecolor="#333333")
        ax.set_yticks(yy); ax.set_yticklabels(h.model, fontsize=8)
        ax.set_xlabel("PR-AUC (mean over held-out vehicles)"); ax.set_title(fs, fontsize=12)
        ax.grid(axis="x", alpha=.3); ax.set_xlim(0, 1)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("In-domain versus cross-vehicle PR-AUC, both feature spaces", fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def fig_heat(t, path, title, cmap="viridis", vmin=0, vmax=1, fmt="{:.2f}", label="PR-AUC"):
    t = t.loc[t.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(1.3 * t.shape[1] + 4.5, .38 * len(t) + 2.6))
    im = ax.imshow(t.values.astype(float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(t.shape[1])); ax.set_xticklabels(t.columns, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(t))); ax.set_yticklabels(t.index, fontsize=8.5)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5,
                        color="w" if (cmap == "viridis" and v < .6) else "k")
    ax.set_title(title, fontsize=12)
    fig.colorbar(im, ax=ax, shrink=.8, label=label)
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def control_A_table(A, df, cf):
    """Three training regimes on identical evaluation windows (in-domain fold 0)."""
    if A is None or not len(A):
        return None
    ind = df[(df.protocol == "in_domain") & (df.fold == 0)].groupby(["feature_set", "model", "test_domain"]).pr_auc.mean().rename("in_domain")
    cm = A[A.eval_set == "in_f0"].groupby(["feature_set", "model", "test_domain"]).pr_auc.mean().rename("cross_matched")
    t = pd.concat([ind, cm], axis=1)
    if cf is not None and len(cf):
        t = t.join(cf.groupby(["feature_set", "model", "test_domain"]).pr_auc.mean().rename("cross_full"))
    return t.dropna(subset=["cross_matched"]).reset_index()


def fig_control_A(tA, path):
    if tA is None or not len(tA):
        return
    g = tA.groupby(["feature_set", "model"])[["in_domain", "cross_matched", "cross_full"]].mean()
    order = g.loc["FS-B"].sort_values("cross_matched").index if "FS-B" in g.index.get_level_values(0) else None
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, fs in zip(axes, ["FS-A", "FS-B"]):
        if fs not in g.index.get_level_values(0):
            continue
        h = g.loc[fs]
        h = h.reindex(order) if order is not None else h.sort_values("cross_matched")
        yy = np.arange(len(h))
        ax.barh(yy + .27, h.in_domain, .26, label="in-domain (same vehicle)", color="#BBBBBB", edgecolor="#555")
        ax.barh(yy, h.cross_matched, .26, label="cross-vehicle, volume-matched", color=PALETTE[fs], edgecolor="#333")
        ax.barh(yy - .27, h.cross_full, .26, label="cross-vehicle, full pool", color=PALETTE[fs], alpha=.45, edgecolor="#333")
        ax.set_yticks(yy); ax.set_yticklabels(h.index, fontsize=9); ax.set_xlim(0, 1)
        ax.set_title(fs); ax.set_xlabel("PR-AUC"); ax.grid(axis="x", alpha=.3)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Control A: is it the vehicle change or the training volume?", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def fig_control_B(B, path):
    if B is None or not len(B):
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, fs in zip(axes, ["FS-B", "FS-A"]):
        sub = B[B.feature_set == fs]
        for m, ls in (("XGBoost", "-"), ("HistGB", "--"), ("Rule:Combined", ":")):
            s = sub[sub.model == m]
            if not len(s):
                continue
            g = s.groupby("k").pr_auc
            mu, sd = g.mean(), g.std()
            ax.errorbar(mu.index, mu.values, yerr=sd.values, ls=ls, marker="o", capsize=3,
                        color=PALETTE[fs], label=m, alpha=.9 if m == "XGBoost" else .6)
        ax.set_xlabel("number of training vehicles k (fixed volume)"); ax.set_title(fs)
        ax.grid(alpha=.3); ax.set_ylim(0, 1)
    axes[0].set_ylabel("cross-vehicle PR-AUC (mean over held-out vehicles)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Control B: fleet diversity at fixed training volume", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def fig_control_C(C, path):
    if C is None or not len(C):
        return
    g = C.groupby(["feature_set", "model"]).pr_auc.mean().unstack(0)
    order = ["FS-A", "FS-A-rank", "FS-B34", "FS-B"]
    g = g[[c for c in order if c in g.columns]]
    g = g.loc[g.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    w = .8 / len(g.columns); yy = np.arange(len(g))
    for i, c in enumerate(g.columns):
        ax.barh(yy + (i - len(g.columns) / 2 + .5) * w, g[c], w, label=c, color=PALETTE.get(c, "#888"), edgecolor="#333")
    ax.set_yticks(yy); ax.set_yticklabels(g.index); ax.set_xlim(0, 1); ax.grid(axis="x", alpha=.3)
    ax.set_xlabel("cross-vehicle PR-AUC"); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Control C: feature-space ablation (cross-vehicle)")
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def fig_ci(dboot, path, fs="FS-B", top=12):
    d = dboot[(dboot.feature_set == fs) & (dboot.metric == "pr_auc")].sort_values("mean").tail(top)
    fig, ax = plt.subplots(figsize=(8, .42 * len(d) + 1.5))
    yy = np.arange(len(d))
    ax.errorbar(d["mean"], yy, xerr=[d["mean"] - d.ci_lo, d.ci_hi - d["mean"]], fmt="o",
                color=PALETTE[fs], capsize=3)
    ax.set_yticks(yy); ax.set_yticklabels(d.model, fontsize=9); ax.grid(axis="x", alpha=.3)
    ax.set_xlabel("cross-vehicle PR-AUC, mean over 8 held-out vehicles (95% bootstrap CI over vehicles)")
    ax.set_title(f"{fs}: the intervals overlap for every learned model")
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--work", required=True)
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()
    R = args.results; figd = os.path.join(R, "figures"); os.makedirs(figd, exist_ok=True)
    pd.set_option("display.width", 250)

    df = pd.read_csv(os.path.join(R, "results_raw.csv"))
    head = table_headline(df); head.to_csv(os.path.join(R, "table_headline.csv"), index=False)
    print("=== headline (cross-vehicle, mean over held-out vehicles) ===")
    print(head[["feature_set", "model", "in_pr", "cross_pr", "gap_pr", "in_rec", "cross_rec",
                "cross_rec_fpr", "cross_event"]].round(3).to_string(index=False))

    dom = table_per_domain(df); dom.to_csv(os.path.join(R, "table_per_domain.csv"))
    table_per_domain(df, "in_domain").to_csv(os.path.join(R, "table_per_domain_indomain.csv"))
    fam = table_per_family(df); fam.to_csv(os.path.join(R, "table_per_family.csv"))
    table_per_family(df, prefix="event_recall").to_csv(os.path.join(R, "table_per_family_event.csv"))
    table_per_family(df, budget="fpr0.001").to_csv(os.path.join(R, "table_per_family_fpr.csv"))
    ri = rank_instability(df); ri.to_csv(os.path.join(R, "table_rank_instability.csv"))
    res = table_resolvability(df); res.to_csv(os.path.join(R, "table_resolvability.csv"), index=False)
    tests = paired_tests(df); tests.to_csv(os.path.join(R, "table_paired_tests.csv"), index=False)
    print("\n=== paired tests ===\n", tests.round(4).to_string(index=False))
    dboot = domain_bootstrap(df); dboot.to_csv(os.path.join(R, "table_ci_domains.csv"), index=False)

    top = head[head.feature_set == "FS-B"].head(6).model.tolist() + ["Rule:Combined", "Rule:Combined@target"]
    cb = capture_bootstrap(args.work, df, sorted(set(top)), n_boot=args.n_boot)
    cb.to_csv(os.path.join(R, "table_ci_captures.csv"), index=False)
    rm = road_without_masquerade(args.work, df); rm.to_csv(os.path.join(R, "table_road_masquerade.csv"), index=False)
    cf = cross_full_on_infold(args.work, df); cf.to_csv(os.path.join(R, "table_control_A_fullpool.csv"), index=False)

    A = B = C = None
    for k in "ABC":
        p = os.path.join(R, f"controls_{k}.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            if k == "A":
                A = d
            elif k == "B":
                B = d
            else:
                C = d
    tA = control_A_table(A, df, cf)
    if tA is not None:
        tA.to_csv(os.path.join(R, "table_control_A.csv"), index=False)
        tA.groupby(["feature_set", "model"])[["in_domain", "cross_matched", "cross_full"]].mean().to_csv(
            os.path.join(R, "table_control_A_summary.csv"))
    if B is not None:
        tB = B.groupby(["feature_set", "model", "k"])[["pr_auc", "roc_auc", PRIMARY, "n_train", "n_train_pos"]].agg(["mean", "std"]).reset_index()
        tB.to_csv(os.path.join(R, "table_control_B.csv"), index=False)
    if C is not None:
        tC = C.groupby(["feature_set", "model"])[["pr_auc", "roc_auc", PRIMARY]].mean().unstack(0)
        tC.to_csv(os.path.join(R, "table_control_C.csv"))

    fig_gap(head, os.path.join(figd, "fig1_generalisation_gap.png"))
    fig_heat(dom.loc["FS-B"], os.path.join(figd, "fig2_domain_heatmap.png"),
             "Cross-vehicle PR-AUC by held-out vehicle (FS-B)")
    fb = fam.loc["FS-B"].dropna(how="all")
    fb = fb.loc[fb.mean(axis=1).sort_values(ascending=False).index[:14]]
    fig_heat(fb, os.path.join(figd, "fig3_family_recall.png"),
             "Cross-vehicle recall by attack family at 10 false alarms/hour (FS-B)",
             cmap="RdYlGn", label="recall")
    fig_control_A(tA, os.path.join(figd, "fig4_control_volume_matched.png"))
    fig_control_B(B, os.path.join(figd, "fig5_control_fleet_size.png"))
    fig_control_C(C, os.path.join(figd, "fig6_control_feature_ablation.png"))
    fig_ci(dboot, os.path.join(figd, "fig7_ci_forest.png"))

    hb = head[head.feature_set == "FS-B"]; ha = head[head.feature_set == "FS-A"]
    summary = dict(
        n_rows=int(len(df)), models=sorted(df.model.unique().tolist()),
        domains=sorted(df.test_domain.unique().tolist()),
        best_cross_FSB=hb.iloc[0].to_dict(), best_cross_FSA=ha.iloc[0].to_dict(),
        best_rule_FSB=hb[hb.model.str.startswith("Rule:")].iloc[0].to_dict(),
        n_configs_cross_exceeds_in_FSB=int((hb.gap_pr < 0).sum()),
        n_configs_with_in_FSB=int(hb.gap_pr.notna().sum()),
        label_fixes={d: json.load(open(os.path.join(args.work, "features", d, "meta.json"))).get("label_fixes", {})
                     for d in sorted(os.listdir(os.path.join(args.work, "features")))},
    )
    json.dump(summary, open(os.path.join(R, "summary.json"), "w"), indent=1, default=str)
    print(f"\nwrote tables and figures to {R}")


if __name__ == "__main__":
    main()
