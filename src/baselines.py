"""
baselines.py -- deterministic, non-AI detectors and one unsupervised learned
baseline. All are fitted on benign windows only and emit a continuous
anomaly score, so every curve is computed exactly as for the learned models.

Two fitting scopes are evaluated cross-vehicle:

  fleet   the envelope is estimated on the *other* vehicles' benign traffic
          (a transfer test of the rule itself, as in the prior run);
  target  the envelope is estimated on the held-out vehicle's own benign
          calibration windows, which the protocol explicitly permits. This is
          how a rule would actually be deployed and is the honest benchmark
          floor.

The scores of a target-fitted rule are computed on the same benign windows
its threshold is calibrated on; for a median/IQR envelope that in-sample effect
is negligible and the realised false-alarm rate on the disjoint evaluation set
is reported for every detector so that any optimism is visible.
"""

from __future__ import annotations

import numpy as np

from features import IDX_B


class _RobustZ:
    def fit(self, x):
        self.med_ = np.median(x, axis=0)
        q1, q3 = np.percentile(x, [25, 75], axis=0)
        self.scale_ = np.maximum(q3 - q1, 1e-6)
        return self

    def z(self, x):
        return (x - self.med_) / self.scale_


class _Detector:
    needs_seq = False
    benign_only = True
    threads = 1


class SingleFeatureRule(_Detector):
    def __init__(self, feature, side="high", name=None):
        self.feature, self.side = feature, side
        self.name = name or f"Rule[{feature}]"

    def fit(self, Fb, y=None, S=None):
        self.s_ = _RobustZ().fit(Fb[:, IDX_B[self.feature]].reshape(-1, 1))
        return self

    def score(self, F, S=None):
        z = self.s_.z(F[:, IDX_B[self.feature]].reshape(-1, 1)).ravel()
        return z if self.side == "high" else -z


class UnseenIdentifierRule(SingleFeatureRule):
    def __init__(self):
        super().__init__("unseen_rate", "high", "Rule:UnseenID")

    def score(self, F, S=None):
        return F[:, IDX_B["unseen_rate"]] * 1e3


class PeriodicityRule(SingleFeatureRule):
    def __init__(self):
        super().__init__("pr_absdev_max", "high", "Rule:Periodicity")


class BusLoadRule(SingleFeatureRule):
    def __init__(self):
        super().__init__("bus_rate_norm", "high", "Rule:BusLoad")


class EntropyRule(SingleFeatureRule):
    def __init__(self):
        super().__init__("id_entropy", "low", "Rule:IDEntropy")


class HammingRule(SingleFeatureRule):
    def __init__(self):
        super().__init__("ham_mean", "high", "Rule:Hamming")


class StalenessRule(SingleFeatureRule):
    """Flags a periodic identifier that has stopped arriving: the suspension
    detector that the prior feature set could not express."""

    def __init__(self):
        super().__init__("stale_max", "high", "Rule:Staleness")


class CombinedRuleIDS(_Detector):
    """Max robust z over a panel of statistics: fires as soon as any monitored
    quantity leaves its attack-free envelope."""
    name = "Rule:Combined"
    HIGH = ["pr_absdev_max", "pr_gt_two", "pr_lt_half", "unseen_rate",
            "bus_rate_norm", "dlc_mismatch_rate", "ham_mean", "dt_cv",
            "id_max_share", "allff_frac", "stale_max"]
    TWO_SIDED = ["id_entropy", "id_unique_frac", "dt_mean_norm", "ent_mean"]

    def __init__(self, name="Rule:Combined", high=None, two_sided=None):
        self.name = name
        self.HIGH = high or self.HIGH
        self.TWO_SIDED = two_sided or self.TWO_SIDED

    def fit(self, Fb, y=None, S=None):
        self.cols_ = [IDX_B[c] for c in self.HIGH + self.TWO_SIDED]
        self.n_high_ = len(self.HIGH)
        self.s_ = _RobustZ().fit(Fb[:, self.cols_])
        return self

    def score(self, F, S=None):
        z = self.s_.z(F[:, self.cols_])
        return np.maximum(z[:, : self.n_high_].max(axis=1),
                          np.abs(z[:, self.n_high_:]).max(axis=1))


class QuantileCombinedRuleIDS(CombinedRuleIDS):
    """Union of the same panel, but each statistic is mapped through its own
    benign empirical CDF to a tail surprisal -log(1 - F) before the maximum is
    taken. The robust-z union is not scale-safe: a fraction-valued statistic
    whose benign interquartile range is ~0 produces z-scores in the tens of
    thousands on ordinary benign windows, and those set the alarm threshold
    for the whole union (on the Opel Astra the 10 FA/h threshold of Rule:Combined
    is a z of 78,000 while a suspension window scores 5,500). Quantiles are
    dimensionless and bounded, so no statistic can drown the others; a small
    robust-z term breaks ties beyond the largest benign value."""
    name = "Rule:CombinedQ"

    def __init__(self, name="Rule:CombinedQ", high=None, two_sided=None):
        super().__init__(name=name, high=high, two_sided=two_sided)

    def fit(self, Fb, y=None, S=None):
        super().fit(Fb)
        Z = self.s_.z(Fb[:, self.cols_])
        Z = np.concatenate([Z[:, : self.n_high_], np.abs(Z[:, self.n_high_:])], axis=1)
        self.ref_ = [np.sort(Z[:, j]) for j in range(Z.shape[1])]
        self.p99_ = np.percentile(Z, 99, axis=0)
        self.mx_ = Z.max(axis=0)
        return self

    def score(self, F, S=None):
        Z = self.s_.z(F[:, self.cols_])
        Z = np.concatenate([Z[:, : self.n_high_], np.abs(Z[:, self.n_high_:])], axis=1)
        out = np.empty_like(Z)
        for j, ref in enumerate(self.ref_):
            n = len(ref)
            q = np.searchsorted(ref, Z[:, j], side="right") / (n + 1.0)
            # beyond the largest benign value the data cannot order features;
            # exceedance is measured in units of the feature's own benign
            # upper-tail spread (p99..max), which is dimensionless and bounded
            # by the tail the data did exhibit
            tail = max(self.mx_[j] - self.p99_[j], 1e-9)
            exc = np.log1p(np.maximum(0.0, (Z[:, j] - self.mx_[j]) / tail))
            out[:, j] = -np.log(1.0 - q) + exc
        return out.max(axis=1)


class IsolationForestDetector(_Detector):
    """Unsupervised learned baseline: benign-only isolation forest."""
    name = "IForest"

    def __init__(self, n_estimators=200, seed=20260907, threads=1):
        self.n_estimators, self.seed, self.threads = n_estimators, seed, threads

    def fit(self, Fb, y=None, S=None):
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        self.sc_ = StandardScaler().fit(Fb)
        self.est_ = IsolationForest(n_estimators=self.n_estimators, max_samples=min(4096, len(Fb)),
                                    random_state=self.seed, n_jobs=self.threads)
        self.est_.fit(self.sc_.transform(Fb))
        return self

    def score(self, F, S=None):
        return -self.est_.score_samples(self.sc_.transform(F))


def rule_baselines():
    return [PeriodicityRule(), UnseenIdentifierRule(), BusLoadRule(), EntropyRule(),
            HammingRule(), StalenessRule(), CombinedRuleIDS(), QuantileCombinedRuleIDS()]


def unsupervised_baselines(threads=1):
    return [IsolationForestDetector(threads=threads)]
