"""
models.py -- learned detectors and ensembles.

Every model exposes fit(X, y, S=None) / score(X, S=None). X is the per-window
feature matrix; S, supplied only to models with needs_seq=True, is the matrix
of short sequences of *temporally consecutive windows of the same capture*
(n, L, d) that the driver gathers for them. The prior run stacked whatever
rows happened to be adjacent after shuffling and subsampling, so its
convolutional, recurrent and attention models were trained on sequences that
crossed capture and vehicle boundaries at random; here sequence context is
always real.
"""

from __future__ import annotations

import copy

import numpy as np
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SEED = 20260907
SEQ_LEN = 8


class SkWrap:
    needs_seq = False
    benign_only = False

    def __init__(self, est, name, scale=False):
        self.est, self.name, self.scale = est, name, scale

    def fit(self, X, y, S=None):
        self.sc_ = StandardScaler().fit(X) if self.scale else None
        self.est.fit(self.sc_.transform(X) if self.sc_ else X, y)
        return self

    def score(self, X, S=None):
        Xs = self.sc_.transform(X) if self.sc_ else X
        if hasattr(self.est, "predict_proba"):
            return self.est.predict_proba(Xs)[:, 1]
        return self.est.decision_function(Xs)


def make_logreg(threads=1):
    return SkWrap(LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced",
                                     random_state=SEED), "LogReg", scale=True)


def make_rf(threads=1):
    return SkWrap(RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, n_jobs=threads,
        class_weight="balanced_subsample", random_state=SEED), "RandomForest")


def make_et(threads=1):
    return SkWrap(ExtraTreesClassifier(
        n_estimators=300, min_samples_leaf=2, n_jobs=threads,
        class_weight="balanced_subsample", random_state=SEED), "ExtraTrees")


def make_hgb(threads=1):
    return SkWrap(HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
        class_weight="balanced", random_state=SEED), "HistGB")


def make_xgb(threads=1):
    import xgboost as xgb
    return SkWrap(xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.85,
        colsample_bytree=0.85, reg_lambda=2.0, tree_method="hist",
        eval_metric="aucpr", n_jobs=threads, random_state=SEED), "XGBoost")


def make_lgbm(threads=1):
    import lightgbm as lgb
    return SkWrap(lgb.LGBMClassifier(
        n_estimators=500, num_leaves=63, learning_rate=0.06, subsample=0.85,
        subsample_freq=1, colsample_bytree=0.85, reg_lambda=2.0, n_jobs=threads,
        verbose=-1, random_state=SEED), "LightGBM")


def make_cat(threads=1):
    from catboost import CatBoostClassifier
    return SkWrap(CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.08, l2_leaf_reg=3.0, verbose=0,
        allow_writing_files=False, random_seed=SEED, thread_count=threads), "CatBoost")


# --------------------------------------------------------------------------
class TorchNet:
    benign_only = False

    def __init__(self, kind="mlp", epochs=8, bs=512, lr=1e-3, name=None,
                 hidden=128, threads=1):
        self.kind, self.epochs, self.bs, self.lr = kind, epochs, bs, lr
        self.hidden, self.threads = hidden, threads
        self.name = name or f"NN:{kind}"
        self.needs_seq = kind != "mlp"

    def _build(self, d):
        import torch.nn as nn
        H = self.hidden
        if self.kind == "mlp":
            return nn.Sequential(
                nn.Linear(d, H), nn.ReLU(), nn.BatchNorm1d(H), nn.Dropout(0.2),
                nn.Linear(H, H // 2), nn.ReLU(), nn.BatchNorm1d(H // 2),
                nn.Dropout(0.2), nn.Linear(H // 2, 1))
        if self.kind == "cnn":
            class C(nn.Module):
                def __init__(s):
                    super().__init__()
                    s.c = nn.Sequential(nn.Conv1d(d, H, 3, padding=1), nn.ReLU(),
                                        nn.BatchNorm1d(H),
                                        nn.Conv1d(H, H, 3, padding=1), nn.ReLU(),
                                        nn.AdaptiveAvgPool1d(1))
                    s.f = nn.Linear(H, 1)

                def forward(s, x):
                    return s.f(s.c(x.transpose(1, 2)).squeeze(-1))
            return C()
        if self.kind == "lstm":
            class L(nn.Module):
                def __init__(s):
                    super().__init__()
                    s.r = nn.LSTM(d, H, batch_first=True)
                    s.f = nn.Linear(H, 1)

                def forward(s, x):
                    o, _ = s.r(x)
                    return s.f(o[:, -1])
            return L()
        if self.kind == "transformer":
            class T(nn.Module):
                def __init__(s):
                    super().__init__()
                    s.p = nn.Linear(d, H)
                    s.pos = nn.Parameter(__import__("torch").zeros(1, SEQ_LEN, H))
                    layer = nn.TransformerEncoderLayer(H, nhead=4, dim_feedforward=2 * H,
                                                       dropout=0.1, batch_first=True,
                                                       norm_first=True)
                    s.e = nn.TransformerEncoder(layer, num_layers=2)
                    s.f = nn.Linear(H, 1)

                def forward(s, x):
                    h = s.p(x) + s.pos[:, : x.shape[1]]
                    return s.f(s.e(h)[:, -1])
            return T()
        raise ValueError(self.kind)

    def _prep(self, X, S):
        Xs = self.sc_.transform(X).astype(np.float32)
        if not self.needs_seq:
            return Xs
        Ss = (S - self.sc_.mean_.astype(np.float32)) / self.sc_.scale_.astype(np.float32)
        return Ss.astype(np.float32)

    def fit(self, X, y, S=None):
        import torch
        import torch.nn as nn
        torch.manual_seed(SEED)
        torch.set_num_threads(max(1, self.threads))
        self.sc_ = StandardScaler().fit(X)
        inp = self._prep(X, S)
        yt = y.astype(np.float32)
        self.net_ = self._build(X.shape[1])
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
        pos = max(float(yt.sum()), 1.0)
        pw = torch.tensor([(len(yt) - pos) / pos], dtype=torch.float32)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        Xt, Yt = torch.from_numpy(inp), torch.from_numpy(yt)
        n = len(Yt)
        g = torch.Generator().manual_seed(SEED)
        self.net_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.bs):
                b = perm[i:i + self.bs]
                if len(b) < 2:
                    continue
                opt.zero_grad()
                loss = lossf(self.net_(Xt[b]).squeeze(-1), Yt[b])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 5.0)
                opt.step()
        return self

    def score(self, X, S=None):
        import torch
        torch.set_num_threads(max(1, self.threads))
        inp = self._prep(X, S)
        self.net_.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(inp), 8192):
                out.append(torch.sigmoid(self.net_(torch.from_numpy(inp[i:i + 8192]))
                                         .squeeze(-1)).numpy())
        return np.concatenate(out) if out else np.zeros(len(inp))


# --------------------------------------------------------------------------
class RankAverageEnsemble:
    """Rank-normalised soft vote. Each member's score is mapped through the
    empirical CDF of that member's scores on the training set, fixed at fit
    time, so the transform is monotone and independent of the batch being
    scored (ranking within the scored batch would make calibration and
    evaluation scores incomparable and would break under chunked scoring)."""
    benign_only = False

    def __init__(self, members, name="Ens:RankAvg"):
        self.members, self.name = members, name
        self.needs_seq = any(getattr(m, "needs_seq", False) for m in members)

    def fit(self, X, y, S=None):
        self.ref_ = []
        for m in self.members:
            Sm = S if getattr(m, "needs_seq", False) else None
            m.fit(X, y, Sm)
            self.ref_.append(np.sort(np.asarray(m.score(X, Sm), dtype=np.float64)))
        return self

    def score(self, X, S=None):
        cols = []
        for m, ref in zip(self.members, self.ref_):
            s = np.asarray(m.score(X, S if getattr(m, "needs_seq", False) else None), dtype=np.float64)
            cols.append(np.searchsorted(ref, s, side="right") / max(len(ref), 1))
        return np.column_stack(cols).mean(axis=1)


class StackingEnsemble:
    benign_only = False
    needs_seq = False

    def __init__(self, members, name="Ens:Stack", n_folds=3):
        self.members, self.name, self.n_folds = members, name, n_folds

    def fit(self, X, y, S=None):
        from sklearn.model_selection import StratifiedKFold
        M = np.zeros((len(y), len(self.members)))
        skf = StratifiedKFold(self.n_folds, shuffle=True, random_state=SEED)
        for tr, te in skf.split(X, y):
            for j, m in enumerate(self.members):
                mm = copy.deepcopy(m)
                mm.fit(X[tr], y[tr])
                M[te, j] = mm.score(X[te])
        self.meta_ = LogisticRegression(max_iter=2000, class_weight="balanced",
                                        random_state=SEED).fit(M, y)
        for m in self.members:
            m.fit(X, y)
        return self

    def score(self, X, S=None):
        return self.meta_.predict_proba(np.column_stack([m.score(X) for m in self.members]))[:, 1]


# --------------------------------------------------------------------------
def model_zoo(threads=1, include_nn=True, include_ens=True, nn_epochs=8):
    zoo = [make_logreg(threads), make_rf(threads), make_et(threads), make_hgb(threads),
           make_xgb(threads), make_lgbm(threads), make_cat(threads)]
    if include_nn:
        zoo += [TorchNet("mlp", epochs=nn_epochs * 2, name="NN:MLP", threads=threads),
                TorchNet("cnn", epochs=nn_epochs, name="NN:CNN1D", threads=threads),
                TorchNet("lstm", epochs=nn_epochs, name="NN:LSTM", threads=threads),
                TorchNet("transformer", epochs=nn_epochs, name="NN:Transformer",
                         hidden=64, threads=threads)]
    if include_ens:
        zoo += [RankAverageEnsemble([make_xgb(threads), make_lgbm(threads), make_rf(threads)],
                                    name="Ens:RankAvg-Trees"),
                StackingEnsemble([make_xgb(threads), make_lgbm(threads), make_rf(threads),
                                  make_et(threads)], name="Ens:Stack-Trees")]
        if include_nn:
            zoo.append(RankAverageEnsemble(
                [make_xgb(threads), make_lgbm(threads), make_rf(threads),
                 TorchNet("mlp", epochs=nn_epochs * 2, threads=threads),
                 TorchNet("cnn", epochs=nn_epochs, threads=threads)],
                name="Ens:RankAvg-Hybrid"))
    return zoo


def fast_zoo(threads=1):
    """The cheap subset used for the controlled experiments and ablations."""
    return [make_logreg(threads), make_rf(threads), make_hgb(threads),
            make_xgb(threads), make_lgbm(threads)]
