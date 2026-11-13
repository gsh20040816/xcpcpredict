"""Baselines (empirical grouped rates, logistic regression) and CatBoost."""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

SEED = 20261108


class EmpiricalModel:
    """M0: smoothed AC frequency per bucket, shrunk hierarchically toward coarser buckets."""

    name = "empirical"

    def __init__(self, k: float = 20.0):
        self.k = k

    @staticmethod
    def _keys(X: pd.DataFrame) -> list[pd.Series]:
        hc = X["hidden_count"].clip(upper=3).astype(int).astype(str)
        pre = X["has_pre_attempt"].astype(int).astype(str)
        team = pd.cut(X["rank_pct"], [-1, 0.1, 0.3, 0.6, 2], labels=False).astype(str)
        prob = pd.cut(X["problem_solve_rate"], [-1, 0.2, 0.4, 0.6, 0.8, 2], labels=False).astype(str)
        age = pd.cut(X["last_hidden_age"], [-2, 0, 5, 15, 30, 1e9], labels=False).astype(str)
        # coarse -> fine
        return [hc, hc + pre, hc + pre + team, hc + pre + team + prob, hc + pre + team + prob + age]

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        y = np.asarray(y, dtype=float)
        self.prior_ = y.mean()
        self.tables_ = []
        for key in self._keys(X):
            g = pd.DataFrame({"k": key.values, "y": y}).groupby("k")["y"].agg(["sum", "count"])
            self.tables_.append(g)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.full(len(X), self.prior_)
        for key, g in zip(self._keys(X), self.tables_):
            s = key.map(g["sum"]).fillna(0).to_numpy()
            n = key.map(g["count"]).fillna(0).to_numpy()
            p = (s + self.k * p) / (n + self.k)
        return p


def _signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


class LogisticModel:
    name = "logistic"

    def __init__(self, columns: list[str], C: float = 1.0):
        self.columns = columns
        self.C = C

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        self.pipe_ = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            FunctionTransformer(_signed_log1p, feature_names_out="one-to-one"),
            StandardScaler(),
            LogisticRegression(C=self.C, max_iter=2000),
        )
        self.pipe_.fit(X[self.columns].astype(float), y, logisticregression__sample_weight=sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipe_.predict_proba(X[self.columns].astype(float))[:, 1]

    def coefficients(self) -> pd.Series:
        names = self.pipe_[:-1].get_feature_names_out(self.columns)
        return pd.Series(self.pipe_[-1].coef_[0], index=names).sort_values()


class CatBoostModel:
    name = "catboost"

    def __init__(self, columns: list[str], iterations: int = 600, depth: int = 6,
                 learning_rate: float = 0.05, seed: int = SEED):
        self.columns = columns
        self.params = dict(
            iterations=iterations, depth=depth, learning_rate=learning_rate, loss_function="Logloss",
            random_seed=seed, verbose=0, thread_count=-1, allow_writing_files=False,
        )

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        self.model_ = CatBoostClassifier(**self.params)
        self.model_.fit(X[self.columns].astype(float), y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X[self.columns].astype(float))[:, 1]

    def importance(self) -> pd.Series:
        return pd.Series(self.model_.get_feature_importance(), index=self.columns).sort_values(ascending=False)
