"""Post-hoc probability calibration fitted on held-out validation contests only."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-4


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Calibrator:
    def __init__(self, method: str = "platt"):
        self.method = method  # "platt", "isotonic" or "none" (identity)

    def fit(self, p: np.ndarray, y: np.ndarray):
        if self.method == "platt":
            self.lr_ = LogisticRegression(C=1e6).fit(_logit(p).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.iso_ = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip").fit(p, y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if self.method == "platt":
            return self.lr_.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        if self.method == "isotonic":
            return self.iso_.predict(p)
        return p


class CalibratedModel:
    """A base model plus a calibrator; ``predict_proba`` returns calibrated p_ac."""

    def __init__(self, base, calibrator: Calibrator):
        self.base = base
        self.calibrator = calibrator

    def predict_proba(self, X) -> np.ndarray:
        return self.calibrator.transform(self.base.predict_proba(X))
