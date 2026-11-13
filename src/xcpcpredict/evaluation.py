"""Contest-grouped splits, probability metrics, calibration tables, bootstrap CIs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-6


# ------------------------------------------------------------------ splits

def time_split(df: pd.DataFrame, test_frac: float = 0.2, val_frac: float = 0.15) -> dict[str, list[str]]:
    """Whole contests only; the most recent contests form the test set."""
    order = (
        df.groupby("contest_id")["contest_start_ms"].first().sort_values().index.tolist()
    )
    n = len(order)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    return {
        "train": order[: n - n_test - n_val],
        "val": order[n - n_test - n_val: n - n_test],
        "test": order[n - n_test:],
    }


def group_folds(contests: list[str], k: int, seed: int) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(sorted(contests)))
    return [perm[i::k] for i in range(k)]


# ------------------------------------------------------------------ metrics

def brier(y, p) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def logloss(y, p) -> float:
    p = np.clip(np.asarray(p), EPS, 1 - EPS)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y, p, bins: int = 10) -> float:
    t = reliability_table(y, p, bins)
    return float((t["n"] * (t["mean_p"] - t["frac_ac"]).abs()).sum() / t["n"].sum())


def metrics(y, p) -> dict[str, float]:
    y = np.asarray(y)
    out = {"n": int(len(y)), "base_rate": float(y.mean()), "brier": brier(y, p), "logloss": logloss(y, p),
           "ece": ece(y, p)}
    if 0 < y.sum() < len(y):
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    return out


def per_contest_metrics(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    d = pd.DataFrame({"contest_id": df["contest_id"].values, "y": df["y"].values, "p": p})
    rows = []
    for cid, g in d.groupby("contest_id"):
        rows.append({"contest_id": cid, "n": len(g), "brier": brier(g.y, g.p), "logloss": logloss(g.y, g.p)})
    return pd.DataFrame(rows)


def reliability_table(y, p, bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = np.minimum((d.p * bins).astype(int), bins - 1)
    t = d.groupby("bin").agg(n=("y", "size"), mean_p=("p", "mean"), frac_ac=("y", "mean")).reset_index()
    t["lo"] = t["bin"] / bins
    t["hi"] = (t["bin"] + 1) / bins
    return t[["lo", "hi", "n", "mean_p", "frac_ac"]]


def bootstrap_delta(df: pd.DataFrame, p_a: np.ndarray, p_b: np.ndarray, metric=brier,
                    n_boot: int = 1000, seed: int = 0) -> dict[str, float]:
    """Contest-cluster bootstrap of metric(b) - metric(a). Negative = b better for losses."""
    d = pd.DataFrame({"c": df["contest_id"].values, "y": df["y"].values, "a": p_a, "b": p_b})
    groups = [g for _, g in d.groupby("c")]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(groups), len(groups))
        s = pd.concat([groups[i] for i in idx])
        deltas.append(metric(s.y, s.b) - metric(s.y, s.a))
    deltas = np.array(deltas)
    point = metric(d.y, d.b) - metric(d.y, d.a)
    return {"delta": float(point), "lo": float(np.quantile(deltas, 0.025)),
            "hi": float(np.quantile(deltas, 0.975)), "p_b_not_better": float(np.mean(deltas >= 0))}


# ------------------------------------------------------------------ subgroups

def subgroups(df: pd.DataFrame) -> dict[str, pd.Series]:
    hc = df["hidden_count"]
    age = df["last_hidden_age"]
    return {
        "hidden=1": hc == 1,
        "hidden=2": hc == 2,
        "hidden>=3": hc >= 3,
        "no pre-freeze attempt": df["has_pre_attempt"] == 0,
        "pre-freeze WA>=2": df["pre_wa_count"] >= 2,
        "easy problem (solve rate>=0.6)": df["problem_solve_rate"] >= 0.6,
        "medium problem": (df["problem_solve_rate"] >= 0.25) & (df["problem_solve_rate"] < 0.6),
        "hard problem (<0.25)": df["problem_solve_rate"] < 0.25,
        "strong team (top 20%)": df["rank_pct"] <= 0.2,
        "weak team (bottom 50%)": df["rank_pct"] > 0.5,
        "just submitted (age<5min)": age < 5,
        "silent >=20min": age >= 20,
        "early freeze (obs_frac<=1/6)": df["obs_frac"] <= 1 / 6 + 1e-9,
        "end of contest (obs_frac=1)": df["obs_frac"] >= 1 - 1e-9,
    }


def subgroup_table(df: pd.DataFrame, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, mask in subgroups(df).items():
        m = mask.to_numpy()
        if m.sum() < 30:
            continue
        row = {"subgroup": name, "n": int(m.sum()), "ac_rate": float(df["y"].to_numpy()[m].mean())}
        for model, p in preds.items():
            row[f"brier[{model}]"] = brier(df["y"].to_numpy()[m], p[m])
            row[f"mean_p[{model}]"] = float(p[m].mean())
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ thresholds

def threshold_table(y, p, thresholds=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7)) -> pd.DataFrame:
    """Confusion matrix at each threshold (predict AC iff p >= threshold)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    rows = []
    for t in thresholds:
        pred = p >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        tn = int((~pred & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        rows.append({
            "threshold": t, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": prec, "recall": rec,
            "specificity": tn / (tn + fp) if tn + fp else float("nan"),
            "accuracy": (tp + tn) / len(y),
            "f1": 2 * prec * rec / (prec + rec) if tp else 0.0,
            "predicted_ac_share": float(pred.mean()),
        })
    return pd.DataFrame(rows)
