"""Would more contests help? Learning curve on a fixed held-out test set, plus extra-data trials."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import bootstrap_delta, brier, metrics, time_split
from .experiment import _md_table
from .features import FULL_COLUMNS
from .models import SEED, CatBoostModel

FRACTIONS = (0.125, 0.25, 0.5, 0.75, 1.0)


def run_learning_curve(df: pd.DataFrame, out: Path, data_version: str, command: str,
                       extra: pd.DataFrame | None = None, repeats: int = 3, n_boot: int = 500) -> pd.DataFrame:
    """``df``: formal contests (defines the split). ``extra``: optional additional training contests
    from another source; only those that started before the first test contest are used."""
    split = time_split(df)
    pool = split["train"] + split["val"]
    te = df[df["contest_id"].isin(split["test"])]
    y = te["y"].to_numpy()
    rng = np.random.default_rng(SEED)
    rows, preds = [], {}
    for frac in FRACTIONS:
        for rep in range(1 if frac == 1.0 else repeats):
            chosen = list(rng.choice(pool, size=int(round(frac * len(pool))), replace=False))
            tr = df[df["contest_id"].isin(chosen)]
            p = CatBoostModel(FULL_COLUMNS).fit(tr, tr["y"].to_numpy()).predict_proba(te)
            m = metrics(y, p)
            rows.append({"variant": "formal contests", "fraction": frac, "repeat": rep, "train_contests": len(chosen),
                         "train_rows": len(tr), "brier": m["brier"], "logloss": m["logloss"], "roc_auc": m["roc_auc"]})
            if frac == 1.0:
                preds["full"] = p
    if extra is not None and len(extra):
        first_test = te["contest_start_ms"].min()
        ex = extra[(extra["contest_start_ms"] < first_test) & ~extra["contest_id"].isin(df["contest_id"])]
        tr = pd.concat([df[df["contest_id"].isin(pool)], ex])
        p = CatBoostModel(FULL_COLUMNS).fit(tr, tr["y"].to_numpy()).predict_proba(te)
        m = metrics(y, p)
        rows.append({"variant": "formal + extra source (pre-test)", "fraction": 1.0, "repeat": 0,
                     "train_contests": tr["contest_id"].nunique(), "train_rows": len(tr), "brier": m["brier"],
                     "logloss": m["logloss"], "roc_auc": m["roc_auc"]})
        d = bootstrap_delta(te, preds["full"], p, brier, n_boot=n_boot)
        extra_delta = d
    else:
        extra_delta = None
    res = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "learning_curve.csv", index=False)
    agg = res.groupby(["variant", "fraction"], sort=False).agg(
        train_contests=("train_contests", "mean"), brier=("brier", "mean"), brier_sd=("brier", "std"),
        logloss=("logloss", "mean"), roc_auc=("roc_auc", "mean")).reset_index()
    L = ["# Learning curve\n",
         f"- Data version: `{data_version}`; seed `{SEED}`; reproduce with `{command}`\n",
         f"Fixed test set: {len(split['test'])} most recent contests ({len(te)} rows). Training contests sampled "
         f"from the {len(pool)} earlier ones ({repeats} random draws per fraction). CatBoost, full features.\n",
         _md_table(agg)]
    if extra_delta:
        L.append(f"\nAdding the extra source: Brier change {extra_delta['delta']:+.5f} "
                 f"(95% CI {extra_delta['lo']:+.5f} … {extra_delta['hi']:+.5f}; negative = better).\n")
    (out / "learning_curve.md").write_text("\n".join(L))
    return agg
