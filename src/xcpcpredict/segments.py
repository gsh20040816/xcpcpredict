"""Does the model transfer across contest types, and does contest quality matter?

1. Per contest type: out-of-fold quality of the all-contest model (M6 vs M7 features).
2. For each evaluated type T (contest-grouped folds over T only), on the same held-out rows:
   train on all types / only on T / all types + contest structure features (M7) /
   M7 with every contest weighted equally / only T + M7.
3. Contest discrimination: are low-discrimination contests harder to predict?
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import bootstrap_delta, brier, group_folds, logloss, metrics, per_contest_metrics
from .experiment import _md_table
from .features import feature_columns
from .models import SEED, CatBoostModel

log = logging.getLogger(__name__)

EVAL_TYPES = ("regional", "invitational", "provincial")


def contest_weights(df: pd.DataFrame) -> np.ndarray:
    """Every contest gets the same total weight (mean weight 1)."""
    n = df.groupby("contest_id")["y"].transform("size").to_numpy()
    w = 1.0 / n
    return w * len(df) / w.sum()


def _fit_predict(cols, train: pd.DataFrame, test: pd.DataFrame, weighted: bool = False) -> np.ndarray:
    w = contest_weights(train) if weighted else None
    return CatBoostModel(cols).fit(train, train["y"].to_numpy(), sample_weight=w).predict_proba(test)


def _skill(y, p) -> float:
    br = float(np.mean(y))
    return 1 - brier(y, p) / (br * (1 - br)) if 0 < br < 1 else float("nan")


def run_segment_study(df: pd.DataFrame, out: Path, data_version: str, command: str, k: int = 5,
                      n_boot: int = 500) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    m6, m7 = feature_columns("M6"), feature_columns("M7")
    y_all = df["y"].to_numpy()

    # ---------------- 1. per-type OOF of the all-contest model
    oof = {"M6": np.zeros(len(df)), "M7": np.zeros(len(df))}
    for fold in group_folds(df["contest_id"].unique().tolist(), k, SEED):
        te = df["contest_id"].isin(fold).to_numpy()
        for name, cols in (("M6", m6), ("M7", m7)):
            oof[name][te] = _fit_predict(cols, df[~te], df[te])
    log.info("per-type OOF done")
    type_rows = []
    for t, g in df.groupby("contest_type"):
        m = (df["contest_type"] == t).to_numpy()
        pc = per_contest_metrics(df[m], oof["M7"][m])
        type_rows.append({
            "contest_type": t, "contests": g["contest_id"].nunique(), "rows": int(m.sum()),
            "share_of_rows": float(m.mean()), "base_rate": float(y_all[m].mean()),
            "brier_M6": brier(y_all[m], oof["M6"][m]), "brier_M7": brier(y_all[m], oof["M7"][m]),
            "skill_M7": _skill(y_all[m], oof["M7"][m]),
            "auc_M7": metrics(y_all[m], oof["M7"][m])["roc_auc"],
            "mean_p_M7": float(oof["M7"][m].mean()), "contest_mean_brier_M7": float(pc["brier"].mean()),
        })
    types = pd.DataFrame(type_rows).sort_values("rows", ascending=False)

    # ---------------- 2. training-set composition per evaluated type
    variants = {
        "all types, M6 (current)": lambda tr, te, T: _fit_predict(m6, tr, te),
        "same type only, M6": lambda tr, te, T: _fit_predict(m6, tr[tr["contest_type"] == T], te),
        "all types, M7 (+type, structure)": lambda tr, te, T: _fit_predict(m7, tr, te),
        "all types, M7, equal contest weight": lambda tr, te, T: _fit_predict(m7, tr, te, weighted=True),
        "same type only, M7": lambda tr, te, T: _fit_predict(m7, tr[tr["contest_type"] == T], te),
    }
    comp_rows, delta_rows = [], []
    for T in EVAL_TYPES:
        contests = sorted(df.loc[df["contest_type"] == T, "contest_id"].unique())
        mask_T = df["contest_type"].eq(T).to_numpy()
        sub = df[mask_T]
        preds = {v: np.zeros(len(sub)) for v in variants}
        for fold in group_folds(contests, k, SEED):
            te_sub = sub["contest_id"].isin(fold).to_numpy()
            tr = df[~df["contest_id"].isin(fold)]
            te = sub[te_sub]
            for v, f in variants.items():
                preds[v][te_sub] = f(tr, te, T)
        log.info("composition study %s done", T)
        y = sub["y"].to_numpy()
        n_same = len(contests)
        for v, p in preds.items():
            mm = metrics(y, p)
            comp_rows.append({"eval_type": T, "variant": v, "eval_contests": len(contests),
                              "train_contests_same_type": n_same - n_same // k,
                              "brier": mm["brier"], "logloss": mm["logloss"], "ece": mm["ece"],
                              "roc_auc": mm["roc_auc"],
                              "contest_mean_brier": float(per_contest_metrics(sub, p)["brier"].mean())})
        base = "all types, M6 (current)"
        for v in variants:
            if v == base:
                continue
            d = bootstrap_delta(sub, preds[base], preds[v], brier, n_boot=n_boot)
            delta_rows.append({"eval_type": T, "candidate": v, "brier_delta_vs_current": d["delta"],
                               "ci_lo": d["lo"], "ci_hi": d["hi"]})
    comp = pd.DataFrame(comp_rows)
    deltas = pd.DataFrame(delta_rows)

    # ---------------- 3. contest discrimination vs predictability (all-contest M7 OOF)
    per = []
    for cid, g in df.assign(p=oof["M7"]).groupby("contest_id"):
        if g["y"].nunique() < 2:
            continue
        end = g[g["obs_frac"] > 0.99]
        per.append({
            "contest_id": cid, "contest_type": g["contest_type"].iloc[0], "rows": len(g),
            "discrimination": float(end["contest_mean_discrimination"].iloc[0]),
            "solved_cv": float(end["contest_solved_cv"].iloc[0]),
            "base_rate": float(g["y"].mean()), "brier": brier(g["y"], g["p"]), "skill": _skill(g["y"], g["p"]),
            "auc": metrics(g["y"], g["p"])["roc_auc"], "logloss": logloss(g["y"], g["p"]),
            "calib_gap": float(g["p"].mean() - g["y"].mean()),
        })
    per = pd.DataFrame(per)
    per.to_csv(out / "segment_per_contest.csv", index=False)
    per["disc_tertile"] = pd.qcut(per["discrimination"], 3, labels=["low", "mid", "high"])
    disc = per.groupby("disc_tertile", observed=True).agg(
        contests=("contest_id", "size"), discrimination=("discrimination", "mean"),
        base_rate=("base_rate", "mean"), brier=("brier", "mean"), skill=("skill", "mean"),
        auc=("auc", "mean"), abs_calib_gap=("calib_gap", lambda s: float(s.abs().mean())),
    ).reset_index()
    corr = per[["discrimination", "solved_cv", "base_rate", "auc", "skill", "brier"]].corr(method="spearman")

    types.to_csv(out / "segment_by_type.csv", index=False)
    comp.to_csv(out / "segment_composition.csv", index=False)
    deltas.to_csv(out / "segment_deltas.csv", index=False)
    disc.to_csv(out / "segment_discrimination.csv", index=False)

    L = ["# Contest types and contest quality\n",
         f"- Data version: `{data_version}`; seed `{SEED}`; reproduce with `{command}`",
         f"- {df['contest_id'].nunique()} contests, {len(df)} rows; CatBoost; contest-grouped {k}-fold CV.\n",
         "## 1. Quality by contest type (model trained on all types, out-of-fold)\n",
         "`skill` = 1 − Brier / (base_rate·(1−base_rate)), comparable across types with different base rates.\n",
         _md_table(types),
         "\n## 2. Which training contests to use (evaluated on held-out contests of one type)\n",
         "All variants of one evaluated type are scored on the same held-out rows. M7 adds contest type "
         "one-hots and contest structure (problem / contest discrimination, solved-count spread, "
         "zero-solve share, problem difficulty spread, unsolved problems).\n",
         _md_table(comp),
         "\nBrier change vs the current setup (negative = better), contest-cluster bootstrap 95% CI:\n",
         _md_table(deltas),
         "\n## 3. Contest discrimination vs predictability\n",
         "Contest discrimination = mean over problems of corr(solved the problem, solved count on the other "
         "problems) on the frozen board. Contests split into tertiles.\n",
         _md_table(disc),
         "\nSpearman correlations across contests:\n",
         _md_table(corr.round(3).reset_index().rename(columns={"index": ""})),
         ""]
    (out / "segment_report.md").write_text("\n".join(L))
    summary = {"by_type": types.to_dict(orient="records"), "composition": comp.to_dict(orient="records"),
               "deltas": deltas.to_dict(orient="records"), "discrimination": disc.to_dict(orient="records")}
    (out / "segment_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return summary
