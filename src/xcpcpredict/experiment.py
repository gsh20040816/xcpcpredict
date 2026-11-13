"""End-to-end offline experiment: baselines, CatBoost, calibration, ablations, report."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .calibration import CalibratedModel, Calibrator  # noqa: E402
from .evaluation import (  # noqa: E402
    bootstrap_delta, brier, group_folds, logloss, metrics, per_contest_metrics, reliability_table,
    subgroup_table, threshold_table, time_split,
)
from .features import COUNT_ONLY_COLUMNS, FULL_COLUMNS, GROUPS, M6_FAMILIES, feature_columns  # noqa: E402
from .models import SEED, CatBoostModel, EmpiricalModel, LogisticModel  # noqa: E402

log = logging.getLogger(__name__)

# Feature families removed one at a time from the full model to measure contribution.
DROP_GROUPS = {
    "dirt/first-try rates": ["team_dirt", "team_dirt_raw", "problem_dirt", "team_first_ac_rate",
                             "problem_first_ac_rate", "problem_first_ac_n"],
    "penalty decomposition": ["penalty", "penalty_wa_count"],
    "hidden submission counts": ["hidden_count", "team_hidden_total", "team_hidden_problems",
                                 "other_hidden_count", "recent_hidden_count", "short_gap_count"],
    "submission gaps": ["min_gap", "max_gap", "mean_gap", "last_gap", "short_gap_count", "hidden_span"],
    "silence since last submission": ["last_hidden_age", "team_last_activity_age", "last_hidden_time"],
    "other-problem activity": GROUPS["M5"] + ["team_hidden_problems", "other_hidden_count"],
    **{f"M6 {name}": cols for name, cols in M6_FAMILIES.items()},
}


def choose_calibration(df_val: pd.DataFrame, p_val: np.ndarray, seed: int = SEED) -> str:
    """Pick Platt vs isotonic by 2-fold contest cross-fitting on the validation contests."""
    folds = group_folds(df_val["contest_id"].unique().tolist(), 2, seed)
    scores = {}
    for method in ("platt", "isotonic"):
        ll = []
        for i in range(2):
            fit_m = df_val["contest_id"].isin(folds[1 - i]).to_numpy()
            ev_m = df_val["contest_id"].isin(folds[i]).to_numpy()
            c = Calibrator(method).fit(p_val[fit_m], df_val["y"].to_numpy()[fit_m])
            ll.append(logloss(df_val["y"].to_numpy()[ev_m], c.transform(p_val[ev_m])))
        scores[method] = float(np.mean(ll))
    return min(scores, key=scores.get)


def fit_crossfit_calibrated(make_model, train: pd.DataFrame, k: int = 5, method: str = "platt") -> CalibratedModel:
    """Train on every training contest; fit the calibrator on contest-grouped out-of-fold predictions."""
    p_oof = oof_predictions(train, make_model, k)
    base = make_model().fit(train, train["y"].to_numpy())
    return CalibratedModel(base, Calibrator(method).fit(p_oof, train["y"].to_numpy()))


def oof_predictions(df: pd.DataFrame, make_model, k: int = 5, seed: int = SEED) -> np.ndarray:
    p = np.zeros(len(df))
    for fold in group_folds(df["contest_id"].unique().tolist(), k, seed):
        te = df["contest_id"].isin(fold).to_numpy()
        m = make_model().fit(df[~te], df["y"].to_numpy()[~te])
        p[te] = m.predict_proba(df[te])
    return p


def _fmt(d: dict) -> str:
    return " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items())


def _md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.astype(object).itertuples(index=False):
        cells = [format(v, floatfmt) if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_reliability(y, preds: dict[str, np.ndarray], path: Path, bins: int = 10) -> None:
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--", label="perfect")
    for name, p in preds.items():
        t = reliability_table(y, p, bins)
        ax.plot(t["mean_p"], t["frac_ac"], marker="o", ms=4, label=name)
        axh.hist(p, bins=bins * 2, range=(0, 1), histtype="step", label=name)
    ax.set_ylabel("observed AC frequency")
    ax.set_title("Reliability on held-out test contests")
    ax.legend(fontsize=8)
    axh.set_xlabel("predicted p_ac")
    axh.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_experiment(df: pd.DataFrame, out: Path, data_version: str, command: str, cv_folds: int = 5,
                   n_boot: int = 1000) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    split = time_split(df)
    tr, va, te = (df[df["contest_id"].isin(split[s])] for s in ("train", "val", "test"))
    y_te = te["y"].to_numpy()
    full = FULL_COLUMNS
    log.info("split: %s", {k: len(v) for k, v in split.items()})

    # ---------------- main comparison on the time-ordered test contests
    models = {
        "empirical(M0)": EmpiricalModel(),
        "logistic": LogisticModel(full),
        "catboost": CatBoostModel(full),
    }
    preds: dict[str, np.ndarray] = {"prior": np.full(len(te), tr["y"].mean())}
    fitted = {}
    for name, m in models.items():
        m.fit(tr, tr["y"].to_numpy())
        fitted[name] = m
        preds[name] = m.predict_proba(te)
        log.info("%s %s", name, _fmt(metrics(y_te, preds[name])))
    calib_choice = {}
    for name in ("logistic", "catboost"):
        p_val = fitted[name].predict_proba(va)
        method = choose_calibration(va, p_val)
        calib_choice[name] = method
        cal = Calibrator(method).fit(p_val, va["y"].to_numpy())
        preds[f"{name}+{method}"] = cal.transform(preds[name])
    cb_cal_name = f"catboost+{calib_choice['catboost']}"

    # Mode B (no hidden submission times)
    mb = CatBoostModel(COUNT_ONLY_COLUMNS).fit(tr, tr["y"].to_numpy())
    mb_cal = Calibrator(choose_calibration(va, mb.predict_proba(va))).fit(mb.predict_proba(va), va["y"].to_numpy())
    preds["catboost ModeB (counts only)"] = mb_cal.transform(mb.predict_proba(te))

    main_rows = []
    for name, p in preds.items():
        row = {"model": name, **metrics(y_te, p)}
        pc = per_contest_metrics(te, p)
        row["contest_mean_brier"] = float(pc["brier"].mean())
        row["contest_mean_logloss"] = float(pc["logloss"].mean())
        main_rows.append(row)
    main = pd.DataFrame(main_rows)
    main.to_csv(out / "test_metrics.csv", index=False)

    deltas = []
    for a, b in [("prior", "empirical(M0)"), ("empirical(M0)", "logistic"), ("logistic", "catboost"),
                 ("empirical(M0)", "catboost"), ("catboost", cb_cal_name),
                 ("catboost ModeB (counts only)", cb_cal_name)]:
        for mname, mf in (("brier", brier), ("logloss", logloss)):
            d = bootstrap_delta(te, preds[a], preds[b], mf, n_boot=n_boot)
            deltas.append({"baseline": a, "candidate": b, "metric": mname, **d})
    deltas = pd.DataFrame(deltas)
    deltas.to_csv(out / "bootstrap_deltas.csv", index=False)

    plot_reliability(y_te, {k: preds[k] for k in ("empirical(M0)", f"logistic+{calib_choice['logistic']}",
                                                  cb_cal_name, "catboost ModeB (counts only)")},
                     out / "reliability.png")
    rel = reliability_table(y_te, preds[cb_cal_name])
    rel.to_csv(out / "reliability_catboost.csv", index=False)

    thr = {name: threshold_table(y_te, preds[name])
           for name in (cb_cal_name, f"logistic+{calib_choice['logistic']}", "empirical(M0)",
                        "catboost ModeB (counts only)")}
    pd.concat([t.assign(model=n) for n, t in thr.items()]).to_csv(out / "confusion_thresholds.csv", index=False)

    sub = subgroup_table(te, {"M0": preds["empirical(M0)"], "LR": preds["logistic"], "CB": preds[cb_cal_name]})
    sub.to_csv(out / "subgroups.csv", index=False)
    pc = per_contest_metrics(te, preds[cb_cal_name]).merge(
        per_contest_metrics(te, preds["empirical(M0)"]), on=["contest_id", "n"], suffixes=("_cb", "_m0"))
    pc.to_csv(out / "per_contest_test.csv", index=False)

    tier_rows = []
    for tier, g in te.groupby("tier"):
        m = te["tier"].eq(tier).to_numpy()
        tier_rows.append({"tier": tier, "contests": g["contest_id"].nunique(),
                          **{k: v for k, v in metrics(y_te[m], preds[cb_cal_name][m]).items()
                             if k in ("n", "base_rate", "brier", "logloss", "roc_auc")}})
    tiers = pd.DataFrame(tier_rows)

    imp = fitted["catboost"].importance()
    imp.to_csv(out / "catboost_importance.csv", header=["importance"])
    coefs = fitted["logistic"].coefficients()

    # ---------------- ablations with contest-grouped CV over all contests
    log.info("ablation: %d-fold contest CV", cv_folds)
    y_all = df["y"].to_numpy()
    abl_rows, oof = [], {}
    oof["M0"] = oof_predictions(df, EmpiricalModel, cv_folds)
    for lvl in GROUPS:
        cols = feature_columns(lvl)
        oof[f"LR-{lvl}"] = oof_predictions(df, lambda c=cols: LogisticModel(c), cv_folds)
        oof[f"CB-{lvl}"] = oof_predictions(df, lambda c=cols: CatBoostModel(c), cv_folds)
        log.info("ablation %s done", lvl)
    for name, cols_drop in DROP_GROUPS.items():
        cols = [c for c in full if c not in cols_drop]
        oof[f"CB-{list(GROUPS)[-1]} minus {name}"] = oof_predictions(df, lambda c=cols: CatBoostModel(c), cv_folds)
        log.info("drop %s done", name)
    for name, p in oof.items():
        m = metrics(y_all, p)
        pc_ = per_contest_metrics(df, p)
        abl_rows.append({"variant": name, "brier": m["brier"], "logloss": m["logloss"], "ece": m["ece"],
                         "roc_auc": m["roc_auc"], "contest_mean_brier": float(pc_["brier"].mean())})
    abl = pd.DataFrame(abl_rows)
    abl.to_csv(out / "ablation_cv.csv", index=False)
    abl_deltas = []
    levels = list(GROUPS)
    top = levels[-1]
    pairs = [("M0", "CB-M1")] + [(f"CB-{a}", f"CB-{b}") for a, b in zip(levels, levels[1:])] + \
            [(f"LR-{a}", f"LR-{b}") for a, b in zip(levels, levels[1:])] + \
            [(f"CB-{top} minus {n}", f"CB-{top}") for n in DROP_GROUPS]
    for a, b in pairs:
        d = bootstrap_delta(df, oof[a], oof[b], brier, n_boot=n_boot // 2)
        abl_deltas.append({"from": a, "to": b, "brier_delta": d["delta"], "ci_lo": d["lo"], "ci_hi": d["hi"]})
    abl_deltas = pd.DataFrame(abl_deltas)
    abl_deltas.to_csv(out / "ablation_deltas.csv", index=False)

    # ---------------- examples
    ex = te.assign(p_ac=preds[cb_cal_name])
    ex_contest = ex.groupby("contest_id").size().idxmax()
    ex = ex[(ex["contest_id"] == ex_contest) & (ex["obs_frac"] > 0.99)].sort_values("p_ac")
    ex_cols = ["team_id", "problem_id", "y", "p_ac", "hidden_count", "pre_wa_count", "solved_count",
               "rank_pct", "problem_solve_rate", "last_hidden_age", "other_subs_after_last"]
    examples = pd.concat([ex.head(5), ex.iloc[len(ex) // 2 - 2: len(ex) // 2 + 3], ex.tail(5)])[ex_cols]

    summary = {
        "data_version": data_version, "seed": SEED, "command": command,
        "split": {k: v for k, v in split.items()},
        "rows": {"train": len(tr), "val": len(va), "test": len(te)},
        "calibration": calib_choice,
        "test_metrics": main.to_dict(orient="records"),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    report = _render_report(df, split, tr, va, te, main, deltas, tiers, sub, rel, imp, coefs, abl, abl_deltas,
                            examples, ex_contest, data_version, command, cv_folds, calib_choice, pc, thr)
    (out / "evaluation_report.md").write_text(report)
    return summary


def _render_report(df, split, tr, va, te, main, deltas, tiers, sub, rel, imp, coefs, abl, abl_deltas,
                   examples, ex_contest, data_version, command, cv_folds, calib_choice, pc, thr) -> str:
    def first_last(ids):
        starts = df.groupby("contest_id")["contest_start_ms"].first()
        d = pd.to_datetime(starts[ids], unit="ms")
        return f"{d.min():%Y-%m-%d} … {d.max():%Y-%m-%d}"

    L = []
    L.append("# Evaluation report\n")
    L.append(f"- Data version (xcpcio/board-data commit): `{data_version}`")
    L.append(f"- Seed: `{SEED}`; reproduce with: `{command}`")
    L.append(f"- Rows (team × problem × observation time): {len(df)} from {df['contest_id'].nunique()} contests "
             f"(tiers {sorted(df['tier'].unique())}), positive rate {df['y'].mean():.3f}")
    L.append(f"- Time split by whole contests: train {len(split['train'])} contests / {len(tr)} rows "
             f"({first_last(split['train'])}), val {len(split['val'])} / {len(va)} ({first_last(split['val'])}), "
             f"test {len(split['test'])} / {len(te)} ({first_last(split['test'])})")
    L.append(f"- Calibration chosen by 2-fold contest cross-fit on val: {calib_choice}\n")
    L.append("## Held-out test contests\n")
    cols = ["model", "n", "base_rate", "brier", "logloss", "ece", "roc_auc", "pr_auc", "contest_mean_brier",
            "contest_mean_logloss"]
    L.append(_md_table(main[cols]))
    L.append("\n`contest_mean_*` averages the metric per contest so large contests do not dominate.\n")
    L.append("### Contest-cluster bootstrap of differences (candidate − baseline; negative = candidate better)\n")
    L.append(_md_table(deltas[["baseline", "candidate", "metric", "delta", "lo", "hi", "p_b_not_better"]]))
    L.append("\n![reliability](reliability.png)\n")
    L.append("### Reliability of the calibrated CatBoost (test)\n")
    L.append(_md_table(rel))
    L.append("\n### Confusion matrices at fixed thresholds (test; predict AC iff p_ac ≥ threshold)\n")
    L.append("The model's product is the probability; thresholds only illustrate the trade-off. With a ~28% "
             "base rate, accuracy of always predicting non-AC is the bar to beat for accuracy.\n")
    for name, t in thr.items():
        L.append(f"**{name}**\n")
        L.append(_md_table(t, ".3f"))
        L.append("")
    L.append("\n### By data quality tier (calibrated CatBoost, test)\n")
    L.append(_md_table(tiers))
    L.append(f"\n### Per-contest Brier on test (calibrated CatBoost vs M0)\n")
    L.append(f"CatBoost better than M0 in {(pc['brier_cb'] < pc['brier_m0']).sum()} of {len(pc)} test contests; "
             f"per-contest Brier range {pc['brier_cb'].min():.3f} – {pc['brier_cb'].max():.3f}.\n")
    L.append("### Subgroups (test)\n")
    L.append(_md_table(sub))
    L.append(f"\n## Ablations ({cv_folds}-fold contest-grouped CV over all {df['contest_id'].nunique()} contests)\n")
    L.append("Cumulative levels: M1 problem solve rate + team solved; M2 + dirt / penalty decomposition / "
             "pre-freeze cell history; M3 + hidden submission counts; M4 + hidden submission times and gaps; "
             "M5 + other-problem activity; M6 + difficulty of problems moved to, crowd behaviour on the same "
             "problem, problem trend before the freeze, medal-line distance, team groups, last-minute "
             "submissions. `minus X` drops a feature family from the full CatBoost model.\n")
    L.append(_md_table(abl))
    L.append("\n### Bootstrap of Brier change (to − from; negative = `to` better)\n")
    L.append(_md_table(abl_deltas))
    L.append("\n## Feature analysis\n")
    L.append("CatBoost importance (top 20):\n")
    L.append(_md_table(imp.head(20).rename("importance").reset_index().rename(columns={"index": "feature"})))
    L.append("\nLogistic regression standardized coefficients (most negative / most positive 8):\n")
    c = pd.concat([coefs.head(8), coefs.tail(8)]).rename("coef").reset_index().rename(columns={"index": "feature"})
    L.append(_md_table(c))
    L.append(f"\n## Example predictions at contest end — `{ex_contest}` (lowest, middle, highest p_ac)\n")
    L.append(_md_table(examples, ".3f"))
    L.append("")
    return "\n".join(L)
