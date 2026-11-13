"""Backtest of final-standings predictions on held-out contests (observed at contest end)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .experiment import _md_table
from .normalize import Contest
from .predict import ModelBundle, predict_snapshot
from .replay import make_snapshot
from .standings import TIERS, build_board, medal_groups, medal_lines, scoreboard, simulate, team_summary


def actual_final(contest: Contest) -> pd.DataFrame:
    solved, pen = scoreboard([(s.team_id, s.problem_id, s.time_ms, s.verdict) for s in contest.submissions],
                             contest.penalty_ms, contest.penalty_rule)
    teams = sorted(contest.teams)
    df = pd.DataFrame({"team_id": teams, "actual_solved": [solved.get(t, 0) for t in teams],
                       "actual_pen": [pen.get(t, 0) for t in teams]})
    df["actual_rank"] = (df["actual_solved"] * 1_000_000 - df["actual_pen"]).rank(ascending=False, method="min")
    return df


def _actual_tiers(board, actual: pd.DataFrame) -> tuple[dict[str, int], list[dict]]:
    """team -> actual tier (0..3), and actual medal lines."""
    a = actual.set_index("team_id")
    tiers, lines = {}, []
    for g, members in medal_groups(board).items():
        teams = [board.teams[i] for i in members]
        sub = a.loc[teams]
        sub = sub.assign(key=sub["actual_solved"] * 1_000_000 - sub["actual_pen"]).sort_values("key", ascending=False)
        rule = board.medal_rule
        if "counts" in rule:
            gc, sc, bc = rule["counts"][g]
        else:
            n_valid = int((sub["actual_solved"] > 0).sum())
            gc, sc, bc = (int(np.ceil(n_valid * f)) for f in rule["ratio"])
        cuts = [gc, gc + sc, gc + sc + bc]
        mr = sub["key"].rank(ascending=False, method="min")
        for t, r, s in zip(sub.index, mr, sub["actual_solved"]):
            tiers[t] = 3 if s == 0 else next((k for k, c in enumerate(cuts) if r <= c), 3)
        for k, name in enumerate(TIERS):
            if 0 < cuts[k] <= len(sub):
                last = sub.iloc[cuts[k] - 1]
                lines.append({"group": g, "tier": name, "actual_solved": int(last["actual_solved"]),
                              "actual_pen": int(last["actual_pen"])})
    return tiers, lines


def backtest_contest(contest: Contest, bundle: ModelBundle, base_rate: float, n_sims: int = 5000) -> dict:
    snap = make_snapshot(contest, contest.duration_ms)
    pred = predict_snapshot(bundle, snap)
    actual = actual_final(contest)
    variants = {
        "model": pred,
        "freeze board = final (all frozen fail)": pred.assign(p_ac=0.0),
        "constant p = base rate": pred.assign(p_ac=base_rate),
    }
    out = {"contest_id": contest.contest_id, "teams": len(contest.teams), "frozen_cells": len(pred), "variants": {}}
    for name, p in variants.items():
        board = build_board(snap, p)
        sim = simulate(board, n_sims)
        summ = team_summary(board, sim).merge(actual, on="team_id")
        r = {
            "rank_mae": float((summ["rank_mean"] - summ["actual_rank"]).abs().mean()),
            "rank_cov90": float(((summ["actual_rank"] >= summ["rank_p05"]) &
                                 (summ["actual_rank"] <= summ["rank_p95"])).mean()),
            "solved_mae": float((summ["expected_solved"] - summ["actual_solved"]).abs().mean()),
        }
        tiers, lines = _actual_tiers(board, actual)
        elig = summ[summ["team_id"].isin(tiers)]
        if len(elig):
            probs = elig[["p_gold", "p_silver", "p_bronze"]].to_numpy()
            probs = np.column_stack([probs, 1 - probs.sum(1)])
            onehot = np.eye(4)[[tiers[t] for t in elig["team_id"]]]
            r["medal_brier"] = float(((probs - onehot) ** 2).sum(1).mean())
            r["medal_modal_acc"] = float((probs.argmax(1) == onehot.argmax(1)).mean())
            # teams whose medal is genuinely uncertain at the freeze
            unsure = probs.max(1) < 0.9
            r["medal_unsure_teams"] = int(unsure.sum())
        if lines:
            ml = medal_lines(board, sim)
            lp, pe = [], []
            for line in lines:
                m = ml[(ml["group"] == line["group"]) & (ml["tier"] == line["tier"])]
                hit = m[m["line_solved"] == line["actual_solved"]]
                lp.append(float(hit["probability"].sum()))
                if len(hit):
                    pe.append(abs(float(hit["penalty_median"].iloc[0]) - line["actual_pen"]))
            r["line_solved_prob"] = float(np.mean(lp))
            r["line_pen_abs_err"] = float(np.mean(pe)) if pe else np.nan
        out["variants"][name] = r
    return out


def run_standings_backtest(contests: list[Contest], bundle: ModelBundle, base_rate: float, out: Path,
                           data_version: str, command: str, n_sims: int = 5000) -> pd.DataFrame:
    rows = []
    for c in contests:
        res = backtest_contest(c, bundle, base_rate, n_sims)
        for v, r in res["variants"].items():
            rows.append({"contest_id": res["contest_id"], "teams": res["teams"],
                         "frozen_cells": res["frozen_cells"], "variant": v, **r})
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "standings_backtest.csv", index=False)
    cols = ["rank_mae", "rank_cov90", "solved_mae", "medal_brier", "medal_modal_acc", "line_solved_prob",
            "line_pen_abs_err"]
    agg = df.groupby("variant", sort=False)[[c for c in cols if c in df]].mean().reset_index()
    wins = df.pivot(index="contest_id", columns="variant", values="medal_brier")
    L = ["# Final-standings backtest\n",
         f"- Data version: `{data_version}`; model `{bundle.model_version}`; reproduce with `{command}`\n",
         f"{len(contests)} held-out contests observed at contest end; {n_sims} simulations each. The model was "
         "trained without these contests. Metrics are averaged over contests.\n",
         "- `rank_mae`: |expected rank − actual final rank|, all teams\n"
         "- `rank_cov90`: share of teams whose actual rank is inside the predicted 5–95% interval (ideal 0.90)\n"
         "- `medal_brier`: 4-class Brier (gold/silver/bronze/none) over medal-eligible teams (lower is better)\n"
         "- `medal_modal_acc`: share of eligible teams whose most likely medal is the actual one\n"
         "- `line_solved_prob`: predicted probability of the actual solved count of each medal line\n"
         "- `line_pen_abs_err`: |predicted median penalty − actual| of the line team, when the solved count "
         "matches\n",
         _md_table(agg),
         ""]
    for v in wins.columns:
        if v != "model":
            L.append(f"- model has lower medal Brier than `{v}` in "
                     f"{int((wins['model'] < wins[v]).sum())} of {wins[v].notna().sum()} contests")
    L.append("\nPer contest: `standings_backtest.csv`.\n")
    (out / "standings_backtest.md").write_text("\n".join(L))
    return agg
