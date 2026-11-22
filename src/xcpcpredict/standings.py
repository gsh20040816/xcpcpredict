"""Final-standings simulation from per-cell p_ac.

Each frozen cell passes with its marginal predicted p_ac, coupled across cells of the same team,
the same problem, and the contest via a Gaussian copula (so unobserved team form and problem
difficulty shifts widen rank intervals appropriately without altering marginal cell probabilities).
If a cell passes, the AC is taken to be the cell's last hidden submission: on tier A/B history
that is exact for 93% of AC frozen cells (penalty exact in 92%, mean over-estimate 5.5 min
otherwise). Without hidden times (Mode B) the AC time is taken as the middle of
[freeze, observation].

Submissions after the observation time are not modelled, so the result is the final board
only when the observation time is the contest end; earlier it is "the board if nothing else
gets submitted".
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd

from .features import STAR_MARKERS
from .normalize import Verdict
from .replay import MODE_FULL_TIMES, PublicSnapshot

TIERS = ("gold", "silver", "bronze")
# Variance shares of the unit-variance Gaussian copula latent variable per cell.
COPULA_TEAM = 0.08
COPULA_PROBLEM = 0.02
COPULA_CONTEST = 0.01


def _norm_ppf(p: np.ndarray) -> np.ndarray:
    """Acklam's rational approximation to the standard normal inverse CDF (max error < 1.15e-9)."""
    p = np.asarray(p, dtype=float)
    out = np.empty_like(p)
    out[p <= 0] = -np.inf
    out[p >= 1] = np.inf
    m = (p > 0) & (p < 1)
    pm = p[m]
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p_low = 0.02425
    z = np.empty_like(pm)
    mid = (pm >= p_low) & (pm <= 1 - p_low)
    if mid.any():
        q = pm[mid] - 0.5
        r = q * q
        z[mid] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    tail = ~mid
    if tail.any():
        pt = pm[tail]
        q = np.sqrt(-2.0 * np.log(np.where(pt < p_low, pt, 1.0 - pt)))
        zt = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
        z[tail] = np.where(pt < p_low, zt, -zt)
    out[m] = z
    return out


def scoreboard(runs, penalty_ms: int, penalty_rule: str) -> tuple[dict[str, int], dict[str, int]]:
    """(team -> solved, team -> penalty minutes) from (team, problem, time_ms, verdict) runs."""
    cells: dict[tuple[str, str], list] = defaultdict(list)
    for team, prob, t, v in runs:
        cells[(team, prob)].append((t, v))
    solved: dict[str, int] = defaultdict(int)
    pen_ms: dict[str, float] = defaultdict(float)
    for (team, _), rs in cells.items():
        rs.sort(key=lambda x: x[0])
        wrong = 0
        for t, v in rs:
            if v == Verdict.AC:
                solved[team] += 1
                t_part = (t // 60000) * 60000 if penalty_rule == "in_minutes" else t
                pen_ms[team] += t_part + wrong * penalty_ms
                break
            wrong += v == Verdict.REJECTED
    return dict(solved), {t: math.floor(v / 60000) for t, v in pen_ms.items()}


@dataclass
class Board:
    teams: list[str]
    solved0: np.ndarray  # public, at the freeze
    pen0: np.ndarray  # minutes, at the freeze (penalty_rule applied per AC)
    cell_team: np.ndarray  # team index of each frozen cell
    cell_problem: list[str]
    cell_p: np.ndarray
    cell_pen: np.ndarray  # penalty minutes added if the cell is AC
    groups: dict[str, tuple[str, ...]] = field(default_factory=dict)
    medal_rule: dict | None = None

    def index(self, team: str) -> int:
        return self.teams.index(team)


def build_board(snap: PublicSnapshot, pred: pd.DataFrame) -> Board:
    visible = [(s.team_id, s.problem_id, s.time_ms, s.verdict) for s in snap.submissions if not s.hidden]
    solved, pen = scoreboard(visible, snap.penalty_ms, snap.penalty_rule)
    teams = sorted(set(snap.team_ids))
    idx = {t: i for i, t in enumerate(teams)}
    pre_wa: dict[tuple[str, str], int] = defaultdict(int)
    hidden: dict[tuple[str, str], list] = defaultdict(list)
    for s in snap.submissions:
        if s.hidden:
            hidden[(s.team_id, s.problem_id)].append(s.time_ms)
        elif s.verdict == Verdict.REJECTED:
            pre_wa[(s.team_id, s.problem_id)] += 1
    mid = (snap.freeze_start_ms + snap.observation_ms) // 2
    pen_min = snap.penalty_ms / 60000
    cell_pen = []
    for team, prob in zip(pred["team_id"], pred["problem_id"]):
        times = hidden[(team, prob)]
        t_ac = max(times) if snap.mode == MODE_FULL_TIMES else mid
        t_part = (t_ac // 60000) if snap.penalty_rule == "in_minutes" else t_ac / 60000
        cell_pen.append(t_part + pen_min * (pre_wa[(team, prob)] + len(times) - 1))
    return Board(
        teams=teams,
        solved0=np.array([solved.get(t, 0) for t in teams]),
        pen0=np.array([pen.get(t, 0) for t in teams], dtype=float),
        cell_team=np.array([idx[t] for t in pred["team_id"]], dtype=int),
        cell_problem=list(pred["problem_id"]),
        cell_p=pred["p_ac"].to_numpy(dtype=float),
        cell_pen=np.array(cell_pen, dtype=float),
        groups=snap.team_groups,
        medal_rule=snap.medal_rule,
    )


@dataclass
class Simulation:
    solved: np.ndarray  # (n_sims, n_teams)
    penalty: np.ndarray  # (n_sims, n_teams) minutes
    outcomes: np.ndarray  # (n_sims, n_cells) bool

    @property
    def score(self) -> np.ndarray:
        return self.solved * 1_000_000 - np.floor(self.penalty)


def simulate(board: Board, n_sims: int = 10000, seed: int = 0) -> Simulation:
    rng = np.random.default_rng(seed)
    n_cells = len(board.cell_p)
    if n_cells == 0:
        outcomes = np.zeros((n_sims, 0), dtype=bool)
    else:
        thresh = _norm_ppf(board.cell_p)
        probs_uniq, prob_idx = np.unique(board.cell_problem, return_inverse=True)
        w_eps = 1.0 - COPULA_TEAM - COPULA_PROBLEM - COPULA_CONTEST
        z = (
            math.sqrt(w_eps) * rng.standard_normal((n_sims, n_cells))
            + math.sqrt(COPULA_CONTEST) * rng.standard_normal((n_sims, 1))
            + math.sqrt(COPULA_PROBLEM) * rng.standard_normal((n_sims, len(probs_uniq)))[:, prob_idx]
            + math.sqrt(COPULA_TEAM) * rng.standard_normal((n_sims, len(board.teams)))[:, board.cell_team]
        )
        outcomes = z < thresh
    onehot = np.zeros((n_cells, len(board.teams)))
    onehot[np.arange(n_cells), board.cell_team] = 1
    solved = board.solved0 + outcomes.astype(float) @ onehot
    penalty = board.pen0 + (outcomes * board.cell_pen) @ onehot
    return Simulation(solved.astype(int), penalty, outcomes)


def _ranks(score: np.ndarray) -> np.ndarray:
    """Competition rank (1 + number of strictly better scores), row-wise."""
    srt = np.sort(score, axis=1)
    n = score.shape[1]
    out = np.empty_like(score, dtype=int)
    for i in range(score.shape[0]):
        out[i] = n - np.searchsorted(srt[i], score[i], side="right") + 1
    return out


def medal_groups(board: Board) -> dict[str, list[int]]:
    """Medal group name -> indices of eligible teams."""
    rule = board.medal_rule
    if not rule:
        return {}
    if "counts" in rule:
        return {g: [i for i, t in enumerate(board.teams) if g == "all" or g in board.groups.get(t, ())]
                for g in rule["counts"]}
    return {"_ratio": [i for i, t in enumerate(board.teams)
                       if not any(m in " ".join(board.groups.get(t, ())).lower() for m in STAR_MARKERS)]}


def _cuts(board: Board, group: str, solved_g: np.ndarray) -> np.ndarray:
    """Cumulative medal cutoffs per simulation, shape (n_sims, 3)."""
    rule = board.medal_rule
    if "counts" in rule:
        g, s, b = rule["counts"][group]
        return np.tile([g, g + s, g + s + b], (solved_g.shape[0], 1))
    n_valid = (solved_g > 0).sum(axis=1)
    fr = np.array(rule["ratio"])
    per = np.ceil(n_valid[:, None] * fr[None, :])
    return np.cumsum(per, axis=1).astype(int)


def team_summary(board: Board, sim: Simulation) -> pd.DataFrame:
    ranks = _ranks(sim.score)
    df = pd.DataFrame({
        "team_id": board.teams,
        "solved_at_freeze": board.solved0,
        "frozen_cells": np.bincount(board.cell_team, minlength=len(board.teams)),
        "expected_solved": sim.solved.mean(0),
        "solved_std": sim.solved.std(0),
        "rank_mean": ranks.mean(0),
        "rank_std": ranks.std(0),
        "rank_p05": np.percentile(ranks, 5, axis=0),
        "rank_p50": np.percentile(ranks, 50, axis=0),
        "rank_p95": np.percentile(ranks, 95, axis=0),
    })
    for tier in TIERS + ("medal",):
        df[f"p_{tier}"] = np.nan
    for g, members in medal_groups(board).items():
        if not members:
            continue
        sc = sim.score[:, members]
        mr = _ranks(sc)
        cuts = _cuts(board, g, sim.solved[:, members])
        tier = np.full(mr.shape, 3)
        for k in (2, 1, 0):
            tier[mr <= cuts[:, [k]]] = k
        tier[sim.solved[:, members] == 0] = 3  # no medal without a solve
        for k, name in enumerate(TIERS):
            df.loc[members, f"p_{name}"] = (tier == k).mean(0)
        df.loc[members, "p_medal"] = (tier < 3).mean(0)
    return df.sort_values("rank_mean").reset_index(drop=True)


def medal_lines(board: Board, sim: Simulation) -> pd.DataFrame:
    """Distribution of the last team inside each medal tier: solved count and penalty."""
    rows = []
    for g, members in medal_groups(board).items():
        if not members:
            continue
        sc = sim.score[:, members]
        order = np.argsort(-sc, axis=1)
        cuts = _cuts(board, g, sim.solved[:, members])
        solved_g, pen_g = sim.solved[:, members], sim.penalty[:, members]
        for k, name in enumerate(TIERS):
            sims = np.flatnonzero(cuts[:, k] > 0)  # simulations in which this tier has a last team
            pos = np.minimum(cuts[sims, k], len(members)) - 1
            who = order[sims, pos]
            ls = solved_g[sims, who]
            lp = np.floor(pen_g[sims, who])
            vals, counts = np.unique(ls, return_counts=True)
            for v, c in zip(vals, counts):
                m = ls == v
                rows.append({"group": g, "tier": name, "line_solved": int(v), "probability": c / len(sc),
                             "penalty_p10": float(np.percentile(lp[m], 10)),
                             "penalty_median": float(np.median(lp[m])),
                             "penalty_p90": float(np.percentile(lp[m], 90))})
    return pd.DataFrame(rows)


def _rank_against_others(sim: Simulation, team_idx: int, score: float | np.ndarray) -> np.ndarray:
    others = np.delete(sim.score, team_idx, axis=1)
    return 1 + (others > np.asarray(score).reshape(-1, 1)).sum(axis=1)


def hypothetical_rank(board: Board, sim: Simulation, team: str, solved: int, penalty: int) -> dict:
    """Rank distribution of `team` if it finished with (solved, penalty), others simulated."""
    r = _rank_against_others(sim, board.index(team), solved * 1_000_000 - penalty)
    return {"team_id": team, "solved": solved, "penalty": penalty, "rank_mean": float(r.mean()),
            "rank_std": float(r.std()), "rank_p05": float(np.percentile(r, 5)),
            "rank_p95": float(np.percentile(r, 95))}


def scenarios(board: Board, sim: Simulation, team: str, max_cells: int = 8) -> pd.DataFrame:
    """Every pass/fail combination of `team`'s frozen cells: probability and rank distribution."""
    ti = board.index(team)
    cells = np.flatnonzero(board.cell_team == ti)
    if len(cells) > max_cells:
        cells = cells[np.argsort(-board.cell_p[cells])][:max_cells]
    groups = {g: m for g, m in medal_groups(board).items() if ti in m}
    rows = []
    for combo in product((0, 1), repeat=len(cells)):
        c = np.array(combo, dtype=bool)
        solved = board.solved0[ti] + c.sum()
        pen = math.floor(board.pen0[ti] + board.cell_pen[cells][c].sum())
        prob = float(np.prod(np.where(c, board.cell_p[cells], 1 - board.cell_p[cells])))
        r = _rank_against_others(sim, ti, solved * 1_000_000 - pen)
        row = {"passed": ",".join(board.cell_problem[i] for i in cells[c]) or "-",
               "probability": prob, "solved": int(solved), "penalty": pen,
               "rank_mean": float(r.mean()), "rank_std": float(r.std()),
               "rank_p05": float(np.percentile(r, 5)), "rank_p95": float(np.percentile(r, 95))}
        for g, members in groups.items():
            others = [m for m in members if m != ti]
            mr = 1 + (sim.score[:, others] > solved * 1_000_000 - pen).sum(axis=1)
            cuts = _cuts(board, g, sim.solved[:, members])
            tier = np.select([mr <= cuts[:, 0], mr <= cuts[:, 1], mr <= cuts[:, 2]], [0, 1, 2], 3)
            if solved == 0:
                tier[:] = 3
            for k, name in enumerate(TIERS):
                row[f"p_{name}"] = float((tier == k).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("probability", ascending=False).reset_index(drop=True)
