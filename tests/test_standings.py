import numpy as np
import pandas as pd

from xcpcpredict.features import extract_features
from xcpcpredict.normalize import normalize_contest
from xcpcpredict.replay import make_snapshot
from xcpcpredict.standings import build_board, hypothetical_rank, medal_lines, scenarios, simulate, team_summary

from .conftest import MIN, make_raw

M = 60_000


def _board(contest, p):
    snap = make_snapshot(contest, contest.duration_ms)
    rows = pd.DataFrame(extract_features(snap))[["team_id", "problem_id"]]
    return build_board(snap, rows.assign(p_ac=p))


def test_extreme_probabilities_are_deterministic(small_contest):
    # Frozen cells: t2/B (WA 250, AC 280), t3/C (AC 245), t1/C (CE 260).
    b1 = _board(small_contest, 1.0)
    s1 = team_summary(b1, simulate(b1, 200)).set_index("team_id")
    assert s1.loc["t2", "expected_solved"] == 2 and s1.loc["t1", "expected_solved"] == 3
    assert s1["rank_std"].max() == 0
    # t2/B if AC: last hidden at 280 min + 1 pre-freeze WA + 1 earlier hidden run -> 280 + 40.
    assert b1.cell_pen[[i for i, (t, p) in enumerate(zip(np.array(b1.teams)[b1.cell_team], b1.cell_problem))
                        if (t, p) == ("t2", "B")][0]] == 320
    b0 = _board(small_contest, 0.0)
    s0 = team_summary(b0, simulate(b0, 200)).set_index("team_id")
    assert s0.loc["t1", "rank_mean"] == 1 and s0.loc["t3", "expected_solved"] == 0


def test_scenarios_cover_all_outcomes(small_contest):
    b = _board(small_contest, 0.5)
    sim = simulate(b, 500)
    sc = scenarios(b, sim, "t2")
    assert len(sc) == 2 and abs(sc["probability"].sum() - 1) < 1e-9
    assert sc.set_index("passed").loc["B", "solved"] == 2
    h = hypothetical_rank(b, sim, "t2", 5, 0)
    assert h["rank_mean"] == 1


def test_medal_probabilities_and_lines():
    runs = [("t1", "A", 10 * MIN, "ACCEPTED"), ("t2", "A", 20 * MIN, "ACCEPTED"),
            ("t3", "A", 30 * MIN, "ACCEPTED"), ("t3", "B", 250 * MIN, "WRONG_ANSWER")]
    cfg, teams, raw = make_raw(runs)
    cfg["medal"] = {"official": {"gold": 1, "silver": 1, "bronze": 1}}
    c = normalize_contest("m", cfg, teams, raw)
    b = _board(c, 1.0)
    sim = simulate(b, 100)
    s = team_summary(b, sim).set_index("team_id")
    assert s.loc["t3", "p_gold"] == 1 and s.loc["t1", "p_silver"] == 1 and s.loc["t2", "p_bronze"] == 1
    lines = medal_lines(b, sim).set_index("tier")
    assert lines.loc["gold", "line_solved"] == 2 and lines.loc["bronze", "line_solved"] == 1
