"""The browser port (web/core.js) against the Python pipeline on a synthetic live contest."""

import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from xcpcpredict.calibration import CalibratedModel, Calibrator
from xcpcpredict.features import COUNT_ONLY_COLUMNS, FULL_COLUMNS, extract_features
from xcpcpredict.models import CatBoostModel
from xcpcpredict.normalize import normalize_contest
from xcpcpredict.predict import ModelBundle
from xcpcpredict.replay import make_snapshot
from xcpcpredict.standings import build_board, medal_lines, simulate, team_summary
from xcpcpredict.userscript import build_userscript, bundle_to_js, web_source

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

START = 1_700_000_000_000
MIN = 60_000
FREEZE = 240 * MIN


def live_contest(seed=0, n_teams=60, problems="ABCDEFGHIJ"):
    """XCPCIO-shaped files of a 5 h contest as served during the freeze: post-freeze runs are FROZEN."""
    rng = np.random.default_rng(seed)
    cfg = {
        "contest_name": "synthetic", "start_time": START, "end_time": START + 300 * MIN, "frozen_time": 3600,
        "penalty": 1200, "problems": [{"id": str(i), "label": p} for i, p in enumerate(problems)],
        "medal": {"official": {"gold": 5, "silver": 10, "bronze": 15}},
        "options": {"submission_timestamp_unit": "millisecond"},
    }
    teams = [{"id": f"T{i:02d}", "name": f"team {i}", "group": ["unofficial"] if i % 11 == 0 else ["official"]}
             for i in range(n_teams)]
    ease = np.linspace(0.9, 0.05, len(problems))
    runs = []
    for i, t in enumerate(teams):
        skill = rng.uniform(0.2, 1.0)
        for j in range(len(problems)):
            if rng.random() > min(1.0, ease[j] + skill * 0.6):
                continue
            k = int(rng.integers(1, 5))
            for ts in np.sort(rng.integers(1, 300 * MIN, size=k)):
                a = skill * ease[j]
                if ts >= FREEZE:
                    status = "FROZEN"
                else:
                    status = rng.choice(["ACCEPTED", "WRONG_ANSWER", "TIME_LIMIT_EXCEEDED", "COMPILATION_ERROR", "PENDING"],
                                        p=[a, (1 - a) * 0.55, (1 - a) * 0.3, (1 - a) * 0.1, (1 - a) * 0.05])
                runs.append({"id": str(len(runs)), "team_id": t["id"], "problem_id": str(j), "timestamp": int(ts),
                             "status": str(status)})
                if status == "ACCEPTED":
                    break
    return cfg, teams, runs


def tiny_bundle(contest) -> ModelBundle:
    rows = []
    for obs in range(FREEZE + 5 * MIN, 300 * MIN + 1, 5 * MIN):
        rows += extract_features(make_snapshot(contest, obs, "A"))
    X = pd.DataFrame(rows)
    y = (np.random.default_rng(1).random(len(X)) < 0.2 + 0.5 * (X["hidden_count"] == 1)).astype(int)

    def fit(cols):
        base = CatBoostModel(cols, iterations=60, depth=4).fit(X, y)
        return CalibratedModel(base, Calibrator("platt").fit(base.predict_proba(X), y))

    return ModelBundle("test", {"A": fit(FULL_COLUMNS), "B": fit(COUNT_ONLY_COLUMNS)})


def run_js(tmp_path, bundle, cfg, teams, runs, cases, sims):
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps({"model": bundle_to_js(bundle), "cfg": cfg, "teams": teams, "runs": runs,
                               "cases": cases, "sims": sims}))
    script = tmp_path / "run.js"
    script.write_text(web_source("core.js") + """
const inp = JSON.parse(require("fs").readFileSync(process.argv[2], "utf8"));
const out = inp.cases.map(([obs, mode]) =>
  XP.run(inp.model, "icpc/test/synthetic", inp.cfg, inp.teams, inp.runs, inp.cfg.start_time + obs, {mode, sims: inp.sims}));
process.stdout.write(JSON.stringify(out, (k, v) => (typeof v === "number" && !Number.isFinite(v) ? null : v)));
""")
    res = subprocess.run([NODE, str(script), str(inp)], capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


@pytest.fixture(scope="module")
def setup():
    cfg, teams, runs = live_contest()
    contest = normalize_contest("icpc/test/synthetic", cfg, teams, runs)
    return cfg, teams, runs, contest, tiny_bundle(contest)


def test_features_and_p_ac_match_python(setup, tmp_path):
    cfg, teams, runs, contest, bundle = setup
    cases = [(FREEZE + 20 * MIN, "A"), (300 * MIN, "A"), (FREEZE + 20 * MIN, "B"), (300 * MIN, "B")]
    for (obs, mode), js in zip(cases, run_js(tmp_path, bundle, cfg, teams, runs, cases, sims=0)):
        py = pd.DataFrame(extract_features(make_snapshot(contest, obs, mode))).set_index(["team_id", "problem_id"])
        jr = pd.DataFrame(js["rows"]).set_index(["team_id", "problem_id"]).reindex(py.index)
        assert len(py) > 20 and len(js["rows"]) == len(py)
        cols = FULL_COLUMNS if mode == "A" else COUNT_ONLY_COLUMNS
        for c in cols:
            a, b = py[c].astype(float).to_numpy(), jr[c].astype(float).to_numpy()
            assert np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True), (mode, c)
        p_py = bundle.models[mode].predict_proba(py.reset_index())
        p_js = pd.DataFrame(js["cells"]).set_index(["team", "problem"]).reindex(py.index)["p"].to_numpy()
        assert np.abs(p_py - p_js).max() < 1e-4


def test_standings_match_python_simulation(setup, tmp_path):
    cfg, teams, runs, contest, bundle = setup
    (js,) = run_js(tmp_path, bundle, cfg, teams, runs, [(300 * MIN, "A")], sims=20000)
    snap = make_snapshot(contest, 300 * MIN, "A")
    pred = pd.DataFrame(extract_features(snap))[["team_id", "problem_id"]]
    pred["p_ac"] = bundle.models["A"].predict_proba(pd.DataFrame(extract_features(snap)))
    board = build_board(snap, pred)
    sim = simulate(board, 20000)
    py = team_summary(board, sim).set_index("team_id")
    jt = pd.DataFrame(js["teams"]).set_index("team").reindex(py.index)
    assert np.allclose(py["solved_at_freeze"], jt["solved0"])
    assert np.abs(py["expected_solved"] - jt["expectedSolved"]).max() < 0.03
    assert np.abs(py["rank_mean"] - jt["rankMean"]).max() < 0.3
    for col, jcol in [("p_gold", "pGold"), ("p_silver", "pSilver"), ("p_medal", "pMedal")]:
        assert np.abs(py[col].fillna(-1) - jt[jcol].fillna(-1)).max() < 0.03, col
    pl = medal_lines(board, sim).set_index(["tier", "line_solved"])["probability"]
    jl = pd.DataFrame(js["lines"]).set_index(["tier", "solved"])["probability"]
    both = pl.index.union(jl.index)
    assert np.abs(pl.reindex(both, fill_value=0) - jl.reindex(both, fill_value=0)).max() < 0.03


def test_no_prediction_after_reveal(setup, tmp_path):
    cfg, teams, runs, contest, bundle = setup
    revealed = [dict(r, status="ACCEPTED" if int(r["id"]) % 3 == 0 else "WRONG_ANSWER") if r["status"] == "FROZEN" else r
                for r in runs]
    (frozen_js,) = run_js(tmp_path, bundle, cfg, teams, runs, [(300 * MIN, "A")], sims=0)
    (revealed_js,) = run_js(tmp_path, bundle, cfg, teams, revealed, [(300 * MIN, "A")], sims=0)
    assert frozen_js["status"] == "finished" and revealed_js == {"status": "revealed"}


def test_userscript_parses(setup, tmp_path):
    *_, bundle = setup
    path = tmp_path / "x.user.js"
    path.write_text(build_userscript(bundle, "https://example.com/x.user.js"))
    text = path.read_text()
    assert "// @updateURL    https://example.com/x.user.js" in text
    subprocess.run([NODE, "--check", str(path)], check=True)


def csg_rank(award_ratio=30020010, base=0):
    """A tiny CSG-OJ rank.json: 5 h contest from 09:00 (UTC+8), 60 min freeze."""
    sol = [  # (id, team, problem, wall clock, result)
        (1, "T1", 101, "09:10:00", 6), (2, "T1", 101, "09:20:00", 11), (3, "T1", 101, "09:30:00", 4),
        (4, "T1", 101, "09:40:00", 6), (5, "T1", 102, "13:30:00", -1), (6, "T2", 101, "10:00:00", 4),
        (8, "T3", 102, "13:10:00", -1), (7, "T3", 102, "09:50:00", 2),
    ]
    return {
        "contest": {"contest_id": 1, "title": "t", "start_time": "2026-11-15 09:00:00", "end_time": "2026-11-15 14:00:00",
                    "frozen_minute": 60, "award_ratio": award_ratio, "flg_award_qty_mode": 0, "flg_award_base": base},
        "team": [{"team_id": "T1", "name": "a", "school": "s1", "tkind": 0, "flg_girls": 0},
                 {"team_id": "T2", "name": "b", "school": "s2", "tkind": 2, "flg_girls": 0},
                 {"team_id": "T3", "name": "c", "school": "s3", "tkind": 0, "flg_girls": 1},
                 {"team_id": "T4", "name": "d", "school": "s4", "tkind": 0, "flg_girls": 0}],
        "problem": [{"problem_id": 102, "num": 1}, {"problem_id": 101, "num": 0}],
        "solution": [{"solution_id": i, "team_id": t, "problem_id": p, "in_date": f"2026-11-15 {w}", "result": r}
                     for i, t, p, w, r in sol],
        "is_multi_group": 0,
    }


def from_csg(tmp_path, cases):
    inp = tmp_path / "csg.json"
    inp.write_text(json.dumps(cases))
    script = tmp_path / "csg.js"
    script.write_text(web_source("core.js") + """
const cases = JSON.parse(require("fs").readFileSync(process.argv[2], "utf8"));
process.stdout.write(JSON.stringify(cases.map(([rank, meta]) => XP.fromCsg(rank, meta))));
""")
    return json.loads(subprocess.run([NODE, str(script), str(inp)], capture_output=True, text=True, check=True).stdout)


def test_csg_adapter(tmp_path):
    meta = {"uuid": "u", "title": "第十二届中国大学生程序设计竞赛（测试）", "ckind": "ccpc", "level": "national",
            "start_time": "2026-11-15T01:00:00Z"}
    d, base1, base2, invite, prov, girl = from_csg(tmp_path, [
        [csg_rank(), meta], [csg_rank(50, base=1), meta], [csg_rank(50, base=2), meta],
        [csg_rank(), {**meta, "level": "national_invite"}], [csg_rank(), {**meta, "level": "provincial"}],
        [csg_rank(), {**meta, "title": "CCPC 女生专场"}],
    ])
    cfg = d["cfg"]
    assert cfg["start_time"] == 1_794_704_400_000  # 2026-11-15 09:00 UTC+8
    assert cfg["end_time"] - cfg["start_time"] == 300 * MIN
    assert cfg["frozen_time"] == 3600 and cfg["status_time_display"] == {"pending": True}
    assert cfg["problems"] == [{"id": "101", "label": "A"}, {"id": "102", "label": "B"}]
    # CE (2) and judging (7) dropped, the WA after T1's AC (4) ignored, frozen runs kept as FROZEN
    assert [(r["id"], r["status"], r["timestamp"]) for r in d["runs"]] == [
        ("1", "WRONG_ANSWER", 10 * MIN), ("3", "ACCEPTED", 30 * MIN), ("6", "ACCEPTED", 60 * MIN),
        ("8", "FROZEN", 250 * MIN), ("5", "FROZEN", 270 * MIN)]
    assert {t["id"]: t["group"] for t in d["teams"]} == {
        "T1": ["official"], "T2": ["unofficial"], "T3": ["official", "girl"], "T4": ["official"]}
    # award base: teams with a solve (moves with the simulation), with a submission (T1, T3), all official
    assert cfg["medal"] == {"csg": {"ratios": [10, 20, 30], "qty": False}}
    assert base1["cfg"]["medal"] == {"official": {"gold": 1, "silver": 0, "bronze": 0}}  # ceil(2 * 50%)
    assert base2["cfg"]["medal"] == {"official": {"gold": 2, "silver": 0, "bronze": 0}}  # ceil(3 * 50%)
    assert (d["contestId"], invite["contestId"], prov["contestId"], girl["contestId"]) == (
        "ccpc/u", "ccpc/invitational/u", "provincial-contest/u", "ccpc/girl/u")
