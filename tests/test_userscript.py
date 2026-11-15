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


def test_userscript_parses(setup, tmp_path):
    *_, bundle = setup
    path = tmp_path / "x.user.js"
    path.write_text(build_userscript(bundle, "https://example.com/x.user.js"))
    text = path.read_text()
    assert "// @updateURL    https://example.com/x.user.js" in text
    subprocess.run([NODE, "--check", str(path)], check=True)
