import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xcpcpredict.audit import audit_contest, mark_duplicates
from xcpcpredict.calibration import CalibratedModel, Calibrator
from xcpcpredict.evaluation import time_split
from xcpcpredict.features import COUNT_ONLY_COLUMNS, FULL_COLUMNS
from xcpcpredict.models import LogisticModel
from xcpcpredict.normalize import normalize_contest
from xcpcpredict.predict import (
    ModelBundle, PredictionUnavailable, export_historical_snapshot, predict_snapshot, snapshot_from_json,
)
from xcpcpredict.replay import make_snapshot

from .conftest import MIN, make_raw

M = 60_000
BOARD = Path("data/board-data")


def test_audit_flags_runs_stamped_at_zero():
    runs = [(f"t{i}", "A", 0, "REJECTED") for i in range(30)] + \
           [(f"t{i}", "A", (241 + i) * MIN, "ACCEPTED") for i in range(30)]
    raw = make_raw(runs, teams=[f"t{i}" for i in range(30)])
    a = audit_contest(normalize_contest("x", *raw), raw[2])
    assert a.tier == "C" and a.zero_time_frac == 0.5


def test_duplicates_are_marked(small_contest):
    a, b = audit_contest(small_contest, []), audit_contest(small_contest, [])
    b.contest_id = "zzz/copy"
    a.tier = b.tier = "A"
    mark_duplicates([a, b])
    assert a.tier == "A" and b.tier == "X" and b.duplicate_of == a.contest_id


def test_time_split_never_shares_contests():
    df = pd.DataFrame({"contest_id": [f"c{i}" for i in range(20) for _ in range(3)],
                       "contest_start_ms": [i for i in range(20) for _ in range(3)]})
    s = time_split(df)
    assert not (set(s["train"]) & set(s["val"])) and not (set(s["val"]) & set(s["test"]))
    assert max(int(c[1:]) for c in s["train"]) < min(int(c[1:]) for c in s["test"])


def _toy_bundle(small_contest):
    from xcpcpredict.features import extract_features

    rows = pd.DataFrame(extract_features(make_snapshot(small_contest, 300 * M)))
    rows = pd.concat([rows] * 4, ignore_index=True)
    y = np.array([0, 1] * (len(rows) // 2) + [0] * (len(rows) % 2))
    models = {}
    for mode, cols in (("A", FULL_COLUMNS), ("B", COUNT_ONLY_COLUMNS)):
        base = LogisticModel(cols).fit(rows, y)
        models[mode] = CalibratedModel(base, Calibrator("none"))
    return ModelBundle("test", models)


def test_predict_roundtrip_and_degradation(small_contest):
    bundle = _toy_bundle(small_contest)
    for mode in ("A", "B"):
        data = json.loads(json.dumps(export_historical_snapshot(small_contest, 300 * M, mode)))
        assert "verdict" not in json.dumps(data.get("hidden", data.get("hidden_counts")))
        out = predict_snapshot(bundle, snapshot_from_json(data))
        assert set(zip(out.team_id, out.problem_id)) == {("t2", "B"), ("t3", "C"), ("t1", "C")}
        assert out.p_ac.between(0, 1).all()
        assert (out.quality_flags == "hidden_times_unavailable").all() == (mode == "B")


def test_predict_rejects_post_freeze_verdicts_and_mode_c(small_contest):
    data = export_historical_snapshot(small_contest, 300 * M, "A")
    late = dict(data, submissions=data["submissions"] + [{"team": "t1", "problem": "A", "time_s": 250 * 60,
                                                          "verdict": "ACCEPTED"}])
    with pytest.raises(ValueError):
        snapshot_from_json(late)
    data["hidden"][0]["verdict"] = "ACCEPTED"
    with pytest.raises(ValueError):
        snapshot_from_json(data)
    data["mode"] = "C"
    with pytest.raises(PredictionUnavailable):
        snapshot_from_json(data)


@pytest.mark.skipif(not (BOARD / "data/icpc/50th/chengdu").exists(), reason="board-data not downloaded")
def test_real_contests_normalize():
    from xcpcpredict.ingest import load_raw

    for cid in ("icpc/50th/chengdu", "icpc/50th/xian"):
        raw = load_raw(cid, BOARD)
        c = normalize_contest(cid, raw.config, raw.teams, raw.runs)
        assert c.duration_ms == 5 * 3600 * 1000 and c.freeze_length_ms == 3600 * 1000
        assert len(c.submissions) == len(raw.runs)
        assert audit_contest(c, raw.runs).tier == "A"
