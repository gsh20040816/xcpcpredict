import pytest

from xcpcpredict.normalize import normalize_contest

MIN = 60


def make_raw(runs, frozen=3600, duration_s=18000, unit=None, problems=("A", "B", "C"), teams=("t1", "t2", "t3")):
    """Raw XCPCIO-shaped dicts. ``runs`` = [(team, problem, seconds, status), ...]."""
    start = 1_700_000_000_000
    cfg = {
        "contest_name": "test",
        "start_time": start,
        "end_time": start + duration_s * 1000,
        "frozen_time": frozen,
        "penalty": 1200,
        "problems": [{"id": p, "label": p} for p in problems],
        "options": {"submission_timestamp_unit": unit} if unit else {},
    }
    mult = 1000 if unit == "millisecond" else 1
    raw_runs = [
        {"id": str(i), "team_id": t, "problem_id": p, "timestamp": int(s * mult), "status": st}
        for i, (t, p, s, st) in enumerate(runs)
    ]
    raw_teams = [{"id": t, "name": t, "group": ["official"]} for t in teams]
    return cfg, raw_teams, raw_runs


@pytest.fixture
def small_contest():
    # 300 min contest, freeze at 240 min.
    runs = [
        ("t1", "A", 10 * MIN, "WRONG_ANSWER"),
        ("t1", "A", 20 * MIN, "ACCEPTED"),
        ("t1", "B", 100 * MIN, "ACCEPTED"),
        ("t2", "A", 30 * MIN, "ACCEPTED"),
        ("t2", "B", 200 * MIN, "WRONG_ANSWER"),
        ("t2", "B", 250 * MIN, "WRONG_ANSWER"),  # hidden
        ("t2", "B", 280 * MIN, "ACCEPTED"),  # hidden
        ("t3", "C", 245 * MIN, "ACCEPTED"),  # hidden
        ("t1", "C", 260 * MIN, "COMPILATION_ERROR"),  # hidden
        ("t1", "A", 270 * MIN, "WRONG_ANSWER"),  # hidden, cell already solved
    ]
    cfg, teams, raw_runs = make_raw(runs)
    return normalize_contest("test/small", cfg, teams, raw_runs)
