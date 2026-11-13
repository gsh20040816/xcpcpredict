import numpy as np

from xcpcpredict.features import COUNT_ONLY_COLUMNS, TIME_COLUMNS, _tier, extract_features
from xcpcpredict.normalize import normalize_contest, parse_medal_rule
from xcpcpredict.replay import make_snapshot

from .conftest import MIN, make_raw

M = 60_000


def _contest(runs, medal=None, groups=None, teams=("t1", "t2", "t3", "t4")):
    cfg, raw_teams, raw_runs = make_raw(runs, problems=("A", "B", "C", "D"), teams=teams)
    if medal is not None:
        cfg["medal"] = medal
    if groups is not None:
        for t in raw_teams:
            t["group"] = groups.get(t["id"], ["official"])
    return normalize_contest("m6", cfg, raw_teams, raw_runs)


def _rows(c, t, mode="A"):
    return {(r["team_id"], r["problem_id"]): r for r in extract_features(make_snapshot(c, t, mode))}


def test_pre_freeze_compile_error_is_invisible():
    c = _contest([("t1", "A", 10 * MIN, "COMPILATION_ERROR"), ("t1", "A", 250 * MIN, "WRONG_ANSWER")])
    snap = make_snapshot(c, 300 * M)
    assert [s.verdict for s in snap.submissions if not s.hidden] == []
    assert _rows(c, 300 * M)[("t1", "A")]["has_pre_attempt"] == 0


def test_medal_rule_parsing():
    assert parse_medal_rule({"official": {"gold": 1, "silver": 2, "bronze": 3}}) == {"counts": {"official": (1, 2, 3)}}
    assert parse_medal_rule("ccpc") == {"ratio": (0.1, 0.2, 0.3)}
    assert parse_medal_rule({}) is None and parse_medal_rule(None) is None


def test_tier_boundaries():
    keys = [(-3, 10), (-2, 5), (-2, 9), (-1, 0)]
    cuts = [1, 2, 3]
    assert [_tier(keys, cuts, k) for k in keys] == [0, 1, 2, 3]
    assert _tier(keys, cuts, (-4, 0)) == 0


def test_medal_distance_and_groups():
    runs = [
        ("t1", "A", 10 * MIN, "ACCEPTED"), ("t1", "B", 20 * MIN, "ACCEPTED"),
        ("t2", "A", 30 * MIN, "ACCEPTED"),
        ("t3", "A", 40 * MIN, "ACCEPTED"),
        ("t3", "B", 250 * MIN, "WRONG_ANSWER"),  # t3 bronze; with B (2 solves) it passes t2 -> silver
        ("t4", "B", 260 * MIN, "WRONG_ANSWER"),  # t4 is unofficial: not medal eligible
    ]
    c = _contest(runs, medal={"official": {"gold": 1, "silver": 1, "bronze": 1}},
                 groups={"t4": ["unofficial"], "t3": ["official", "girl"]})
    rows = _rows(c, 300 * M)
    t3, t4 = rows[("t3", "B")], rows[("t4", "B")]
    assert t3["medal_eligible"] == 1 and t3["medal_tier_now"] == 2
    assert t3["solved_gap_next_tier"] == 0 and t3["rank_gap_next_tier"] == 1
    assert t3["tier_gain_if_ac"] == 1 and t3["rank_gain_if_ac"] > 0
    assert t3["team_girl"] == 1 and t3["team_unofficial"] == 0 and t4["team_unofficial"] == 1
    assert t4["medal_eligible"] == 0 and np.isnan(t4["medal_tier_now"])


CROWD_RUNS = [
        ("t1", "A", 10 * MIN, "ACCEPTED"), ("t2", "A", 12 * MIN, "ACCEPTED"),
        ("t3", "A", 14 * MIN, "ACCEPTED"), ("t1", "B", 200 * MIN, "ACCEPTED"),
        # problem C frozen cells: t1 once, t2 twice, t3 once then moves on to B (harder than C? no: B solved by 1)
        ("t1", "C", 245 * MIN, "WRONG_ANSWER"),
        ("t2", "C", 250 * MIN, "WRONG_ANSWER"), ("t2", "C", 255 * MIN, "WRONG_ANSWER"),
        ("t3", "C", 250 * MIN, "WRONG_ANSWER"), ("t3", "D", 290 * MIN, "WRONG_ANSWER"),
]


def test_crowd_and_moved_to_problems():
    c = _contest(CROWD_RUNS)
    r = _rows(c, 300 * M)
    t1 = r[("t1", "C")]
    # others on C: t2 (2 hidden, silent 45 min, not moved) and t3 (1 hidden, silent 50 min, moved to D)
    assert t1["crowd_mean_hidden"] == 1.5 and t1["crowd_multi_frac"] == 0.5
    assert t1["crowd_silent_frac"] == 1.0 and t1["crowd_moved_on_frac"] == 0.5
    t3 = r[("t3", "C")]
    # D and C both have 0 pre-freeze ACs -> neither harder nor easier
    assert t3["after_last_harder_problems"] == 0 and t3["after_last_easier_problems"] == 0
    assert t3["last_hidden_before_end"] == 50 and t3["hidden_last5_count"] == 0
    assert r[("t3", "D")]["hidden_last5_count"] == 0 and r[("t3", "D")]["last_hidden_before_end"] == 10


def test_problem_trend():
    runs = [("t1", "A", 10 * MIN, "ACCEPTED"), ("t2", "A", 200 * MIN, "ACCEPTED"),
            ("t3", "A", 230 * MIN, "WRONG_ANSWER"), ("t3", "A", 250 * MIN, "WRONG_ANSWER")]
    r = _rows(_contest(runs), 300 * M)[("t3", "A")]
    # window = the 60 min before the freeze (180..240)
    assert r["problem_recent_ac"] == 1 and r["problem_recent_ac_share"] == 0.5
    assert r["problem_recent_new_attempts"] == 2 and r["problem_last_ac_age"] == 40


def test_mode_b_time_columns_are_nan():
    rows = _rows(_contest(CROWD_RUNS), 300 * M, "B")
    for row in rows.values():
        assert all(np.isnan(row[c]) for c in TIME_COLUMNS)
    # count-based crowd features still work without hidden times
    assert rows[("t1", "C")]["crowd_mean_hidden"] == 1.5 and rows[("t1", "C")]["crowd_multi_frac"] == 0.5
    assert not set(TIME_COLUMNS) & set(COUNT_ONLY_COLUMNS)


def test_contest_classification():
    from xcpcpredict.normalize import classify_contest

    assert classify_contest("icpc/51st/nanjing", "南京站 - 正式赛") == "regional"
    assert classify_contest("icpc/51st/xian-invitational", "") == "invitational"
    assert classify_contest("ccpc/11th/northeastern", "2025 年 CCPC 全国邀请赛（东北）") == "invitational"
    assert classify_contest("provincial-contest/2024/shandong", "CCPC 全国邀请赛（山东）暨山东省赛") == "provincial"
    assert classify_contest("icpc/49th/online-qualification-1", "") == "online"
    assert classify_contest("icpc/49th/ecfinal", "") == classify_contest("ccpc/9th/final", "") == "final"
    assert classify_contest("ccpc/9th/girl", "") == "special"


def test_problem_discrimination():
    # A is solved exactly by the strong teams (high discrimination); B by a weak team only (negative).
    runs = [
        ("t1", "C", 5 * MIN, "ACCEPTED"), ("t2", "C", 6 * MIN, "ACCEPTED"), ("t3", "C", 7 * MIN, "ACCEPTED"),
        ("t1", "D", 8 * MIN, "ACCEPTED"), ("t2", "D", 9 * MIN, "ACCEPTED"),
        ("t1", "A", 10 * MIN, "ACCEPTED"), ("t2", "A", 11 * MIN, "ACCEPTED"),
        ("t4", "B", 12 * MIN, "ACCEPTED"),
        ("t3", "A", 250 * MIN, "WRONG_ANSWER"), ("t3", "B", 251 * MIN, "WRONG_ANSWER"),
    ]
    r = _rows(_contest(runs), 300 * M)
    assert r[("t3", "A")]["problem_discrimination"] > 0.8
    assert r[("t3", "B")]["problem_discrimination"] < 0
    assert r[("t3", "A")]["ctype_regional"] == 0 or np.isnan(r[("t3", "A")]["ctype_regional"])
