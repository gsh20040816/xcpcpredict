import pytest

from xcpcpredict.normalize import NormalizationError, Verdict, map_verdict, normalize_contest

from .conftest import make_raw


def test_verdict_mapping_both_vocabularies():
    assert map_verdict("ACCEPTED") == map_verdict("correct") == Verdict.AC
    assert map_verdict("WRONG_ANSWER") == map_verdict("incorrect") == map_verdict("REJECTED") == Verdict.REJECTED
    assert map_verdict("COMPILATION_ERROR") == Verdict.REJECTED_NO_PENALTY
    assert map_verdict("pending") == map_verdict("FROZEN") == Verdict.UNRESOLVED
    with pytest.raises(NormalizationError):
        map_verdict("SOMETHING_NEW")


def test_seconds_and_milliseconds_runs_agree():
    runs = [("t1", "A", 600, "ACCEPTED"), ("t2", "B", 17000, "WRONG_ANSWER")]
    a = normalize_contest("s", *make_raw(runs))
    b = normalize_contest("ms", *make_raw(runs, unit="millisecond"))
    assert [s.time_ms for s in a.submissions] == [s.time_ms for s in b.submissions] == [600_000, 17_000_000]


def test_frozen_time_in_seconds_or_milliseconds():
    s = normalize_contest("s", *make_raw([], frozen=3600))
    ms = normalize_contest("ms", *make_raw([], frozen=3_600_000))
    assert s.freeze_start_ms == ms.freeze_start_ms == 4 * 3600 * 1000


def test_start_end_in_seconds():
    cfg, teams, runs = make_raw([("t1", "A", 60, "ACCEPTED")])
    cfg["start_time"] //= 1000
    cfg["end_time"] //= 1000
    c = normalize_contest("x", cfg, teams, runs)
    assert c.duration_ms == 18000 * 1000


def test_no_freeze_is_simulated_and_flagged():
    c = normalize_contest("x", *make_raw([], frozen=0))
    assert c.freeze_simulated and "freeze_simulated" in c.quality_flags
    assert c.freeze_length_ms == 3600 * 1000


def test_timestamps_beyond_duration_raise_instead_of_guessing():
    # Seconds-looking values far beyond the contest under the declared unit.
    with pytest.raises(NormalizationError):
        normalize_contest("x", *make_raw([("t1", "A", 30000, "ACCEPTED")]))


def test_team_dict_form_and_numeric_ids():
    cfg, _, runs = make_raw([("1", "A", 60, "ACCEPTED")])
    runs[0]["team_id"] = 1
    teams = {"1": {"name": "one", "official": 1}}
    c = normalize_contest("x", cfg, teams, runs)
    assert c.submissions[0].team_id == "1"
    assert c.teams["1"].groups == ("official",)


def test_unknown_team_is_kept_and_flagged():
    c = normalize_contest("x", *make_raw([("ghost", "A", 60, "ACCEPTED")]))
    assert "ghost" in c.teams and "unknown_team_refs" in c.quality_flags


def test_i18n_contest_name():
    cfg, teams, runs = make_raw([])
    cfg["contest_name"] = {"fallback_lang": "zh-CN", "texts": {"zh-CN": "南京站 - 热身赛", "en": "Nanjing warmup"}}
    assert normalize_contest("x", cfg, teams, runs).name == "南京站 - 热身赛"
