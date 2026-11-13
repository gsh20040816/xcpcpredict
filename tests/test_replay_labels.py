import dataclasses

import numpy as np
import pytest

from xcpcpredict.features import extract_features
from xcpcpredict.labels import label_cells
from xcpcpredict.normalize import Submission, Verdict, normalize_contest
from xcpcpredict.replay import MODE_COUNTS_ONLY, make_snapshot, observation_times

from .conftest import MIN, make_raw

M = 60_000


def test_snapshot_hides_post_freeze_verdicts_and_future(small_contest):
    snap = make_snapshot(small_contest, 265 * M)
    hidden = [s for s in snap.submissions if s.hidden]
    assert {(s.team_id, s.problem_id, s.time_ms) for s in hidden} == {
        ("t2", "B", 250 * M), ("t3", "C", 245 * M), ("t1", "C", 260 * M)}
    assert all(s.verdict is None for s in hidden)
    assert max(s.time_ms for s in snap.submissions) <= 265 * M


def test_mode_b_drops_hidden_times(small_contest):
    snap = make_snapshot(small_contest, 300 * M, MODE_COUNTS_ONLY)
    assert all(s.time_ms is None for s in snap.submissions if s.hidden)
    rows = {(r["team_id"], r["problem_id"]): r for r in extract_features(snap)}
    assert rows[("t2", "B")]["hidden_count"] == 2
    assert np.isnan(rows[("t2", "B")]["last_hidden_age"])


def test_spec_label_example():
    # 250 WA, 280 AC: label 0 at 260 min, 1 at 290 min.
    c = normalize_contest("x", *make_raw([("t1", "A", 250 * MIN, "WRONG_ANSWER"), ("t1", "A", 280 * MIN, "ACCEPTED")]))
    assert label_cells(c, 260 * M)[("t1", "A")].y == 0
    assert label_cells(c, 290 * M)[("t1", "A")].y == 1


def test_labels_skip_cells_solved_before_freeze(small_contest):
    labels = label_cells(small_contest, 300 * M)
    assert ("t1", "A") not in labels
    assert labels[("t2", "B")].y == 1 and labels[("t3", "C")].y == 1 and labels[("t1", "C")].y == 0


def test_unresolved_hidden_verdict_is_marked():
    c = normalize_contest("x", *make_raw([("t1", "A", 250 * MIN, "PENDING")]))
    assert label_cells(c, 300 * M)[("t1", "A")].unresolved


def _flip_hidden(contest):
    fz = contest.freeze_start_ms
    flip = {Verdict.AC: Verdict.REJECTED, Verdict.REJECTED: Verdict.AC,
            Verdict.REJECTED_NO_PENALTY: Verdict.AC, Verdict.UNRESOLVED: Verdict.AC}
    subs = [dataclasses.replace(s, verdict=flip[s.verdict]) if s.time_ms >= fz else s for s in contest.submissions]
    return dataclasses.replace(contest, submissions=subs)


def _rows(contest, t, mode="A"):
    return sorted(extract_features(make_snapshot(contest, t, mode)), key=lambda r: (r["team_id"], r["problem_id"]))


def _same(a, b):
    assert len(a) == len(b)
    for ra, rb in zip(a, b):
        assert ra.keys() == rb.keys()
        for k in ra:
            va, vb = ra[k], rb[k]
            assert (va == vb) or (isinstance(va, float) and np.isnan(va) and np.isnan(vb)), k


@pytest.mark.parametrize("mode", ["A", "B"])
def test_features_invariant_to_hidden_verdicts(small_contest, mode):
    for t in observation_times(small_contest):
        _same(_rows(small_contest, t, mode), _rows(_flip_hidden(small_contest), t, mode))


def test_features_invariant_to_future_submissions(small_contest):
    extra = Submission("t3", "A", 295 * M, Verdict.AC)
    later = dataclasses.replace(small_contest, submissions=small_contest.submissions + [extra])
    _same(_rows(small_contest, 290 * M), _rows(later, 290 * M))


def test_scoreboard_reconstruction(small_contest):
    rows = {(r["team_id"], r["problem_id"]): r for r in _rows(small_contest, 300 * M)}
    t2 = rows[("t2", "B")]
    # t2 solved A at 30 min, 0 WA -> penalty 30; t1 solved A (20 + 20) and B (100) -> penalty 140.
    assert t2["solved_count"] == 1 and t2["penalty"] == 30 and t2["pre_wa_count"] == 1
    t1 = rows[("t1", "C")]
    assert t1["solved_count"] == 2 and t1["penalty"] == 140 and t1["penalty_wa_count"] == 1
    assert t1["rank_pct"] == 0.0 and t2["rank_pct"] == 0.5
    # t1's hidden WA on already-solved A is not a frozen cell.
    assert ("t1", "A") not in rows
    assert t2["hidden_count"] == 2 and t2["min_gap"] == 30.0 and t2["last_hidden_age"] == 20.0
