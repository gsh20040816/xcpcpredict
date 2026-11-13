"""Feature extraction from a ``PublicSnapshot`` only.

One row per frozen cell: a (team, problem) with no visible AC before the freeze and at
least one hidden submission at the observation time. Times are in minutes. Features
that the snapshot's visibility mode cannot support are NaN, never back-filled.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import defaultdict
from statistics import median

import numpy as np

from .normalize import Verdict
from .replay import MODE_FULL_TIMES, PublicSnapshot

FEATURE_VERSION = "f3"

MIN = 60000.0
SMOOTH_K = 3.0  # Beta prior strength (pseudo-observations) for rate smoothing
SHORT_GAP_MIN = 5.0
RECENT_HIDDEN_MIN = 15.0
SILENT_MIN = 20.0
LAST_MINUTES = 5.0
STAR_MARKERS = ("unofficial", "star", "打星")
GIRL_MARKERS = ("girl", "female", "女")

# M6 feature families (all from public information; see docs in extract_features).
M6_FAMILIES: dict[str, list[str]] = {
    "moved-to problems": [
        "other_hidden_harder_problems", "other_hidden_easier_problems",
        "after_last_harder_problems", "after_last_easier_problems", "after_last_min_ac_frac",
    ],
    "crowd behaviour": [
        "crowd_mean_hidden", "crowd_multi_frac", "crowd_silent_frac", "crowd_moved_on_frac",
        "crowd_recent_teams",
    ],
    "problem trend": [
        "problem_recent_ac", "problem_recent_ac_share", "problem_recent_new_attempts", "problem_last_ac_age",
    ],
    "medal distance": [
        "medal_eligible", "medal_tier_now", "solved_gap_next_tier", "rank_gap_next_tier",
        "tier_gain_if_ac", "rank_gain_if_ac",
    ],
    "team groups": ["team_unofficial", "team_girl"],
    "last minutes": ["last_hidden_before_end", "first_hidden_before_end", "hidden_last5_count"],
}

# Contest-level structure (public at the freeze) and contest type (known to any observer).
CONTEST_TYPE_FEATURES = ["ctype_regional", "ctype_invitational", "ctype_provincial", "ctype_online", "ctype_final"]
M7_COLUMNS = CONTEST_TYPE_FEATURES + [
    "problem_discrimination", "contest_mean_discrimination", "contest_solved_cv", "contest_zero_frac",
    "contest_levels_frac", "contest_problem_spread", "contest_unsolved_problems",
]

# Ablation groups, cumulative: M1 ⊂ M2 ⊂ ... ⊂ M7.
GROUPS: dict[str, list[str]] = {
    "M1": [
        "problem_ac_count", "problem_attempt_count", "problem_solve_rate", "problem_ac_frac",
        "solved_count", "rank_pct", "n_problems", "n_active_teams",
    ],
    "M2": [
        "penalty", "penalty_wa_count", "team_dirt", "team_dirt_raw", "team_first_ac_rate",
        "attempted_count", "recent_ac_count", "recent_submission_count", "weighted_solved",
        "minutes_since_last_ac", "problem_dirt", "problem_first_ac_rate", "problem_first_ac_n",
        "first_ac_time", "median_ac_time", "strong_team_solve_rate", "strong_team_n",
        "problem_ease_rank", "pre_wa_count", "has_pre_attempt",
        "pre_first_time", "pre_last_fail_age", "pre_span", "ability_gap",
        "n_harder_solved", "n_easier_unsolved",
    ],
    "M3": [
        "hidden_count", "team_hidden_total", "team_hidden_problems", "other_hidden_count",
        "problem_hidden_teams", "problem_hidden_subs", "obs_since_freeze", "obs_frac",
        "time_remaining",
    ],
    "M4": [
        "first_hidden_time", "last_hidden_time", "last_hidden_age", "hidden_span", "min_gap",
        "max_gap", "mean_gap", "last_gap", "short_gap_count", "recent_hidden_count",
        "gap_pre_to_first_hidden",
    ],
    "M5": [
        "other_subs_after_last", "other_problems_after_last", "other_subs_before_first_hidden",
        "team_last_activity_age",
    ],
    "M6": [c for fam in M6_FAMILIES.values() for c in fam],
    "M7": M7_COLUMNS,
}


def feature_columns(level: str = "M6") -> list[str]:
    cols: list[str] = []
    for g, names in GROUPS.items():
        cols.extend(names)
        if g == level:
            return cols
    raise KeyError(level)


FULL_COLUMNS = feature_columns(list(GROUPS)[-1])
# Features that need hidden submission times; NaN in Mode B.
TIME_COLUMNS = GROUPS["M4"] + GROUPS["M5"] + [
    "after_last_harder_problems", "after_last_easier_problems", "after_last_min_ac_frac",
    "crowd_silent_frac", "crowd_moved_on_frac", "crowd_recent_teams",
    "last_hidden_before_end", "first_hidden_before_end", "hidden_last5_count",
]
# Features usable when hidden submission times are not public (Mode B).
COUNT_ONLY_COLUMNS = [c for c in FULL_COLUMNS if c not in TIME_COLUMNS]


def _smooth(num: float, den: float, prior: float, k: float = SMOOTH_K) -> float:
    return (num + k * prior) / (den + k)


class _Cell:
    __slots__ = ("pre", "hidden_times", "hidden_n")

    def __init__(self):
        self.pre: list = []  # visible (time_ms, verdict), sorted
        self.hidden_times: list[int] = []
        self.hidden_n = 0

    # pre-freeze summaries
    def first_ac_idx(self):
        return next((i for i, (_, v) in enumerate(self.pre) if v == Verdict.AC), None)

    def judged(self):
        return [(t, v) for t, v in self.pre if v != Verdict.UNRESOLVED]


def _group_flags(snap: PublicSnapshot, teams) -> tuple[dict, dict]:
    labelled = any(snap.team_groups.get(t) for t in teams)
    star, girl = {}, {}
    for t in teams:
        g = " ".join(snap.team_groups.get(t, ())).lower()
        star[t] = float(any(m in g for m in STAR_MARKERS)) if labelled else np.nan
        girl[t] = float(any(m in g for m in GIRL_MARKERS)) if labelled else np.nan
    return star, girl


def _medal_groups(snap: PublicSnapshot, teams, keys: dict, star: dict) -> dict[str, tuple[list, list[int]]]:
    """team -> (sorted keys of its medal group, cumulative cutoffs [gold, +silver, +bronze])."""
    rule = snap.medal_rule
    if not rule:
        return {}
    groups: dict[str, list[str]] = {}
    if "counts" in rule:
        for g in rule["counts"]:
            groups[g] = list(teams) if g == "all" else [t for t in teams if g in snap.team_groups.get(t, ())]
    else:  # ratio preset: non-star teams, cutoffs relative to teams with a solve
        groups["_ratio"] = [t for t in teams if star.get(t) != 1.0]
    out = {}
    for g, members in groups.items():
        if not members:
            continue
        if "counts" in rule:
            gold, silver, bronze = rule["counts"][g]
        else:
            n_valid = sum(1 for t in members if keys[t][0] < 0)
            gold, silver, bronze = (math.ceil(f * n_valid) for f in rule["ratio"])
        cuts = [gold, gold + silver, gold + silver + bronze]
        sorted_keys = sorted(keys[t] for t in members)
        for t in members:
            out.setdefault(t, (sorted_keys, cuts))
    return out


def _contest_structure(snap: PublicSnapshot, active_teams, team_solved, prob_ac_frac) -> tuple[dict, dict]:
    """Contest-level spread / discrimination of the frozen board, and per-problem discrimination.

    Problem discrimination is the corrected item-total correlation used for exam items:
    corr(team solved this problem, team's solved count on the other problems).
    """
    solved = np.array([len(team_solved[t]) for t in active_teams], dtype=float)
    disc = {}
    for p in snap.problem_ids:
        x = np.array([p in team_solved[t] for t in active_teams], dtype=float)
        rest = solved - x
        disc[p] = float(np.corrcoef(x, rest)[0, 1]) if x.std() > 0 and rest.std() > 0 else np.nan
    vals = [v for v in disc.values() if not np.isnan(v)]
    ct = snap.contest_type
    out = {f"ctype_{k}": (float(ct == k) if ct else np.nan)
           for k in ("regional", "invitational", "provincial", "online", "final")}
    out.update({
        "contest_mean_discrimination": float(np.mean(vals)) if vals else np.nan,
        "contest_solved_cv": float(solved.std() / solved.mean()) if solved.size and solved.mean() > 0 else np.nan,
        "contest_zero_frac": float((solved == 0).mean()) if solved.size else np.nan,
        "contest_levels_frac": len(set(solved.tolist())) / (len(snap.problem_ids) + 1),
        "contest_problem_spread": float(np.std(list(prob_ac_frac.values()))),
        "contest_unsolved_problems": sum(1 for v in prob_ac_frac.values() if v == 0),
    })
    return out, disc


def _tier(sorted_keys: list, cuts: list[int], key) -> int:
    """0 gold, 1 silver, 2 bronze, 3 none; key ranks at the number of strictly better keys."""
    pos = bisect_left(sorted_keys, key)
    for i, c in enumerate(cuts):
        if pos < c:
            return i
    return 3


def extract_features(snap: PublicSnapshot) -> list[dict]:
    fz, obs, dur = snap.freeze_start_ms, snap.observation_ms, snap.duration_ms
    mode_a = snap.mode == MODE_FULL_TIMES
    cells: dict[tuple[str, str], _Cell] = defaultdict(_Cell)
    team_hidden: dict[str, list] = defaultdict(list)  # (time or None, problem)
    team_pre_times: dict[str, list[int]] = defaultdict(list)
    for s in snap.submissions:
        c = cells[(s.team_id, s.problem_id)]
        if s.hidden:
            c.hidden_n += 1
            if s.time_ms is not None:
                c.hidden_times.append(s.time_ms)
            team_hidden[s.team_id].append((s.time_ms, s.problem_id))
        else:
            c.pre.append((s.time_ms, s.verdict))
            team_pre_times[s.team_id].append(s.time_ms)
    for c in cells.values():
        c.pre.sort(key=lambda x: x[0])
        c.hidden_times.sort()

    problems = snap.problem_ids
    n_problems = len(problems)
    active_teams = sorted({t for t, _ in cells})
    n_active = max(1, len(active_teams))

    # ---------------- pre-freeze scoreboard per team
    team_solved: dict[str, list[str]] = defaultdict(list)
    team_pen_ms: dict[str, float] = defaultdict(float)
    team_wa_pen: dict[str, int] = defaultdict(int)
    team_first_try: dict[str, list[int]] = defaultdict(list)  # 1/0 per attempted problem
    team_ac_times: dict[str, list[int]] = defaultdict(list)
    prob_ac_teams: dict[str, list[str]] = defaultdict(list)
    prob_attempt: dict[str, int] = defaultdict(int)
    prob_first_try: dict[str, list[int]] = defaultdict(list)
    prob_wrong_before_ac: dict[str, int] = defaultdict(int)
    prob_ac_times: dict[str, list[int]] = defaultdict(list)

    for (team, prob), c in cells.items():
        judged = c.judged()
        if not judged:
            continue
        prob_attempt[prob] += 1
        first_try = int(judged[0][1] == Verdict.AC)
        team_first_try[team].append(first_try)
        prob_first_try[prob].append(first_try)
        idx = c.first_ac_idx()
        if idx is not None:
            ac_ms = c.pre[idx][0]
            wrong = sum(1 for _, v in c.pre[:idx] if v == Verdict.REJECTED)
            team_solved[team].append(prob)
            team_wa_pen[team] += wrong
            team_ac_times[team].append(ac_ms)
            if snap.penalty_rule == "in_minutes":
                team_pen_ms[team] += (ac_ms // 60000) * 60000 + wrong * snap.penalty_ms
            else:
                team_pen_ms[team] += ac_ms + wrong * snap.penalty_ms
            prob_ac_teams[prob].append(team)
            prob_wrong_before_ac[prob] += wrong
            prob_ac_times[prob].append(ac_ms)

    team_penalty = {t: math.floor(team_pen_ms[t] / MIN) for t in active_teams}
    order = sorted(active_teams, key=lambda t: (-len(team_solved[t]), team_penalty[t]))
    rank: dict[str, int] = {}
    prev_key, prev_rank = None, 0
    for i, t in enumerate(order):
        key = (len(team_solved[t]), team_penalty[t])
        if key != prev_key:
            prev_key, prev_rank = key, i + 1
        rank[t] = prev_rank
    rank_pct = {t: (rank[t] - 1) / max(1, n_active - 1) for t in active_teams}

    total_attempt = sum(prob_attempt.values())
    total_ac = sum(len(v) for v in prob_ac_teams.values())
    g_solve = total_ac / total_attempt if total_attempt else 0.5
    g_wrong = sum(prob_wrong_before_ac.values())
    g_dirt = g_wrong / (g_wrong + total_ac) if total_ac else 0.5
    all_first = [x for v in prob_first_try.values() for x in v]
    g_first = sum(all_first) / len(all_first) if all_first else 0.5

    prob_ac_frac = {p: len(prob_ac_teams[p]) / n_active for p in problems}
    ease_order = sorted(problems, key=lambda p: -len(prob_ac_teams[p]))
    ease_rank = {p: i / max(1, n_problems - 1) for i, p in enumerate(ease_order)}
    strong = {t for t in active_teams if rank_pct[t] <= 0.25 and team_solved[t]}

    # ---------------- per problem hidden activity (public: pending counts per cell)
    prob_hidden_teams: dict[str, int] = defaultdict(int)
    prob_hidden_subs: dict[str, int] = defaultdict(int)
    for (team, prob), c in cells.items():
        if c.hidden_n and c.first_ac_idx() is None:
            prob_hidden_teams[prob] += 1
            prob_hidden_subs[prob] += c.hidden_n

    recent_window = fz - (dur - fz)  # same length as the freeze, just before it

    # ---------------- M6 aggregates
    star, girl = _group_flags(snap, active_teams)
    board_teams = sorted(set(snap.team_ids))
    keys = {t: (-len(team_solved[t]), team_penalty.get(t, 0)) for t in board_teams}
    medal = _medal_groups(snap, board_teams, keys, star)
    all_keys = sorted(keys[t] for t in active_teams)
    structure, discrimination = _contest_structure(snap, active_teams, team_solved, prob_ac_frac)

    prob_first_attempt: dict[str, list[int]] = defaultdict(list)
    for (team, prob), c in cells.items():
        judged = c.judged()
        if judged:
            prob_first_attempt[prob].append(judged[0][0])

    # last hidden submission time of each team on each problem (Mode A)
    team_last_by_prob: dict[str, dict[str, int]] = defaultdict(dict)
    for (team, prob), c in cells.items():
        if c.hidden_times:
            team_last_by_prob[team][prob] = c.hidden_times[-1]

    def moved_on(team: str, prob: str, last_h: int) -> bool:
        return any(t > last_h for p, t in team_last_by_prob[team].items() if p != prob)

    # crowd: per problem, indicators over frozen cells, summed so each row can subtract itself
    crowd: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(6))  # n, hidden, multi, silent, moved, recent
    cell_ind: dict[tuple[str, str], np.ndarray] = {}
    for (team, prob), c in cells.items():
        if not c.hidden_n or c.first_ac_idx() is not None:
            continue
        v = np.array([1.0, c.hidden_n, c.hidden_n >= 2, np.nan, np.nan, np.nan])
        if mode_a:
            last_h = c.hidden_times[-1]
            v[3] = obs - last_h >= SILENT_MIN * MIN
            v[4] = moved_on(team, prob, last_h)
            v[5] = obs - last_h <= RECENT_HIDDEN_MIN * MIN
        cell_ind[(team, prob)] = v
        crowd[prob] += v  # NaN propagates in Mode B, which is what we want

    rows = []
    for (team, prob), c in cells.items():
        if not c.hidden_n or c.first_ac_idx() is not None:
            continue
        solved = team_solved[team]
        n_solved = len(solved)
        wrong_solved = team_wa_pen[team]
        tf = team_first_try[team]
        pft = prob_first_try[prob]
        p_ac = len(prob_ac_teams[prob])
        p_att = prob_attempt[prob]
        p_wrong = prob_wrong_before_ac[prob]
        ac_times = sorted(prob_ac_times[prob])
        judged = c.judged()
        fails = [t for t, v in judged if v != Verdict.AC]

        r = {
            "team_id": team,
            "problem_id": prob,
            # M1
            "problem_ac_count": p_ac,
            "problem_attempt_count": p_att,
            "problem_solve_rate": _smooth(p_ac, p_att, g_solve),
            "problem_ac_frac": prob_ac_frac[prob],
            "solved_count": n_solved,
            "rank_pct": rank_pct[team],
            "n_problems": n_problems,
            "n_active_teams": n_active,
            # M2
            "penalty": team_penalty[team],
            "penalty_wa_count": wrong_solved,
            "team_dirt": _smooth(wrong_solved, wrong_solved + n_solved, g_dirt),
            "team_dirt_raw": wrong_solved / (wrong_solved + n_solved) if n_solved else np.nan,
            "team_first_ac_rate": _smooth(sum(tf), len(tf), g_first),
            "attempted_count": len(tf),
            "recent_ac_count": sum(1 for t in team_ac_times[team] if t >= recent_window),
            "recent_submission_count": sum(1 for t in team_pre_times[team] if t >= recent_window),
            "weighted_solved": sum(1 - prob_ac_frac[p] for p in solved),
            "minutes_since_last_ac": (fz - max(team_ac_times[team])) / MIN if n_solved else np.nan,
            "problem_dirt": _smooth(p_wrong, p_wrong + p_ac, g_dirt),
            "problem_first_ac_rate": _smooth(sum(pft), len(pft), g_first),
            "problem_first_ac_n": len(pft),
            "first_ac_time": ac_times[0] / MIN if ac_times else np.nan,
            "median_ac_time": median(ac_times) / MIN if ac_times else np.nan,
            "strong_team_solve_rate": (
                _smooth(sum(1 for t in strong if prob in team_solved[t]), len(strong), g_solve)
            ),
            "strong_team_n": len(strong),
            "problem_ease_rank": ease_rank[prob],
            "pre_wa_count": sum(1 for _, v in c.pre if v == Verdict.REJECTED),
            "has_pre_attempt": int(bool(judged)),
            "pre_first_time": c.pre[0][0] / MIN if c.pre else np.nan,
            "pre_last_fail_age": (fz - fails[-1]) / MIN if fails else np.nan,
            "pre_span": (c.pre[-1][0] - c.pre[0][0]) / MIN if c.pre else np.nan,
            "ability_gap": prob_ac_frac[prob] - rank_pct[team],
            "n_harder_solved": sum(1 for p in solved if len(prob_ac_teams[p]) < p_ac),
            "n_easier_unsolved": sum(
                1 for p in problems if p not in solved and len(prob_ac_teams[p]) > p_ac
            ),
        }

        # M3: counts (Mode A and B)
        th = team_hidden[team]
        r.update({
            "hidden_count": c.hidden_n,
            "team_hidden_total": len(th),
            "team_hidden_problems": len({p for _, p in th}),
            "other_hidden_count": len(th) - c.hidden_n,
            "problem_hidden_teams": prob_hidden_teams[prob],
            "problem_hidden_subs": prob_hidden_subs[prob],
            "obs_since_freeze": (obs - fz) / MIN,
            "obs_frac": (obs - fz) / max(1, dur - fz),
            "time_remaining": (dur - obs) / MIN,
        })

        # M6 (count-based part; usable in Mode B)
        tot = crowd[prob] - cell_ind[(team, prob)]
        n_other = tot[0]
        other_probs = {p for _, p in th if p != prob}
        recent_ac = sum(1 for t in ac_times if t >= recent_window)
        r.update({
            "other_hidden_harder_problems": sum(1 for p in other_probs if len(prob_ac_teams[p]) < p_ac),
            "other_hidden_easier_problems": sum(1 for p in other_probs if len(prob_ac_teams[p]) > p_ac),
            "crowd_mean_hidden": tot[1] / n_other if n_other else np.nan,
            "crowd_multi_frac": tot[2] / n_other if n_other else np.nan,
            "problem_recent_ac": recent_ac,
            "problem_recent_ac_share": recent_ac / p_ac if p_ac else np.nan,
            "problem_recent_new_attempts": sum(1 for t in prob_first_attempt[prob] if t >= recent_window),
            "problem_last_ac_age": (fz - ac_times[-1]) / MIN if ac_times else np.nan,
            "team_unofficial": star[team],
            "team_girl": girl[team],
        })
        r.update(structure)
        r["problem_discrimination"] = discrimination[prob]
        my_key = keys[team]
        pre_wa = r["pre_wa_count"]
        key_if_ac = (my_key[0] - 1, my_key[1] + math.floor(fz / MIN) + pre_wa * snap.penalty_ms / MIN)
        r["rank_gain_if_ac"] = (bisect_left(all_keys, my_key) - bisect_left(all_keys, key_if_ac)) / n_active
        if snap.medal_rule is None:
            r.update({k: np.nan for k in ("medal_eligible", "medal_tier_now", "solved_gap_next_tier",
                                          "rank_gap_next_tier", "tier_gain_if_ac")})
        elif team not in medal:
            r.update(medal_eligible=0.0, medal_tier_now=np.nan, solved_gap_next_tier=np.nan,
                     rank_gap_next_tier=np.nan, tier_gain_if_ac=np.nan)
        else:
            sk, cuts = medal[team]
            tier = _tier(sk, cuts, my_key)
            pos = bisect_left(sk, my_key)
            if tier > 0 and cuts[tier - 1] > 0:
                cutoff_key = sk[min(cuts[tier - 1], len(sk)) - 1]
                solved_gap = my_key[0] - cutoff_key[0]  # keys store -solved
                rank_gap = pos + 1 - cuts[tier - 1]
            else:
                solved_gap = rank_gap = np.nan
            r.update(medal_eligible=1.0, medal_tier_now=float(tier), solved_gap_next_tier=solved_gap,
                     rank_gap_next_tier=rank_gap, tier_gain_if_ac=float(tier - _tier(sk, cuts, key_if_ac)))

        # M4/M5 and time-based M6: need hidden times
        nan_keys = TIME_COLUMNS
        if mode_a:
            ht = c.hidden_times
            gaps = [(b - a) / MIN for a, b in zip(ht, ht[1:])]
            last_pre = c.pre[-1][0] if c.pre else None
            first_h, last_h = ht[0], ht[-1]
            others = [(t, p) for t, p in th if p != prob]
            after = [(t, p) for t, p in others if t > last_h]
            team_times = [t for t, _ in th] + team_pre_times[team]
            r.update({
                "first_hidden_time": (first_h - fz) / MIN,
                "last_hidden_time": (last_h - fz) / MIN,
                "last_hidden_age": (obs - last_h) / MIN,
                "hidden_span": (last_h - first_h) / MIN,
                "min_gap": min(gaps) if gaps else np.nan,
                "max_gap": max(gaps) if gaps else np.nan,
                "mean_gap": sum(gaps) / len(gaps) if gaps else np.nan,
                "last_gap": gaps[-1] if gaps else np.nan,
                "short_gap_count": sum(1 for g in gaps if g < SHORT_GAP_MIN),
                "recent_hidden_count": sum(1 for t in ht if obs - t <= RECENT_HIDDEN_MIN * MIN),
                "gap_pre_to_first_hidden": (first_h - last_pre) / MIN if last_pre is not None else np.nan,
                "other_subs_after_last": len(after),
                "other_problems_after_last": len({p for _, p in after}),
                "other_subs_before_first_hidden": sum(1 for t, _ in others if t < first_h),
                "team_last_activity_age": (obs - max(team_times)) / MIN,
                "after_last_harder_problems": sum(1 for p in {p for _, p in after} if len(prob_ac_teams[p]) < p_ac),
                "after_last_easier_problems": sum(1 for p in {p for _, p in after} if len(prob_ac_teams[p]) > p_ac),
                "after_last_min_ac_frac": min((prob_ac_frac[p] for _, p in after), default=np.nan),
                "crowd_silent_frac": tot[3] / n_other if n_other else np.nan,
                "crowd_moved_on_frac": tot[4] / n_other if n_other else np.nan,
                "crowd_recent_teams": tot[5],
                "last_hidden_before_end": (dur - last_h) / MIN,
                "first_hidden_before_end": (dur - first_h) / MIN,
                "hidden_last5_count": sum(1 for t in ht if dur - t <= LAST_MINUTES * MIN),
            })
        else:
            r.update({k: np.nan for k in nan_keys})
        rows.append(r)
    return rows
