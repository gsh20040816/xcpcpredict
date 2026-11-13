"""Unified internal data model and normalization of raw XCPCIO contest files.

All times are integer milliseconds relative to contest start. Time units are
inferred per field; ambiguous cases raise ``NormalizationError``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class NormalizationError(ValueError):
    pass


class Verdict(str, Enum):
    AC = "AC"
    REJECTED = "REJECTED"  # confirmed non-AC, counts for penalty
    REJECTED_NO_PENALTY = "REJECTED_NO_PENALTY"  # confirmed non-AC, no penalty (e.g. CE)
    UNRESOLVED = "UNRESOLVED"  # pending / frozen / unknown final verdict


_AC = {"ACCEPTED", "CORRECT", "AC"}
_REJECTED = {
    "REJECTED", "WRONG_ANSWER", "INCORRECT", "TIME_LIMIT_EXCEEDED", "RUNTIME_ERROR",
    "MEMORY_LIMIT_EXCEEDED", "OUTPUT_LIMIT_EXCEEDED", "NO_OUTPUT", "PRESENTATION_ERROR",
    "IDLENESS_LIMIT_EXCEEDED", "HACKED", "PARTIALLY_CORRECT",
}
_NO_PENALTY = {
    "REJECTED_NO_PENALTY",
    "COMPILATION_ERROR", "SYSTEM_ERROR", "JUDGEMENT_FAILED", "CONFIGURATION_ERROR",
    "FILE_ERROR", "CANCELED", "SKIPPED", "SECURITY_VIOLATION", "DENIAL_OF_JUDGEMENT",
}
_UNRESOLVED = {
    "UNRESOLVED",
    "PENDING", "FROZEN", "WAITING", "PREPARING", "COMPILING", "RUNNING", "JUDGING",
    "UNKNOWN", "UNDEFINED",
}


def map_verdict(raw: str) -> Verdict:
    s = str(raw).strip().upper()
    if s in _AC:
        return Verdict.AC
    if s in _REJECTED:
        return Verdict.REJECTED
    if s in _NO_PENALTY:
        return Verdict.REJECTED_NO_PENALTY
    if s in _UNRESOLVED:
        return Verdict.UNRESOLVED
    raise NormalizationError(f"unknown verdict {raw!r}")


@dataclass(frozen=True)
class Submission:
    team_id: str
    problem_id: str
    time_ms: int
    verdict: Verdict
    submission_id: str | None = None


@dataclass
class Team:
    team_id: str
    groups: tuple[str, ...] = ()


@dataclass
class Contest:
    contest_id: str
    name: str
    start_time_ms: int  # unix epoch ms
    duration_ms: int
    freeze_start_ms: int  # relative to start; == duration_ms when there is no freeze
    penalty_ms: int
    penalty_rule: str  # "in_minutes" | "accumulate_in_seconds_and_finally_to_the_minute"
    problem_ids: list[str]
    teams: dict[str, Team]
    submissions: list[Submission]
    medal_rule: dict | None = None  # see parse_medal_rule
    contest_type: str | None = None  # see classify_contest
    freeze_simulated: bool = False
    quality_flags: list[str] = field(default_factory=list)
    submissions_out_of_range: int = 0

    @property
    def freeze_length_ms(self) -> int:
        return self.duration_ms - self.freeze_start_ms


# ---------------------------------------------------------------- time units

# Contests without a freeze get one over this final share of the contest. Team behaviour does
# not depend on the public board being frozen (teams always see their own verdicts), so a
# simulated freeze still yields valid samples.
SIMULATED_FREEZE_FRACTION = 0.2


def _abs_time_to_ms(value: int | float) -> int:
    """Unix timestamps: seconds are ~1.7e9, milliseconds ~1.7e12."""
    v = int(value)
    if 1e11 <= v < 1e14:
        return v
    if 1e8 <= v < 1e11:
        return v * 1000
    raise NormalizationError(f"cannot infer unit of absolute time {value!r}")


def _frozen_to_ms(frozen: int | float, duration_ms: int) -> int:
    """``frozen_time`` is the freeze length, usually seconds, occasionally ms."""
    f = int(frozen)
    if f * 1000 <= duration_ms:
        # Could still be ms if f <= duration_ms/1e6, i.e. a sub-second freeze in ms;
        # treat seconds as the documented default.
        return f * 1000
    if f <= duration_ms:
        return f
    raise NormalizationError(f"frozen_time {frozen!r} longer than contest ({duration_ms} ms)")


def _submission_multiplier(opts: dict, raw_runs: list[dict], duration_ms: int) -> int:
    """Multiplier converting run timestamps to ms, validated against the contest duration."""
    declared = opts.get("submission_timestamp_unit")
    mult = {"millisecond": 1, "second": 1000, None: 1000}.get(declared)
    if mult is None:
        raise NormalizationError(f"unknown submission_timestamp_unit {declared!r}")
    mx = max((int(r["timestamp"]) for r in raw_runs), default=0)
    if mx * mult > duration_ms * 1.05:
        raise NormalizationError(
            f"max timestamp {mx} with unit {declared or 'second(default)'} exceeds duration {duration_ms} ms")
    return mult


# ----------------------------------------------------------------- loading

def _team_groups(t: dict) -> tuple[str, ...]:
    groups = []
    g = t.get("group")
    if isinstance(g, list):
        groups.extend(str(x) for x in g)
    elif isinstance(g, str):
        groups.append(g)
    for k in ("official", "unofficial", "girl", "undergraduate", "vocational"):
        if t.get(k) in (True, 1, "1", "true"):
            groups.append(k)
    return tuple(dict.fromkeys(groups))


def classify_contest(contest_id: str, name: str) -> str:
    """Coarse contest type from the board-data path and contest name."""
    cid, low = contest_id.lower(), name.lower()
    if any(m in cid for m in ("online", "qualification", "preliminary")):
        return "online"
    if any(m in cid for m in ("world-finals", "ec-final", "ecfinal")) or cid.endswith("/final"):
        return "final"
    if any(m in cid for m in ("girl", "vocational", "gaozhi")):
        return "special"
    if cid.startswith("provincial-contest/"):
        return "provincial"
    if "invitational" in cid or "邀请赛" in name or "invitational" in low:
        return "invitational"
    if cid.startswith(("icpc/", "ccpc/")):
        return "regional"
    return "special"


def parse_medal_rule(raw) -> dict | None:
    """XCPCIO ``medal`` config -> {"counts": {group: (gold, silver, bronze)}} or {"ratio": (g, s, b)}.

    Counts are per medal colour (gold = ranks 1..g, silver the next s, ...). The "ccpc"
    preset is approximated as 10/20/30% of eligible teams with at least one solve.
    """
    if isinstance(raw, str):
        return {"ratio": (0.1, 0.2, 0.3)} if raw.lower() == "ccpc" else None
    if isinstance(raw, dict):
        counts = {
            str(g): (int(v.get("gold", 0)), int(v.get("silver", 0)), int(v.get("bronze", 0)))
            for g, v in raw.items() if isinstance(v, dict)
        }
        return {"counts": counts} if counts else None
    return None


def _text(value) -> str:
    """Plain string from either a string or an XCPCIO i18n object {fallback_lang, texts}."""
    if isinstance(value, dict) and "texts" in value:
        texts = value["texts"]
        return str(texts.get(value.get("fallback_lang")) or next(iter(texts.values()), ""))
    return str(value)


def normalize_contest(
    contest_id: str,
    raw_cfg: dict,
    raw_teams: list | dict,
    raw_runs: list[dict],
) -> Contest:
    flags: list[str] = []

    start_ms = _abs_time_to_ms(raw_cfg["start_time"])
    duration_ms = _abs_time_to_ms(raw_cfg["end_time"]) - start_ms
    if duration_ms <= 0:
        raise NormalizationError("non-positive duration")

    freeze_len_ms = _frozen_to_ms(raw_cfg.get("frozen_time", 0) or 0, duration_ms)
    freeze_simulated = freeze_len_ms == 0
    if freeze_simulated:
        freeze_len_ms = int(round(duration_ms * SIMULATED_FREEZE_FRACTION))
        flags.append("freeze_simulated")

    penalty_ms = int(raw_cfg.get("penalty", 1200)) * 1000
    opts = raw_cfg.get("options") or {}
    penalty_rule = opts.get("calculation_of_penalty") or "in_minutes"

    problem_ids = [str(p.get("id", i)) for i, p in enumerate(raw_cfg["problems"])]

    teams: dict[str, Team] = {}
    if isinstance(raw_teams, dict):
        team_list = [dict(v, team_id=v.get("team_id", k)) for k, v in raw_teams.items()]
    else:
        team_list = raw_teams
    for t in team_list:
        tid = str(t.get("team_id", t.get("id")))
        teams[tid] = Team(tid, _team_groups(t))

    mult = _submission_multiplier(opts, raw_runs, duration_ms)

    subs: list[Submission] = []
    unknown_team = unknown_problem = out_of_range = 0
    pid_set = set(problem_ids)
    for r in raw_runs:
        tid = str(r["team_id"])
        pid = str(r["problem_id"])
        if pid not in pid_set:
            unknown_problem += 1
            continue
        if tid not in teams:
            unknown_team += 1
            teams[tid] = Team(tid, ("unknown",))
        t_ms = int(r["timestamp"]) * mult
        if t_ms < 0 or t_ms > duration_ms:
            out_of_range += 1
            continue
        sid = r.get("submission_id", r.get("id"))
        subs.append(Submission(tid, pid, t_ms, map_verdict(r["status"]), str(sid) if sid is not None else None))
    subs.sort(key=lambda s: (s.time_ms, s.submission_id or ""))
    if unknown_team:
        flags.append("unknown_team_refs")
    if unknown_problem:
        flags.append("unknown_problem_refs")
    if out_of_range:
        flags.append("submissions_out_of_range")

    name = _text(raw_cfg.get("contest_name", contest_id))
    return Contest(
        contest_id=contest_id,
        name=name,
        start_time_ms=start_ms,
        duration_ms=duration_ms,
        freeze_start_ms=duration_ms - freeze_len_ms,
        penalty_ms=penalty_ms,
        penalty_rule=penalty_rule,
        problem_ids=problem_ids,
        teams=teams,
        submissions=subs,
        medal_rule=parse_medal_rule(raw_cfg.get("medal")),
        contest_type=classify_contest(contest_id, name),
        freeze_simulated=freeze_simulated,
        quality_flags=flags,
        submissions_out_of_range=out_of_range,
    )
