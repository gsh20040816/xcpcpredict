"""Historical replay: build the public view of a contest at an observation time.

``PublicSnapshot`` is the only object feature extraction may see. It is built so that
post-freeze verdicts cannot be represented at all (``verdict is None`` for hidden
submissions), and in Mode B hidden submission times are dropped too.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .normalize import Contest, Verdict

MODE_FULL_TIMES = "A"  # hidden submission times visible
MODE_COUNTS_ONLY = "B"  # only the number of hidden submissions per cell is visible

# Observation offsets as fractions of the freeze length: for a 60 min freeze this is
# +5, +10, +20, +30, +40, +50, +60 minutes.
DEFAULT_OBS_FRACTIONS = (1 / 12, 2 / 12, 4 / 12, 6 / 12, 8 / 12, 10 / 12, 1.0)


@dataclass(frozen=True)
class PublicSubmission:
    team_id: str
    problem_id: str
    time_ms: int | None  # None only for hidden submissions in Mode B
    verdict: Verdict | None  # None <=> hidden (submitted at/after freeze)

    @property
    def hidden(self) -> bool:
        return self.verdict is None


@dataclass
class PublicSnapshot:
    contest_id: str
    duration_ms: int
    freeze_start_ms: int
    observation_ms: int
    penalty_ms: int
    penalty_rule: str
    problem_ids: list[str]
    team_ids: list[str]
    mode: str
    submissions: list[PublicSubmission] = field(default_factory=list)
    team_groups: dict[str, tuple[str, ...]] = field(default_factory=dict)
    medal_rule: dict | None = None
    contest_type: str | None = None  # known to any observer: regional / invitational / ...

    def __post_init__(self):
        if not (self.freeze_start_ms <= self.observation_ms <= self.duration_ms):
            raise ValueError("observation time must lie inside the freeze period")
        for s in self.submissions:
            if s.hidden and s.time_ms is not None and not (self.freeze_start_ms <= s.time_ms <= self.observation_ms):
                raise ValueError("hidden submission outside [freeze, observation]")


def make_snapshot(contest: Contest, observation_ms: int, mode: str = MODE_FULL_TIMES) -> PublicSnapshot:
    fz = contest.freeze_start_ms
    subs: list[PublicSubmission] = []
    for s in contest.submissions:
        if s.time_ms > observation_ms:
            continue  # not submitted yet
        if s.time_ms < fz:
            # Public boards show only AC / penalized attempts; a CE before the freeze is
            # invisible. During the freeze every run, CE included, appears as pending.
            if s.verdict != Verdict.REJECTED_NO_PENALTY:
                subs.append(PublicSubmission(s.team_id, s.problem_id, s.time_ms, s.verdict))
        else:
            t = s.time_ms if mode == MODE_FULL_TIMES else None
            subs.append(PublicSubmission(s.team_id, s.problem_id, t, None))
    return PublicSnapshot(
        contest_id=contest.contest_id,
        duration_ms=contest.duration_ms,
        freeze_start_ms=fz,
        observation_ms=observation_ms,
        penalty_ms=contest.penalty_ms,
        penalty_rule=contest.penalty_rule,
        problem_ids=list(contest.problem_ids),
        team_ids=list(contest.teams),
        mode=mode,
        submissions=subs,
        team_groups={tid: t.groups for tid, t in contest.teams.items()},
        medal_rule=contest.medal_rule,
        contest_type=contest.contest_type,
    )


def observation_times(contest: Contest, fractions=DEFAULT_OBS_FRACTIONS) -> list[int]:
    fz, L = contest.freeze_start_ms, contest.freeze_length_ms
    return sorted({fz + int(round(L * f)) for f in fractions})
