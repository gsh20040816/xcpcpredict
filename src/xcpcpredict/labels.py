"""Supervised labels from private (true) historical verdicts.

y = 1 iff at least one submission made in [freeze_start, observation] on the cell is
finally AC. Submissions made after the observation time never influence the label.
"""

from __future__ import annotations

from dataclasses import dataclass

from .normalize import Contest, Verdict


@dataclass(frozen=True)
class CellLabel:
    y: int
    unresolved: bool  # some hidden submission has no final verdict -> label unusable


def label_cells(contest: Contest, observation_ms: int) -> dict[tuple[str, str], CellLabel]:
    fz = contest.freeze_start_ms
    pre_ac: set[tuple[str, str]] = set()
    hidden: dict[tuple[str, str], list[Verdict]] = {}
    for s in contest.submissions:
        key = (s.team_id, s.problem_id)
        if s.time_ms < fz:
            if s.verdict == Verdict.AC:
                pre_ac.add(key)
        elif s.time_ms <= observation_ms:
            hidden.setdefault(key, []).append(s.verdict)
    out = {}
    for key, verdicts in hidden.items():
        if key in pre_ac:
            continue
        out[key] = CellLabel(
            y=int(Verdict.AC in verdicts),
            unresolved=Verdict.UNRESOLVED in verdicts and Verdict.AC not in verdicts,
        )
    return out
