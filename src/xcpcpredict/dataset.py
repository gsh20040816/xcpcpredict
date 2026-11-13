"""Load all contests, audit them, and build the supervised replay dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .audit import ContestAudit, audit_contest, mark_duplicates
from .features import extract_features
from .ingest import DEFAULT_DATA_DIR, discover_contests, load_raw
from .labels import label_cells
from .normalize import Contest, NormalizationError, normalize_contest
from .replay import MODE_FULL_TIMES, make_snapshot, observation_times

log = logging.getLogger(__name__)

META_COLUMNS = ["contest_id", "team_id", "problem_id", "observation_ms", "y", "tier", "contest_type",
                "contest_start_ms"]


@dataclass
class LoadedContest:
    contest: Contest | None
    audit: ContestAudit


def load_all(repo: Path = DEFAULT_DATA_DIR, contest_ids: list[str] | None = None) -> list[LoadedContest]:
    out = []
    for cid in contest_ids or discover_contests(repo):
        raw = load_raw(cid, repo)
        try:
            c = normalize_contest(cid, raw.config, raw.teams, raw.runs)
        except NormalizationError as e:
            a = ContestAudit(cid, name=str(raw.config.get("contest_name", cid)))
            a.reasons.append(f"normalization_error:{e}")
            out.append(LoadedContest(None, a))
            continue
        out.append(LoadedContest(c, audit_contest(c, raw.runs)))
    mark_duplicates([lc.audit for lc in out])
    return out


def audit_frame(loaded: list[LoadedContest]) -> pd.DataFrame:
    return pd.DataFrame([lc.audit.to_dict() for lc in loaded])


def contest_rows(contest: Contest, audit: ContestAudit, mode: str = MODE_FULL_TIMES) -> tuple[list[dict], int]:
    rows, dropped = [], 0
    for t in observation_times(contest):
        snap = make_snapshot(contest, t, mode)
        labels = label_cells(contest, t)
        for r in extract_features(snap):
            lab = labels[(r["team_id"], r["problem_id"])]
            if lab.unresolved:
                dropped += 1
                continue
            r.update(
                contest_id=contest.contest_id, observation_ms=t, y=lab.y, tier=audit.tier,
                contest_type=contest.contest_type, contest_start_ms=contest.start_time_ms,
            )
            rows.append(r)
    return rows, dropped


def build_dataset(loaded: list[LoadedContest], tiers=("A", "B", "C"), mode: str = MODE_FULL_TIMES) -> pd.DataFrame:
    rows: list[dict] = []
    for lc in loaded:
        if lc.contest is None or lc.audit.tier not in tiers:
            continue
        r, dropped = contest_rows(lc.contest, lc.audit, mode)
        if dropped:
            log.info("%s: dropped %d rows with unresolved hidden verdicts", lc.audit.contest_id, dropped)
        rows.extend(r)
    df = pd.DataFrame(rows)
    return df[META_COLUMNS + [c for c in df.columns if c not in META_COLUMNS]]
