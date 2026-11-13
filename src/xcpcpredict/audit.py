"""Per-contest data quality and provenance checks.

Quality tiers:
  A  second (or finer) resolution, no sign of synthesis — preferred for training/eval
  B  minute-resolution timestamps, otherwise plausible real logs
  C  suspected synthesized from a scoreboard (e.g. rejected runs stamped at t=0)
  X  unusable (normalization failure, unresolved hidden verdicts, no frozen cells, duplicate)

Provenance is never asserted: every contest carries ``provenance_unknown`` because
board-data does not record whether runs come from a judge log or a scoreboard.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

import pandas as pd

from .normalize import Contest, Verdict

WARMUP_MARKERS = ("warmup", "practice", "dress-rehearsal", "热身")
ONLINE_MARKERS = ("online", "qualification", "网络")


@dataclass
class ContestAudit:
    contest_id: str
    name: str = ""
    tier: str = "X"
    reasons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    start_time_ms: int = 0
    duration_min: float = 0.0
    freeze_min: float = 0.0
    freeze_simulated: bool = False
    is_warmup: bool = False
    is_online: bool = False
    contest_type: str = ""
    n_teams: int = 0
    n_problems: int = 0
    n_submissions: int = 0
    raw_statuses: str = ""
    resolution: str = ""
    zero_time_frac: float = 0.0
    same_time_cell_frac: float = 0.0
    pre_unresolved: int = 0
    post_submissions: int = 0
    post_unresolved: int = 0
    frozen_cells: int = 0
    frozen_cell_ac_rate: float = float("nan")
    post_ac_resubmit_frac: float = float("nan")
    fingerprint: str = ""
    duplicate_of: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["reasons"] = ";".join(self.reasons)
        d["flags"] = ";".join(self.flags)
        return d


def fingerprint(contest: Contest) -> str:
    h = hashlib.sha1()
    for s in contest.submissions:
        h.update(f"{s.team_id}|{s.problem_id}|{s.time_ms}|{s.verdict.value}\n".encode())
    return h.hexdigest()


def _resolution(times: list[int]) -> str:
    n = len(times)
    if sum(1 for t in times if t % 60000 == 0) / n > 0.9:
        return "minute"
    if sum(1 for t in times if t % 1000 == 0) / n > 0.99:
        return "second"
    return "millisecond"


def audit_contest(contest: Contest, raw_runs: list[dict]) -> ContestAudit:
    a = ContestAudit(contest.contest_id, name=contest.name)
    a.flags = list(contest.quality_flags) + ["provenance_unknown"]
    a.start_time_ms = contest.start_time_ms
    a.duration_min = contest.duration_ms / 60000
    a.freeze_min = contest.freeze_length_ms / 60000
    a.freeze_simulated = contest.freeze_simulated
    low = f"{contest.contest_id} {contest.name}".lower()
    a.is_warmup = any(m in low for m in WARMUP_MARKERS)
    a.is_online = any(m in low for m in ONLINE_MARKERS)
    a.contest_type = contest.contest_type
    a.n_teams = len(contest.teams)
    a.n_problems = len(contest.problem_ids)
    subs = contest.submissions
    a.n_submissions = len(subs)
    a.raw_statuses = ",".join(f"{k}:{v}" for k, v in Counter(r["status"] for r in raw_runs).most_common())
    a.fingerprint = fingerprint(contest)

    if not subs:
        a.reasons.append("no_submissions")
        return a

    times = [s.time_ms for s in subs]
    a.resolution = _resolution(times)
    a.zero_time_frac = sum(1 for t in times if t == 0) / len(times)

    cells: dict[tuple[str, str], list] = defaultdict(list)
    for s in subs:
        cells[(s.team_id, s.problem_id)].append(s)
    multi = [v for v in cells.values() if len(v) >= 2]
    a.same_time_cell_frac = (
        sum(1 for v in multi if len({s.time_ms for s in v}) == 1) / len(multi) if multi else 0.0
    )

    fz = contest.freeze_start_ms
    a.pre_unresolved = sum(1 for s in subs if s.time_ms < fz and s.verdict == Verdict.UNRESOLVED)
    post = [s for s in subs if s.time_ms >= fz]
    a.post_submissions = len(post)
    a.post_unresolved = sum(1 for s in post if s.verdict == Verdict.UNRESOLVED)

    frozen = ac = 0
    ac_cells = resub = 0
    for v in cells.values():
        pre_ac = any(s.verdict == Verdict.AC and s.time_ms < fz for s in v)
        hidden = [s for s in v if s.time_ms >= fz]
        if not pre_ac and hidden:
            frozen += 1
            ac += any(s.verdict == Verdict.AC for s in hidden)
        first_ac = next((i for i, s in enumerate(v) if s.verdict == Verdict.AC), None)
        if first_ac is not None:
            ac_cells += 1
            resub += first_ac < len(v) - 1
    a.frozen_cells = frozen
    a.frozen_cell_ac_rate = ac / frozen if frozen else float("nan")
    a.post_ac_resubmit_frac = resub / ac_cells if ac_cells else float("nan")

    # ---- tiering
    if a.post_submissions and a.post_unresolved / a.post_submissions > 0.02:
        a.reasons.append("unresolved_hidden_verdicts")
    if frozen < 5:
        a.reasons.append("too_few_frozen_cells")
    if contest.submissions_out_of_range > 0.01 * (len(subs) + 1):
        a.reasons.append("many_submissions_out_of_range")
    if a.reasons:
        a.tier = "X"
    elif a.zero_time_frac > 0.01 or a.same_time_cell_frac > 0.25:
        a.tier = "C"
        a.reasons.append("suspected_synthesized")
    elif frozen >= 20 and a.frozen_cell_ac_rate > 0.95:
        # Real freezes always contain failed attempts; a log where every hidden run is
        # AC was almost certainly rebuilt from a final scoreboard.
        a.tier = "C"
        a.reasons.append("no_hidden_rejections")
    elif a.resolution == "minute":
        a.tier = "B"
    else:
        a.tier = "A"
    return a


def mark_duplicates(audits: list[ContestAudit]) -> None:
    """Identical submission logs published under several ids leak across splits."""
    seen: dict[str, str] = {}
    for a in sorted(audits, key=lambda x: x.contest_id):
        if not a.n_submissions:
            continue
        if a.fingerprint in seen:
            a.duplicate_of = seen[a.fingerprint]
            a.tier = "X"
            a.reasons.append("duplicate")
        else:
            seen[a.fingerprint] = a.contest_id


def resubmission_stats(contests: list[Contest], windows_min=(5, 10, 30, 60)) -> pd.DataFrame:
    """P(same team resubmits the same problem within Δ | verdict of this run).

    Tests H4/H5 on real logs. Runs with less than Δ left before the contest end are
    excluded for that Δ (right censoring).
    """
    counts = {(v, w): [0, 0] for v in ("AC", "REJECTED") for w in windows_min}
    for c in contests:
        cells: dict[tuple[str, str], list] = defaultdict(list)
        for s in c.submissions:
            cells[(s.team_id, s.problem_id)].append(s)
        for runs in cells.values():
            for i, s in enumerate(runs):
                if s.verdict not in (Verdict.AC, Verdict.REJECTED):
                    continue
                nxt = runs[i + 1].time_ms if i + 1 < len(runs) else None
                for w in windows_min:
                    if s.time_ms + w * 60000 > c.duration_ms:
                        continue
                    cnt = counts[(s.verdict.value, w)]
                    cnt[1] += 1
                    cnt[0] += nxt is not None and nxt - s.time_ms <= w * 60000
    rows = [{"verdict": v, "window_min": w, "runs": n, "resubmit_rate": k / n if n else float("nan")}
            for (v, w), (k, n) in counts.items()]
    return pd.DataFrame(rows)


def render_quality_report(audit: pd.DataFrame, behavior: pd.DataFrame, data_version: str, command: str) -> str:
    def table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
        for r in df.astype(object).itertuples(index=False):
            out.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
        return "\n".join(out)

    a = audit.copy()
    L = ["# Data quality report\n",
         f"Source: xcpcio/board-data commit `{data_version}`, {len(a)} contests; reproduce with `{command}`.\n"]
    L.append("## Tiers\n")
    L.append("- **A**: second/ms-resolution timestamps, no synthesis signal\n- **B**: minute-resolution timestamps\n"
             "- **C**: suspected synthesized from a scoreboard (rejected runs at t=0, identical timestamps "
             "per cell, or no rejected hidden runs at all) — labels unusable\n"
             "- **X**: unusable (normalization error, unresolved hidden verdicts, duplicate log, <5 frozen cells)\n")
    g = a.groupby(["tier", "is_warmup"]).agg(
        contests=("contest_id", "size"), submissions=("n_submissions", "sum"),
        frozen_cells=("frozen_cells", "sum"), mean_frozen_ac_rate=("frozen_cell_ac_rate", "mean"),
    ).reset_index()
    L.append(table(g))
    L.append("\nThe frozen-cell AC rate of tier C is ≈1.0: synthesized logs put every rejected run "
             "before the freeze, so their hidden cells are all AC. Training on them would teach the model "
             "that hidden submissions always pass, so tier C is excluded by default.\n")
    reasons = a.loc[a["reasons"] != "", "reasons"].str.split(";").explode().value_counts()
    L.append("## Reasons for downgrading\n")
    L.append(table(reasons.rename_axis("reason").reset_index(name="contests")))
    L.append("\n## Format variation observed\n")
    L.append(table(a.groupby("resolution").size().rename_axis("timestamp_resolution").reset_index(name="contests")))
    L.append("")
    L.append(f"- Contests with no freeze (`frozen_time=0`, freeze simulated as the last 20%): "
             f"{int(a['freeze_simulated'].sum())}")
    L.append(f"- Warmup / practice contests: {int(a['is_warmup'].sum())} (excluded from the main dataset; their "
             f"frozen-cell AC rate is much higher)")
    L.append(f"- Online contests: {int(a['is_online'].sum())}")
    L.append(f"- Duplicate logs published under two ids: {int((a['duplicate_of'] != '').sum())}")
    L.append(f"- Contests referencing teams missing from team.json: "
             f"{int(a['flags'].str.contains('unknown_team_refs').sum())}")
    L.append("- Every contest is flagged `provenance_unknown`: board-data does not state whether runs come from "
             "a judge log or were rebuilt from a scoreboard.\n")
    use = a[a["tier"].isin(["A", "B"]) & ~a["is_warmup"]]
    L.append("## Usable formal contests (tier A/B, non-warmup)\n")
    L.append(f"{len(use)} contests, {int(use['n_submissions'].sum())} submissions, "
             f"{int(use['frozen_cells'].sum())} frozen cells at contest end; median frozen cells per contest "
             f"{use['frozen_cells'].median():.0f}; frozen-cell AC rate mean {use['frozen_cell_ac_rate'].mean():.3f} "
             f"(range {use['frozen_cell_ac_rate'].min():.3f}–{use['frozen_cell_ac_rate'].max():.3f}).\n")
    L.append(f"Share of solved cells with another submission after the first AC: median "
             f"{use['post_ac_resubmit_frac'].median():.3f}, max {use['post_ac_resubmit_frac'].max():.3f}.\n")
    L.append("## Resubmission behaviour (tier A formal contests)\n")
    L.append("P(the team submits the same problem again within Δ minutes | verdict of the run). This is the "
             "mechanism behind H4/H5: a follow-up hidden submission is strong evidence the earlier one failed.\n")
    L.append(table(behavior.pivot(index="window_min", columns="verdict", values="resubmit_rate").reset_index()))
    L.append("\n## Limitations\n")
    L.append("- No official frozen scoreboards are stored in board-data, so reconstructed solved/penalty/rank "
             "cannot be compared against an independent official source; XCPCIO itself renders boards from "
             "the same run.json, so agreement with the live board is by construction.\n"
             "- Judging completion times are not recorded; a run judged after the next submission is "
             "indistinguishable from one judged instantly.\n"
             "- Whether a real public board showed hidden submission times (Mode A) or only pending counts "
             "(Mode B) is not recorded per contest; both modes are modelled.\n"
             "- Compile errors during the freeze are treated as visible pending submissions (non-AC). Boards "
             "that hide CE would show fewer hidden submissions.\n")
    L.append("## Per-contest table\n")
    L.append("See `audit.csv` for every contest's metrics and reasons.\n")
    return "\n".join(L)
