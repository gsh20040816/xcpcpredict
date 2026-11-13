"""Inference on a public frozen-scoreboard snapshot, plus model artifact I/O.

Input JSON (times in seconds from contest start):

{
  "contest_id": "my-contest",
  "contest_type": "regional",              # optional: regional/invitational/provincial/online/final/special
  "duration_s": 18000, "freeze_start_s": 14400, "observation_s": 16200,
  "penalty_s": 1200,                       # optional, default 1200
  "penalty_rule": "in_minutes",            # optional
  "mode": "A",                             # A: hidden times known; B: pending counts only; C: board stopped
  "problems": ["A", "B", ...],
  "teams": ["t1", {"id": "t2", "groups": ["unofficial"]}, ...],   # optional; groups optional
  "medal": {"official": {"gold": 32, "silver": 64, "bronze": 96}},  # optional, XCPCIO format or "ccpc"
  "submissions": [{"team": "t1", "problem": "A", "time_s": 600, "verdict": "WRONG_ANSWER"}, ...],
  "hidden": [{"team": "t1", "problem": "B", "time_s": 14700}, ...],          # Mode A
  "hidden_counts": [{"team": "t1", "problem": "B", "count": 2}, ...]         # Mode B
}

``submissions`` holds only pre-freeze runs with their public verdicts (non-penalized ones such
as CE are ignored, as public boards do not show them). Hidden entries must not carry a
verdict; such input is rejected rather than silently used.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .features import FEATURE_VERSION, extract_features
from .normalize import Contest, Verdict, map_verdict, parse_medal_rule
from .replay import MODE_COUNTS_ONLY, MODE_FULL_TIMES, PublicSnapshot, PublicSubmission, make_snapshot

MODEL_FILE = "model.pkl"


class PredictionUnavailable(RuntimeError):
    pass


@dataclass
class ModelBundle:
    model_version: str
    models: dict  # mode -> CalibratedModel
    meta: dict = field(default_factory=dict)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / MODEL_FILE, "wb") as f:
            pickle.dump(self, f)
        (directory / "meta.json").write_text(json.dumps(
            {"model_version": self.model_version, "feature_version": FEATURE_VERSION, **self.meta},
            indent=2, ensure_ascii=False))
        return directory / MODEL_FILE

    @staticmethod
    def load(directory: Path) -> "ModelBundle":
        with open(Path(directory) / MODEL_FILE, "rb") as f:
            return pickle.load(f)


def _ms(seconds) -> int:
    return int(round(float(seconds) * 1000))


def snapshot_from_json(data: dict) -> PublicSnapshot:
    mode = str(data.get("mode", MODE_FULL_TIMES)).upper()
    if mode == "C":
        raise PredictionUnavailable(
            "Mode C: the public board stopped updating, so hidden submissions are not observable; "
            "no frozen cell can be identified and no prediction is made.")
    fz = _ms(data["freeze_start_s"])
    subs: list[PublicSubmission] = []
    for s in data.get("submissions", []):
        t = _ms(s["time_s"])
        if t >= fz:
            raise ValueError(f"submission at {s['time_s']}s is after the freeze; put it in 'hidden'")
        v = map_verdict(s["verdict"])
        if v != Verdict.REJECTED_NO_PENALTY:
            subs.append(PublicSubmission(str(s["team"]), str(s["problem"]), t, v))
    if mode not in (MODE_FULL_TIMES, MODE_COUNTS_ONLY):
        raise ValueError(f"unknown mode {mode!r}")
    hidden = data.get("hidden" if mode == MODE_FULL_TIMES else "hidden_counts", [])
    if any("verdict" in h or "status" in h for h in hidden):
        raise ValueError("hidden submissions must not carry a verdict")
    for h in hidden:
        if mode == MODE_FULL_TIMES:
            subs.append(PublicSubmission(str(h["team"]), str(h["problem"]), _ms(h["time_s"]), None))
        else:
            subs.extend(PublicSubmission(str(h["team"]), str(h["problem"]), None, None)
                        for _ in range(int(h["count"])))
    teams, groups = [], {}
    for t in data.get("teams", []):
        tid = str(t["id"]) if isinstance(t, dict) else str(t)
        teams.append(tid)
        if isinstance(t, dict):
            groups[tid] = tuple(str(g) for g in t.get("groups", ()))
    teams += sorted({s.team_id for s in subs} - set(teams))
    return PublicSnapshot(
        contest_id=str(data.get("contest_id", "unknown")),
        duration_ms=_ms(data["duration_s"]),
        freeze_start_ms=fz,
        observation_ms=_ms(data["observation_s"]),
        penalty_ms=_ms(data.get("penalty_s", 1200)),
        penalty_rule=str(data.get("penalty_rule", "in_minutes")),
        problem_ids=[str(p) for p in data["problems"]],
        team_ids=teams,
        mode=mode,
        submissions=subs,
        team_groups=groups,
        medal_rule=parse_medal_rule(data.get("medal")),
        contest_type=data.get("contest_type"),
    )


def _medal_to_json(rule: dict | None):
    if not rule:
        return None
    if "ratio" in rule:
        return "ccpc"
    return {g: {"gold": a, "silver": b, "bronze": c} for g, (a, b, c) in rule["counts"].items()}


def snapshot_to_json(snap: PublicSnapshot) -> dict:
    out = {
        "contest_id": snap.contest_id,
        "contest_type": snap.contest_type,
        "duration_s": snap.duration_ms / 1000,
        "freeze_start_s": snap.freeze_start_ms / 1000,
        "observation_s": snap.observation_ms / 1000,
        "penalty_s": snap.penalty_ms / 1000,
        "penalty_rule": snap.penalty_rule,
        "mode": snap.mode,
        "problems": snap.problem_ids,
        "teams": [{"id": t, "groups": list(snap.team_groups.get(t, ()))} for t in snap.team_ids],
        "medal": _medal_to_json(snap.medal_rule),
        "submissions": [
            {"team": s.team_id, "problem": s.problem_id, "time_s": s.time_ms / 1000, "verdict": s.verdict.value}
            for s in snap.submissions if not s.hidden
        ],
    }
    hidden = [s for s in snap.submissions if s.hidden]
    if snap.mode == MODE_FULL_TIMES:
        out["hidden"] = [{"team": s.team_id, "problem": s.problem_id, "time_s": s.time_ms / 1000} for s in hidden]
    else:
        counts: dict[tuple[str, str], int] = {}
        for s in hidden:
            counts[(s.team_id, s.problem_id)] = counts.get((s.team_id, s.problem_id), 0) + 1
        out["hidden_counts"] = [{"team": t, "problem": p, "count": n} for (t, p), n in sorted(counts.items())]
    return out


def export_historical_snapshot(contest: Contest, observation_ms: int, mode: str) -> dict:
    return snapshot_to_json(make_snapshot(contest, observation_ms, mode))


def predict_snapshot(bundle: ModelBundle, snap: PublicSnapshot) -> pd.DataFrame:
    rows = extract_features(snap)
    cols = ["contest_id", "team_id", "problem_id", "observation_time", "hidden_submissions", "p_ac",
            "model_version", "feature_version", "quality_flags"]
    if not rows:
        return pd.DataFrame(columns=cols)
    X = pd.DataFrame(rows)
    model = bundle.models[snap.mode]
    flags = "" if snap.mode == MODE_FULL_TIMES else "hidden_times_unavailable"
    out = pd.DataFrame({
        "contest_id": snap.contest_id,
        "team_id": X["team_id"],
        "problem_id": X["problem_id"],
        "observation_time": snap.observation_ms / 1000,
        "hidden_submissions": X["hidden_count"].astype(int),
        "p_ac": model.predict_proba(X),
        "model_version": bundle.model_version,
        "feature_version": FEATURE_VERSION,
        "quality_flags": flags,
    })
    return out[cols].sort_values(["team_id", "problem_id"]).reset_index(drop=True)
