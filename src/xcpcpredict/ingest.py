"""Download, discover and read XCPCIO contests.

All contest data comes from the xcpcio/board-data git repository (MIT licensed), checked out
at a release tag so that every report and model can be reproduced from the same data.
board.xcpcio.com is never crawled: its robots.txt disallows all automated access, and every
contest it shows is published to board-data.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

BOARD_DATA_URL = "https://github.com/xcpcio/board-data.git"
# Release tag of board-data behind the committed reports and the default model.
BOARD_DATA_REV = "v2026.11.11.2057"
DEFAULT_DATA_DIR = Path("data/board-data")


@dataclass
class RawContest:
    contest_id: str
    config: dict
    teams: list | dict
    runs: list[dict]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def fetch_board_data(dest: Path = DEFAULT_DATA_DIR, rev: str = BOARD_DATA_REV) -> str:
    """Shallow checkout of ``rev`` (tag, branch or commit) of board-data at ``dest``; returns the commit."""
    if not (dest / ".git").exists():
        dest.mkdir(parents=True, exist_ok=True)
        _git(dest, "init", "-q")
        _git(dest, "remote", "add", "origin", BOARD_DATA_URL)
    _git(dest, "fetch", "-q", "--depth", "1", "origin", rev)
    _git(dest, "checkout", "-q", "--detach", "FETCH_HEAD")
    return data_version(dest)


def data_version(repo: Path = DEFAULT_DATA_DIR) -> str:
    return _git(repo, "rev-parse", "HEAD").strip()


def discover_contests(repo: Path = DEFAULT_DATA_DIR) -> list[str]:
    root = repo / "data"
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found; run `xcpcpredict fetch` first")
    return sorted(str(p.parent.relative_to(root)) for p in root.rglob("config.json"))


def load_raw(contest_id: str, repo: Path = DEFAULT_DATA_DIR) -> RawContest:
    d = repo / "data" / contest_id

    def read(name: str):
        with open(d / name, encoding="utf-8") as f:
            return json.load(f)

    return RawContest(contest_id, read("config.json"), read("team.json"), read("run.json"))
