"""Command line: fetch -> audit -> build -> experiment / train -> predict."""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
from pathlib import Path

import pandas as pd

from . import ingest
from .audit import render_quality_report, resubmission_stats
from .dataset import audit_frame, build_dataset, load_all
from .replay import MODE_FULL_TIMES

DEFAULT_DATASET = Path("data/dataset.pkl")
DEFAULT_MODEL_DIR = Path("models/v4")


def _command() -> str:
    return "xcpcpredict " + " ".join(shlex.quote(a) for a in sys.argv[1:])


def cmd_fetch(args):
    commit = ingest.fetch_board_data(args.data, args.rev)
    print(f"board-data {args.rev} ({commit}) at {args.data}")


def cmd_audit(args):
    loaded = load_all(args.data)
    a = audit_frame(loaded)
    args.out.mkdir(parents=True, exist_ok=True)
    a.drop(columns=["fingerprint"]).to_csv(args.out / "audit.csv", index=False)
    real = [lc.contest for lc in loaded if lc.audit.tier == "A" and not lc.audit.is_warmup]
    beh = resubmission_stats(real)
    beh.to_csv(args.out / "resubmission_stats.csv", index=False)
    (args.out / "data_quality_report.md").write_text(render_quality_report(a, beh, ingest.data_version(args.data), _command()))
    print(a["tier"].value_counts().to_string())
    print(f"wrote {args.out / 'data_quality_report.md'}")


def cmd_build(args):
    loaded = load_all(args.data)
    if not args.include_warmup:
        loaded = [lc for lc in loaded if not lc.audit.is_warmup]
    df = build_dataset(loaded, tiers=tuple(args.tiers.split(",")), mode=MODE_FULL_TIMES)
    df.attrs["data_version"] = ingest.data_version(args.data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(args.out)
    print(f"{len(df)} rows from {df['contest_id'].nunique()} contests, positive rate {df['y'].mean():.3f} -> {args.out}")


def cmd_experiment(args):
    from .experiment import run_experiment

    df = pd.read_pickle(args.dataset)
    s = run_experiment(df, args.out, df.attrs["data_version"], _command(), cv_folds=args.folds,
                       n_boot=args.boot)
    print(pd.DataFrame(s["test_metrics"])[["model", "brier", "logloss", "ece", "roc_auc"]].to_string(index=False))
    print(f"wrote {args.out / 'evaluation_report.md'}")


def cmd_segment_study(args):
    from .segments import run_segment_study

    df = pd.read_pickle(args.dataset)
    s = run_segment_study(df, args.out, df.attrs["data_version"], _command(), k=args.folds,
                          n_boot=args.boot)
    print(pd.DataFrame(s["deltas"]).to_string(index=False))
    print(f"wrote {args.out / 'segment_report.md'}")


def cmd_learning_curve(args):
    from .learning_curve import run_learning_curve

    df = pd.read_pickle(args.dataset)
    extra = pd.read_pickle(args.extra) if args.extra else None
    agg = run_learning_curve(df, args.out, df.attrs["data_version"], _command(), extra=extra,
                             repeats=args.repeats)
    print(agg.round(4).to_string(index=False))
    print(f"wrote {args.out / 'learning_curve.md'}")


def cmd_train(args):
    from .evaluation import time_split
    from .experiment import fit_crossfit_calibrated
    from .features import COUNT_ONLY_COLUMNS, FEATURE_VERSION, FULL_COLUMNS
    from .models import SEED, CatBoostModel
    from .predict import ModelBundle

    df = pd.read_pickle(args.dataset)
    split = time_split(df)
    held_out = [] if args.all_contests else split["test"]
    tr = df[~df["contest_id"].isin(held_out)]
    bundle = ModelBundle(
        model_version=args.version,
        models={
            "A": fit_crossfit_calibrated(lambda: CatBoostModel(FULL_COLUMNS), tr),
            "B": fit_crossfit_calibrated(lambda: CatBoostModel(COUNT_ONLY_COLUMNS), tr),
        },
        meta={"data_version": df.attrs["data_version"], "seed": SEED,
              "feature_version": FEATURE_VERSION, "command": _command(),
              "train_contests": sorted(tr["contest_id"].unique()),
              "calibration": "platt on contest-grouped 5-fold out-of-fold predictions",
              "held_out_test_contests": held_out},
    )
    path = bundle.save(args.out)
    print(f"saved {path}")


def cmd_export_snapshot(args):
    from .normalize import normalize_contest
    from .predict import export_historical_snapshot

    raw = ingest.load_raw(args.contest, args.data)
    c = normalize_contest(args.contest, raw.config, raw.teams, raw.runs)
    t = c.freeze_start_ms + int(args.minutes_after_freeze * 60000)
    data = export_historical_snapshot(c, min(t, c.duration_ms), args.mode)
    args.out.write_text(json.dumps(data, ensure_ascii=False))
    print(f"wrote public snapshot of {args.contest} at freeze+{args.minutes_after_freeze}min (mode {args.mode}) "
          f"-> {args.out}")


def _load_snapshot(args):
    """Model bundle and public snapshot from the input JSON; Mode C exits with code 2."""
    from .predict import ModelBundle, PredictionUnavailable, snapshot_from_json

    try:
        snap = snapshot_from_json(json.loads(args.input.read_text()))
    except PredictionUnavailable as e:
        print(json.dumps({"error": "unavailable", "detail": str(e)}, ensure_ascii=False))
        sys.exit(2)
    return ModelBundle.load(args.model), snap


def cmd_predict(args):
    from .predict import predict_snapshot

    bundle, snap = _load_snapshot(args)
    out = predict_snapshot(bundle, snap)
    if args.format == "csv":
        out.to_csv(sys.stdout, index=False)
    else:
        print(out.to_json(orient="records", force_ascii=False, indent=1))


def cmd_standings(args):
    from .predict import predict_snapshot
    from .standings import build_board, hypothetical_rank, medal_lines, scenarios, simulate, team_summary

    bundle, snap = _load_snapshot(args)
    board = build_board(snap, predict_snapshot(bundle, snap))
    sim = simulate(board, args.sims, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    summ = team_summary(board, sim)
    summ.to_csv(args.out / "standings_teams.csv", index=False)
    lines = medal_lines(board, sim)
    lines.to_csv(args.out / "medal_lines.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ.head(args.top).round(3).to_string(index=False))
    if len(lines):
        print("\nmedal lines (P >= 5%):")
        print(lines[lines["probability"] >= 0.05].round(3).to_string(index=False))
    if args.team:
        sc = scenarios(board, sim, args.team)
        sc.to_csv(args.out / f"scenarios_{args.team}.csv", index=False)
        print(f"\nscenarios for {args.team}:")
        print(sc.round(3).to_string(index=False))
        if args.hypothetical:
            solved, penalty = args.hypothetical
            print(hypothetical_rank(board, sim, args.team, solved, penalty))
    print(f"\nwrote {args.out}")


def cmd_export_userscript(args):
    from .predict import ModelBundle
    from .userscript import build_userscript

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_userscript(ModelBundle.load(args.model), args.download_url), encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


def cmd_standings_backtest(args):
    from .evaluation import time_split
    from .predict import ModelBundle
    from .standings_eval import run_standings_backtest

    df = pd.read_pickle(args.dataset)
    split = time_split(df)
    base_rate = float(df[df["contest_id"].isin(split["train"])]["y"].mean())
    loaded = {lc.audit.contest_id: lc.contest for lc in load_all(args.data, contest_ids=split["test"])}
    agg = run_standings_backtest([loaded[c] for c in split["test"]], ModelBundle.load(args.model), base_rate,
                                 args.out, df.attrs["data_version"], _command(), args.sims)
    print(agg.round(4).to_string(index=False))
    print(f"wrote {args.out / 'standings_backtest.md'}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="xcpcpredict", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def data_arg(sp):
        sp.add_argument("--data", type=Path, default=ingest.DEFAULT_DATA_DIR, help="board-data checkout")

    sp = sub.add_parser("fetch", help="shallow checkout of xcpcio/board-data at a release tag")
    data_arg(sp)
    sp.add_argument("--rev", default=ingest.BOARD_DATA_REV,
                    help="tag, branch or commit (default: the release behind the committed reports)")
    sp.set_defaults(func=cmd_fetch)

    sp = sub.add_parser("audit", help="data quality report")
    data_arg(sp)
    sp.add_argument("--out", type=Path, default=Path("reports"))
    sp.set_defaults(func=cmd_audit)

    sp = sub.add_parser("build", help="replay contests into a supervised dataset")
    data_arg(sp)
    sp.add_argument("--tiers", default="A,B")
    sp.add_argument("--include-warmup", action="store_true")
    sp.add_argument("--out", type=Path, default=DEFAULT_DATASET)
    sp.set_defaults(func=cmd_build)

    sp = sub.add_parser("experiment", help="baselines, CatBoost, calibration, ablations, report")
    sp.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    sp.add_argument("--out", type=Path, default=Path("reports"))
    sp.add_argument("--folds", type=int, default=5)
    sp.add_argument("--boot", type=int, default=1000)
    sp.set_defaults(func=cmd_experiment)

    sp = sub.add_parser("segment-study", help="contest types, training composition, contest discrimination")
    sp.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    sp.add_argument("--out", type=Path, default=Path("reports"))
    sp.add_argument("--folds", type=int, default=5)
    sp.add_argument("--boot", type=int, default=500)
    sp.set_defaults(func=cmd_segment_study)

    sp = sub.add_parser("learning-curve", help="test-set quality vs number of training contests")
    sp.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    sp.add_argument("--extra", type=Path, help="dataset of additional contests (e.g. a new data source)")
    sp.add_argument("--repeats", type=int, default=3)
    sp.add_argument("--out", type=Path, default=Path("reports"))
    sp.set_defaults(func=cmd_learning_curve)

    sp = sub.add_parser("train", help="train and save the calibrated Mode A / Mode B models")
    sp.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    sp.add_argument("--out", type=Path, default=DEFAULT_MODEL_DIR)
    sp.add_argument("--version", default="v4")
    sp.add_argument("--all-contests", action="store_true",
                    help="also train on the test contests (for deployment, after evaluation)")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("export-snapshot", help="write the public snapshot of a historical contest as input JSON")
    data_arg(sp)
    sp.add_argument("--contest", required=True)
    sp.add_argument("--minutes-after-freeze", type=float, default=30)
    sp.add_argument("--mode", choices=["A", "B"], default="A")
    sp.add_argument("--out", type=Path, required=True)
    sp.set_defaults(func=cmd_export_snapshot)

    sp = sub.add_parser("predict", help="predict p_ac for every frozen cell of a public snapshot")
    sp.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    sp.add_argument("--input", type=Path, required=True)
    sp.add_argument("--format", choices=["json", "csv"], default="json")
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser("standings", help="simulate final standings, medal lines and team scenarios")
    sp.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    sp.add_argument("--input", type=Path, required=True)
    sp.add_argument("--sims", type=int, default=10000)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--team", help="team id for pass/fail scenarios")
    sp.add_argument("--hypothetical", type=int, nargs=2, metavar=("SOLVED", "PENALTY"),
                    help="with --team: rank distribution if the team ends with this solved count and penalty")
    sp.add_argument("--top", type=int, default=20)
    sp.add_argument("--out", type=Path, default=Path("standings_out"))
    sp.set_defaults(func=cmd_standings)

    sp = sub.add_parser("export-userscript", help="Tampermonkey script that predicts in the browser on board.xcpcio.com")
    sp.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    sp.add_argument("--out", type=Path, default=Path("out/xcpcpredict.user.js"))
    sp.add_argument("--download-url", help="URL the script is published at; lets Tampermonkey check for updates")
    sp.set_defaults(func=cmd_export_userscript)

    sp = sub.add_parser("standings-backtest", help="backtest final standings on held-out test contests")
    data_arg(sp)
    sp.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    sp.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    sp.add_argument("--sims", type=int, default=5000)
    sp.add_argument("--out", type=Path, default=Path("reports"))
    sp.set_defaults(func=cmd_standings_backtest)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
