# Data quality report

Source: xcpcio/board-data commit `9075040bc9f190e3faa01ecca859bae428b1fc5f`, 472 contests; reproduce with `xcpcpredict audit`.

## Tiers

- **A**: second/ms-resolution timestamps, no synthesis signal
- **B**: minute-resolution timestamps
- **C**: suspected synthesized from a scoreboard (rejected runs at t=0, identical timestamps per cell, or no rejected hidden runs at all) — labels unusable
- **X**: unusable (normalization error, unresolved hidden verdicts, duplicate log, <5 frozen cells)

| tier | is_warmup | contests | submissions | frozen_cells | mean_frozen_ac_rate |
|---|---|---|---|---|---|
| A | False | 121 | 672986 | 64860 | 0.278 |
| A | True | 79 | 257634 | 15753 | 0.447 |
| B | False | 43 | 283330 | 26315 | 0.285 |
| B | True | 13 | 68896 | 2271 | 0.340 |
| C | False | 140 | 696609 | 18353 | 0.991 |
| C | True | 22 | 89928 | 1887 | 1.000 |
| X | False | 30 | 1063879 | 2308 | 0.800 |
| X | True | 24 | 61554 | 3612 | 0.089 |

The frozen-cell AC rate of tier C is ≈1.0: synthesized logs put every rejected run before the freeze, so their hidden cells are all AC. Training on them would teach the model that hidden submissions always pass, so tier C is excluded by default.

## Reasons for downgrading

| reason | contests |
|---|---|
| suspected_synthesized | 172 |
| unresolved_hidden_verdicts | 23 |
| duplicate | 11 |
| too_few_frozen_cells | 9 |
| normalization_error:max timestamp 19560 with unit second(default) exceeds duration 18000000 ms | 3 |
| no_submissions | 3 |
| normalization_error:max timestamp 8385000 with unit millisecond exceeds duration 7200000 ms | 1 |
| normalization_error:max timestamp 55500 with unit second(default) exceeds duration 18000000 ms | 1 |
| many_submissions_out_of_range | 1 |
| normalization_error:max timestamp 99960 with unit second(default) exceeds duration 18000000 ms | 1 |
| normalization_error:max timestamp 7140 with unit second(default) exceeds duration 5400000 ms | 1 |
| no_hidden_rejections | 1 |

## Format variation observed

| timestamp_resolution | contests |
|---|---|
|  | 10 |
| millisecond | 126 |
| minute | 122 |
| second | 214 |

- Contests with no freeze (`frozen_time=0`, freeze simulated as the last 20%): 123
- Warmup / practice contests: 138 (excluded from the main dataset; their frozen-cell AC rate is much higher)
- Online contests: 55
- Duplicate logs published under two ids: 11
- Contests referencing teams missing from team.json: 17
- Every contest is flagged `provenance_unknown`: board-data does not state whether runs come from a judge log or were rebuilt from a scoreboard.

## Usable formal contests (tier A/B, non-warmup)

164 contests, 956316 submissions, 91175 frozen cells at contest end; median frozen cells per contest 436; frozen-cell AC rate mean 0.280 (range 0.102–0.468).

Share of solved cells with another submission after the first AC: median 0.017, max 0.206.

## Resubmission behaviour (tier A formal contests)

P(the team submits the same problem again within Δ minutes | verdict of the run). This is the mechanism behind H4/H5: a follow-up hidden submission is strong evidence the earlier one failed.

| window_min | AC | REJECTED |
|---|---|---|
| 5 | 0.016 | 0.356 |
| 10 | 0.016 | 0.513 |
| 30 | 0.015 | 0.760 |
| 60 | 0.014 | 0.873 |

## Limitations

- No official frozen scoreboards are stored in board-data, so reconstructed solved/penalty/rank cannot be compared against an independent official source; XCPCIO itself renders boards from the same run.json, so agreement with the live board is by construction.
- Judging completion times are not recorded; a run judged after the next submission is indistinguishable from one judged instantly.
- Whether a real public board showed hidden submission times (Mode A) or only pending counts (Mode B) is not recorded per contest; both modes are modelled.
- Compile errors during the freeze are treated as visible pending submissions (non-AC). Boards that hide CE would show fewer hidden submissions.

## Per-contest table

See `audit.csv` for every contest's metrics and reasons.
