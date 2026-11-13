# Final-standings backtest

- Data version: `9075040bc9f190e3faa01ecca859bae428b1fc5f`; model `v4`; reproduce with `xcpcpredict standings-backtest`

33 held-out contests observed at contest end; 5000 simulations each. The model was trained without these contests. Metrics are averaged over contests.

- `rank_mae`: |expected rank − actual final rank|, all teams
- `rank_cov90`: share of teams whose actual rank is inside the predicted 5–95% interval (ideal 0.90)
- `medal_brier`: 4-class Brier (gold/silver/bronze/none) over medal-eligible teams (lower is better)
- `medal_modal_acc`: share of eligible teams whose most likely medal is the actual one
- `line_solved_prob`: predicted probability of the actual solved count of each medal line
- `line_pen_abs_err`: |predicted median penalty − actual| of the line team, when the solved count matches

| variant | rank_mae | rank_cov90 | solved_mae | medal_brier | medal_modal_acc | line_solved_prob | line_pen_abs_err |
|---|---|---|---|---|---|---|---|
| model | 10.4711 | 0.8598 | 0.2960 | 0.1428 | 0.8980 | 0.9135 | 31.8025 |
| freeze board = final (all frozen fail) | 16.0235 | 0.0557 | 0.3897 | 0.3219 | 0.8390 | 0.4753 | 68.6069 |
| constant p = base rate | 15.3956 | 0.8799 | 0.4375 | 0.2004 | 0.8490 | 0.8247 | 61.9605 |

- model has lower medal Brier than `constant p = base rate` in 25 of 27 contests
- model has lower medal Brier than `freeze board = final (all frozen fail)` in 27 of 27 contests

Per contest: `standings_backtest.csv`.
