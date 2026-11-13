# Contest types and contest quality

- Data version: `9075040bc9f190e3faa01ecca859bae428b1fc5f`; seed `20261108`; reproduce with `xcpcpredict segment-study`
- 164 contests, 330114 rows; CatBoost; contest-grouped 5-fold CV.

## 1. Quality by contest type (model trained on all types, out-of-fold)

`skill` = 1 − Brier / (base_rate·(1−base_rate)), comparable across types with different base rates.

| contest_type | contests | rows | share_of_rows | base_rate | brier_M6 | brier_M7 | skill_M7 | auc_M7 | mean_p_M7 | contest_mean_brier_M7 |
|---|---|---|---|---|---|---|---|---|---|---|
| regional | 56 | 102771 | 0.3113 | 0.3206 | 0.1568 | 0.1568 | 0.2802 | 0.8163 | 0.3159 | 0.1573 |
| online | 10 | 87906 | 0.2663 | 0.2733 | 0.1455 | 0.1457 | 0.2662 | 0.8164 | 0.2821 | 0.1439 |
| provincial | 64 | 86256 | 0.2613 | 0.2402 | 0.1344 | 0.1341 | 0.2652 | 0.8211 | 0.2419 | 0.1337 |
| invitational | 18 | 32335 | 0.0980 | 0.2725 | 0.1510 | 0.1501 | 0.2427 | 0.7998 | 0.2744 | 0.1500 |
| final | 11 | 16880 | 0.0511 | 0.3287 | 0.1646 | 0.1624 | 0.2642 | 0.8042 | 0.3115 | 0.1660 |
| special | 5 | 3966 | 0.0120 | 0.2705 | 0.1460 | 0.1455 | 0.2629 | 0.8194 | 0.2614 | 0.1522 |

## 2. Which training contests to use (evaluated on held-out contests of one type)

All variants of one evaluated type are scored on the same held-out rows. M7 adds contest type one-hots and contest structure (problem / contest discrimination, solved-count spread, zero-solve share, problem difficulty spread, unsolved problems).

| eval_type | variant | eval_contests | train_contests_same_type | brier | logloss | ece | roc_auc | contest_mean_brier |
|---|---|---|---|---|---|---|---|---|
| regional | all types, M6 (current) | 56 | 45 | 0.1565 | 0.4741 | 0.0124 | 0.8174 | 0.1570 |
| regional | same type only, M6 | 56 | 45 | 0.1590 | 0.4811 | 0.0080 | 0.8109 | 0.1596 |
| regional | all types, M7 (+type, structure) | 56 | 45 | 0.1565 | 0.4742 | 0.0051 | 0.8170 | 0.1570 |
| regional | all types, M7, equal contest weight | 56 | 45 | 0.1565 | 0.4743 | 0.0082 | 0.8173 | 0.1571 |
| regional | same type only, M7 | 56 | 45 | 0.1592 | 0.4814 | 0.0061 | 0.8107 | 0.1598 |
| invitational | all types, M6 (current) | 18 | 15 | 0.1497 | 0.4598 | 0.0119 | 0.8014 | 0.1496 |
| invitational | same type only, M6 | 18 | 15 | 0.1565 | 0.4816 | 0.0242 | 0.7808 | 0.1561 |
| invitational | all types, M7 (+type, structure) | 18 | 15 | 0.1492 | 0.4584 | 0.0038 | 0.8028 | 0.1490 |
| invitational | all types, M7, equal contest weight | 18 | 15 | 0.1500 | 0.4602 | 0.0066 | 0.8008 | 0.1499 |
| invitational | same type only, M7 | 18 | 15 | 0.1557 | 0.4792 | 0.0232 | 0.7832 | 0.1553 |
| provincial | all types, M6 (current) | 64 | 52 | 0.1341 | 0.4173 | 0.0134 | 0.8217 | 0.1338 |
| provincial | same type only, M6 | 64 | 52 | 0.1377 | 0.4285 | 0.0117 | 0.8104 | 0.1373 |
| provincial | all types, M7 (+type, structure) | 64 | 52 | 0.1338 | 0.4166 | 0.0052 | 0.8219 | 0.1337 |
| provincial | all types, M7, equal contest weight | 64 | 52 | 0.1346 | 0.4189 | 0.0035 | 0.8194 | 0.1345 |
| provincial | same type only, M7 | 64 | 52 | 0.1377 | 0.4288 | 0.0096 | 0.8101 | 0.1375 |

Brier change vs the current setup (negative = better), contest-cluster bootstrap 95% CI:

| eval_type | candidate | brier_delta_vs_current | ci_lo | ci_hi |
|---|---|---|---|---|
| regional | same type only, M6 | 0.0025 | 0.0018 | 0.0034 |
| regional | all types, M7 (+type, structure) | 0.0000 | -0.0003 | 0.0004 |
| regional | all types, M7, equal contest weight | 0.0000 | -0.0003 | 0.0005 |
| regional | same type only, M7 | 0.0027 | 0.0021 | 0.0035 |
| invitational | same type only, M6 | 0.0068 | 0.0008 | 0.0116 |
| invitational | all types, M7 (+type, structure) | -0.0005 | -0.0013 | 0.0003 |
| invitational | all types, M7, equal contest weight | 0.0002 | -0.0007 | 0.0013 |
| invitational | same type only, M7 | 0.0060 | 0.0006 | 0.0100 |
| provincial | same type only, M6 | 0.0036 | 0.0025 | 0.0049 |
| provincial | all types, M7 (+type, structure) | -0.0003 | -0.0008 | 0.0002 |
| provincial | all types, M7, equal contest weight | 0.0005 | -0.0001 | 0.0010 |
| provincial | same type only, M7 | 0.0036 | 0.0026 | 0.0046 |

## 3. Contest discrimination vs predictability

Contest discrimination = mean over problems of corr(solved the problem, solved count on the other problems) on the frozen board. Contests split into tertiles.

| disc_tertile | contests | discrimination | base_rate | brier | skill | auc | abs_calib_gap |
|---|---|---|---|---|---|---|---|
| low | 55 | 0.3261 | 0.2880 | 0.1512 | 0.2459 | 0.8055 | 0.0214 |
| mid | 54 | 0.3900 | 0.2985 | 0.1506 | 0.2601 | 0.8134 | 0.0228 |
| high | 55 | 0.4665 | 0.2588 | 0.1390 | 0.2561 | 0.8174 | 0.0221 |

Spearman correlations across contests:

|  | discrimination | solved_cv | base_rate | auc | skill | brier |
|---|---|---|---|---|---|---|
| discrimination | 1.0000 | 0.5400 | -0.2270 | 0.1710 | 0.0710 | -0.2490 |
| solved_cv | 0.5400 | 1.0000 | -0.1770 | 0.1800 | 0.0890 | -0.2230 |
| base_rate | -0.2270 | -0.1770 | 1.0000 | -0.2010 | 0.2410 | 0.8700 |
| auc | 0.1710 | 0.1800 | -0.2010 | 1.0000 | 0.8140 | -0.5800 |
| skill | 0.0710 | 0.0890 | 0.2410 | 0.8140 | 1.0000 | -0.1890 |
| brier | -0.2490 | -0.2230 | 0.8700 | -0.5800 | -0.1890 | 1.0000 |
