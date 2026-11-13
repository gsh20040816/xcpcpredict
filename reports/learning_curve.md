# Learning curve

- Data version: `9075040bc9f190e3faa01ecca859bae428b1fc5f`; seed `20261108`; reproduce with `xcpcpredict learning-curve`

Fixed test set: 33 most recent contests (60581 rows). Training contests sampled from the 131 earlier ones (3 random draws per fraction). CatBoost, full features.

| variant | fraction | train_contests | brier | brier_sd | logloss | roc_auc |
|---|---|---|---|---|---|---|
| formal contests | 0.1250 | 16.0000 | 0.1580 | 0.0024 | 0.4816 | 0.7916 |
| formal contests | 0.2500 | 33.0000 | 0.1536 | 0.0005 | 0.4689 | 0.8024 |
| formal contests | 0.5000 | 66.0000 | 0.1519 | 0.0001 | 0.4647 | 0.8065 |
| formal contests | 0.7500 | 98.0000 | 0.1512 | 0.0002 | 0.4625 | 0.8083 |
| formal contests | 1.0000 | 131.0000 | 0.1503 | nan | 0.4602 | 0.8104 |