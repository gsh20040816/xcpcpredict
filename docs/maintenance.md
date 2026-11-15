# 维护指南

## 开发环境与测试

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q
```

测试几秒钟就能跑完。`test_real_contests_normalize` 读取 `data/board-data` 里的两场真实比赛，本地没有这份数据时跳过；其余测试用合成的小比赛。`tests/test_userscript.py` 用 node 运行 `web/core.js`，和 Python 逐列比对特征、`p_ac` 和排名模拟；没装 node 时跳过。GitHub Actions 的 ubuntu 环境自带 node。

每次推送到 `main` 或提交 PR 时，GitHub Actions（`.github/workflows/tests.yml`）会在 Python 3.11 和 3.14 上运行测试。

`pyproject.toml` 里的依赖没有固定版本，复现报告用的精确版本在 `requirements-lock.txt`。

## 仓库里放什么

| 路径 | 是否提交 | 说明 |
|---|---|---|
| `src/`、`tests/`、`docs/` | 提交 | 代码和文档 |
| `requirements-lock.txt` | 提交 | 复现报告时使用的精确依赖版本 |
| `reports/` | 提交 | 命令生成的实验报告 |
| `data/` | 不提交 | board-data 检出约 440 MB，`dataset.pkl` 约 270 MB，都能用命令重新生成 |
| `models/` | 不提交 | 用 `train` 重新生成；需要分发时放到 GitHub Release，见下文 |
| `out/`、`standings_out/`、`reports_*/` | 不提交 | 临时输出。单独跑的实验先写到 `reports_<名字>/`，确认要保留再放进 `reports/` |

## 数据来源

- 比赛数据只来自 [xcpcio/board-data](https://github.com/xcpcio/board-data) 带 tag 的 release，还没收录的比赛等上游发布。board.xcpcio.com 的 robots.txt 禁止自动访问。
- 新增数据来源时，先确认许可和 robots.txt，并在 README 的“数据来源与致谢”里写明来源和许可。
- 原始数据不进仓库，报告里不放队名和个人姓名。

## 更新数据

当前报告和默认模型使用的数据版本是 `ingest.BOARD_DATA_REV`，即 board-data 的 release `v2026.11.11.2057`。`xcpcpredict fetch` 默认检出这个版本。

换用更新的数据：

1. 在 [board-data 的 tags](https://github.com/xcpcio/board-data/tags) 里找到新的 release，运行 `xcpcpredict fetch --rev <tag>`。
2. 运行 `xcpcpredict audit`，看新比赛分在哪一档。档位不对时，在 `audit.csv` 的 `reasons` 列查原因。
3. 运行 `xcpcpredict build` 重建数据集。board-data 的 commit 会作为数据版本记进数据集，之后的报告和模型元数据都会带上它。
4. 决定用新数据作为正式结果时，把 `ingest.BOARD_DATA_REV` 改成新 tag，按 README 的“端到端复现”重跑全部实验，训练新版本的模型，把代码、报告和文档一起提交。

## 重跑实验

完整命令见 README 的“端到端复现”。各报告和命令的对应关系见 [`results.md`](results.md)。在 16 核台式机上，各步骤耗时如下：

| 步骤 | 耗时 |
|---|---|
| `fetch` | 13 秒 |
| `audit` | 12 秒 |
| `build` | 48 秒 |
| `experiment` | 10 分钟 |
| `segment-study` | 6 分钟 |
| `learning-curve` | 38 秒 |
| `train` | 50 秒 |
| `standings-backtest` | 23 秒 |

升级依赖以后，先重跑全部实验，再更新锁文件：

```bash
{ head -2 requirements-lock.txt; pip freeze | grep -v -E '^(-e |# Editable)'; } > lock.tmp && mv lock.tmp requirements-lock.txt
```

锁文件开头两行注释写着验证过的 Python 版本和平台，更新时一起改。

## 修改特征

特征都在 `features.py` 的 `extract_features(snap)` 里计算。它只拿得到 `PublicSnapshot`，里面没有拒绝类型、语言、封榜前的编译错误、隐藏提交的结果和观察时刻之后的提交。新特征只从它计算。

1. 把列名加进 `GROUPS` 中合适的级别。M6 的特征加进 `M6_FAMILIES` 的某个特征族，比赛级的结构特征加进 `M7_COLUMNS`。这样它才会进入 `FULL_COLUMNS` 和消融实验。
2. 需要隐藏提交时间的特征加进 `TIME_COLUMNS`。Mode B 的模型不使用这些列，它们在 Mode B 下是 NaN，`test_mode_b_time_columns_are_nan` 会检查。
3. 改 `FEATURE_VERSION`，比如从 `f3` 改成 `f4`。
4. 在 `src/xcpcpredict/web/core.js` 的 `extractFeatures` 里做同样的修改，浏览器用户脚本靠它计算特征。
5. 跑 `pytest`。`tests/test_replay_labels.py` 检查特征不受隐藏结果和未来提交的影响，`tests/test_userscript.py` 检查 JavaScript 和 Python 的特征逐列一致。
6. 重建数据集，跑 `experiment`。消融表和 bootstrap 会显示新特征有没有显著贡献。

`predict` 不检查模型训练时的特征版本，输出的 `feature_version` 取的是代码里的值。改了特征以后，要重新训练模型再用。

## 训练和发布模型

```bash
xcpcpredict train --version v5 --out models/v5
```

`models/v5/meta.json` 会记录以下信息：

- `model_version`、`feature_version`
- `data_version`、`seed`、生成命令
- 训练用的比赛列表 `train_contests`
- 校准方式
- 留出的测试比赛列表 `held_out_test_contests`

换默认模型时，要同时修改 `cli.py` 里的 `DEFAULT_MODEL_DIR` 和 `train` 子命令的 `--version` 默认值。

`models/` 不进 git。需要分享模型时，上传到 GitHub Release，tag 用 `model-v5` 这样的名字，和代码版本区分开。发布用的模型加 `--all-contests`，用全部比赛训练，版本号加 `-all` 后缀；对外报的指标沿用不加该参数的同版本模型。每个模型 release 同时附上用这个模型生成的用户脚本：

```bash
xcpcpredict train --all-contests --version v5-all --out models/v5-all
xcpcpredict export-userscript --model models/v5-all --out out/xcpcpredict.user.js \
  --download-url https://github.com/gsh20040816/xcpcpredict/releases/latest/download/xcpcpredict.user.js
gh release create model-v5-all models/v5-all/model.pkl models/v5-all/meta.json out/xcpcpredict.user.js \
  --title "模型 v5-all" --notes "数据版本、训练比赛数、对应的测试集指标……"
```

脚本的 `@updateURL` 指向最新 release 里的 `xcpcpredict.user.js`，所以最新的 release 必须带这个文件。脚本的 `@version` 取生成时间，新生成的脚本总比旧的新，Tampermonkey 会提示更新。

`model.pkl` 是 pickle 文件，加载时会执行其中的代码，只加载自己训练或可信来源的模型。

## 版本号约定

| 版本号 | 位置 | 什么时候改 |
|---|---|---|
| 特征版本 `fN` | `features.FEATURE_VERSION` | 特征的列或含义发生变化 |
| 模型版本 `vN` | `train --version`，记进 `meta.json` | 每次训练一个准备使用或发布的模型 |
| 包版本 | `pyproject.toml` 的 `version` | 输入输出格式或命令行接口发生变化 |

## 提交约定

- 提交说明用 `类型: 说明` 的格式。类型包括 `feat`、`fix`、`docs`、`test`、`ci`、`chore`、`report`。
- 改了影响结果的代码以后，重跑相关命令，把新报告和代码一起提交。每份报告开头都记录了数据版本和生成命令。
