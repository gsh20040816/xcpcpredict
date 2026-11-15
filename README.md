# xcpcpredict：XCPC 封榜 AC 概率预测

ICPC、CCPC 和各省赛通常在最后一小时封榜。封榜以后，场外观众能看到每支队伍在哪道题上又交了几次，有的榜单还显示提交时间，但看不到评测结果。本项目只用这些公开信息，对每个**冻结格子**给出校准过的概率：

> `p_ac`：截至观察时刻，这支队伍在这道题上已经交的隐藏提交里，**至少有一次 AC** 的概率。

冻结格子指同时满足两个条件的（队伍，题目）：封榜前这道题还没有 AC，截至观察时刻至少交过一次隐藏提交。在 `p_ac` 的基础上，项目用蒙特卡洛模拟最终榜单，给出每支队伍的排名分布、奖牌概率、奖牌线分布，以及某支队伍各题过或不过时的排名。

## 主要结果

数据来自 XCPCIO 公开发布的 [xcpcio/board-data](https://github.com/xcpcio/board-data)，release `v2026.11.11.2057`（commit `9075040`）。共审计 472 场比赛，其中 164 场正式赛的提交时间可信，用于训练和评估。按开赛时间取最近的 33 场作为留出测试集，共 60,581 个样本，其中 28.5% 最终有 AC：

| 模型 | Brier ↓ | Log loss ↓ | ECE ↓ | AUC ↑ |
|---|---|---|---|---|
| 基础比例（所有格子给同一个概率） | 0.2040 | 0.598 | 0.007 | 0.500 |
| 经验分组概率 M0 | 0.1672 | 0.506 | 0.012 | 0.761 |
| Logistic 回归 + Platt 校准 | 0.1541 | 0.471 | 0.005 | 0.801 |
| CatBoost + Platt 校准 | **0.1513** | **0.463** | **0.004** | **0.808** |
| CatBoost，只有隐藏提交次数（Mode B） | 0.1685 | 0.507 | 0.008 | 0.758 |

- 这几个模型用 106 场比赛训练，在之后的 25 场上校准。默认模型 v4 用测试集之前的全部 131 场训练；用 131 场训练时，测试集 Brier 是 0.1503，AUC 是 0.810（见学习曲线报告）。
- 从 M0 到 Logistic，再到 CatBoost，每一步 Brier 的下降，95% 置信区间都不含 0（按比赛重抽样的 bootstrap）。33 场测试赛里，CatBoost 每一场都比 M0 好。
- 在测试集上按预测概率分成 10 档，每一档的实际 AC 比例和平均预测概率最多相差 0.014。
- 最有用的信息是隐藏提交的时间。在 164 场比赛的 5 折交叉验证里，加入提交时间、提交间隔和沉默时长后，Brier 从 0.1653 降到 0.1487，比其余各组特征的提升加起来还多。

最终排名预测在同样 33 场测试赛上的回测结果如下，观察时刻是比赛结束：

| 方法 | 排名平均误差 | 实际排名落入 90% 区间的比例 | 奖牌 Brier | 最可能的奖牌猜中率 | 实际奖牌线题数的预测概率 | 奖牌线罚时误差（分钟） |
|---|---|---|---|---|---|---|
| **模型（v4）** | **10.5** | 0.86 | **0.143** | **0.898** | **0.91** | **32** |
| 把封榜榜单当最终榜 | 16.0 | 0.06 | 0.322 | 0.839 | 0.48 | 69 |
| 所有冻结格子用同一个基础概率 | 15.4 | 0.88 | 0.200 | 0.849 | 0.82 | 62 |

各项实验的中文解读见 [`docs/results.md`](docs/results.md)，自动生成的英文报告原文在 [`reports/`](reports/) 目录。

## 安装

需要 Python 3.11 或更高版本。

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
source .venv/bin/activate
```

要逐位复现 `reports/`，按下文“端到端复现”安装锁定的依赖版本。

## 快速开始

仓库里不带数据和模型，第一次使用要先下载数据并训练：

```bash
# 1. 浅检出 board-data 的 release v2026.11.11.2057 到 data/board-data（约 440 MB）
xcpcpredict fetch

# 2. 审计数据质量，回放比赛，生成训练集
xcpcpredict audit        # reports/data_quality_report.md
xcpcpredict build        # data/dataset.pkl

# 3. 训练默认模型
xcpcpredict train        # models/v4/{model.pkl,meta.json}
```

下面用第 51 届 ICPC 西安站演示。这场在测试集里，v4 训练时没见过。先把封榜 30 分钟时的公开榜单导出成输入文件，再做预测：

```bash
mkdir -p out
xcpcpredict export-snapshot --contest icpc/51st/xian --minutes-after-freeze 30 --out out/snap.json
xcpcpredict predict --input out/snap.json --format csv
```

```text
contest_id,team_id,problem_id,observation_time,hidden_submissions,p_ac,model_version,feature_version,quality_flags
icpc/51st/xian,Team003,7,16200.0,3,0.2304,v4,f3,
icpc/51st/xian,Team004,7,16200.0,1,0.2886,v4,f3,
...
```

比赛结束时可以模拟最终榜单：

```bash
xcpcpredict export-snapshot --contest icpc/51st/xian --minutes-after-freeze 60 --out out/end.json
xcpcpredict standings --input out/end.json --team Team139 --hypothetical 7 900 --out out/standings
```

输出包括：

- 每支队伍的排名分布和奖牌概率
- 各档奖牌线的题数和罚时分布
- Team139 每种过题组合下的排名
- `--hypothetical 7 900` 给出 Team139 以 7 题、900 分钟罚时收尾时的排名

实际比赛中，按 [`docs/usage.md`](docs/usage.md) 里的格式把公开榜单整理成 JSON，交给 `predict` 或 `standings`。

## 浏览器用户脚本

装好 [Tampermonkey](https://www.tampermonkey.net/) 后打开 <https://github.com/gsh20040816/xcpcpredict/releases/latest/download/xcpcpredict.user.js> 安装脚本。之后在 [XCPCIO 榜单](https://board.xcpcio.com) 上看比赛时，冻结格子会显示 AC 概率，队名下方显示奖牌概率，右下角面板给出各档奖牌线。模型嵌在脚本里，预测和排名模拟都在浏览器里完成。详见 [`docs/usage.md`](docs/usage.md#浏览器用户脚本)。

## 端到端复现

下面的命令重新生成 `reports/` 里的全部报告和默认模型：

```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt && pip install --no-deps -e .
xcpcpredict fetch                # board-data release v2026.11.11.2057
xcpcpredict audit
xcpcpredict build
xcpcpredict experiment
xcpcpredict segment-study
xcpcpredict learning-curve
xcpcpredict train                # models/v4
xcpcpredict standings-backtest
git diff --stat reports/
```

`reports/` 是在 Linux（x86_64）上按这组命令生成的，在 16 核台式机上全程约 20 分钟。在 macOS（arm64）上，M0、CatBoost 和排名模拟的结果逐位相同。Logistic 回归依赖底层数值库，个别指标会在小数点后第 4 位上不同。

## 只使用公开信息

模型输入只包含场外观众在公开榜单上能看到的内容：

- 封榜前的提交和公开结果（AC 或非 AC）。拒绝类型（WA、TLE、RE 等）和编程语言不作为特征。
- 封榜前的编译错误不计入，公开榜单不显示它们。封榜后的每次提交，包括编译错误，都显示为待判。
- 封榜后的提交只用时间（Mode A）或次数（Mode B）。
- 榜单封榜后停止更新（Mode C）时看不到新提交，不做预测。

## 目录结构

```text
src/xcpcpredict/
  ingest.py          按指定版本下载 board-data，读取比赛
  normalize.py       统一的比赛、队伍、提交模型；推断时间单位；映射评测状态
  audit.py           数据质量分档（A/B/C/X）、合成数据识别、重复检测、重交行为统计
  dataset.py         读取并审计全部比赛，回放生成训练集
  replay.py          生成任意观察时刻的公开快照（PublicSnapshot）
  labels.py          标签：截至观察时刻的隐藏提交里是否有 AC
  features.py        从公开快照提取特征（M1–M7 分组，用于消融）
  models.py          经验分组概率 M0、Logistic 回归、CatBoost
  calibration.py     Platt / 保序回归校准
  evaluation.py      按比赛划分数据集、评估指标、bootstrap
  experiment.py      完整离线实验和评估报告
  segments.py        比赛类型、训练集构成、比赛区分度研究
  learning_curve.py  训练比赛数量与效果的学习曲线
  predict.py         JSON 快照转成每格 p_ac；模型文件读写
  standings.py       最终排名蒙特卡洛模拟
  standings_eval.py  最终排名回测
  userscript.py      生成浏览器用户脚本，把模型导出成 JSON 嵌进去
  web/core.js        预测流程的 JavaScript 移植：解析 XCPCIO 数据、特征、模型推断、排名模拟
  web/board.js       用户脚本的页面部分：取数据、标注榜单、面板
  cli.py             命令行入口
tests/               单元测试（除一条读取 board-data 的测试外，都用合成数据）
reports/             实验报告（由命令生成，提交进仓库）
```

## 文档

- [`docs/usage.md`](docs/usage.md)：全部命令、输入输出格式、Mode A/B/C、排名模拟的输出、浏览器用户脚本
- [`docs/results.md`](docs/results.md)：数据质量、模型评估、消融、比赛类型、学习曲线和排名回测的解读
- [`docs/maintenance.md`](docs/maintenance.md)：更新数据、修改特征、训练和发布模型、版本号
- [`docs/SPEC.md`](docs/SPEC.md)：实现前的需求与设计记录

## 已知限制

- 排名模拟假设各冻结格子相互独立，并且观察时刻之后不再有新提交。所以只有在比赛结束时观察，结果才是最终榜单；比赛中途得到的是“之后不再交题时的榜单”。
- board-data 没有说明提交记录来自评测日志还是由榜单反推。疑似反推的比赛（C 档）已经排除，但无法确认剩下的全是真实日志。

## 数据来源与致谢

本项目的全部比赛数据来自 [XCPCIO](https://xcpcio.com)：

- **[xcpcio/board-data](https://github.com/xcpcio/board-data)**：XCPCIO 整理并公开发布的历届 ICPC、CCPC 和各省赛的榜单与提交记录，MIT 许可，Copyright © 2020–present XCPCIO。本项目使用 release `v2026.11.11.2057`。感谢 [XCPCIO 项目](https://github.com/xcpcio)的维护者和贡献者长期收集、整理和公开这些数据。
- 这些记录最初来自各场比赛的主办方和评测系统，感谢他们公开比赛数据。

仓库不包含原始数据，`xcpcpredict fetch` 用 git 下载 board-data 的版本化发布。

模型和实验依赖 [NumPy](https://numpy.org)、[pandas](https://pandas.pydata.org)、[scikit-learn](https://scikit-learn.org)、[CatBoost](https://catboost.ai) 和 [matplotlib](https://matplotlib.org) 等开源项目。

## 许可证

代码以 [MIT 许可证](LICENSE) 发布。比赛数据不在本仓库中，使用时遵守 board-data 的 MIT 许可。
