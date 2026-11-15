# 使用指南

所有命令都通过 `xcpcpredict` 执行。全局参数 `-v` 打印进度日志，要写在子命令前面，例如 `xcpcpredict -v build`。路径都相对当前目录，请在仓库根目录下运行。

## 命令一览

| 命令 | 作用 | 常用参数 | 输出 |
|---|---|---|---|
| `fetch` | 浅检出 xcpcio/board-data 的指定版本 | `--rev`（默认是报告所用的 release `v2026.11.11.2057`）、`--data` | `data/board-data/` |
| `audit` | 审计数据质量 | `--out` | `data_quality_report.md`、`audit.csv`、`resubmission_stats.csv` |
| `build` | 回放比赛，生成监督学习数据集 | `--tiers A,B`、`--include-warmup`、`--out` | `data/dataset.pkl` |
| `experiment` | 跑基线、CatBoost、校准和消融，生成评估报告 | `--folds 5`、`--boot 1000` | `evaluation_report.md` 及配套 csv/png |
| `segment-study` | 研究比赛类型、训练集构成和比赛区分度 | `--folds 5`、`--boot 500` | `segment_report.md` 及配套 csv/json |
| `learning-curve` | 研究训练比赛数量对测试效果的影响。`--extra` 传入另一个数据集时，会比较加入它之后的效果（只用开赛早于第一场测试赛的比赛） | `--extra`、`--repeats 3` | `learning_curve.md`、`learning_curve.csv` |
| `train` | 训练并保存 Mode A / Mode B 两个模型 | `--version v4`、`--out models/v4`、`--all-contests` | `model.pkl`、`meta.json` |
| `export-snapshot` | 把历史比赛某一时刻的公开榜单导出成预测输入 | `--contest`、`--minutes-after-freeze 30`、`--mode A` | JSON 文件 |
| `predict` | 预测每个冻结格子的 `p_ac` | `--model`、`--input`、`--format json/csv` | 标准输出 |
| `standings` | 模拟最终排名、奖牌线和单队情景 | `--sims 10000`、`--team`、`--hypothetical`、`--out` | CSV 文件和标准输出 |
| `standings-backtest` | 在测试赛上回测最终排名预测 | `--model`、`--sims 5000` | `standings_backtest.md`、`standings_backtest.csv` |
| `export-userscript` | 生成在 board.xcpcio.com 上运行的 Tampermonkey 脚本，模型嵌在脚本里 | `--model`、`--out`、`--download-url` | `out/xcpcpredict.user.js` |

各命令的默认路径如下：

- 数据：`--data data/board-data`
- 数据集：`--dataset data/dataset.pkl`
- 模型：`--model models/v4`
- 报告：`--out reports`

`fetch --rev` 可以是 tag、分支或完整的 commit SHA，例如 `--rev main` 取最新数据。

`train` 默认不用测试集训练。测试集是按开赛时间最近的 20% 比赛，当前是 33 场。要用于新比赛时，可以加 `--all-contests` 把测试赛也纳入训练；这样的模型没有留出测试集，对外报的指标沿用不加该参数时的结果。

## 预测输入

`predict` 和 `standings` 读取同一种 JSON。所有时间都是**从比赛开始算起的秒数**。

```json
{
  "contest_id": "demo",
  "contest_type": "regional",
  "duration_s": 18000,
  "freeze_start_s": 14400,
  "observation_s": 16200,
  "penalty_s": 1200,
  "penalty_rule": "in_minutes",
  "mode": "A",
  "problems": ["A", "B", "C"],
  "teams": ["t1", {"id": "t2", "groups": ["official"]}, {"id": "t3", "groups": ["unofficial"]}],
  "medal": {"official": {"gold": 32, "silver": 64, "bronze": 96}},
  "submissions": [
    {"team": "t1", "problem": "A", "time_s": 600, "verdict": "WRONG_ANSWER"},
    {"team": "t1", "problem": "A", "time_s": 900, "verdict": "ACCEPTED"}
  ],
  "hidden": [
    {"team": "t1", "problem": "B", "time_s": 14700},
    {"team": "t2", "problem": "C", "time_s": 15100}
  ]
}
```

| 字段 | 是否必填 | 说明 |
|---|---|---|
| `contest_id` | 否 | 原样写进输出，默认是 `unknown` |
| `contest_type` | 否 | `regional`、`invitational`、`provincial`、`online`、`final`、`special` 之一。不填时比赛类型特征按缺失处理 |
| `duration_s` | 是 | 比赛总时长 |
| `freeze_start_s` | 是 | 封榜开始的时刻 |
| `observation_s` | 是 | 观察时刻，必须在 `freeze_start_s` 和 `duration_s` 之间 |
| `penalty_s` | 否 | 每次错误提交的罚时，默认 1200 秒 |
| `penalty_rule` | 否 | `in_minutes`（默认，AC 时间按分钟取整）或 `accumulate_in_seconds_and_finally_to_the_minute` |
| `mode` | 否 | `A`（默认）、`B` 或 `C`，见下一节 |
| `problems` | 是 | 题号列表 |
| `teams` | 否 | 队伍列表，每项可以是字符串，也可以是 `{"id", "groups"}`。提交里出现了但没列出的队伍会自动补上 |
| `medal` | 否 | 奖牌规则，见下文。不填时不计算奖牌概率 |
| `submissions` | 否 | 封榜前的提交和公开结果。封榜后的提交放在这里会报错 |
| `hidden` | Mode A | 封榜后到观察时刻为止的每次提交，只有队伍、题号和时间。带 `verdict` 或 `status` 字段时报错 |
| `hidden_counts` | Mode B | 每个格子的隐藏提交次数：`{"team", "problem", "count"}`。带 `verdict` 或 `status` 字段时报错 |

`verdict` 支持 XCPCIO 的状态名，也支持 `correct` / `incorrect`，不区分大小写：

- 算 AC：`ACCEPTED`、`CORRECT`、`AC`
- 算错误提交，计罚时：`WRONG_ANSWER`、`INCORRECT`、`TIME_LIMIT_EXCEEDED`、`RUNTIME_ERROR`、`MEMORY_LIMIT_EXCEEDED`、`OUTPUT_LIMIT_EXCEEDED`、`PRESENTATION_ERROR`、`REJECTED` 等
- 忽略，公开榜单上不显示：`COMPILATION_ERROR`、`SYSTEM_ERROR`、`CANCELED` 等
- 不认识的状态名报错

### Mode A / B / C

| 模式 | 适用的公开榜单 | 输入 | 使用的模型 |
|---|---|---|---|
| A | 能看到每次隐藏提交的时间 | `hidden` | 完整模型 |
| B | 只能看到每格待判了几次 | `hidden_counts` | 只用提交次数的模型，所有时间类特征都是缺失值；输出带 `hidden_times_unavailable` 标记 |
| C | 封榜后整个榜单停止更新 | 无 | 不做预测，返回 `{"error": "unavailable", ...}`，退出码为 2 |

### 奖牌规则

`medal` 有两种写法：

- **按名额**：`{"组名": {"gold": 金牌数, "silver": 银牌数, "bronze": 铜牌数}}`，这是 XCPCIO 的格式。一支队伍的 `groups` 里含有这个组名，才参与这个组的奖牌计算；组名 `all` 表示所有队伍。没写 `groups` 的队伍不属于任何组，按组名发奖时要给队伍写上对应的 `groups`，或者用 `all`。
- **CCPC 比例**：写成字符串 `"ccpc"`，按 10% / 20% / 30% 发金、银、铜牌。基数是至少过了一题的正式队伍；`groups` 里含有 `unofficial`、`star` 或 `打星` 的队伍不算正式队伍。

用 `export-snapshot` 从历史比赛导出的输入文件，`teams`、`groups` 和 `medal` 都已经按原比赛配置填好，可以当作参考。

## 预测输出

`predict` 对每个冻结格子输出一行：

| 字段 | 含义 |
|---|---|
| `contest_id` | 比赛 ID |
| `team_id` / `problem_id` | 队伍和题目 |
| `observation_time` | 观察时刻（秒） |
| `hidden_submissions` | 观察到的隐藏提交次数 |
| `p_ac` | 这些隐藏提交里至少有一次 AC 的概率 |
| `model_version` / `feature_version` | 模型版本和特征版本，例如 `v4` / `f3` |
| `quality_flags` | Mode B 下是 `hidden_times_unavailable`，表示没有隐藏提交时间；Mode A 下为空 |

`p_ac` 的变化规律：

- 新提交出现时 `p_ac` 可能下降。比如 250 分钟交了一次，280 分钟同一道题又交了一次，后一次提交说明前一次多半没过。
- 刚交上的提交 `p_ac` 偏低，交完以后长时间没有动静的格子 `p_ac` 偏高。在测试集里，最后一次提交不到 5 分钟的格子，AC 比例是 14%；沉默超过 20 分钟的格子，AC 比例是 51%。

## 最终排名模拟

`standings` 先对输入做一次 `predict`，再进行 `--sims` 次（默认 10000 次）模拟。每次模拟中，每个冻结格子以各自的 `p_ac` 独立决定过或不过。格子过了的话，AC 时间取这个格子的最后一次隐藏提交；Mode B 没有时间信息，取封榜开始和观察时刻的中点。观察时刻之后的提交不在模拟范围内，所以只有在比赛结束时观察，结果才是最终榜单。

`--out` 目录下生成以下文件：

- `standings_teams.csv`：每支队伍一行，按平均排名排序

  | 列 | 含义 |
  |---|---|
  | `solved_at_freeze` | 封榜时的过题数 |
  | `frozen_cells` | 冻结格子数 |
  | `expected_solved` / `solved_std` | 最终过题数的期望和标准差 |
  | `rank_mean` / `rank_std` | 排名的期望和标准差 |
  | `rank_p05` / `rank_p50` / `rank_p95` | 排名的 5%、50%、95% 分位数 |
  | `p_gold` / `p_silver` / `p_bronze` / `p_medal` | 获得金、银、铜牌及任意奖牌的概率。不参与奖牌计算的队伍为空 |

- `medal_lines.csv`：每个奖牌组每一档（金、银、铜）最后一名获奖队伍的情况，包括过题数 `line_solved` 及其概率 `probability`，以及在该过题数下罚时的 10%、50%、90% 分位数。名额为 0 的档没有这一行。
- `scenarios_<队伍>.csv`：指定 `--team` 时生成。列出这支队伍冻结格子所有过或不过的组合，包括每种组合的概率、最终过题数和罚时、排名分布和奖牌概率。冻结格子超过 8 个时，只取 `p_ac` 最高的 8 个。

`--hypothetical 题数 罚时` 要和 `--team` 一起用。其他队伍照常模拟，输出这支队伍以给定题数和罚时（分钟）收尾时的排名分布。

## 浏览器用户脚本

用户脚本在 [XCPCIO 榜单](https://board.xcpcio.com) 和 [CCPC 外榜](https://rank.iccpc.cn)（CSG-OJ）上直接显示预测，不需要本地跑命令。模型嵌在脚本里，预测和模拟都在浏览器里完成。

### 安装

先装 [Tampermonkey](https://www.tampermonkey.net/)，再打开 <https://github.com/gsh20040816/xcpcpredict/releases/latest/download/xcpcpredict.user.js>，Tampermonkey 会弹出安装页面。这个地址指向最新的模型 release，Tampermonkey 会按它检查更新。

也可以用自己训练的模型生成：

```bash
xcpcpredict export-userscript --model models/v4-all --out out/xcpcpredict.user.js
```

把生成的文件拖进 Tampermonkey 的管理页面即可安装。不加 `--download-url` 时脚本不会自动更新。

### 页面上显示什么

打开任意一场比赛的榜单（XCPCIO 的比赛页，或 CCPC 外榜 `/r/<比赛>` 页），右下角会出现“封榜预测”面板。

- **冻结格子**：从格子底部往上填充绿色，填充高度等于 `p_ac`，右上角标出百分比。鼠标悬停显示隐藏提交次数。
- **队伍名**：底部一条细条按金、银、铜概率分段着色。鼠标悬停显示期望过题数、排名中位数和 90% 区间，以及各档奖牌概率。打星队伍不参与发奖，只显示排名。
- **面板**：观察时刻、冻结格子数、模型版本、计算耗时，以及概率不低于 5% 的各档奖牌线（题数和罚时中位数）。“隐藏标注”恢复原始榜单，“刷新”立即重算，“–”收起面板。

封榜前面板只显示封榜时间，到点后自动开始。封榜期间每分钟重算一次，页面在后台时暂停。比赛结束后改为每 5 分钟检查一次；一旦封榜后的提交出现最终结果（榜单解封），就清掉标注、不再预测，面板显示“榜单已经解封”。

### 数据和计算

- 脚本读取的是榜单页面本身也会加载的数据，封榜期间每分钟取一次，数据不会发往别处。
  - XCPCIO：`https://board.xcpcio.com/data/<比赛路径>/` 下的 `run.json`；`config.json`、`team.json` 和 `organizations.json` 只在打开页面时取一次。
  - CCPC 外榜：页面 `OR_BOOT.Prefetch` 指向的 `rank.json` 快照，比赛、队伍、题目和提交都在里面。
- XCPCIO 的榜单显示待判提交的时间时（`status_time_display.pending` 没有设成 `false`，这也是 XCPCIO 的默认值）用 Mode A，否则用 Mode B。CCPC 外榜点开格子能看到每次提交的时间，用 Mode A。观察时刻取当前时间，比赛结束后取结束时刻。
- CCPC 外榜的数据按榜单自己的口径转换（`core.js` 的 `fromCsg`）：
  - 结果码 4 为 AC，5–10 为计罚时的错误，-1 为封榜后的提交。编译错误（11）和评测中的提交（0–3）不显示，丢掉。同一题可见的 AC 之后的提交也忽略。
  - 罚时按秒累加，每次错误加 20 分钟，最后折成分钟。
  - `tkind=2` 是打星队，`tkind=1` 或 `flg_girls=1` 是女队。
  - 奖牌按比赛设置：比例或名额（`flg_award_qty_mode`），基数（`flg_award_base`）可以是过题队、交过题的队或全部正式队，名次线和榜单一样按累计比例向上取整。前两种基数在模拟里固定；按过题队算时，每次模拟重新数基数。
  - 设了多个奖项分组（`is_multi_group`）的比赛不算奖牌概率。
- 排名模拟默认 5000 次，随机种子固定，数据不变时每次刷新的结果相同。400 支队伍左右的比赛，`p_ac` 约 30 ms，模拟约 0.15 秒。
- `src/xcpcpredict/web/core.js` 是 `normalize`、`replay`、`features`、模型推断和 `standings` 的 JavaScript 移植。在西安、南京、沈阳和一场 CCPC 上，它和 Python 算出的 `p_ac` 最大相差 2e-7；奖牌概率的差别在 1 万次模拟的抽样误差以内。

### 限制

- 只在 `board.xcpcio.com` 和 `rank.iccpc.cn` 上运行。
- XCPCIO 的榜单行上没有队伍 ID，脚本按队名匹配，重名时再比对学校名。仍然无法区分的行不做标注。CCPC 外榜的行带队伍 ID（`data-row-id`），格子带题号（`d-pro-idx`），直接对应。
- CCPC 外榜只标注“队伍排名”视图，学校排名和滚榜视图不标注。
- CCPC 外榜在封榜前会隐藏还没被足够多队伍做出的题；只在封榜后预测，这时所有题都已公开。
- 脚本依赖榜单的页面结构：XCPCIO 表头的 `team-column`、`problem-column` 等 class 和冻结格子的 `pending` class；CCPC 外榜的 `data-row-id`、`d-pro-idx`、`pro-pending` 和 `.team-info-section`。网站改版后标注可能失效，面板的计算不受影响。

