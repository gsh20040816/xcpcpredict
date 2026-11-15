"""Tampermonkey userscript that runs the predictor in the browser on public scoreboards.

The calibrated CatBoost models are embedded as plain oblivious trees. ``web/core.js`` ports
normalization, feature extraction, the standings simulation and the CSG-OJ data adapter;
``web/board.js`` reads the board's public data and draws p_ac and medal odds on the page
(board.xcpcio.com and rank.iccpc.cn).
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

from .features import FEATURE_VERSION
from .predict import ModelBundle

HEADER = """\
// ==UserScript==
// @name         xcpcpredict 封榜过题概率
// @namespace    xcpcpredict
// @version      {version}
// @description  在 XCPCIO 榜单和 CCPC 外榜的冻结格子上显示 AC 概率，并模拟最终排名和奖牌线（模型 {model_version}，特征 {feature_version}）
// @match        https://board.xcpcio.com/*
// @match        https://rank.iccpc.cn/r/*
// @grant        none
// @run-at       document-idle
// @noframes
{update}// ==/UserScript==
"""


def model_to_js(model) -> dict:
    """A Platt-calibrated CatBoostModel as {columns, trees, scale, bias, platt}."""
    cal = model.calibrator
    if not hasattr(model.base, "model_") or cal.method != "platt":
        raise ValueError("only Platt-calibrated CatBoost models can be exported")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.json"
        model.base.model_.save_model(str(path), format="json")
        raw = json.loads(path.read_text())
    column = {f["feature_index"]: f["flat_feature_index"] for f in raw["features_info"]["float_features"]}
    trees = []
    for t in raw["oblivious_trees"]:
        if any(s["split_type"] != "FloatFeature" for s in t["splits"]):
            raise ValueError("only float-feature splits are supported")
        trees.append([
            [column[s["float_feature_index"]] for s in t["splits"]],
            [s["border"] for s in t["splits"]],
            [float(f"{v:.6g}") for v in t["leaf_values"]],
        ])
    scale, bias = raw["scale_and_bias"]
    return {"columns": list(model.base.columns), "trees": trees, "scale": scale, "bias": bias[0],
            "platt": [float(cal.lr_.coef_[0][0]), float(cal.lr_.intercept_[0])]}


def bundle_to_js(bundle: ModelBundle) -> dict:
    return {"model_version": bundle.model_version, "feature_version": FEATURE_VERSION,
            "models": {mode: model_to_js(m) for mode, m in bundle.models.items()}}


def web_source(name: str) -> str:
    return resources.files("xcpcpredict").joinpath("web", name).read_text(encoding="utf-8")


def build_userscript(bundle: ModelBundle, download_url: str | None = None) -> str:
    """The whole script; ``download_url`` (where the script is published) enables Tampermonkey updates."""
    model = json.dumps(bundle_to_js(bundle), separators=(",", ":"))
    update = f"// @downloadURL  {download_url}\n// @updateURL    {download_url}\n" if download_url else ""
    # build time as the version, so every published build is newer than the last
    version = datetime.now(UTC).strftime("%Y.%m.%d.%H%M")
    return (HEADER.format(version=version, model_version=bundle.model_version, feature_version=FEATURE_VERSION,
                          update=update)
            + "(() => {\n\"use strict\";\nconst XP_MODEL = " + model + ";\n"
            + web_source("core.js") + "\n" + web_source("board.js") + "})();\n")
