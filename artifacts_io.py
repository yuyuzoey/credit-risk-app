"""artifacts_io.py —— 队友交付物接入层（E 同学：界面只展示，不重算）

约定目录：项目根目录下 artifacts/
    C 同学（管模型）：
        best_model.pkl            锁定的模型（RandomForest + 预处理，joblib 格式）
        threshold.json            选定的阈值 + 选择依据 + 验证集结果
        model_comparison.csv      train/valid 上三模型对比（含基线）
        threshold_comparison.csv  验证集阈值扫描（阈值 -> 精确率/召回率/F1）
        feature_importance.csv    特征重要性排名
    D 同学（管评价）：
        test_result.json          测试集最终评价结果（只跑一次，锁定）
    B 同学（管特征）：
        字段口径.md                字段表与编码说明（文案用）

注意：best_model.pkl 由 joblib 写盘；反序列化时 pickl 会 import B 同学的
      preprocess 模块（内含 _pay_status_to_string 函数），因此项目根目录必须
      存在 preprocess.py，否则加载会报 ModuleNotFoundError。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

C_BUNDLE = ARTIFACTS_DIR / "best_model.pkl"
C_THRESHOLD = ARTIFACTS_DIR / "threshold.json"
C_MODEL_CMP = ARTIFACTS_DIR / "model_comparison.csv"
C_THRESH_CMP = ARTIFACTS_DIR / "threshold_comparison.csv"
C_FEAT_IMP = ARTIFACTS_DIR / "feature_importance.csv"
D_TEST = ARTIFACTS_DIR / "test_result.json"
B_FIELDS = ARTIFACTS_DIR / "字段口径.md"


def artifacts_status() -> dict[str, bool]:
    """侧边栏用：各个交付物是否就位。"""
    return {
        "C 模型 best_model.pkl": C_BUNDLE.exists(),
        "C 阈值 threshold.json": C_THRESHOLD.exists(),
        "C 模型比较 model_comparison.csv": C_MODEL_CMP.exists(),
        "C 阈值扫描 threshold_comparison.csv": C_THRESH_CMP.exists(),
        "C 特征重要性 feature_importance.csv": C_FEAT_IMP.exists(),
        "D 测试集结果 test_result.json": D_TEST.exists(),
        "B 字段口径 字段口径.md": B_FIELDS.exists(),
    }


# ---------------------------------------------------------------------------
# 通用读取
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# C：模型 / 阈值 / 比较表
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_c_bundle() -> dict | None:
    """加载 C 同学锁定的模型包。返回 None 表示文件缺失或依赖缺失。"""
    if not C_BUNDLE.exists():
        return None
    return joblib.load(C_BUNDLE)


def c_bundle_error() -> str | None:
    """返回加载 C 模型时的错误说明（成功或文件缺失返回 None）。"""
    try:
        load_c_bundle()
        return None
    except ModuleNotFoundError as e:
        return f"缺少模块：{e.name}（需要 B 同学的 preprocess.py）"
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def load_c_threshold() -> dict | None:
    """C 同学选定阈值 + 依据 + 验证集指标。"""
    return _read_json(C_THRESHOLD)


def load_c_model_comparison(split: str | None = None) -> pd.DataFrame | None:
    df = _read_csv(C_MODEL_CMP)
    if df is None or split is None:
        return df
    return df[df["split"] == split].reset_index(drop=True)


def load_c_threshold_curve(model: str = "RandomForest") -> pd.DataFrame | None:
    df = _read_csv(C_THRESH_CMP)
    if df is None:
        return None
    sub = df[df["model"] == model]
    return sub.reset_index(drop=True) if len(sub) else df


def load_c_feature_importance(top_n: int | None = None) -> pd.DataFrame | None:
    df = _read_csv(C_FEAT_IMP)
    if df is None or top_n is None:
        return df
    return df.head(top_n)


def predict_with_c_model(bundle: dict, record_df: pd.DataFrame) -> float:
    """用 C 同学的模型预测单条记录的违约概率（原始 23 列 -> 内部预处理）。"""
    X = record_df[bundle["feature_cols"]]
    return float(bundle["model"].predict_proba(bundle["preprocessor"].transform(X))[0, 1])


def predict_test_with_c_model(bundle: dict, test_df: pd.DataFrame):
    """用 C 同学的模型对整段测试集打分，返回 (y_true, y_prob)。"""
    X = test_df[bundle["feature_cols"]]
    prob = bundle["model"].predict_proba(bundle["preprocessor"].transform(X))[:, 1]
    return test_df[bundle["label_col"]].to_numpy(), prob


# ---------------------------------------------------------------------------
# D：测试集最终评价
# ---------------------------------------------------------------------------

def load_d_test_result() -> dict | None:
    return _read_json(D_TEST)


def d_result_to_metrics(d: dict) -> dict:
    """把 D 的 test_result.json 拍平成界面好用的指标字典。"""
    cm = d.get("confusion_matrix", {})
    tn, fp, fn, tp = cm.get("tn", 0), cm.get("fp", 0), cm.get("fn", 0), cm.get("tp", 0)
    prec = d.get("precision_default")
    rec = d.get("recall_default")
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec else None
    n = d.get("test_sample_count", tn + fp + fn + tp)
    return {
        "roc_auc": d.get("roc_auc"),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": (tn + tp) / n if n else None,
        "alarm_rate": d.get("predicted_positive_count", tp + fp) / n if n else None,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "误判人数": fp, "漏判人数": fn,
        "n_test": n,
        "threshold": d.get("threshold"),
        "model_name": d.get("model_name"),
    }


def read_b_field_spec() -> str | None:
    """B 同学的字段口径原文（Markdown）。"""
    if not B_FIELDS.exists():
        return None
    return B_FIELDS.read_text(encoding="utf-8")
