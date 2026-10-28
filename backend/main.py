"""backend/main.py —— FastAPI 后端服务（前后端分离的后端）

职责边界（这是分离的验收标准）：
    本文件及其依赖（credit_pipeline / artifacts_io）**不 import streamlit**，
    只做计算与数据读取，通过 HTTP 把结果（JSON）交给任意前端。

用法：
    uvicorn backend.main:app --reload --port 8000
    # 打开 http://127.0.0.1:8000/docs 可以看到全部接口并直接试

几个刻意的设计：
    1. /train 一次训练同时返回 **三模型在验证集上的概率** 和整条阈值扫描曲线。
       前端切模型、拖阈值都在本地算，不需要反复请求 —— 否则每次点一下都要等训练。
       （前端仍然只做展示：切模型 / 拖阈值只是从已返回的概率里重新算指标，
        指标公式与后端 compute_metrics 完全一致，属于"展示层的筛选"而非"重算模型"。）
    2. 训练结果的管道（模型对象）留在后端内存，前端只拿到 job_id。
       /predict 用 job_id 找回管道再预测，模型对象不出后端。
    3. 测试集结果永远从 artifacts/test_result.json 读，接口不提供"重新评价测试集"的能力
       —— 课程红线：测试集只评价一次。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import artifacts_io as ai
import credit_pipeline as cp
from backend import state

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "01_credit_default.csv"

app = FastAPI(
    title="信用卡客户下月违约风险分析 · 后端 API",
    description="课程项目题目一。仅作教学分析，不用于真实授信决策。",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 教学项目，允许本地任意端口的前端访问
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 统一错误格式：{"error": "...", "detail": "..."}
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def _unhandled(request, exc: Exception):  # noqa: ANN001
    return JSONResponse(
        status_code=500,
        content={"error": "服务器内部错误", "detail": f"{type(exc).__name__}: {exc}"},
    )


def _jsonable(obj: Any) -> Any:
    """把 numpy / pandas / NaN 递归转成可 JSON 化的原生类型。"""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if obj is pd.NA or obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return _jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return _jsonable(obj.tolist())
    return obj


# ---------------------------------------------------------------------------
# 数据加载（进程内缓存，避免每次请求都读 CSV）
# ---------------------------------------------------------------------------

_DF_CACHE: dict[str, pd.DataFrame] = {}


def _get_df(force: bool = False) -> pd.DataFrame:
    key = str(DEFAULT_CSV)
    if force or key not in _DF_CACHE:
        if not DEFAULT_CSV.exists():
            raise HTTPException(
                status_code=404,
                detail=f"数据文件不存在：{DEFAULT_CSV}",
            )
        try:
            _DF_CACHE[key] = cp.load_data(DEFAULT_CSV)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"数据读取失败：{exc}") from exc
    return _DF_CACHE[key]


# ---------------------------------------------------------------------------
# 请求 / 响应模型
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    exclude_sensitive: bool = Field(False, description="是否剔除性别/教育/婚姻/年龄")
    pay_encoding: str = Field("numeric+binary", description="PAY_* 编码方式")


class ThresholdRequest(BaseModel):
    job_id: Optional[str] = None
    model: str = Field("tree", description="baseline / logreg / tree")
    criterion: str = Field("max_f1", description="max_f1 / recall_ge / min_cost")
    recall_target: float = 0.60
    cost_ratio: tuple[float, float] = (5.0, 1.0)


class LockRequest(BaseModel):
    model: str
    threshold: float
    reason: str = ""


class RecordRequest(BaseModel):
    record_id: int


class CustomRequest(BaseModel):
    """手工输入一条新客户（演示页"输入"模式）。字段与 CSV 一一对应。"""

    credit_limit_ntd: float = 50000
    sex_code: int = 1
    education_code: int = 2
    marriage_code: int = 2
    age_years: int = 35
    pay_status: list[int] = Field(
        default=[0, 0, 0, 0, 0, 0], description="近 6 个月还款状态，从最新月份开始",
        min_length=6, max_length=6)
    bill_amount: list[float] = Field(
        default=[0, 0, 0, 0, 0, 0], min_length=6, max_length=6)
    paid_amount: list[float] = Field(
        default=[0, 0, 0, 0, 0, 0], min_length=6, max_length=6)


# ---------------------------------------------------------------------------
# 1) 健康检查
# ---------------------------------------------------------------------------

@app.get("/health", summary="健康检查 + 交付物到达状态")
def health():
    thr = ai.load_c_threshold() or {}
    return _jsonable({
        "status": "ok",
        "data_file": str(DEFAULT_CSV.name),
        "data_ready": DEFAULT_CSV.exists(),
        "artifacts": ai.artifacts_status(),
        "c_model_error": ai.c_bundle_error(),
        "c_model": thr.get("model_name"),
        "c_threshold": thr.get("threshold"),
        "locked": state.get_locked(),
        "job_id": state.current_job_id(),
    })


# ---------------------------------------------------------------------------
# 2) 数据概览
# ---------------------------------------------------------------------------

@app.get("/data/meta", summary="样本量、列名、划分统计")
def data_meta():
    df = _get_df()
    audit = cp.audit_data(df)
    return _jsonable({
        "total": int(len(df)),
        "columns": list(df.columns),
        "overall_default_rate": float(df[cp.LABEL].mean()),
        "audit": audit,
    })


def _hist_payload(series: pd.Series, bins: int = 30, clip_pct: float | None = None):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if clip_pct is not None and len(s):
        lo, hi = s.quantile(0.01), s.quantile(clip_pct)
        s = s.clip(lo, hi)
    if not len(s):
        return {"counts": [], "centers": []}
    counts, edges = np.histogram(s, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    return {"counts": counts.tolist(), "centers": centers.tolist()}


@app.get("/data/distribution", summary="授信额度 / 年龄 / 还款状态的分布（按 split 分组）")
def data_distribution():
    df = _get_df()
    out: dict[str, Any] = {}
    for split in ["train", "valid", "test"]:
        sub = df[df[cp.SPLIT_COL] == split]
        out[split] = {
            "n": int(len(sub)),
            "limit_bal": _hist_payload(sub[cp.COL_LIMIT_BAL], bins=30, clip_pct=0.99),
            "age": _hist_payload(sub[cp.COL_AGE], bins=30),
        }
    # 还款状态：各编码的人数（用最近一个月）
    pay_col = cp.PAY_COLS[0] if cp.PAY_COLS else None
    pay_dist = {}
    if pay_col:
        for split in ["train", "valid", "test"]:
            sub = df[df[cp.SPLIT_COL] == split]
            vc = sub[pay_col].value_counts().sort_index()
            pay_dist[split] = {"labels": [str(i) for i in vc.index],
                               "counts": vc.values.tolist()}
    return _jsonable({
        "by_split": out,
        "pay_status_column": pay_col,
        "pay_status_by_split": pay_dist,
    })


@app.get("/data/pay-codes", summary="还款状态编码含义表")
def data_pay_codes():
    df = _get_df()
    return _jsonable({
        "meaning": cp.PAY_CODE_MEANING,
        "table": cp.describe_pay_codes(df),
    })


@app.get("/data/default-rate", summary="额度使用率分箱 vs 违约率")
def data_default_rate():
    df = _get_df()
    df_f = cp.add_derived_features(df)
    df_f["util_bin"] = pd.cut(
        df_f["utilization_1"],
        bins=[-np.inf, 0, 0.2, 0.4, 0.6, 0.8, 1.0, np.inf],
        labels=["≤0", "0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0", ">1.0"],
    )
    return _jsonable({
        "table": cp.default_rate_by_group(df_f, "util_bin"),
        "total": int(len(df)),
    })


@app.post("/data/reload", summary="重新读取 CSV（清空缓存与训练结果）")
def data_reload():
    df = _get_df(force=True)
    state.clear_jobs()
    state.set_locked(None)
    return _jsonable({"status": "reloaded", "rows": int(len(df))})


# ---------------------------------------------------------------------------
# 3) 训练（一次返回三模型的验证集概率 + 阈值曲线）
# ---------------------------------------------------------------------------

@app.post("/train", summary="在 train 上训练三模型，返回验证集指标与阈值曲线")
def train(req: TrainRequest):
    df = _get_df()
    train_df = df[df[cp.SPLIT_COL] == "train"]
    valid_df = df[df[cp.SPLIT_COL] == "valid"]

    try:
        result = cp.train_models(
            train_df, valid_df,
            pay_encoding=req.pay_encoding,
            exclude_sensitive=req.exclude_sensitive,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"训练失败：{exc}") from exc

    y_valid = result["y_valid"]
    models_payload: dict[str, Any] = {}
    for name, m in result["models"].items():
        p_valid = m["y_prob_valid"]
        curve = cp.threshold_sweep(y_valid, p_valid)
        models_payload[name] = {
            "metrics_at_0_5": cp.compute_metrics(y_valid, p_valid, 0.5),
            "auc": float(cp.compute_metrics(y_valid, p_valid, 0.5)["roc_auc"] or 0.0),
            "threshold_curve": curve.where(pd.notna(curve), None).to_dict(orient="records"),
        }

    job = {
        "models": {k: v["pipe"] for k, v in result["models"].items()},
        "y_valid": y_valid,
        "prob_valid": {k: np.asarray(v["y_prob_valid"]) for k, v in result["models"].items()},
        "pay_encoding": result["pay_encoding"],
        "exclude_sensitive": result["exclude_sensitive"],
        "feature_cols": result["feature_cols"],
    }
    job_id = state.save_job(job)

    return _jsonable({
        "job_id": job_id,
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "default_rate_valid": float(np.mean(y_valid)),
        "pay_encoding": result["pay_encoding"],
        "exclude_sensitive": result["exclude_sensitive"],
        "n_features": len(result["feature_cols"]),
        "models": models_payload,
    })


@app.post("/threshold/pick", summary="按准则选阈值（仅用验证集）")
def threshold_pick(req: ThresholdRequest):
    job = state.get_job(req.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="训练结果不存在，请先调用 /train")
    if req.model not in job["models"]:
        raise HTTPException(status_code=400, detail=f"未知模型：{req.model}")

    y_valid = job["y_valid"]
    p = job["prob_valid"].get(req.model)
    if p is None:
        p = np.zeros(len(y_valid))

    sweep = cp.threshold_sweep(y_valid, p)
    threshold, reason = cp.pick_threshold(
        sweep, req.criterion,
        recall_target=req.recall_target,
        cost_ratio=tuple(req.cost_ratio),
    )
    return _jsonable({
        "model": req.model,
        "criterion": req.criterion,
        "threshold": threshold,
        "reason": reason,
        "valid": threshold is not None,
        "metrics_at_threshold": (
            cp.compute_metrics(y_valid, p, float(threshold))
            if threshold is not None else None),
        "metrics_at_0_5": cp.compute_metrics(y_valid, p, 0.5),
    })


# ---------------------------------------------------------------------------
# 4) 锁定 / 重置（对应课程"锁定模型与阈值"）
# ---------------------------------------------------------------------------

@app.get("/model/lock", summary="查询当前锁定状态")
def lock_get():
    return _jsonable({"locked": state.get_locked()})


@app.post("/model/lock", summary="锁定模型与阈值")
def lock_post(req: LockRequest):
    if req.model not in ("baseline", "logreg", "tree"):
        raise HTTPException(status_code=400, detail=f"未知模型：{req.model}")
    payload = {
        "model": req.model,
        "threshold": float(req.threshold),
        "reason": req.reason,
        "job_id": state.current_job_id(),
    }
    state.set_locked(payload)
    return _jsonable({"status": "locked", "locked": payload})


@app.delete("/model/lock", summary="重置锁定与测试结果（会清空已评价的测试结果）")
def lock_delete():
    state.set_locked(None)
    state.clear_jobs()
    cp.reset_evaluation()
    return _jsonable({"status": "reset"})


# ---------------------------------------------------------------------------
# 5) 队友交付物（只读）
# ---------------------------------------------------------------------------

@app.get("/artifacts/status", summary="各交付物是否就位")
def artifacts_status():
    return _jsonable({
        "status_map": ai.artifacts_status(),
        "c_model_error": ai.c_bundle_error(),
    })


@app.get("/artifacts/c-model", summary="正式模型的信息与阈值")
def artifacts_c_model():
    thr = ai.load_c_threshold()
    if thr is None:
        raise HTTPException(status_code=404, detail="threshold.json 不存在")
    bundle = ai.load_c_bundle()
    return _jsonable({
        "threshold_info": thr,
        "model_loaded": bundle is not None,
        "load_error": ai.c_bundle_error(),
        "model_name": bundle.get("model_name") if bundle else thr.get("model_name"),
    })


@app.get("/artifacts/comparison", summary="模型对比表 / 阈值曲线 / 特征重要性")
def artifacts_comparison(split: str = "valid", top_n: int = 20):
    mc = ai.load_c_model_comparison(split)
    thr = ai.load_c_threshold() or {}
    tc = ai.load_c_threshold_curve(thr.get("model_name", "RandomForest"))
    fi = ai.load_c_feature_importance(top_n)
    return _jsonable({
        "model_comparison": mc,
        "threshold_curve": tc,
        "feature_importance": fi,
        "threshold_info": thr,
    })


@app.get("/artifacts/test-result", summary="测试集锁定结果（只读，不重算）")
def artifacts_test_result():
    d = ai.load_d_test_result()
    if d is None:
        raise HTTPException(status_code=404, detail="test_result.json 不存在")
    m = ai.d_result_to_metrics(d)
    return _jsonable({"raw": d, "metrics": m})


@app.get("/artifacts/field-spec", summary="字段口径说明（Markdown 原文）")
def artifacts_field_spec():
    text = ai.read_b_field_spec()
    if not text:
        raise HTTPException(status_code=404, detail="字段口径.md 不存在")
    return _jsonable({"markdown": text})


# ---------------------------------------------------------------------------
# 6) 预测
# ---------------------------------------------------------------------------

@app.post("/predict/record", summary="按 record_id 预测（读真实 CSV 记录）")
def predict_record(req: RecordRequest):
    df = _get_df()
    row = df[df["record_id"] == req.record_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"record_id={req.record_id} 不存在")
    rec = row.iloc[[0]]
    payload = _predict_any(rec, record_id=req.record_id,
                           split=rec[cp.SPLIT_COL].iloc[0],
                           true_label=rec[cp.LABEL].iloc[0])
    payload["record"] = _record_card(rec)
    return payload


@app.get("/records/random", summary="从测试集随机抽一条（演示页用）")
def records_random():
    df = _get_df()
    test = df[df[cp.SPLIT_COL] == "test"]
    if test.empty:
        raise HTTPException(status_code=404, detail="测试集为空")
    pos = int(np.random.randint(len(test)))
    rec = test.iloc[[pos]]          # 保持 DataFrame（含原列名），不能传 Series
    payload = _predict_any(rec, record_id=int(rec["record_id"].iloc[0]),
                           split="test", true_label=int(rec[cp.LABEL].iloc[0]))
    payload["record"] = _record_card(rec)
    return payload


@app.post("/records/get", summary="按 record_id 取记录详情（不预测）")
def records_get(req: RecordRequest):
    df = _get_df()
    row = df[df["record_id"] == req.record_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"record_id={req.record_id} 不存在")
    return _jsonable(_record_card(row.iloc[[0]]))


def _record_card(rec: pd.DataFrame) -> dict:
    """把一条记录整理成前端好展示的形式：原始字段 + 衍生特征。

    衍生特征的公式由后端 credit_pipeline 统一定义，前端不重复实现
    —— 否则两边算错一个公式就前后不一致了。
    """
    feat = cp.add_derived_features(rec).iloc[0]
    raw = rec.iloc[0]
    info = {
        "record_id": int(raw["record_id"]),
        "split": str(raw[cp.SPLIT_COL]),
        cp.COL_LIMIT_BAL: float(raw[cp.COL_LIMIT_BAL]),
        cp.COL_SEX: int(raw[cp.COL_SEX]),
        cp.COL_EDUCATION: int(raw[cp.COL_EDUCATION]),
        cp.COL_MARRIAGE: int(raw[cp.COL_MARRIAGE]),
        cp.COL_AGE: float(raw[cp.COL_AGE]),
        cp.PAY_COLS[0]: int(raw[cp.PAY_COLS[0]]),
        cp.BILL_COLS[0]: float(raw[cp.BILL_COLS[0]]),
        cp.PAY_AMT_COLS[0]: float(raw[cp.PAY_AMT_COLS[0]]),
    }
    derived = {
        "额度使用率": round(float(feat["utilization_1"]), 4),
        "还款占账单比": round(float(feat["pay_ratio_1"]), 4),
        "最大逾期月数": int(feat["max_delinq"]),
        "逾期月数个数": int(feat["n_delinq_months"]),
    }
    return {"info": info, "derived": derived,
            "label": (None if pd.isna(raw[cp.LABEL]) else int(raw[cp.LABEL]))}



@app.post("/predict/custom", summary="手工输入一条新客户并预测")
def predict_custom(req: CustomRequest):
    df = _get_df()
    row: dict[str, Any] = {
        "record_id": -1,
        cp.SPLIT_COL: "custom",
        cp.LABEL: np.nan,
        cp.COL_LIMIT_BAL: req.credit_limit_ntd,
        cp.COL_SEX: req.sex_code,
        cp.COL_EDUCATION: req.education_code,
        cp.COL_MARRIAGE: req.marriage_code,
        cp.COL_AGE: req.age_years,
    }
    # 近 6 个月日期从新到旧：PAY_COLS / BILL_COLS / PAY_AMT_COLS 已按此顺序定义
    for col, val in zip(cp.PAY_COLS, req.pay_status):
        row[col] = val
    for col, val in zip(cp.BILL_COLS, req.bill_amount):
        row[col] = val
    for col, val in zip(cp.PAY_AMT_COLS, req.paid_amount):
        row[col] = val
    # 补齐 CSV 中其它列，保持列顺序一致
    for c in df.columns:
        row.setdefault(c, np.nan)
    rec = pd.DataFrame([row])[list(df.columns)]
    return _predict_any(rec, record_id=None, split="custom", true_label=None)


def _predict_any(rec: pd.DataFrame, record_id, split: str, true_label) -> dict:
    """统一预测入口：优先用正式模型（C 交付），不可用时退回后端现场训练模型。"""
    thr_info = ai.load_c_threshold() or {}
    threshold = float(thr_info.get("threshold", 0.5))
    bundle = ai.load_c_bundle()

    if bundle is not None:
        prob = float(ai.predict_with_c_model(bundle, rec))
        source = "artifacts/best_model.pkl（正式模型）"
        model_key = bundle.get("model_name", "RandomForest")
    else:
        job = state.get_job(None)
        if job is None:
            raise HTTPException(
                status_code=503,
                detail="正式模型不可用且尚未训练，请先调用 /train 或检查 artifacts/",
            )
        locked = state.get_locked() or {}
        model_key = locked.get("model", "tree")
        threshold = float(locked.get("threshold", 0.5))
        pipe = job["models"].get(model_key)
        if pipe is None:
            raise HTTPException(status_code=400, detail=f"模型 {model_key} 不可用")
        prepared = cp._prepare_features(rec, job["pay_encoding"])
        drop = list(cp.EXCLUDE_COLS)
        if job["exclude_sensitive"]:
            drop += [c for c in cp.SENSITIVE_COLS if c in prepared.columns]
        X = prepared.drop(columns=drop)
        prob = float(pipe.predict_proba(X)[0, 1])
        source = f"后端现场训练模型（{model_key}）"

    is_alarm = bool(prob >= threshold)
    # 风险分档（仅用于文字提示，不参与任何计算）
    if prob >= 0.6:
        band = "高"
    elif prob >= 0.3:
        band = "中"
    else:
        band = "低"

    payload = {
        "record_id": record_id,
        "split": split,
        "probability": round(prob, 6),
        "threshold": threshold,
        "alarm": is_alarm,
        "risk_band": band,
        "model_source": source,
        "model_name": model_key,
        "true_label": None if true_label is None or pd.isna(true_label) else int(true_label),
        "label_note": (
            "该记录来自测试集：真实标签仅事后核验，不参与预测"
            if split == "test" else
            ("该记录来自验证集/训练集，同样不参与预测" if split in ("train", "valid") else
             "手工输入的记录没有真实标签")
        ),
    }
    return _jsonable(payload)


# ---------------------------------------------------------------------------
# 7) 字段字典（给前端"手工输入"模式做控件）
# ---------------------------------------------------------------------------

@app.get("/schema/fields", summary="可手工输入的字段及其取值范围")
def schema_fields():
    df = _get_df()
    fields = []
    for col in [cp.COL_LIMIT_BAL, cp.COL_SEX, cp.COL_EDUCATION,
                cp.COL_MARRIAGE, cp.COL_AGE]:
        s = pd.to_numeric(df[col], errors="coerce")
        fields.append({
            "name": col,
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median()),
        })
    return _jsonable({
        "numeric_fields": fields,
        "pay_cols": cp.PAY_COLS,
        "bill_cols": cp.BILL_COLS,
        "pay_amt_cols": cp.PAY_AMT_COLS,
        "pay_code_meaning": cp.PAY_CODE_MEANING,
    })
