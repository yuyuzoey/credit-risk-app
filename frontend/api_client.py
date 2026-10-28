"""frontend/api_client.py —— 前端唯一的数据入口（前端与后端之间的唯一通道）

**这是前后端分离的验收核心：**
    本文件（以及整个 frontend/ 目录）**不 import credit_pipeline / artifacts_io / sklearn**。
    所有数据都是通过 HTTP 从后端拿到的 JSON。

    可以用这条命令自检：
        grep -r "import credit_pipeline\\|import artifacts_io\\|import sklearn\\|import joblib" frontend/
        → 应当没有任何输出

设计要点：
    - 后端地址从环境变量 BACKEND_URL 读，默认 http://127.0.0.1:8000
    - 所有请求带超时；连不上时抛 BackendError，界面据此显示"后端未连接"提示
    - 训练是重活，单独给更长的超时（TIMEOUT_TRAIN）
"""

from __future__ import annotations

import os
from typing import Any

import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 30
TIMEOUT_TRAIN = 300


class BackendError(RuntimeError):
    """后端不可用或返回错误时抛出。界面统一捕获后显示友好提示。"""

    def __init__(self, message: str, *, detail: str = "", status: int | None = None):
        super().__init__(message)
        self.detail = detail
        self.status = status


def _request(method: str, path: str, *, json_body: Any = None,
             timeout: int = TIMEOUT) -> Any:
    url = f"{BACKEND_URL}{path}"
    try:
        resp = requests.request(method, url, json=json_body, timeout=timeout)
    except requests.exceptions.ConnectionError as exc:
        raise BackendError(
            "后端服务未连接",
            detail=(f"请先在项目根目录运行：\n"
                    f"uvicorn backend.main:app --reload --port 8000\n\n"
                    f"当前尝试的地址：{url}\n\n原始错误：{exc}"),
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise BackendError("后端请求超时",
                           detail=f"{path} 超过 {timeout}s 未返回") from exc

    if resp.status_code >= 400:
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001
            body = {"detail": resp.text[:400]}
        raise BackendError(
            body.get("error") or body.get("detail") or f"HTTP {resp.status_code}",
            detail=str(body.get("detail", "")),
            status=resp.status_code,
        )
    try:
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        raise BackendError("返回内容不是合法 JSON", detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------

def health() -> dict:
    return _request("GET", "/health", timeout=8)


def is_online() -> tuple[bool, str]:
    """返回 (是否在线, 失败原因)。"""
    try:
        health()
        return True, ""
    except BackendError as exc:
        return False, f"{exc}。{exc.detail}"


# ---------------------------------------------------------------------------
# 数据概览
# ---------------------------------------------------------------------------

def data_meta() -> dict:
    return _request("GET", "/data/meta")


def data_distribution() -> dict:
    return _request("GET", "/data/distribution")


def data_pay_codes() -> dict:
    return _request("GET", "/data/pay-codes")


def data_default_rate() -> dict:
    return _request("GET", "/data/default-rate")


def reload_data() -> dict:
    return _request("POST", "/data/reload")


# ---------------------------------------------------------------------------
# 训练 / 阈值 / 锁定
# ---------------------------------------------------------------------------

def train(exclude_sensitive: bool = False,
          pay_encoding: str = "numeric+binary") -> dict:
    return _request("POST", "/train",
                    json_body={"exclude_sensitive": exclude_sensitive,
                               "pay_encoding": pay_encoding},
                    timeout=TIMEOUT_TRAIN)


def pick_threshold(job_id: str | None, model: str, criterion: str,
                   recall_target: float = 0.60,
                   cost_ratio: tuple[float, float] = (5.0, 1.0)) -> dict:
    return _request("POST", "/threshold/pick", json_body={
        "job_id": job_id, "model": model, "criterion": criterion,
        "recall_target": recall_target,
        "cost_ratio": list(cost_ratio),
    })


def get_lock() -> dict:
    return _request("GET", "/model/lock")


def lock(model: str, threshold: float, reason: str = "") -> dict:
    return _request("POST", "/model/lock",
                    json_body={"model": model, "threshold": threshold,
                               "reason": reason})


def unlock() -> dict:
    return _request("DELETE", "/model/lock")


# ---------------------------------------------------------------------------
# 专家交付物
# ---------------------------------------------------------------------------

def artifacts_status() -> dict:
    return _request("GET", "/artifacts/status")


def c_model_info() -> dict:
    return _request("GET", "/artifacts/c-model")


def artifacts_comparison(split: str = "valid", top_n: int = 20) -> dict:
    return _request("GET", f"/artifacts/comparison?split={split}&top_n={top_n}")


def test_result() -> dict:
    return _request("GET", "/artifacts/test-result")


def field_spec() -> dict:
    return _request("GET", "/artifacts/field-spec")


# ---------------------------------------------------------------------------
# 预测
# ---------------------------------------------------------------------------

def predict_record(record_id: int) -> dict:
    """返回 {probability, threshold, alarm, record:{info, derived, label}, ...}"""
    return _request("POST", "/predict/record", json_body={"record_id": int(record_id)})


def random_test_record() -> dict:
    """随机抽一条测试集记录并预测。"""
    return _request("GET", "/records/random")


def predict_custom(payload: dict) -> dict:
    return _request("POST", "/predict/custom", json_body=payload)


def schema_fields() -> dict:
    return _request("GET", "/schema/fields")


# ---------------------------------------------------------------------------
# 前端本地的"展示层计算"：从后端已返回的概率与阈值里挑选指标
#
# 说明：这里不做任何模型计算，只是把后端返回的阈值扫描表按阈值筛选 / 取最优，
#      公式与后端 compute_metrics 的阈值扫描完全一致（后端已算好整张表）。
# ---------------------------------------------------------------------------

def metrics_at(curve: list[dict], threshold: float) -> dict:
    """从阈值扫描表里取最接近该阈值的一行（表步长 0.01）。"""
    if not curve:
        return {}
    target = round(float(threshold), 2)
    best = min(curve, key=lambda r: abs(float(r["threshold"]) - target))
    return best


def curve_to_frame(curve: list[dict], cols: list[str]) -> dict[str, list]:
    """把阈值扫描表转成 {列名: 值列表}，方便前端直接喂给图表。"""
    return {c: [row.get(c) for row in curve] for c in cols}
