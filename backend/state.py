"""backend/state.py —— 后端内存状态（服务端唯一的全局可变状态）

为什么需要它：
    Streamlit 原来把训练结果存在 st.session_state 里。前后端分离后，
    训练结果住在后端进程内存中，前端通过 job_id 引用它。
    前端每次交互都会重新跑一遍脚本，没有这个状态就得反复训练。

设计要点：
    - _JOBS：一次训练产出的全部结果（三模型管道、验证集指标、阈值曲线）
    - _LOCKED：锁定的模型名与阈值，落盘到 results/locked_model.json
      （与 credit_pipeline.load_locked 同一文件，重启后仍能读到，课程要求的
        "锁定后不得随意改动" 才真正成立）
    - 全部加锁，防止并发请求同时改写
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
LOCK_FILE = ROOT / "results" / "locked_model.json"

_LOCK = threading.RLock()
_JOBS: dict[str, dict[str, Any]] = {}
_LOCKED: Optional[dict[str, Any]] = None
_CURRENT_JOB: Optional[str] = None


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


def save_job(job: dict[str, Any]) -> str:
    """保存一次训练结果，返回 job_id。"""
    global _CURRENT_JOB
    jid = new_job_id()
    with _LOCK:
        _JOBS.clear()          # 只保留最近一次，避免内存膨胀
        _JOBS[jid] = job
        _CURRENT_JOB = jid
    return jid


def get_job(job_id: Optional[str]) -> Optional[dict[str, Any]]:
    with _LOCK:
        if job_id and job_id in _JOBS:
            return _JOBS[job_id]
        if _CURRENT_JOB:
            return _JOBS.get(_CURRENT_JOB)
    return None


def current_job_id() -> Optional[str]:
    with _LOCK:
        return _CURRENT_JOB


def clear_jobs() -> None:
    global _CURRENT_JOB
    with _LOCK:
        _JOBS.clear()
        _CURRENT_JOB = None


def set_locked(payload: Optional[dict[str, Any]]) -> None:
    """设置/清除锁定。写入内存同时落盘，保证后端进程重启后锁定依然有效。"""
    global _LOCKED
    with _LOCK:
        _LOCKED = payload
        LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        if payload is None:
            if LOCK_FILE.exists():
                LOCK_FILE.unlink()
        else:
            with open(LOCK_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)


def get_locked() -> Optional[dict[str, Any]]:
    """读锁定状态；内存里没有就尝试从磁盘恢复（后端重启后的第一次调用）。"""
    global _LOCKED
    with _LOCK:
        if _LOCKED is not None:
            return _LOCKED
        if LOCK_FILE.exists():
            try:
                with open(LOCK_FILE, encoding="utf-8") as f:
                    _LOCKED = json.load(f)
                return _LOCKED
            except Exception:  # noqa: BLE001
                return None
    return None
