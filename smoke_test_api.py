"""smoke_test_api.py —— 后端接口冒烟测试（跑一遍前端会用到的每个接口）

用法（先启动后端）：
    uvicorn backend.main:app --port 8000
    python smoke_test_api.py

这个文件是"前后端分离"的验收脚本：只要它全绿，就说明前端页面要的数据后端都给得出来。
"""

from __future__ import annotations

import sys

import requests

BASE = "http://127.0.0.1:8000"


def _get(path: str, **kw):
    return requests.get(BASE + path, timeout=kw.pop("timeout", 30), **kw)


def _post(path: str, **kw):
    return requests.post(BASE + path, timeout=kw.pop("timeout", 300), **kw)


def main() -> int:
    ok = 0
    fail: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        if cond:
            ok += 1
            print(f"  [OK]   {name}")
        else:
            fail.append(name)
            print(f"  [FAIL] {name}  {detail}")

    print("== 1. 健康检查 ==")
    r = _get("/health")
    check("GET /health", r.status_code == 200, r.text[:200])
    h = r.json()
    check("所有交付物就位", all(h["artifacts"].values()), str(h["artifacts"]))
    check("C 模型可加载", h["c_model_error"] is None, str(h["c_model_error"]))

    print("== 2. 数据概览 ==")
    r = _get("/data/meta")
    check("GET /data/meta", r.status_code == 200, r.text[:200])
    meta = r.json()
    check("总样本 30000", meta["total"] == 30000, str(meta.get("total")))
    check("三组划分齐全",
          all(k in meta["audit"] for k in ("train", "valid", "test")))

    r = _get("/data/distribution")
    check("GET /data/distribution", r.status_code == 200, r.text[:200])
    dd = r.json()
    check("每组都有额度直方图",
          all(dd["by_split"][s]["limit_bal"]["counts"] for s in ("train", "valid", "test")))
    check("每组都有还款状态分布",
          all(dd["pay_status_by_split"][s]["counts"] for s in ("train", "valid", "test")))

    r = _get("/data/pay-codes")
    check("GET /data/pay-codes", r.status_code == 200, r.text[:200])
    check("编码表含负值", any(int(row["取值"]) < 0 for row in r.json()["table"]))

    r = _get("/data/default-rate")
    check("GET /data/default-rate", r.status_code == 200, r.text[:200])
    check("分箱表非空", len(r.json()["table"]) >= 5)

    print("== 3. 训练 / 阈值 / 锁定 ==")
    r = _post("/train", json={"exclude_sensitive": False, "pay_encoding": "numeric+binary"})
    check("POST /train", r.status_code == 200, r.text[:300])
    tr = r.json()
    job_id = tr["job_id"]
    check("三模型都有阈值曲线",
          all(len(tr["models"][m]["threshold_curve"]) == 99
              for m in ("baseline", "logreg", "tree")))
    check("树模型 AUC 合理 (0.75~0.85)",
          0.75 <= tr["models"]["tree"]["metrics_at_0_5"]["roc_auc"] <= 0.85,
          str(tr["models"]["tree"]["metrics_at_0_5"]["roc_auc"]))

    r = _post("/threshold/pick", json={"job_id": job_id, "model": "tree",
                                       "criterion": "max_f1"})
    check("POST /threshold/pick", r.status_code == 200, r.text[:300])
    picked = r.json()
    check("选出阈值", picked["threshold"] is not None, str(picked))

    r = _post("/threshold/pick", json={"job_id": job_id, "model": "baseline",
                                       "criterion": "max_f1"})
    check("基线选阈值应返回 valid=false（不崩）",
          r.status_code == 200 and r.json()["valid"] is False, r.text[:200])

    r = _get("/model/lock")
    check("GET /model/lock", r.status_code == 200, r.text[:200])

    r = _post("/model/lock", json={"model": "tree",
                                   "threshold": picked["threshold"],
                                   "reason": "冒烟测试"})
    check("POST /model/lock", r.status_code == 200, r.text[:200])
    check("锁定已生效", _get("/model/lock").json()["locked"] is not None)

    r = _get("/health")
    check("health 里能看到锁定", r.json()["locked"] is not None, r.text[:300])

    print("== 4. 交付物 ==")
    for path in ["/artifacts/status", "/artifacts/c-model",
                 "/artifacts/comparison?split=valid&top_n=20",
                 "/artifacts/test-result", "/artifacts/field-spec"]:
        r = _get(path)
        check(f"GET {path}", r.status_code == 200, r.text[:200])

    cmp_ = _get("/artifacts/comparison?split=valid&top_n=20").json()
    check("模型比较表有内容", bool(cmp_["model_comparison"]))
    check("阈值曲线有内容", bool(cmp_["threshold_curve"]))
    check("特征重要性 Top20", len(cmp_["feature_importance"]) == 20,
          str(len(cmp_["feature_importance"])))

    d = _get("/artifacts/test-result").json()
    check("D 的 TN/FP/FN/TP = 2899/606/400/596",
          (d["metrics"]["TN"], d["metrics"]["FP"],
           d["metrics"]["FN"], d["metrics"]["TP"]) == (2899, 606, 400, 596),
          str({k: d["metrics"][k] for k in ("TN", "FP", "FN", "TP")}))
    check("D 的 ROC-AUC = 0.782545", abs(d["metrics"]["roc_auc"] - 0.782545) < 1e-9,
          str(d["metrics"]["roc_auc"]))

    print("== 5. 预测 ==")
    r = _post("/predict/record", json={"record_id": 6})
    check("POST /predict/record", r.status_code == 200, r.text[:200])
    pr = r.json()
    check("record_id=6 概率 0.327124", abs(pr["probability"] - 0.327124) < 1e-6,
          str(pr.get("probability")))
    check("记录详情里有衍生特征", "utilization_1" not in pr["record"]["derived"]
          and "额度使用率" in pr["record"]["derived"], str(pr["record"]["derived"]))

    r = _get("/records/random")
    check("GET /records/random", r.status_code == 200, r.text[:200])
    check("随机记录来自测试集", r.json()["split"] == "test", r.text[:200])

    r = _post("/predict/record", json={"record_id": 999999})
    check("不存在的 record_id 返回 404", r.status_code == 404, r.text[:200])

    r = _post("/predict/custom", json={
        "credit_limit_ntd": 20000, "sex_code": 1, "education_code": 1,
        "marriage_code": 2, "age_years": 25,
        "pay_status": [2, 2, 3, 3, 4, 4],
        "bill_amount": [18000, 17500, 17000, 16500, 16000, 15500],
        "paid_amount": [0, 0, 0, 0, 0, 0]})
    check("POST /predict/custom", r.status_code == 200, r.text[:300])
    pc = r.json()
    check("高风险客户判为高", pc["risk_band"] == "高" and pc["alarm"] is True,
          str({k: pc[k] for k in ("probability", "risk_band", "alarm")}))

    print("== 6. 字段字典 ==")
    r = _get("/schema/fields")
    check("GET /schema/fields", r.status_code == 200, r.text[:200])
    sf = r.json()
    check("6 个还款状态列", len(sf["pay_cols"]) == 6)
    check("有数值字段范围", len(sf["numeric_fields"]) == 5)

    print("== 7. 重置 ==")
    r = requests.delete(BASE + "/model/lock", timeout=60)
    check("DELETE /model/lock", r.status_code == 200, r.text[:200])
    check("重置后无锁定", _get("/model/lock").json()["locked"] is None)

    print()
    total = ok + len(fail)
    print(f"结果：{ok}/{total} 通过")
    if fail:
        print("失败项：")
        for f in fail:
            print("  -", f)
        return 1
    print("全部通过 ✅ 前端页面所需数据后端都能提供。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
