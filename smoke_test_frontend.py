"""smoke_test_frontend.py —— 前端四页冒烟测试（不需要打开浏览器）

原理：用 Streamlit 官方的 AppTest 无头执行 frontend/app.py，逐个页面点过去，
      检查没有异常抛出。需要后端已启动。

用法：
    uvicorn backend.main:app --port 8000    # 另一个窗口
    python smoke_test_frontend.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent / "frontend" / "app.py")


def run_page(page_name: str, extra_setup=None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, f"首页加载即报错：{at.exception}"

    # 侧边栏第一组 radio 是页面导航
    at.radio[0].set_value(page_name).run()

    if extra_setup is not None:
        extra_setup(at)
        at.run()

    assert not at.exception, f"页面「{page_name}」报错：{at.exception}"
    return at


def main() -> int:
    fails: list[str] = []
    ok = 0

    def check(name: str, cond=True, detail: str = "") -> None:
        nonlocal ok
        if cond:
            ok += 1
            print(f"  [OK]   {name}")
        else:
            fails.append(name)
            print(f"  [FAIL] {name}  {detail}")

    # ---- 数据概览 ----
    try:
        at = run_page("数据概览")
        check("数据概览：能渲染", not at.exception, str(at.exception))
        check(f"数据概览：5 个统计卡片（实际 {len(at.metric)}）", len(at.metric) >= 5)
        check(f"数据概览：4 张图（实际 {len(at.get('plotly_chart'))}）",
              len(at.get("plotly_chart")) == 4)
    except Exception as exc:  # noqa: BLE001
        fails.append("数据概览")
        print(f"  [FAIL] 数据概览：{exc}")

    # ---- 模型比较 ----
    try:
        at = run_page("模型比较")
        check("模型比较：能渲染（含后端训练）", not at.exception, str(at.exception))
        check(f"模型比较：≥3 张图（实际 {len(at.get('plotly_chart'))}）",
              len(at.get("plotly_chart")) >= 3)
        check(f"模型比较：≥2 张表（实际 {len(at.dataframe)}）", len(at.dataframe) >= 2)
    except Exception as exc:  # noqa: BLE001
        fails.append("模型比较")
        print(f"  [FAIL] 模型比较：{exc}")

    # ---- 最终评价 ----
    try:
        at = run_page("最终评价（测试集）")
        check("最终评价：能渲染", not at.exception, str(at.exception))
        has_d = any("D 同学锁定的测试集最终结果" in (s.value or "")
                    for s in at.success)
        check("最终评价：读到了 D 的 test_result.json", has_d)
        check(f"最终评价：≥8 个指标卡片（实际 {len(at.metric)}）", len(at.metric) >= 8)
    except Exception as exc:  # noqa: BLE001
        fails.append("最终评价")
        print(f"  [FAIL] 最终评价：{exc}")

    # ---- 客户风险演示 ----
    try:
        at = run_page("客户风险演示")
        check("客户风险演示：能渲染", not at.exception, str(at.exception))
        # 点「查询」按钮
        clicked = False
        for b in at.button:
            if "查询" in (b.label or ""):
                b.click().run()
                clicked = True
                break
        check("客户风险演示：找到并点击查询按钮", clicked)
        check("客户风险演示：查询后无异常", not at.exception, str(at.exception))
        txt = " ".join(str(m.value) for m in at.markdown)
        check("客户风险演示：出现风险概率", "风险概率" in txt, txt[:300])
        check(f"客户风险演示：4 个衍生特征卡片（实际 {len(at.metric)}）",
              len(at.metric) >= 4)
    except Exception as exc:  # noqa: BLE001
        fails.append("客户风险演示")
        print(f"  [FAIL] 客户风险演示：{exc}")

    print()
    print(f"结果：{ok} 项通过，{len(fails)} 项失败")
    if fails:
        for f in fails:
            print("  -", f)
        return 1
    print("前端四页全部通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
