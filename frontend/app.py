"""信用卡客户下月违约风险分析 —— 前端（Streamlit）· 蓝金风控仪表盘版

═══ 前后端分离说明（本次 UI 优化后依然成立）═══
本文件是**纯前端**：
    - 不 import credit_pipeline / artifacts_io / sklearn / joblib / pandas
    - 所有数据都通过 frontend/api_client.py 用 HTTP 从后端（backend/main.py）取
    - 页面里只做展示：画图、排版、把后端返回的数字摆出来；不做任何模型运算 / 指标计算

后端启动：
    uvicorn backend.main:app --reload --port 8000
    （接口文档 http://127.0.0.1:8000/docs）

前端启动：
    streamlit run frontend/app.py

自检（应当没有任何输出）：
    grep -r "import credit_pipeline\\|import artifacts_io\\|import sklearn\\|import joblib" frontend/

【UI 优化】样式、徽章、指标卡、表格、环形仪表盘等"展示组件"都放在 ui_const.py，
本文件只负责：取数（走 api_client）→ 摆放组件。每处 UI 改动都有「【UI】」注释。
"""

from __future__ import annotations

from pathlib import Path
import sys

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# 允许 `streamlit run frontend/app.py` 时能 import 同目录的模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

import api_client as api  # noqa: E402
import ui_const as U  # noqa: E402

st.set_page_config(page_title="信用卡违约风险分析", page_icon="💳", layout="wide",
                   initial_sidebar_state="expanded")

# 【UI】一次性注入全局样式（蓝金配色、卡片、按钮微交互、tooltip、侧边栏）
st.markdown(U.GLOBAL_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _fmt(v, nd: int = 4) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _pct(v) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(v)


def _md(html: str) -> None:
    """渲染 HTML 组件的统一出口。"""
    st.markdown(html, unsafe_allow_html=True)


def _card(name: str, bar: str | None = None):
    """【UI】独立卡片容器：白底 / 8px 圆角 / 18px 内边距 / 柔和阴影（样式见 GLOBAL_CSS）。
    bar 可选 navy / wine / bronze，控制卡片左侧状态细条颜色。"""
    key = f"card_{name}" + (f"_bar-{bar}" if bar else "")
    return st.container(key=key)


def _page_title(icon: str, text: str) -> None:
    _md(f'<div class="st-t" style="font-size:1.35rem;margin-top:6px">{icon}<span>{U.esc(text)}</span></div>')


# ---- 【UI】Toast 反馈：跨 st.rerun() 也能弹出（先入队，下一轮运行开头统一弹） ----

def _flash(kind: str, msg: str) -> None:
    st.session_state.setdefault("_flash", []).append((kind, msg))


def _drain_flash() -> None:
    icons = {"success": "✅", "warn": "⚠️", "error": "❗"}
    for kind, msg in st.session_state.pop("_flash", []):
        st.toast(msg, icon=icons.get(kind, "ℹ️"))


def _friendly(e: api.BackendError) -> str:
    """【UI】把后端错误翻译成可读中文，不向用户展示 Python 堆栈。"""
    s = e.status
    if s == 404:
        return "没有找到对应的数据，请确认编号或 artifacts/ 交付物是否已就位。"
    if s == 422:
        return "提交的内容不符合要求，请按表单提示修正后重试。"
    if s == 503:
        return "后端模型暂时不可用，请检查 artifacts/ 或先在「模型比较」页锁定模型。"
    if s is not None and s >= 500:
        return "后端处理时出错了，请查看运行后端的终端日志。"
    return str(e)


def _offline_block(err: api.BackendError) -> None:
    """【UI】后端没起来 → 友好空状态卡片（图标 + 说明 + 启动命令），不抛堆栈。"""
    _md(U.empty_state(
        "🔌", "后端服务未连接，界面暂时无法取数",
        f"请先在项目根目录启动后端，启动后刷新本页即可。<br>当前后端地址：<code>{U.esc(api.BACKEND_URL)}</code>"))
    st.code(U.BACKEND_CMD, language="bash")
    with st.expander("技术详情（可忽略）"):
        st.code(str(err.detail or err), language="text")


def _show_error(e: api.BackendError, *, toast: bool = True) -> None:
    """【UI】统一错误出口：连接类问题显示空状态卡片，其余显示酒红侧边条提示 + toast。"""
    if e.status is None and "未连接" in str(e):
        _offline_block(e)
        if toast:
            st.toast("后端服务未连接，请先启动后端", icon="❗")
        return
    msg = _friendly(e)
    _md(U.callout("danger", "操作未成功", U.esc(msg), icon="❗", tag="错误"))
    if toast:
        st.toast(msg, icon="❗")


def _missing_state(title: str, desc: str, files: list[str]) -> None:
    """【UI】artifacts 文件缺失的空状态卡片：写明缺失文件 + 启动命令。"""
    _md(U.empty_state("📂", title, desc, files))
    st.code(U.BACKEND_CMD, language="bash")


def _sensitive_notice() -> None:
    _md('<div style="height:8px"></div>')
    _md(U.callout(
        "warn", "使用须知",
        "本结果仅为教学示例，不构成授信决定。<br>"
        "模型中使用的性别、婚姻、教育、年龄等属于敏感属性；"
        "相关性不等于因果关系，模型输出不得作为对任何个人或群体的差别对待依据。",
        icon="⚖️", tag="敏感属性"))


# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------

with st.sidebar:
    _md(U.sidebar_brand())      # 【UI】顶部项目徽章「📊风控模拟演示」
    with st.spinner(U.SPINNER_TEXT):
        online, why = api.is_online()
    _md(U.sidebar_pill(online))
    st.caption(f"后端地址：`{api.BACKEND_URL}`")

    if st.button("🔄 重新加载数据", width="stretch"):
        with st.spinner(U.SPINNER_TEXT):
            try:
                api.reload_data()
                st.cache_data.clear()
                st.session_state.clear()
                _flash("success", "已让后端重新读取 CSV 并清空训练结果。")
                st.rerun()
            except api.BackendError as e:
                _show_error(e)

    st.divider()
    # 【UI】页面导航：值仍是原来的 4 个页面名，只在显示时加图标
    page = st.radio("页面导航", U.PAGES,
                    format_func=lambda p: f"{U.PAGE_ICONS.get(p, '')}  {p}")
    st.divider()

    # 【UI】交付物状态：✅ 就绪 / ⚠️ 异常 / ❗ 缺失，附简短说明；数据仍由后端读取 artifacts/ 后告知
    if online:
        try:
            with st.spinner(U.SPINNER_TEXT):
                h = api.health()
            arts = h.get("artifacts") or {}
            ready = sum(1 for ok in arts.values() if ok)
            st.markdown("**队友交付物状态**")
            st.caption(f"artifacts/ 共 {len(arts)} 项，已就绪 {ready} 项"
                       + ("" if ready == len(arts) else "；缺失项请向对应同学索取"))
            for name, ok in arts.items():
                _md(U.sidebar_status_row(name, "ok" if ok else "miss",
                                         "已就绪" if ok else "文件缺失"))
            if h.get("c_model_error"):
                _md(U.sidebar_status_row("C 模型", "warn", f"加载失败：{h['c_model_error']}"))
            if h.get("locked"):
                lk = h["locked"]
                st.caption(
                    f"🔒 已锁定：{U.MODEL_LABELS.get(lk.get('model'), lk.get('model'))}"
                    f"，阈值 {lk.get('threshold')}")
        except api.BackendError:
            pass

_drain_flash()


# ---------------------------------------------------------------------------
# 顶部标题
# ---------------------------------------------------------------------------

_md(U.page_header("信用卡客户下月违约风险分析"))      # 【UI】标题旁小徽章
_md(U.callout("warn", "", "本项目仅作教学分析，不用于真实授信决策。", icon="⚠️", tag="教学演示"))
_md('<div style="height:8px"></div>')

if not online:
    _offline_block(api.BackendError(why.split("。")[0] if why else "后端未连接", detail=why))
    _md(U.footer_html())
    st.stop()


# ===========================================================================
# 页面一：数据概览
# ===========================================================================

def page_overview() -> None:
    _page_title("📊", "数据概览")
    # 【UI】敏感属性提示：💡 角标，悬浮出现说明
    _md(f'<div class="sub">{U.tip("💡 敏感属性提示", U.SENSITIVE_TIP)}</div>')

    with st.spinner(U.SPINNER_TEXT):
        try:
            meta = api.data_meta()
            dist = api.data_distribution()
            pay_codes = api.data_pay_codes()
            util = api.data_default_rate()
        except api.BackendError as e:
            _show_error(e)
            return

    audit = meta["audit"]
    total = meta["total"]

    # 【UI】样本统计 → 横向指标卡片组；违约率数字用点缀金加粗 + 📈
    def _rate_sub(name: str) -> str:
        return f"违约率 <b>{_fmt(audit[name]['default_rate'])}</b>"

    _md(U.kpi_row([
        U.kpi_card("总样本量", f"{total:,}", icon="🗂️", tip_text=U.METRIC_TIPS["样本量"]),
        U.kpi_card("训练集", f"{audit['train']['n']:,}", icon="🧪",
                   tip_text="用来训练模型的数据。", sub=_rate_sub("train")),
        U.kpi_card("验证集", f"{audit['valid']['n']:,}", icon="🧭",
                   tip_text="用来选模型、选阈值的数据。", sub=_rate_sub("valid")),
        U.kpi_card("测试集", f"{audit['test']['n']:,}", icon="🎯",
                   tip_text="只在最后评价一次的数据，不参与调参。", sub=_rate_sub("test")),
        U.kpi_card("整体违约率", _fmt(meta["overall_default_rate"]), icon="📈",
                   tip_text=U.METRIC_TIPS["违约率"], gold=True),
    ]))

    # ---- 按划分分组的统计表 ----
    rows = [{
        "划分": name,
        "样本量": audit[name]["n"],
        "违约人数": audit[name]["default_n"],
        "违约率": audit[name]["default_rate"],
        "缺失值个数": audit[name]["missing_total"],
        "record_id重复数": audit[name]["record_id_dup"],
    } for name in ["train", "valid", "test"]]

    def _stat_cell(k, v, _row):
        if k == "划分":
            return U.badge(v, "navy")
        if k == "违约率":
            return U.esc(_fmt(v))
        if isinstance(v, int):
            return f"{v:,}"
        return None

    with _card("stat_table"):
        _md(U.section_title("🧾", "按划分分组的统计表"))
        _md(U.table_html(
            ["划分", "样本量", "违约人数", "违约率", "缺失值个数", "record_id重复数"], rows,
            headers={"样本量": ("样本量", "", "样本量"), "违约率": ("违约率", "", "违约率")},
            bold_cols=("违约率",), cell_fmt=_stat_cell))
        _md(U.footnote("缺失值只统计原始 CSV 中的真正空值；还款状态 -2 / -1 是有效编码，不计入缺失。"))

    # ---- 图1：授信额度分布（三组叠加）----
    with _card("chart_limit"):
        _md(U.section_title("💰", "授信额度分布"))
        fig = go.Figure()
        for split, cn in [("train", "训练集"), ("valid", "验证集"), ("test", "测试集")]:
            d = dist["by_split"][split]["limit_bal"]
            fig.add_trace(go.Bar(x=d["centers"], y=d["counts"], name=cn, opacity=0.75,
                                 marker_color=U.SPLIT_COLORS[split]))
        fig.update_layout(xaxis_title="授信额度（新台币）", yaxis_title="客户数", barmode="overlay")
        U.style_fig(fig, height=340)
        st.plotly_chart(fig, width="stretch")
        _md(U.footnote(f"授信额度分布（全部 n={total}，横轴按 1%–99% 分位截断）"))

    # ---- 图2：年龄分布 ----
    with _card("chart_age"):
        _md(U.section_title("🎂", "年龄分布"))
        fig = go.Figure()
        for split, cn in [("train", "训练集"), ("valid", "验证集"), ("test", "测试集")]:
            d = dist["by_split"][split]["age"]
            fig.add_trace(go.Bar(x=d["centers"], y=d["counts"], name=cn, opacity=0.75,
                                 marker_color=U.SPLIT_COLORS[split]))
        fig.update_layout(xaxis_title="年龄", yaxis_title="客户数", barmode="overlay")
        U.style_fig(fig, height=340)
        st.plotly_chart(fig, width="stretch")
        _md(U.footnote(f"年龄分布（全部 n={total}）"))

    # ---- 图3：还款状态取值分布 ----
    with _card("chart_pay"):
        _md(U.section_title("🧭", f"最近月还款状态 {dist['pay_status_column']} 取值分布"))
        pbs = dist["pay_status_by_split"]
        fig = go.Figure()
        for split, cn in [("train", "训练集"), ("valid", "验证集"), ("test", "测试集")]:
            fig.add_trace(go.Bar(x=pbs[split]["labels"], y=pbs[split]["counts"],
                                 name=cn, opacity=0.75, marker_color=U.SPLIT_COLORS[split]))
        fig.update_layout(xaxis_title="还款状态取值（负值为有效编码）", yaxis_title="客户数",
                          barmode="overlay")
        U.style_fig(fig, height=340)
        st.plotly_chart(fig, width="stretch")
        _md(U.footnote(f"最近月还款状态取值分布（全部 n={total}）"))

    # ---- 图4：额度使用率分箱 vs 违约率 ----
    with _card("chart_util"):
        _md(U.section_title("📈", "额度使用率分箱 vs 违约率", U.METRIC_TIPS["额度使用率"]))
        tbl = util["table"]
        fig = px.bar(
            x=[r["组值"] for r in tbl], y=[r["违约率"] for r in tbl],
            text=[f"{_pct(r['违约率'])}\n(n={r['样本量']})" for r in tbl])
        fig.update_traces(marker_color=U.NAVY, textposition="outside")
        fig.update_layout(xaxis_title="额度使用率区间", yaxis_title="违约率")
        U.style_fig(fig, height=340, legend=False)
        st.plotly_chart(fig, width="stretch")
        _md(U.footnote(f"额度使用率分箱 vs 违约率（全部 n={util['total']}）。"
                       "额度使用率明显越高，违约率越高 —— 这是衍生特征 utilization_1 的依据。"))

    # ---- 还款状态编码含义 ----
    with _card("pay_codes"):
        _md(U.section_title("🏷️", "还款状态编码含义"))
        # 【UI】callout 高亮重点备注
        _md(U.callout("note", "重点备注",
                      "pay_status 中的 <b>-2、-1、0</b> 是有效业务编码，<b>不是缺失值</b>；负值不作为缺失值处理。",
                      icon="📌", tag="有效编码"))
        _md('<div style="height:10px"></div>')
        table = pay_codes["table"]
        cols = ["取值", "状态"] + [k for k in (table[0].keys() if table else []) if k != "取值"]
        rows_pc = [{**r, "状态": ""} for r in table]

        def _pc_cell(k, v, row):
            if k == "取值":
                return f"<b>{U.esc(v)}</b>"
            if k == "状态":      # 【UI】正常状态🔵蓝色角标，逾期🟤古铜角标；不整格染色
                return U.pay_kind_badge(row["取值"])
            if k == "训练集占比":
                return U.esc(_fmt(v))
            return None

        _md(U.table_html(cols, rows_pc, cell_fmt=_pc_cell))

    _md(U.callout("info", "", "数据无逐笔申请日期，划分为分层随机划分，结论不可外推到未来时段。",
                  icon="ℹ️", tag="说明"))


# ===========================================================================
# 页面二：模型比较
# ===========================================================================

def _cm_fig(cm: list[list]) -> go.Figure:
    """【UI】蓝调热力图混淆矩阵：保留原始数字，方便报告复制。"""
    fig = px.imshow(
        cm, text_auto=True, aspect="auto",
        x=["预测不违约(0)", "预测违约(1)"], y=["实际不违约(0)", "实际违约(1)"],
        color_continuous_scale=U.BLUE_SCALE)
    fig.update_traces(textfont_size=20)
    fig.update_layout(xaxis_title="预测", yaxis_title="实际", coloraxis_showscale=False)
    U.style_fig(fig, height=330, legend=False)
    return fig


def page_model_compare() -> None:
    _page_title("🔍", "模型比较")

    # ---- 上半部分：C 同学交付的团队正式口径（后端读文件，页面不重算）----
    with st.spinner(U.SPINNER_TEXT):
        try:
            art = api.artifacts_comparison("valid", 10)     # 【UI】特征重要性只展示 Top10
        except api.BackendError as e:
            st.toast(f"读取 C 同学交付物失败：{_friendly(e)}", icon="⚠️")
            art = {}

    thr_info = art.get("threshold_info") or {}
    mc = art.get("model_comparison")

    if not thr_info and not mc:
        _missing_state("未找到 C 同学的交付物",
                       "模型比较、阈值曲线、特征重要性都依赖下面的文件，请向 C 同学索取并放入 artifacts/ 后刷新本页。",
                       ["artifacts/model_comparison.csv", "artifacts/threshold.json"])
    else:
        best_model = thr_info.get("model_name") or "RandomForest"
        thr_val = thr_info.get("threshold")

        _md(U.section_title("🏆", "C 同学交付：验证集模型比较"))
        if thr_info:
            _md(U.callout(
                "note", "选定模型与阈值",
                f"C 同学选定模型：<b>{U.esc(U.MODEL_LABELS_EN.get(thr_info.get('model_name'), thr_info.get('model_name')))}</b>，"
                f"阈值 <b>{U.esc(thr_val)}</b>；依据：{U.esc(thr_info.get('selection_reason', '-'))}",
                icon="🧭", tag="验证集选定"))
            _md('<div style="height:12px"></div>')

        if mc:
            # 【UI】美化对比表：最优模型 👑 金色角标 + 左侧 2px 海军蓝强调线；ROC-AUC / Recall / F1 加粗
            def _cmp_cell(k, v, row):
                if k == "model":
                    crown = " " + U.badge("👑 最优", "gold") if row.get("model") == best_model else ""
                    return f"{U.esc(v)}{crown}"
                if k in ("tn", "fp", "fn", "tp"):
                    return f"{int(v):,}" if v is not None else "N/A"
                if k in ("roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy"):
                    return U.esc(_fmt(v))
                return None

            cols = [c for c in U.CMP_HEADERS if c in mc[0]] + \
                   [c for c in mc[0] if c not in U.CMP_HEADERS]
            with _card("cmp_table"):
                _md(U.section_title("📋", "模型对比表（验证集）"))
                _md(U.table_html(
                    cols, mc, headers=U.CMP_HEADERS, bold_cols=("roc_auc", "recall", "f1"),
                    cell_fmt=_cmp_cell,
                    row_class=lambda r: "hl" if r.get("model") == best_model else ""))
                _md(U.footnote("来源：artifacts/model_comparison.csv（split=valid，由 C 同学计算）"))

            # 【UI】混淆矩阵：每个模型一个 tab，蓝调热力图
            with _card("cmp_cm"):
                _md(U.section_title("🧩", "验证集混淆矩阵"))
                labels = [U.MODEL_LABELS_EN.get(r["model"], r["model"]) for r in mc]
                tabs = st.tabs(labels)
                for tab, r in zip(tabs, mc):
                    with tab:
                        cm = [[r["tn"], r["fp"]], [r["fn"], r["tp"]]]
                        st.plotly_chart(_cm_fig(cm), width="stretch", key=f"cm_{r['model']}")
                _md(U.cm_legend_html())
                _md(U.footnote("正类=违约。矩阵中的数字与上方对比表的 tn / fp / fn / tp 一一对应。"))

        tc = art.get("threshold_curve")
        if tc:
            with _card("cmp_thr"):
                _md(U.section_title("📉", "阈值—指标曲线（C 同学，验证集）"))
                fig = go.Figure()
                for col, lab, color in [("precision", "精确率", U.NAVY),
                                        ("recall", "召回率", U.OK_BLUE),
                                        ("f1", "F1", U.WARN_BRONZE)]:
                    fig.add_trace(go.Scatter(
                        x=[r["threshold"] for r in tc], y=[r.get(col) for r in tc],
                        name=lab, mode="lines+markers", line=dict(color=color, width=2),
                        marker=dict(size=5)))
                if thr_val is not None:
                    # 【UI】虚线标注选定阈值，线上附 🏷️ 角标说明选择规则
                    fig.add_vline(
                        x=thr_val, line_dash="dash", line_color=U.GOLD, line_width=1.6,
                        annotation_text=f"🏷️ 选定业务阈值：{thr_val}<br>{U.THRESHOLD_RULE_NOTE}",
                        annotation_position="top", annotation_font_color=U.NAVY,
                        annotation_bgcolor="rgba(255,255,255,.9)", annotation_bordercolor=U.GOLD)
                fig.update_layout(
                    xaxis_title=f"判定阈值（C 同学只保存了 {len(tc)} 个候选点）", yaxis_title="指标值")
                U.style_fig(fig, height=380)
                fig.update_layout(margin=dict(l=10, r=10, t=70, b=10),
                                  legend=dict(orientation="h", yanchor="bottom", y=1.18, x=0))
                st.plotly_chart(fig, width="stretch")
                _md(U.footnote("来源：artifacts/threshold_comparison.csv"))

        fi = art.get("feature_importance")
        if fi:
            with _card("cmp_fi"):
                _md(U.section_title("⭐", "特征重要性 Top10（C 同学的随机森林）"))
                fi_sorted = sorted(fi, key=lambda r: r["importance_or_coefficient"])
                vals = [r["importance_or_coefficient"] for r in fi_sorted]
                fig = go.Figure(go.Bar(
                    x=vals, y=[r["feature"] for r in fi_sorted], orientation="h",
                    marker=dict(color=vals, colorscale=[[0, "#9FB5D0"], [1, U.NAVY]])))
                fig.update_layout(xaxis_title="重要性", yaxis_title="特征（编码后）")
                U.style_fig(fig, height=400, legend=False)
                st.plotly_chart(fig, width="stretch")
                _md(U.footnote("💬 " + U.FEATURE_NOTE + "　来源：artifacts/feature_importance.csv"))

    # ---- 下半部分：后端现场训练（通过 API 拿结果）----
    _md('<div style="height:4px"></div>')
    _md(U.section_title("🛠️", "后端在线复算（后端在 train 上现场训练）"))
    _md(U.callout("info", "",
                  "此处的模型由后端在训练集上现场拟合，参数与 C 同学的随机森林不同，指标会有小差异；"
                  "团队正式口径以上方 C 同学的交付物为准。", icon="ℹ️", tag="仅供对照"))
    _md('<div style="height:8px"></div>')

    with st.spinner(U.SPINNER_TEXT):
        try:
            locked_payload = api.get_lock()
        except api.BackendError as e:
            _show_error(e)
            return
    locked = locked_payload.get("locked")
    is_locked = locked is not None

    # 训练结果缓存在 session_state（一次训练要几秒，切页面不必重训）
    if "train" not in st.session_state:
        with st.spinner("正在获取数据，请稍候…（后端训练中：训练集 2 万条 × 三模型）"):
            try:
                st.session_state["train"] = api.train()
                st.toast("后端训练完成", icon="✅")
            except api.BackendError as e:
                _show_error(e)
                return
    tr = st.session_state["train"]
    job_id = tr["job_id"]
    curves = {name: m["threshold_curve"] for name, m in tr["models"].items()}
    metrics05 = {name: m["metrics_at_0_5"] for name, m in tr["models"].items()}

    _md(U.footnote(f"本次训练 job_id={job_id}　特征数={tr['n_features']}　"
                   f"训练集 n={tr['n_train']}　验证集 n={tr['n_valid']}　"
                   f"验证集违约率={_fmt(tr['default_rate_valid'])}"))

    if is_locked:
        model_name = locked["model"]
        threshold = float(locked["threshold"])
        _md(U.callout(
            "note", "已锁定",
            f"模型={U.esc(U.MODEL_LABELS.get(model_name, model_name))}，阈值={threshold}。"
            "如需更改请先在「最终评价」页点击重置（将清空测试结果）。",
            icon="🔒", tag="已锁定"))
        reason = locked.get("reason", "")
    else:
        with _card("pick"):
            _md(U.section_title("🎛️", "选择模型与阈值准则（仅用验证集）", U.METRIC_TIPS["阈值"]))
            model_name = st.selectbox(
                "选择模型", ["baseline", "logreg", "tree"], index=1,
                format_func=lambda x: U.MODEL_LABELS[x])
            criterion = st.radio("选择准则", ["max_f1", "recall_ge", "min_cost"], horizontal=True)
            recall_target = 0.60
            cost_ratio = (5.0, 1.0)
            if criterion == "recall_ge":
                recall_target = float(st.slider("目标召回率", 0.30, 0.95, 0.60, 0.05))
            elif criterion == "min_cost":
                fn_cost = st.slider("漏判成本（误判成本固定为 1）", 1, 20, 5)
                cost_ratio = (float(fn_cost), 1.0)

            with st.spinner(U.SPINNER_TEXT):
                try:
                    picked = api.pick_threshold(job_id, model_name, criterion,
                                                recall_target, cost_ratio)
                except api.BackendError as e:
                    _show_error(e)
                    picked = {"threshold": None, "reason": _friendly(e)}

            threshold = picked.get("threshold")
            reason = picked.get("reason", "")
            if threshold is None:
                _md(U.callout(
                    "warn", "无法选出有意义阈值",
                    f"当前模型「{U.esc(U.MODEL_LABELS[model_name])}」无法选出有意义阈值：{U.esc(reason)}。"
                    "已临时按 0.5 展示，请在上方切换到「逻辑回归」或「树模型」后再选阈值。",
                    icon="⚠️", tag="注意"))
                threshold = 0.5
            else:
                _md(U.callout("note", "",
                              f"<b>选中阈值 {U.esc(threshold)}</b>，依据：{U.esc(reason)}",
                              icon="🎯", tag="已选阈值"))

    curve = curves.get(model_name, [])
    m_05 = metrics05.get(model_name, {})
    m_sel = api.metrics_at(curve, threshold) if curve else {}

    with _card("val_metrics"):
        _md(U.section_title("📏", "验证集指标对比"))
        col1, col2 = st.columns(2)
        for col, head, m in [(col1, "阈值 0.5", m_05), (col2, f"选定阈值 {threshold}", m_sel)]:
            with col:
                _md(f'<div style="font-weight:650;color:{U.NAVY};margin-bottom:8px">{U.esc(head)}</div>')
                _md(U.kpi_row([
                    U.kpi_card("ROC-AUC", _fmt(m.get("roc_auc"))),
                    U.kpi_card("精确率", _fmt(m.get("precision"))),
                    U.kpi_card("召回率", _fmt(m.get("recall"))),
                    U.kpi_card("F1", _fmt(m.get("f1"))),
                ]))
                if m.get("reason"):
                    _md(U.footnote(f"原因：{m['reason']}"))

    with _card("val_curve"):
        _md(U.section_title("📉", "阈值—指标曲线（验证集）"))
        if curve:
            fig = go.Figure()
            for col_name, label, color in [("precision", "精确率", U.NAVY),
                                           ("recall", "召回率", U.OK_BLUE),
                                           ("f1", "F1", U.WARN_BRONZE)]:
                fig.add_trace(go.Scatter(
                    x=[r["threshold"] for r in curve], y=[r[col_name] for r in curve],
                    name=label, mode="lines", line=dict(color=color, width=2)))
            fig.add_vline(x=threshold, line_dash="dash", line_color=U.GOLD, line_width=1.6,
                          annotation_text=f"🏷️ 阈值={threshold}", annotation_font_color=U.NAVY)
            fig.update_layout(xaxis_title="判定阈值", yaxis_title="指标值")
            U.style_fig(fig, height=360)
            st.plotly_chart(fig, width="stretch")
            _md(U.footnote(f"阈值—精确率/召回率/F1 曲线（验证集 n={tr['n_valid']}）"))
        else:
            _md(U.footnote("该模型无可用的阈值曲线。"))

    with _card("val_cmp3"):
        _md(U.section_title("📋", "三模型验证集对比（阈值 0.5）"))
        cmp_rows = [{
            "模型": U.MODEL_LABELS[name],
            "AUC": metrics05[name].get("roc_auc"),
            "精确率": metrics05[name].get("precision"),
            "召回率": metrics05[name].get("recall"),
            "F1": metrics05[name].get("f1"),
            "训练集口径": "—",
        } for name in ["baseline", "logreg", "tree"]]
        sel_label = U.MODEL_LABELS[model_name]

        def _c3(k, v, _row):
            if k in ("AUC", "精确率", "召回率", "F1"):
                return U.esc(_fmt(v))
            return None

        _md(U.table_html(
            ["模型", "AUC", "精确率", "召回率", "F1", "训练集口径"], cmp_rows,
            headers={"AUC": ("AUC", "", "ROC-AUC"), "精确率": ("精确率", "", "精确率"),
                     "召回率": ("召回率", "", "召回率"), "F1": ("F1", "", "F1")},
            bold_cols=("AUC", "召回率", "F1"), cell_fmt=_c3,
            row_class=lambda r: "hl" if r["模型"] == sel_label else ""))
        _md(U.footnote(f"当前选中的模型：{sel_label}"))

    # ---- 锁定 ----
    if not is_locked:
        with _card("lock", "bronze"):
            _md(U.section_title("🔒", "锁定模型与阈值"))
            _md(U.footnote("锁定后测试集评价使用该模型与阈值，且在「最终评价」页重置前不可更改 —— "
                           "这是课程要求的\"测试集只评价一次\"的执行保障。"))
            if st.button("🔒 锁定模型与阈值", type="primary"):
                with st.spinner(U.SPINNER_TEXT):
                    try:
                        api.lock(model_name, float(threshold), reason)
                        _flash("success", f"已锁定：模型={U.MODEL_LABELS[model_name]}，阈值={threshold}")
                        st.rerun()
                    except api.BackendError as e:
                        _show_error(e)
    else:
        _md(U.footnote("已锁定，如需更改请先在「最终评价」页点击重置（将清空测试结果）。"))


# ===========================================================================
# 页面三：最终评价（测试集）
# ===========================================================================

def _dash(v, nd: int = 4) -> str:
    """基线侧后端没给数据时显示「—」。"""
    return "—" if v is None else _fmt(v, nd)


def page_final_evaluate() -> None:
    _page_title("🏁", "最终评价（测试集）")

    # 【UI】⚠️ 警告 callout：阈值来自验证集，测试集只评价一次
    _md(U.callout(
        "warn", "评价规则",
        "阈值来自<b>验证集</b>；测试集仅做<b>一次性最终评估</b>，<b>没有参与调参</b>。"
        "不得反复调参后把测试集当作独立评估。",
        icon="⚠️", tag="重要"))
    _md('<div style="height:8px"></div>')

    # ---- D 同学交付：锁定的测试集最终结果（只读，页面不重算）----
    with st.spinner(U.SPINNER_TEXT):
        try:
            d = api.test_result()
        except api.BackendError:
            d = None

    if d is not None:
        raw = d["raw"]
        m_d = d["metrics"]
        _md(U.callout(
            "note", "D 同学锁定的测试集最终结果",
            "D 同学锁定的测试集最终结果（artifacts/test_result.json，只评价一次，本页直接读取，不重算）<br>"
            f"<b>模型</b>：{U.esc(U.MODEL_LABELS_EN.get(raw.get('model_name'), raw.get('model_name')))}　"
            f"<b>阈值</b>：{U.esc(raw.get('threshold'))}（来源：{U.esc(raw.get('threshold_source'))}）　"
            f"<b>测试集样本量</b>：n={U.esc(m_d['n_test'])}<br>"
            f"<b>评价时间</b>：{U.esc(raw.get('evaluated_at'))}　|　"
            f"<b>数据指纹</b>：<code>{U.esc(raw.get('data_fingerprint'))}</code>　|　"
            f"<b>代码版本</b>：{U.esc(raw.get('code_version'))}",
            icon="🔐", tag="已锁定 · 只评价一次"))
        if not raw.get("feature_order_verified", True):
            _md('<div style="height:8px"></div>')
            _md(U.callout("warn", "特征顺序未核对",
                          f"D 同学标注 feature_order_verified=false。原文说明：{U.esc(raw.get('feature_note'))}",
                          icon="⚠️", tag="注意"))
        if raw.get("unavailable_reasons"):
            _md('<div style="height:8px"></div>')
            _md(U.callout("info", "", f"不可计算的指标：{U.esc(raw['unavailable_reasons'])}",
                          icon="ℹ️", tag="说明"))
        _md('<div style="height:8px"></div>')

        # 【UI】基线 vs 随机森林并排对比（金色仅用于关键数字）
        base = raw.get("baseline") or {}
        col_b, col_r = st.columns(2)
        with col_b:
            with _card("test_base", "bronze"):
                _md(U.section_title("⚪", "基线（全判不违约）", right=U.badge("对照组", "silver")))
                _md(U.kpi_row([
                    U.kpi_card("ROC-AUC", _dash(base.get("roc_auc")), dim=True),
                    U.kpi_card("违约类精确率", _dash(base.get("precision")), dim=True),
                    U.kpi_card("召回率", _dash(base.get("recall")), dim=True),
                    U.kpi_card("F1", _dash(base.get("f1")), dim=True),
                    U.kpi_card("报警率", _dash(base.get("alarm_rate")), dim=True),
                    U.kpi_card("准确率", _dash(base.get("accuracy")), dim=True),
                ]))
                if not base:
                    _md(U.footnote("后端暂未提供基线模型的测试集指标（test_result.json 里只有锁定模型的结果），"
                                   "此处不做前端估算，故显示「—」。"))
        with col_r:
            with _card("test_rf", "navy"):
                _md(U.section_title("🌲", U.MODEL_LABELS_EN.get(raw.get("model_name"), "随机森林"),
                                    right=U.badge("锁定模型", "navy")))
                _md(U.kpi_row([
                    U.kpi_card("ROC-AUC", _fmt(m_d["roc_auc"]), gold=True),
                    U.kpi_card("违约类精确率", _fmt(m_d["precision"]), gold=True),
                    U.kpi_card("召回率", _fmt(m_d["recall"]), gold=True),
                    U.kpi_card("F1", _fmt(m_d["f1"]), gold=True),
                    U.kpi_card("报警率", _fmt(m_d["alarm_rate"])),
                    U.kpi_card("准确率", _fmt(m_d["accuracy"])),
                ]))

        _md('<div style="height:2px"></div>')
        _md(U.kpi_row([
            U.kpi_card("误判人数（FP）", m_d["误判人数"], icon="🟤",
                       tip_text=U.METRIC_TIPS["误判人数（FP）"]),
            U.kpi_card("漏判人数（FN）", m_d["漏判人数"], icon="🔻",
                       tip_text=U.METRIC_TIPS["漏判人数（FN）"]),
        ]))

        cm = [[m_d["TN"], m_d["FP"]], [m_d["FN"], m_d["TP"]]]
        with _card("test_cm"):
            _md(U.section_title("🧩", f"混淆矩阵（测试集 n={m_d['n_test']}，D 同学交付）"))
            st.plotly_chart(_cm_fig(cm), width="stretch", key="cm_test")
            _md(U.cm_legend_html())
            _md(U.footnote("正类=违约。测试集仅评价一次，不得反复调参后当作独立评估。"))
    else:
        _missing_state("未找到测试集评价结果",
                       "请向 D 同学索取测试集评价结果，放入 artifacts/ 后刷新本页。",
                       ["artifacts/test_result.json"])

    # ---- 后端现场锁定流程的说明 ----
    _md('<div style="height:4px"></div>')
    _md(U.section_title("🧪", "后端的交互式选型与锁定"))
    with st.spinner(U.SPINNER_TEXT):
        try:
            locked = api.get_lock().get("locked")
        except api.BackendError as e:
            _show_error(e)
            return

    if locked is None:
        _md(U.callout(
            "warn", "后端尚未锁定模型",
            "请先在「模型比较」页锁定模型与阈值。<br>"
            "说明：后端锁定 ≠ D 同学的 test_result.json。D 的结果是团队正式口径的最终成品；"
            "后端锁定用于演示\"先验证集选型、再测试集评价一次\"的完整流程。",
            icon="🔓", tag="未锁定"))
        return

    with _card("lock_info", "navy"):
        _md(f"<b>锁定模型</b>：{U.esc(U.MODEL_LABELS.get(locked.get('model'), locked.get('model')))}　"
            f"<b>阈值</b>：{U.esc(locked.get('threshold'))}")
        _md(U.footnote("锁定已记录在 results/locked_model.json。"
                       "测试集指标请以上方 D 同学的 test_result.json 为团队正式口径。"))
        if locked.get("reason"):
            _md(U.footnote(f"选阈值依据：{locked['reason']}"))

    with _card("reset", "bronze"):
        _md(U.section_title("♻️", "重置"))
        confirm = st.checkbox("我确认要重置（将清空后端锁定与训练结果）")
        if st.button("重置锁定（将删除测试结果并允许重新选型）"):
            if not confirm:
                st.toast("请先勾选上面的确认框。", icon="⚠️")
                _md(U.field_hint("请先勾选上面的确认框，再点击重置。", bad=True))
            else:
                with st.spinner(U.SPINNER_TEXT):
                    try:
                        api.unlock()
                        st.session_state.clear()
                        _flash("success", "已重置，请重新在「模型比较」页选型与锁定。")
                        st.rerun()
                    except api.BackendError as e:
                        _show_error(e)


# ===========================================================================
# 页面四：客户风险演示
# ===========================================================================

def _pay_meaning() -> dict[int, str]:
    """还款状态编码含义：以后端返回为准，取不到时用 ui_const 里的兜底文案。"""
    if "pay_meaning" not in st.session_state:
        m = dict(U.PAY_CODE_FALLBACK)
        try:
            for r in api.data_pay_codes()["table"]:
                m[int(r["取值"])] = r["含义"]
        except (api.BackendError, KeyError, ValueError, TypeError):
            pass
        st.session_state["pay_meaning"] = m
    return st.session_state["pay_meaning"]


def _month_label(col: str) -> str:
    """pay_status_2005_09 / bill_amount_2005_09_ntd → 2005年09月"""
    parts = [p for p in col.split("_") if p.isdigit()]
    return f"{parts[0]}年{parts[1]}月" if len(parts) >= 2 else col


def _custom_form() -> None:
    """【UI】自定义输入表单：非法值行内即时提示（图标 + 一句话），有问题时按钮置灰，不抛报错。"""
    with st.spinner(U.SPINNER_TEXT):
        try:
            schema = api.schema_fields()
        except api.BackendError as e:
            _show_error(e)
            return
    nf = {f["name"]: f for f in schema["numeric_fields"]}
    pay_cols, bill_cols, amt_cols = schema["pay_cols"], schema["bill_cols"], schema["pay_amt_cols"]
    meaning = {int(k): v for k, v in schema["pay_code_meaning"].items()}
    codes = sorted(meaning)
    errs: list[str] = []

    def _range_hint(name: str, val: float, label: str) -> None:
        f = nf.get(name)
        if f is not None and (val < f["min"] or val > f["max"]):
            _md(U.field_hint(f"{label}超出样本范围（{f['min']:g} ～ {f['max']:g}），请调整。", bad=True))
            errs.append(name)

    with _card("custom_form"):
        _md(U.section_title("📝", "手动输入客户资料"))
        c1, c2, c3 = st.columns(3)
        with c1:
            limit = st.number_input("授信额度（新台币）", value=float(nf[U.COL_LIMIT_BAL]["median"]),
                                    step=1000.0, format="%.0f", key="cu_limit")
            if limit <= 0:
                _md(U.field_hint("授信额度必须大于 0。", bad=True))
                errs.append("limit_pos")
            else:
                _range_hint(U.COL_LIMIT_BAL, limit, "授信额度")
            age = st.number_input("年龄", value=int(nf[U.COL_AGE]["median"]), step=1, key="cu_age")
            _range_hint(U.COL_AGE, age, "年龄")
        with c2:
            sex = st.number_input("性别编码", value=int(nf[U.COL_SEX]["median"]), step=1, key="cu_sex")
            _range_hint(U.COL_SEX, sex, "性别编码")
            edu = st.number_input("教育程度编码", value=int(nf[U.COL_EDUCATION]["median"]), step=1,
                                  key="cu_edu")
            _range_hint(U.COL_EDUCATION, edu, "教育程度编码")
        with c3:
            mar = st.number_input("婚姻状况编码", value=int(nf[U.COL_MARRIAGE]["median"]), step=1,
                                  key="cu_mar")
            _range_hint(U.COL_MARRIAGE, mar, "婚姻状况编码")
            _md(f'<div class="sub" style="margin-top:6px">{U.tip("💡 敏感属性提示", U.SENSITIVE_TIP)}</div>')

        t_pay, t_bill, t_amt = st.tabs(["🗓️ 近 6 月还款状态", "🧾 近 6 月账单金额", "💵 近 6 月已还金额"])
        pay_vals, bill_vals, amt_vals = [], [], []
        with t_pay:
            cols = st.columns(3)
            for i, col in enumerate(pay_cols):
                with cols[i % 3]:
                    pay_vals.append(int(st.selectbox(
                        _month_label(col), codes, index=codes.index(0) if 0 in codes else 0,
                        format_func=lambda c: f"{c}｜{meaning[c]}", key=f"cu_pay_{i}")))
        with t_bill:
            _md(U.footnote("账单金额允许为负（多还了款会出现负账单），不做非负限制。"))
            cols = st.columns(3)
            for i, col in enumerate(bill_cols):
                with cols[i % 3]:
                    bill_vals.append(float(st.number_input(
                        _month_label(col), value=0.0, step=1000.0, format="%.0f", key=f"cu_bill_{i}")))
        with t_amt:
            cols = st.columns(3)
            for i, col in enumerate(amt_cols):
                with cols[i % 3]:
                    v = float(st.number_input(
                        _month_label(col), value=0.0, step=1000.0, format="%.0f", key=f"cu_amt_{i}"))
                    if v < 0:
                        _md(U.field_hint("已还金额不能为负数。", bad=True))
                        errs.append(f"amt_{i}")
                    amt_vals.append(v)

        if errs:
            _md(U.callout("warn", "", f"有 <b>{len(errs)}</b> 项输入需要修正，修正后即可计算。",
                          icon="⚠️", tag="待修正"))
        if st.button("🧮 计算违约风险", type="primary", disabled=bool(errs), key="cu_go"):
            payload = {
                "credit_limit_ntd": float(limit), "sex_code": int(sex), "education_code": int(edu),
                "marriage_code": int(mar), "age_years": int(age),
                "pay_status": pay_vals, "bill_amount": bill_vals, "paid_amount": amt_vals,
            }
            with st.spinner(U.SPINNER_TEXT):
                try:
                    st.session_state["demo_result"] = api.predict_custom(payload)
                    st.session_state["demo_inputs"] = {
                        "payload": payload, "pay_cols": pay_cols}
                    st.toast("✅客户风险预测计算完成", icon="✅")
                except api.BackendError as e:
                    _show_error(e)


def _render_result(result: dict) -> None:
    prob = result["probability"]
    threshold = result["threshold"]
    meaning = _pay_meaning()

    # 【UI】风险结果卡：左侧 4px 状态侧边条 + 金色大号概率 + 🏷️阈值角标 + 环形仪表盘
    _md(U.risk_card_html(prob, threshold, bool(result["alarm"]), result["risk_band"]))
    _md(U.footnote(f"预测来源：{result['model_source']}"))

    # ---- 主要输入信息（原始字段 + 后端算好的衍生特征）----
    with _card("cust_info"):
        _md(U.section_title("🧑‍💼", "主要输入信息"))
        card = result.get("record") or {}
        info = card.get("info") or {}
        custom = st.session_state.get("demo_inputs") if result.get("split") == "custom" else None

        if info:
            rows = [{"字段": U.FIELD_LABELS.get(k, k), "取值": v, "_k": k} for k, v in info.items()]
        elif custom:
            p = custom["payload"]
            rows = [{"字段": U.FIELD_LABELS.get(k, k), "取值": p[k], "_k": k}
                    for k in (U.COL_LIMIT_BAL, U.COL_SEX, U.COL_EDUCATION, U.COL_MARRIAGE, U.COL_AGE)]
        else:
            rows = []

        def _info_cell(k, v, row):
            if k == "取值" and row["_k"] == U.PAY_COL_1:      # 【UI】还款状态配图标角标
                return f"{U.esc(v)} {U.pay_chip(v, meaning.get(int(v), ''))}"
            return None

        _md(U.table_html(["字段", "取值"], rows, cell_fmt=_info_cell))

        if custom:
            _md('<div style="height:10px"></div>')
            _md('<div class="sub" style="margin-bottom:6px">近 6 个月还款状态（从最新月份开始）</div>')
            chips = "".join(
                f'<span>{U.esc(_month_label(c))}&nbsp;{U.pay_chip(v, meaning.get(int(v), ""))}</span>'
                for c, v in zip(custom["pay_cols"], custom["payload"]["pay_status"]))
            _md(f'<div class="chips">{chips}</div>')

    derived = card.get("derived") or {}
    if derived:
        _md(U.section_title("🧮", "衍生特征", "由后端统一计算，前后端不重复实现公式"))
        _md(U.kpi_row([
            U.kpi_card("额度使用率", derived.get("额度使用率")),
            U.kpi_card("还款占账单比", derived.get("还款占账单比")),
            U.kpi_card("最大逾期月数", derived.get("最大逾期月数")),
            U.kpi_card("逾期月数个数", derived.get("逾期月数个数")),
        ]))

    # ---- 事后真实标签 ----
    split = result.get("split")
    true_label = result.get("true_label")
    if split == "test" and true_label is not None:
        with _card("cust_label", "bronze"):
            _md(U.section_title("🧷", "事后真实标签（仅用于对照，不参与预测）"))
            _md(f"<b>真实标签</b>：{true_label}（{'违约' if true_label == 1 else '正常'}）")
            _md(U.footnote("此标签为事后结果，仅用于评估模型预测准确性，不得用于调整任何参数。"))
    else:
        _md(U.footnote(result.get("label_note") or f"该记录来自 {split} 集，无事后标签展示。"))


def page_customer_demo() -> None:
    _page_title("👤", "客户风险演示")

    with st.spinner(U.SPINNER_TEXT):
        try:
            h = api.health()
        except api.BackendError as e:
            _show_error(e)
            return
    locked = h.get("locked")
    has_c_model = bool(h.get("c_model")) and not h.get("c_model_error")

    if has_c_model:
        _md(U.callout(
            "note", "当前预测模型",
            f"当前使用 <b>C 同学交付的正式模型</b>（{U.esc(h.get('c_model'))}），"
            f"阈值 = {U.esc(h.get('c_threshold'))}。模型放在后端，前端只拿预测结果。",
            icon="🧠", tag="正式模型"))
    else:
        _md(U.callout(
            "warn", "正式模型不可用",
            "C 同学的正式模型不可用，预测会退回后端现场训练模型。"
            f"原因：{U.esc(h.get('c_model_error') or '未找到 artifacts/best_model.pkl')}",
            icon="⚠️", tag="已降级"))
        if locked is None:
            # 【UI】模型 / 后端都不可用 → 友好空状态卡片
            _missing_state("暂时无法预测",
                           "后端也尚未锁定模型，无法预测。请先到「模型比较」页锁定，或补齐下面的文件。",
                           ["artifacts/best_model.pkl"])
            _sensitive_notice()
            return
    _md('<div style="height:8px"></div>')

    with _card("pick_customer"):
        _md(U.section_title("🎯", "选择客户"))
        method = st.radio("取数方式", ["输入 record_id", "从测试集随机抽取", "手动输入客户资料"],
                          horizontal=True)

        if method == "输入 record_id":
            rid = st.number_input("record_id", min_value=1, value=1, step=1)
            if st.button("🔍 查询", type="primary"):
                with st.spinner(U.SPINNER_TEXT):
                    try:
                        st.session_state["demo_result"] = api.predict_record(int(rid))
                        st.session_state.pop("demo_inputs", None)
                        st.toast("✅客户风险预测计算完成", icon="✅")
                    except api.BackendError as e:
                        if e.status == 404:
                            st.toast(f"record_id={int(rid)} 不存在，请确认编号。", icon="⚠️")
                            _md(U.field_hint(f"record_id={int(rid)} 不存在，请确认编号。"))
                        else:
                            _show_error(e)
        elif method == "从测试集随机抽取":
            if st.button("🎲 随机抽取一条测试集记录", type="primary"):
                with st.spinner(U.SPINNER_TEXT):
                    try:
                        st.session_state["demo_result"] = api.random_test_record()
                        st.session_state.pop("demo_inputs", None)
                        st.toast("✅客户风险预测计算完成", icon="✅")
                    except api.BackendError as e:
                        _show_error(e)

    if method == "手动输入客户资料":
        _custom_form()

    result = st.session_state.get("demo_result")
    if result is None:
        _md(U.footnote("请选择客户后查看风险演示。"))
        _sensitive_notice()
        return

    _render_result(result)
    _sensitive_notice()


# ---------------------------------------------------------------------------
# 页面分发
# ---------------------------------------------------------------------------

if page == "数据概览":
    page_overview()
elif page == "模型比较":
    page_model_compare()
elif page == "最终评价（测试集）":
    page_final_evaluate()
elif page == "客户风险演示":
    page_customer_demo()

# 【UI】页脚小字角标
_md(U.footer_html())
