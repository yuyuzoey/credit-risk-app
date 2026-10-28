"""frontend/ui_const.py —— 前端的展示常量 + 纯展示组件（只放标签、列名、样式、HTML 拼装，不含任何模型/指标算法）

为什么这里要写死列名：
    前后端分离后，前端不再 import credit_pipeline，拿不到 cp.COL_LIMIT_BAL 这些常量。
    列名属于"接口契约"的一部分（后端 /schema/fields 也会返回），前端只用来做展示的
    列名映射，不参与任何计算，所以在这里集中声明。

    ⚠️ 如果 B 同学后期改了列名，这里要同步改；后端 /schema/fields 是权威来源。

【UI 优化说明】本文件新增三类内容（原有常量一行未改）：
    1. 蓝金配色常量 + 全局 CSS（GLOBAL_CSS）
    2. 指标通俗释义（METRIC_TIPS，用于悬浮 tooltip）
    3. 纯字符串的 HTML 组件函数（徽章、指标卡、表格、环形仪表盘、空状态…），
       只负责"把后端给的数字摆好看"，不做任何业务计算。
"""

from __future__ import annotations

import html as _html

# ===========================================================================
# 一、原有常量（保持不变）
# ===========================================================================

# ---- 列名（与后端 CSV 一致）----
COL_LIMIT_BAL = "credit_limit_ntd"
COL_SEX = "sex_code"
COL_EDUCATION = "education_code"
COL_MARRIAGE = "marriage_code"
COL_AGE = "age_years"

PAY_COL_1 = "pay_status_2005_09"
BILL_COL_1 = "bill_amount_2005_09_ntd"
PAY_AMT_COL_1 = "paid_amount_2005_09_ntd"

# ---- 模型标签 ----
MODEL_LABELS = {
    "baseline": "基线（全判不违约）",
    "logreg": "逻辑回归",
    "tree": "树模型（梯度提升）",
}
MODEL_LABELS_EN = {
    "Baseline_AllNoDefault": "基线（全判不违约）",
    "LogisticRegression": "逻辑回归",
    "RandomForest": "随机森林",
}

# ---- 字段中文名（展示用）----
FIELD_LABELS = {
    "record_id": "记录编号",
    COL_LIMIT_BAL: "授信额度（新台币）",
    COL_SEX: "性别编码",
    COL_EDUCATION: "教育程度编码",
    COL_MARRIAGE: "婚姻状况编码",
    COL_AGE: "年龄",
    PAY_COL_1: "最近月还款状态",
    BILL_COL_1: "最近月账单金额",
    PAY_AMT_COL_1: "最近月已还金额",
}

# ---- 页面名 ----
PAGES = ["数据概览", "模型比较", "最终评价（测试集）", "客户风险演示"]

# ===========================================================================
# 二、新增：色彩 / 文案常量
# ===========================================================================

# ---- 蓝金配色（金色只做数字、徽章、细边框，禁止整块填充）----
NAVY = "#0F2B4D"       # 主色：标题、模块边框、选中态
GOLD = "#D4AF37"       # 点缀金：仅关键数字 / 角标 / 最优标记
OK_BLUE = "#4A6F9B"    # 正常 / 就绪
WARN_BRONZE = "#B48C48"  # 警告 / 注意（哑光古铜）
RISK_WINE = "#9E4B4B"  # 高风险 / 违约（暗酒红）
TXT = "#233954"
TXT_SUB = "#647894"
TXT_PH = "#889CB8"
BG = "#F0F4F8"
LINE = "#CBD5E1"

# 图表里 train / valid / test 三组的固定配色（海军蓝 → 浅海军蓝 → 冷银灰）
SPLIT_COLORS = {"train": NAVY, "valid": OK_BLUE, "test": "#A8B8CC"}

# 蓝调热力图色阶（混淆矩阵用，金融冷色调，不用红绿）
BLUE_SCALE = [[0.0, "#EEF3F9"], [0.5, "#7E9CC2"], [1.0, NAVY]]

BACKEND_CMD = "uvicorn backend.main:app --reload --port 8000"
SPINNER_TEXT = "正在获取数据，请稍候…"
FOOTER_TEXT = "本系统仅课程教学演示，不可用于真实信贷业务"
TITLE_BADGE = "🔹课程小组作业｜风控模拟系统"
SIDEBAR_BADGE = "📊风控模拟演示"
THRESHOLD_RULE_NOTE = "Recall≥0.6 约束下最优 Precision"
FEATURE_NOTE = "One-Hot 编码会拆分原始字段，单个特征的重要性不等于原始字段的总贡献。"
SENSITIVE_TIP = ("age_years、sex_code、education_code、marriage_code 属于敏感属性，"
                 "仅用于教学演示，不用于真实授信。")

PAGE_ICONS = {
    "数据概览": "📊",
    "模型比较": "🔍",
    "最终评价（测试集）": "🏁",
    "客户风险演示": "👤",
}

# ---- 指标通俗释义（悬浮 tooltip；key 同时覆盖中英文写法）----
METRIC_TIPS = {
    "ROC-AUC": "衡量模型把「会违约」的人排在「不会违约」的人前面的能力。0.5≈瞎猜，越接近 1 越好。",
    "PR-AUC": "在违约客户占少数时，更看重「找得准且找得全」的综合表现，越大越好。",
    "精确率": "模型喊「会违约」的人里，真的违约的比例。越高，误伤越少。",
    "Precision": "模型喊「会违约」的人里，真的违约的比例。越高，误伤越少。",
    "违约类精确率": "模型喊「会违约」的人里，真的违约的比例。越高，误伤越少。",
    "召回率": "真正会违约的人里，被模型抓出来的比例。越高，漏掉的坏客户越少。",
    "Recall": "真正会违约的人里，被模型抓出来的比例。越高，漏掉的坏客户越少。",
    "F1": "精确率和召回率的折中分数，两者都高它才高。",
    "准确率": "所有客户里预测对的比例。违约客户占少数时，这个数字会「虚高」，需配合其他指标看。",
    "Accuracy": "所有客户里预测对的比例。违约客户占少数时，这个数字会「虚高」，需配合其他指标看。",
    "报警率": "被模型判为「会违约」（触发报警）的客户占全部客户的比例。",
    "TP": "真阳性：实际违约，模型也判违约（抓对了）。",
    "FP": "假阳性：实际没违约，模型却判违约（误伤好客户）。",
    "FN": "假阴性：实际违约，模型却判不违约（漏掉坏客户）。",
    "TN": "真阴性：实际没违约，模型也判不违约（放对了）。",
    "误判人数（FP）": "实际没违约却被模型判为违约的人数（误伤）。",
    "漏判人数（FN）": "实际违约却被模型判为不违约的人数（漏网）。",
    "违约率": "样本中下个月发生违约的客户占比。",
    "阈值": "违约概率达到或超过这个数值，就判定为「高风险 / 报警」。",
    "额度使用率": "最近月账单金额 ÷ 授信额度，越高说明越接近用满额度。",
    "还款占账单比": "最近月已还金额 ÷ 最近月账单金额，越低说明还得越少。",
    "最大逾期月数": "近 6 个月里最长的一次逾期月数。",
    "逾期月数个数": "近 6 个月里出现逾期的月份个数。",
    "样本量": "该数据集里包含的客户记录条数。",
    "违约概率": "模型估计的该客户下个月违约的可能性，范围 0～1。",
}

# ---- 列名 → (主标题, 副标题, tooltip键)：模型对比表表头用 ----
CMP_HEADERS = {
    "model": ("model", "模型", None),
    "split": ("split", "数据集", None),
    "roc_auc": ("roc_auc", "ROC-AUC", "ROC-AUC"),
    "pr_auc": ("pr_auc", "PR-AUC", "PR-AUC"),
    "precision": ("precision", "精确率", "精确率"),
    "recall": ("recall", "召回率", "召回率"),
    "f1": ("f1", "F1", "F1"),
    "accuracy": ("accuracy", "准确率", "准确率"),
    "tn": ("tn", "真阴性", "TN"),
    "fp": ("fp", "假阳性", "FP"),
    "fn": ("fn", "假阴性", "FN"),
    "tp": ("tp", "真阳性", "TP"),
}

# ---- 还款状态展示：编码 → (图标, 简称)；含义以后端返回为准，这里只是兜底 ----
PAY_CODE_FALLBACK = {
    -2: "无消费 / 未使用额度",
    -1: "账单已全额还清",
    0: "使用循环信用，最低还款额已付、未逾期",
    1: "逾期 1 个月", 2: "逾期 2 个月", 3: "逾期 3 个月", 4: "逾期 4 个月",
    5: "逾期 5 个月", 6: "逾期 6 个月", 7: "逾期 7 个月",
    8: "逾期 8 个月及以上", 9: "逾期 8 个月及以上",
}


# ===========================================================================
# 三、全局样式（一次性注入）
# ===========================================================================

GLOBAL_CSS = """
<style>
:root{
  --navy:#0F2B4D; --gold:#D4AF37; --ok:#4A6F9B; --warn:#B48C48; --risk:#9E4B4B;
  --txt:#233954; --sub:#647894; --ph:#889CB8; --bg:#F0F4F8; --line:#CBD5E1;
  --shadow:0 4px 16px rgba(15,43,77,.08); --t:240ms;
}
/* ---------- 页面底色 / 字体 ---------- */
.stApp{background:var(--bg);color:var(--txt);
  font-family:-apple-system,"PingFang SC","Microsoft YaHei","Segoe UI",sans-serif;}
.block-container{padding-top:2.4rem;padding-bottom:2rem;max-width:1280px;}
[data-testid="stHeader"]{background:transparent;}
footer{visibility:hidden;}
hr{border-color:var(--line)!important;}
p,li,label{color:var(--txt);}
[data-testid="stCaptionContainer"]{color:var(--sub);}

/* ---------- 卡片：st.container(key="card_xxx") 统一玻璃微拟态 ---------- */
div[class*="st-key-card_"]{
  background:#fff;border-radius:8px;padding:18px;
  border:1px solid rgba(203,213,225,.7);border-left:3px solid var(--navy);
  box-shadow:var(--shadow);backdrop-filter:blur(6px);
  transition:box-shadow var(--t) ease,transform var(--t) ease;}
div[class*="st-key-card_"]:hover{box-shadow:0 6px 20px rgba(15,43,77,.12);}
div[class*="bar-wine"]{border-left:4px solid var(--risk)!important;}
div[class*="bar-navy"]{border-left:4px solid var(--navy)!important;}
div[class*="bar-bronze"]{border-left:4px solid var(--warn)!important;}
div[class*="bar-none"]{border-left:1px solid rgba(203,213,225,.7)!important;}

/* ---------- 标题层级 ---------- */
.ph{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:10px;}
.ph h1{margin:0;padding:0;font-size:1.9rem;font-weight:700;color:var(--navy);letter-spacing:.5px;}
.st-t{display:flex;align-items:center;gap:8px;font-size:1.08rem;font-weight:650;
  color:var(--navy);margin:0 0 10px 0;}
.st-t .st-r{margin-left:auto;}
.foot{font-size:.78rem;color:var(--sub);margin-top:6px;line-height:1.5;}
.sub{font-size:.8rem;color:var(--sub);}

/* ---------- 徽章 ---------- */
.bdg{display:inline-flex;align-items:center;gap:4px;padding:2px 9px;border-radius:999px;
  font-size:.74rem;font-weight:600;line-height:1.5;white-space:nowrap;border:1px solid transparent;}
.bdg-navy{background:rgba(15,43,77,.08);color:var(--navy);border-color:rgba(15,43,77,.18);}
.bdg-ok{background:rgba(74,111,155,.12);color:var(--ok);border-color:rgba(74,111,155,.3);}
.bdg-bronze{background:rgba(180,140,72,.12);color:#8A6A30;border-color:rgba(180,140,72,.4);}
.bdg-wine{background:rgba(158,75,75,.1);color:var(--risk);border-color:rgba(158,75,75,.3);}
.bdg-gold{background:#fff;color:#A8872A;border-color:var(--gold);}   /* 金：只用细边框+文字 */
.bdg-silver{background:rgba(203,213,225,.35);color:var(--sub);border-color:var(--line);}

/* ---------- tooltip（纯 CSS，悬浮 / 键盘聚焦均可弹出） ---------- */
.tip{position:relative;display:inline-flex;align-items:center;gap:3px;cursor:help;}
.tip-i{font-size:.72em;color:var(--ph);transition:color var(--t);}
.tip:hover .tip-i,.tip:focus .tip-i{color:var(--navy);}
.tip::after{content:attr(data-tip);position:absolute;left:50%;bottom:calc(100% + 8px);
  transform:translateX(-50%) translateY(4px);width:max-content;max-width:250px;
  background:var(--navy);color:#EAF0F7;font-size:.78rem;font-weight:400;line-height:1.5;
  padding:8px 11px;border-radius:6px;white-space:normal;text-align:left;
  box-shadow:0 6px 18px rgba(15,43,77,.25);opacity:0;pointer-events:none;
  transition:opacity var(--t) ease,transform var(--t) ease;z-index:9999;}
.tip:hover::after,.tip:focus::after{opacity:1;transform:translateX(-50%) translateY(0);}

/* ---------- 指标卡（横向卡片组） ---------- */
.kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:16px;}
.kpi{background:#fff;border-radius:8px;padding:14px 16px;box-shadow:var(--shadow);
  border:1px solid rgba(203,213,225,.7);border-left:3px solid var(--navy);
  transition:box-shadow var(--t) ease,transform var(--t) ease;}
.kpi:hover{transform:translateY(-1px);box-shadow:0 6px 20px rgba(15,43,77,.12);}
.kpi-lab{font-size:.82rem;color:var(--sub);display:flex;align-items:center;gap:5px;margin-bottom:6px;}
.kpi-val{font-size:1.6rem;font-weight:700;color:var(--navy);line-height:1.2;
  font-variant-numeric:tabular-nums;}
.kpi-val.gold{color:var(--gold);}           /* 金色仅用于核心数字 */
.kpi-sub{font-size:.76rem;color:var(--sub);margin-top:4px;}
.kpi-sub b{color:var(--gold);}
.kpi.dim .kpi-val{color:var(--ph);}

/* ---------- callout 提示框：白底 + 侧边细条 + 角标 ---------- */
.co{display:flex;gap:12px;background:#fff;border-radius:8px;padding:12px 16px;
  box-shadow:var(--shadow);border:1px solid rgba(203,213,225,.7);border-left:4px solid var(--navy);
  font-size:.9rem;line-height:1.65;color:var(--txt);}
.co-warn{border-left-color:var(--warn);} .co-danger{border-left-color:var(--risk);}
.co-info{border-left-color:var(--ok);}
.co-ic{font-size:1.2rem;line-height:1.4;}
.co-t{font-weight:650;color:var(--navy);margin-bottom:2px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;}
.co code,.es code{background:rgba(15,43,77,.06);padding:1px 6px;border-radius:4px;font-size:.84em;}

/* ---------- 空状态卡片 ---------- */
.es{background:#fff;border-radius:8px;padding:28px 24px;text-align:center;
  box-shadow:var(--shadow);border:1px dashed var(--line);border-top:3px solid var(--warn);}
.es-ic{font-size:2.4rem;margin-bottom:6px;}
.es-t{font-size:1.1rem;font-weight:650;color:var(--navy);margin-bottom:6px;}
.es-d{color:var(--sub);font-size:.9rem;line-height:1.7;}
.es-f{margin-top:10px;display:flex;gap:6px;justify-content:center;flex-wrap:wrap;}

/* ---------- 美化表格（替代裸 dataframe） ---------- */
table.tb{width:100%;border-collapse:separate;border-spacing:0;font-size:.86rem;color:var(--txt);}
table.tb th{background:#F6F8FB;color:var(--navy);font-weight:650;text-align:left;
  padding:9px 12px;border-bottom:1px solid var(--line);line-height:1.3;}
table.tb th small{display:block;font-weight:400;color:var(--sub);font-size:.72rem;}
table.tb td{padding:9px 12px;border-bottom:1px solid #E6ECF3;word-break:break-word;
  overflow-wrap:anywhere;font-variant-numeric:tabular-nums;vertical-align:middle;}
table.tb tr:last-child td{border-bottom:none;}
table.tb tbody tr{transition:background var(--t) ease;}
table.tb tbody tr:hover{background:#F6F9FC;}
table.tb td.b{font-weight:700;color:var(--navy);}      /* 核心指标列加粗 */
table.tb tr.hl td{border-top:1px solid var(--navy);border-bottom:1px solid var(--navy);background:#FAFCFE;}
table.tb tr.hl td:first-child{box-shadow:inset 2px 0 0 var(--navy);border-left:1px solid var(--navy);}
table.tb tr.hl td:last-child{border-right:1px solid var(--navy);}

/* ---------- 风险结果卡：左侧 4px 状态条 + 环形仪表盘 ---------- */
.rk{position:relative;display:flex;align-items:center;gap:28px;flex-wrap:wrap;background:#fff;
  border-radius:8px;padding:20px 22px 20px 26px;box-shadow:var(--shadow);
  border:1px solid rgba(203,213,225,.7);overflow:hidden;}
.rk::before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--navy);}
.rk.high::before{background:var(--risk);}
.rk-main{flex:1;min-width:230px;}
.rk-lab{font-size:.85rem;color:var(--sub);display:flex;gap:8px;align-items:center;flex-wrap:wrap;}
.rk-num{font-size:3rem;font-weight:800;color:var(--gold);line-height:1.15;
  font-variant-numeric:tabular-nums;margin:4px 0;}
.rk-res{font-size:.95rem;font-weight:600;color:var(--navy);display:flex;gap:8px;align-items:center;flex-wrap:wrap;}
.ring{--p:0;--c:var(--navy);width:132px;height:132px;border-radius:50%;flex:none;
  background:conic-gradient(var(--c) calc(var(--p)*1%),#E2E8F0 0);
  display:flex;align-items:center;justify-content:center;
  box-shadow:0 0 14px rgba(15,43,77,.12);}
.ring.high{--c:var(--risk);}
.ring-in{width:100px;height:100px;border-radius:50%;background:#fff;display:flex;
  flex-direction:column;align-items:center;justify-content:center;}
.ring-in b{font-size:1.45rem;color:var(--navy);line-height:1.1;}
.ring-in span{font-size:.7rem;color:var(--sub);}

/* ---------- 还款状态小标签 / 行内校验提示 ---------- */
.chips{display:flex;gap:8px;flex-wrap:wrap;}
.hint{font-size:.78rem;color:#8A6A30;margin:-8px 0 8px 0;display:flex;gap:4px;align-items:center;}
.hint.bad{color:var(--risk);}

/* ---------- 侧边栏：深海军蓝基底 ---------- */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0F2B4D 0%,#0B2140 100%);}
[data-testid="stSidebar"] *{color:#DCE6F2;}
[data-testid="stSidebar"] hr{border-color:rgba(203,213,225,.18)!important;}
[data-testid="stSidebar"] code{background:rgba(255,255,255,.1);color:#fff;}
.sb-brand{display:flex;flex-direction:column;gap:8px;margin-bottom:6px;}
.sb-brand .bdg{background:rgba(255,255,255,.08);color:#EAF0F7;border-color:rgba(212,175,55,.55);}
.sb-title{font-size:1.05rem;font-weight:700;color:#fff;}
.sb-row{display:flex;align-items:flex-start;gap:8px;font-size:.8rem;padding:5px 0;line-height:1.4;}
.sb-row .n{flex:1;word-break:break-all;}
.sb-row .s{font-size:.72rem;color:#9FB3CC;}
.sb-pill{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:999px;
  font-size:.76rem;font-weight:600;border:1px solid rgba(255,255,255,.25);}
.sb-pill.ok{background:rgba(74,111,155,.35);}
.sb-pill.bad{background:rgba(180,140,72,.28);border-color:rgba(180,140,72,.6);}
[data-testid="stSidebar"] label[data-baseweb="radio"]{padding:7px 10px;border-radius:6px;
  transition:background var(--t) ease;margin-bottom:2px;}
[data-testid="stSidebar"] label[data-baseweb="radio"]:hover{background:rgba(255,255,255,.07);}
[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked){
  background:rgba(255,255,255,.12);box-shadow:inset 3px 0 0 var(--gold);}

/* ---------- 按钮：悬浮微上浮 + 点击涟漪（克制，220~300ms） ---------- */
[data-testid^="stBaseButton"]{position:relative;overflow:hidden;border-radius:8px;
  transition:transform var(--t) ease,box-shadow var(--t) ease,background var(--t) ease,border-color var(--t) ease;}
[data-testid^="stBaseButton"]:hover{transform:translateY(-1px);box-shadow:0 6px 16px rgba(15,43,77,.18);}
[data-testid^="stBaseButton"]:active{transform:translateY(0);box-shadow:0 2px 6px rgba(15,43,77,.18);}
[data-testid^="stBaseButton"]::after{content:"";position:absolute;inset:0;pointer-events:none;
  background:radial-gradient(circle,rgba(212,175,55,.35) 12%,transparent 13%) center/1200% no-repeat;
  opacity:0;transition:background-size .3s ease,opacity .45s ease;}
[data-testid^="stBaseButton"]:active::after{background-size:0%;opacity:1;transition:0s;}
[data-testid="stBaseButton-primary"]{background:var(--navy);border-color:var(--navy);color:#fff;}
[data-testid="stBaseButton-primary"]:hover{background:#163A66;border-color:#163A66;color:#fff;}
[data-testid="stBaseButton-primary"] p{color:#fff;}
[data-testid="stBaseButton-secondary"]{background:#fff;border:1px solid var(--navy);color:var(--navy);}
[data-testid="stBaseButton-secondary"] p{color:var(--navy);}
[data-testid="stSidebar"] [data-testid^="stBaseButton"]{background:rgba(255,255,255,.06);
  border:1px solid rgba(203,213,225,.4);}
[data-testid="stSidebar"] [data-testid^="stBaseButton"] p{color:#EAF0F7;}
[data-testid="stSidebar"] [data-testid^="stBaseButton"]:hover{background:rgba(255,255,255,.14);}

/* spinner / toast 颜色微调 */
[data-testid="stSpinner"]{color:var(--navy);}
</style>
"""


# ===========================================================================
# 四、纯展示组件（返回 HTML 字符串，配合 st.markdown(..., unsafe_allow_html=True)）
#     注意：HTML 内不能出现空行，否则 Markdown 会把后半段当成代码块。
# ===========================================================================

def esc(x) -> str:
    """HTML 转义，防止字段里的特殊字符破坏页面。"""
    return _html.escape(str(x), quote=True)


def badge(text: str, kind: str = "navy") -> str:
    """小徽章。kind: navy / ok / bronze / wine / gold / silver。"""
    return f'<span class="bdg bdg-{kind}">{esc(text)}</span>'


def tip(label: str, tip_text: str | None = None, *, raw: bool = False) -> str:
    """悬浮释义。tip_text 缺省时按 label 去 METRIC_TIPS 里找；找不到就只显示文字。"""
    inner = label if raw else esc(label)
    text = tip_text or METRIC_TIPS.get(label)
    if not text:
        return inner
    return (f'<span class="tip" tabindex="0" data-tip="{esc(text)}">{inner}'
            f'<span class="tip-i">ⓘ</span></span>')


def page_header(title: str, badge_text: str = TITLE_BADGE) -> str:
    """页面大标题 + 小徽章（课程小组作业）。"""
    return f'<div class="ph"><h1>{esc(title)}</h1>{badge(badge_text, "gold")}</div>'


def section_title(icon: str, text: str, tip_text: str | None = None, right: str = "") -> str:
    """模块子标题：图标 + 文字（+ 可选 ⓘ 提示、右侧角标）。"""
    t = tip(text, tip_text) if tip_text else esc(text)
    r = f'<span class="st-r">{right}</span>' if right else ""
    return f'<div class="st-t"><span>{icon}</span><span>{t}</span>{r}</div>'


def footnote(text: str) -> str:
    """图表 / 表格底部的灰色小字脚注。"""
    return f'<div class="foot">{esc(text)}</div>'


def kpi_card(label: str, value, *, tip_text: str | None = None, icon: str = "",
             sub: str = "", gold: bool = False, dim: bool = False) -> str:
    """单个指标卡。gold=True 时数字用点缀金（仅数字）。sub 为受信任的 HTML。"""
    ic = f"<span>{icon}</span>" if icon else ""
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    cls = "kpi dim" if dim else "kpi"
    vcls = "kpi-val gold" if gold else "kpi-val"
    return (f'<div class="{cls}"><div class="kpi-lab">{ic}{tip(label, tip_text)}</div>'
            f'<div class="{vcls}">{esc(value)}</div>{sub_html}</div>')


def kpi_row(cards: list[str]) -> str:
    """把多张指标卡横向排成一组（自适应列数）。"""
    return f'<div class="kpi-row">{"".join(cards)}</div>'


def callout(kind: str, title: str, body: str, icon: str = "💡", tag: str = "") -> str:
    """提示框。kind: note(海军蓝) / info(浅海军蓝) / warn(古铜) / danger(酒红)。
    body 为受信任 HTML；不整块染色，只用侧边条 + 图标 + 角标。"""
    tg = badge(tag, {"warn": "bronze", "danger": "wine", "info": "ok"}.get(kind, "navy")) if tag else ""
    ttl = f'<div class="co-t">{esc(title)} {tg}</div>' if title else ""
    return (f'<div class="co co-{kind}"><div class="co-ic">{icon}</div>'
            f'<div>{ttl}<div>{body}</div></div></div>')


def empty_state(icon: str, title: str, desc: str, missing: list[str] | None = None) -> str:
    """友好的空状态卡片：图标 + 说明 + 缺失文件徽章。启动命令由 app.py 用 st.code 另外展示（可复制）。"""
    files = ""
    if missing:
        files = '<div class="es-f">' + "".join(badge(f, "bronze") for f in missing) + "</div>"
    return (f'<div class="es"><div class="es-ic">{icon}</div><div class="es-t">{esc(title)}</div>'
            f'<div class="es-d">{desc}</div>{files}</div>')


def table_html(columns: list[str], rows: list[dict], *, headers: dict | None = None,
               bold_cols: tuple = (), cell_fmt=None, row_class=None) -> str:
    """美化表格。
    columns   —— 要显示的列（键名）
    headers   —— {键: (主标题, 副标题, tooltip键)}，缺省则直接用键名
    bold_cols —— 需要加粗高亮的核心指标列
    cell_fmt  —— f(键, 值, 整行) -> 受信任 HTML；返回 None 则用默认转义
    row_class —— f(整行) -> 'hl' 等 CSS 类（用于重要行细边框强调）
    """
    headers = headers or {}
    ths = []
    for c in columns:
        h = headers.get(c)
        if h:
            main, subt, tk = h
            main_html = tip(main, METRIC_TIPS.get(tk)) if tk else esc(main)
            sub_html = f"<small>{esc(subt)}</small>" if subt else ""
            ths.append(f"<th>{main_html}{sub_html}</th>")
        else:
            ths.append(f"<th>{esc(c)}</th>")
    trs = []
    for r in rows:
        tds = []
        for c in columns:
            v = r.get(c)
            cell = cell_fmt(c, v, r) if cell_fmt else None
            if cell is None:
                cell = esc("" if v is None else v)
            tds.append(f'<td class="b">{cell}</td>' if c in bold_cols else f"<td>{cell}</td>")
        rc = row_class(r) if row_class else ""
        trs.append(f'<tr class="{rc}">{"".join(tds)}</tr>')
    return (f'<table class="tb"><thead><tr>{"".join(ths)}</tr></thead>'
            f'<tbody>{"".join(trs)}</tbody></table>')


def ring_html(prob: float, high: bool) -> str:
    """风险环形进度仪表盘：只把后端返回的概率画成圆环（展示映射，不是计算）。"""
    pct = max(0.0, min(float(prob), 1.0)) * 100
    cls = "ring high" if high else "ring"
    return (f'<div class="{cls}" style="--p:{pct:.2f}"><div class="ring-in">'
            f'<b>{pct:.1f}%</b><span>违约概率</span></div></div>')


def risk_card_html(prob: float, threshold, alarm: bool, band: str) -> str:
    """风险结果卡：左侧 4px 状态条（低风险=海军蓝，高风险=暗酒红）+ 金色大号概率 + 环形仪表盘。"""
    cls = "high" if alarm else "low"
    result = (badge("⚠️ 报警（预测违约）", "wine") if alarm else badge("✅ 不报警（预测正常）", "ok"))
    return (
        f'<div class="rk {cls}"><div class="rk-main">'
        f'<div class="rk-lab">{tip("风险概率", METRIC_TIPS["违约概率"])}'
        f'{badge("🏷️ 判定阈值 " + str(threshold), "navy")}{badge("风险分档：" + str(band), "silver")}</div>'
        f'<div class="rk-num">{float(prob):.4f}</div>'
        f'<div class="rk-res">{result}'
        f'<span class="sub">风险概率 {float(prob):.4f} {"≥" if alarm else "<"} 阈值 {esc(threshold)}</span></div>'
        f'</div>{ring_html(prob, alarm)}</div>')


def pay_chip(code, meaning: str = "") -> str:
    """还款状态小标签：正常编码(-2/-1/0)🔵，逾期≥1 🟤，悬浮显示含义。"""
    try:
        c = int(code)
    except (TypeError, ValueError):
        return badge(f"❔ {code}", "silver")
    if c == -2:
        icon, kind = "💤", "ok"
    elif c == -1:
        icon, kind = "✅", "ok"
    elif c == 0:
        icon, kind = "🔵", "ok"
    else:
        icon, kind = "🟤", "bronze"
    text = meaning or PAY_CODE_FALLBACK.get(c, f"未定义取值 {c}")
    return (f'<span class="tip" tabindex="0" data-tip="{esc(text)}">'
            f'{badge(f"{icon} {c}", kind)}</span>')


def pay_kind_badge(code) -> str:
    """还款状态编码表的「状态」列角标。"""
    try:
        c = int(code)
    except (TypeError, ValueError):
        return badge("—", "silver")
    return badge("🔵 正常状态", "ok") if c <= 0 else badge("🟤 逾期", "bronze")


def field_hint(msg: str, bad: bool = False) -> str:
    """表单字段下方的行内校验提示（图标 + 一句话，不抛报错）。"""
    return f'<div class="hint{" bad" if bad else ""}">{"❗" if bad else "⚠️"} {esc(msg)}</div>'


def cm_legend_html() -> str:
    """混淆矩阵下方的 TN/FP/FN/TP 图例，每个都带悬浮释义。"""
    items = [("TN", "真阴性", "ok"), ("FP", "假阳性", "bronze"),
             ("FN", "假阴性", "wine"), ("TP", "真阳性", "navy")]
    chips = "".join(
        f'<span class="tip" tabindex="0" data-tip="{esc(METRIC_TIPS[k])}">{badge(f"{k} {n}", kd)}</span>'
        for k, n, kd in items)
    return f'<div class="chips" style="margin-top:6px">{chips}</div>'


# ---- 侧边栏组件 ----

def sidebar_brand() -> str:
    return (f'<div class="sb-brand"><span class="bdg">{esc(SIDEBAR_BADGE)}</span>'
            f'<div class="sb-title">💳 信用卡违约风险分析</div></div>')


def sidebar_pill(online: bool) -> str:
    return ('<span class="sb-pill ok">🟢 后端已连接</span>' if online
            else '<span class="sb-pill bad">🟠 后端未连接</span>')


def sidebar_status_row(name: str, state: str, note: str) -> str:
    """交付物状态行。state: ok(✅ 浅海军蓝角标) / warn(⚠️ 古铜角标) / miss(❗ 古铜角标)。"""
    icon = {"ok": "✅", "warn": "⚠️", "miss": "❗"}[state]
    kind = "ok" if state == "ok" else "bronze"
    return (f'<div class="sb-row"><span>{icon}</span><div class="n">{esc(name)}'
            f'<div class="s">{esc(note)}</div></div>{badge("就绪" if state == "ok" else "注意", kind)}</div>')


def footer_html() -> str:
    """页脚小字角标：仅课程教学演示。"""
    return (f'<div style="text-align:center;margin:28px 0 6px 0">{badge("🎓 " + FOOTER_TEXT, "silver")}</div>')


# ---- Plotly 统一样式（图表也走蓝金冷色调）----

def style_fig(fig, height: int | None = None, legend: bool = True):
    """统一图表外观：透明底、冷灰网格、海军蓝文字。"""
    fig.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TXT, family="PingFang SC, Microsoft YaHei, sans-serif", size=13),
        margin=dict(l=10, r=10, t=36, b=10), hoverlabel=dict(bgcolor=NAVY, font_color="#fff"),
        colorway=[NAVY, OK_BLUE, "#A8B8CC", WARN_BRONZE],
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        transition=dict(duration=250),
    )
    fig.update_xaxes(gridcolor="#E6ECF3", linecolor=LINE, zerolinecolor=LINE)
    fig.update_yaxes(gridcolor="#E6ECF3", linecolor=LINE, zerolinecolor=LINE)
    if height:
        fig.update_layout(height=height)
    return fig
