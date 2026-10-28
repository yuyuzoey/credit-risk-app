"""信用卡客户下月违约风险分析 —— 数据管线（算法/数据层）。

本文件只负责数据处理、建模、评估，不包含任何界面代码。
界面层 app.py 通过 import 本模块的函数来展示结果。

当前实现：Step 1 —— 项目骨架与数据加载。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

# 标签列：1 = 下月违约，0 = 正常
LABEL = "default_next_month"

# 绝对不能进入特征矩阵的列（标识列 + 划分列 + 标签列）
EXCLUDE_COLS = ["record_id", "split", "default_next_month"]

# 数据集划分列
SPLIT_COL = "split"

# ---- 实际 CSV 列名（本数据集列名与 UCI 经典命名不同，统一在此声明）----
COL_LIMIT_BAL = "credit_limit_ntd"          # 授信额度（新台币，教学用途仅作数值）
COL_SEX = "sex_code"                        # 性别（敏感属性）
COL_EDUCATION = "education_code"            # 教育程度（敏感属性）
COL_MARRIAGE = "marriage_code"              # 婚姻状况（敏感属性）
COL_AGE = "age_years"                       # 年龄（敏感属性）

# 近 6 个月还款状态（第 1 个为最近月，依次往前；负值是有效编码，非缺失）
PAY_COLS = [
    "pay_status_2005_09",  # 最近月（对应 UCI 的 PAY_0）
    "pay_status_2005_08",
    "pay_status_2005_07",
    "pay_status_2005_06",
    "pay_status_2005_05",
    "pay_status_2005_04",
]

# 近 6 个月账单金额
BILL_COLS = [
    "bill_amount_2005_09_ntd",
    "bill_amount_2005_08_ntd",
    "bill_amount_2005_07_ntd",
    "bill_amount_2005_06_ntd",
    "bill_amount_2005_05_ntd",
    "bill_amount_2005_04_ntd",
]

# 近 6 个月已还金额
PAY_AMT_COLS = [
    "paid_amount_2005_09_ntd",
    "paid_amount_2005_08_ntd",
    "paid_amount_2005_07_ntd",
    "paid_amount_2005_06_ntd",
    "paid_amount_2005_05_ntd",
    "paid_amount_2005_04_ntd",
]

# 类别型特征列
CATEGORICAL_COLS = [COL_SEX, COL_EDUCATION, COL_MARRIAGE]

# 敏感属性列（界面演示与公平性讨论时使用）
SENSITIVE_COLS = [COL_SEX, COL_EDUCATION, COL_MARRIAGE, COL_AGE]

# 固定随机种子，保证可复现
RANDOM_STATE = 42

# 负值还款状态编码含义（界面必须原样说明，负值是有效编码，非缺失）
PAY_CODE_MEANING = {
    -2: "无消费 / 未使用额度",
    -1: "账单已全额还清",
    0: "使用循环信用，最低还款额已付、未逾期",
    1: "逾期 1 个月",
    2: "逾期 2 个月",
    3: "逾期 3 个月",
    4: "逾期 4 个月",
    5: "逾期 5 个月",
    6: "逾期 6 个月",
    7: "逾期 7 个月",
    8: "逾期 8 个月及以上",
    9: "逾期 8 个月及以上",
}


# ---------------------------------------------------------------------------
# Step 1：数据加载
# ---------------------------------------------------------------------------

def load_data(path: str | Path) -> pd.DataFrame:
    """读取信用卡违约数据 CSV，返回完整 DataFrame。

    - 使用 encoding="utf-8-sig" 读取，兼容带 BOM 的 UTF-8 文件。
    - 文件不存在 / 解析失败时抛出信息明确的中文异常，不静默吞掉。
    - 读取后打印：总行数、列名、split 三组的行数。

    参数:
        path: CSV 文件路径。

    返回:
        包含全部样本的 pandas.DataFrame。
    """
    csv_path = Path(path)
    # 文件存在性检查
    if not csv_path.exists():
        raise FileNotFoundError(f"数据文件不存在：{csv_path.resolve()}，请检查路径。")
    if not csv_path.is_file():
        raise ValueError(f"路径不是文件：{csv_path.resolve()}")

    # 读取 CSV，解析失败时包装为中文异常
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        # 若 utf-8-sig 失败，尝试 gbk（Windows 常见编码）并提示
        try:
            df = pd.read_csv(csv_path, encoding="gbk")
        except Exception as e:
            raise ValueError(f"CSV 解析失败，编码问题或其他格式错误：{e}") from e
    except Exception as e:
        raise ValueError(f"CSV 解析失败：{e}") from e

    # 必需列检查
    required = [LABEL, SPLIT_COL, "record_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必需列：{missing}，实际列名为：{list(df.columns)}")

    # 打印基本信息
    total = len(df)
    split_counts = df[SPLIT_COL].value_counts().to_dict()
    print(f"[load_data] 读取成功：{csv_path.name}")
    print(f"  总行数：{total}")
    print(f"  列名（{len(df.columns)} 列）：{list(df.columns)}")
    print(f"  split 三组行数：{split_counts}")

    return df


# ---------------------------------------------------------------------------
# Step 2：数据核查统计
# ---------------------------------------------------------------------------

def audit_data(df: pd.DataFrame) -> dict:
    """按 split 分组做数据核查，返回每组统计信息。

    每组统计：样本量、违约人数、违约率、缺失值个数（逐列缺失合计 +
    有缺失的列名清单）、record_id 重复数。

    约束：缺失值只统计原始 CSV 中的真正空值（NaN/None），
    PAY_* 的负值（-2/-1/0）是有效编码，不计入缺失。

    返回:
        {"train": {n, default_n, default_rate, missing_total, missing_cols,
                   record_id_dup, missing_by_col}, "valid": {...}, "test": {...}}
    """
    result: dict[str, dict] = {}
    for name in ["train", "valid", "test"]:
        sub = df[df[SPLIT_COL] == name]
        n = len(sub)
        if n == 0:
            result[name] = {"n": 0, "reason": f"split={name} 无样本，统计无法计算"}
            continue

        default_n = int(sub[LABEL].sum())
        default_rate = round(float(sub[LABEL].mean()), 4)

        # 缺失值统计：只统计真正空值，负值还款状态是编码不算缺失
        missing_by_col = sub.isna().sum().to_dict()
        missing_total = int(sum(missing_by_col.values()))
        missing_cols = [c for c, v in missing_by_col.items() if v > 0]

        # record_id 重复数
        record_id_dup = int(sub["record_id"].duplicated().sum())

        result[name] = {
            "n": n,
            "default_n": default_n,
            "default_rate": default_rate,
            "missing_total": missing_total,
            "missing_cols": missing_cols,
            "missing_by_col": {c: int(v) for c, v in missing_by_col.items() if v > 0},
            "record_id_dup": record_id_dup,
        }
    return result


def describe_pay_codes(df: pd.DataFrame) -> pd.DataFrame:
    """返回 PAY_* 各取值的含义表与训练集中客户数占比。

    用于向用户说明 -2/-1/0/1..9 的编码含义，负值不当缺失。
    统计最近月还款状态 PAY_0（pay_status_2005_09）在训练集中的取值分布。

    返回:
        DataFrame，列：取值、含义、训练集客户数、训练集占比
    """
    train = df[df[SPLIT_COL] == "train"]
    col = PAY_COLS[0]  # 最近月还款状态
    vc = train[col].value_counts().sort_index()
    total = int(vc.sum())

    rows = []
    for code, count in vc.items():
        meaning = PAY_CODE_MEANING.get(int(code), f"未定义取值 {code}")
        ratio = round(count / total, 4) if total > 0 else None
        rows.append({
            "取值": int(code),
            "含义": meaning,
            "训练集客户数": int(count),
            "训练集占比": ratio,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 2：画图所需数据函数（返回 DataFrame，画图交给界面层）
# ---------------------------------------------------------------------------

def _filter_split(df: pd.DataFrame, split: str | None) -> pd.DataFrame:
    """按 split 过滤，split 为 None 时返回全部。"""
    if split is None:
        return df
    return df[df[SPLIT_COL] == split]


def dist_limit_bal(df: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    """授信额度分布，返回该组全部样本的额度值 DataFrame（供界面画直方图）。"""
    sub = _filter_split(df, split)
    return sub[[COL_LIMIT_BAL]].copy()


def dist_age(df: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    """年龄分布，返回该组全部样本的年龄值 DataFrame（供界面画直方图）。"""
    sub = _filter_split(df, split)
    return sub[[COL_AGE]].copy()


def dist_pay_status(df: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    """最近月还款状态 PAY_0 的取值分布，返回取值 + 计数 DataFrame。"""
    sub = _filter_split(df, split)
    col = PAY_COLS[0]
    vc = sub[col].value_counts().sort_index()
    return pd.DataFrame({"取值": vc.index.astype(int), "客户数": vc.values.astype(int)})


def default_rate_by_group(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """按某列分组的违约率，用于「额度使用率分箱 vs 违约率」等分析。

    返回:
        DataFrame，列：组值、样本量、违约人数、违约率
        若分组后无样本返回空 DataFrame。
    """
    if col not in df.columns:
        return pd.DataFrame(columns=["组值", "样本量", "违约人数", "违约率"])

    grouped = df.groupby(col, observed=True)
    rows = []
    for val, grp in grouped:
        n = len(grp)
        if n == 0:
            rows.append({"组值": val, "样本量": 0, "违约人数": 0, "违约率": None})
            continue
        d_n = int(grp[LABEL].sum())
        rate = round(d_n / n, 4)
        rows.append({"组值": val, "样本量": n, "违约人数": d_n, "违约率": rate})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 3：预处理 + 衍生特征 + 建模
# ---------------------------------------------------------------------------

def clean_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """归并类别异常值：EDUCATION 的 0/4/5/6 归并为 4（其他），MARRIAGE 的 0 归并为 3（其他）。

    在建模前调用，train/valid/test 用同一套规则（不涉及拟合）。
    """
    out = df.copy()
    if COL_EDUCATION in out.columns:
        out[COL_EDUCATION] = out[COL_EDUCATION].replace({0: 4, 5: 4, 6: 4})
    if COL_MARRIAGE in out.columns:
        out[COL_MARRIAGE] = out[COL_MARRIAGE].replace({0: 3})
    return out


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """行内衍生特征（不涉及拟合，train/valid/test 用同一套公式）。

    - utilization_1 = BILL_AMT1 / LIMIT_BAL（分母为 0 时置 0）
    - pay_ratio_1 = PAY_AMT1 / (BILL_AMT1 + 1)
    - max_delinq = PAY_0..PAY_6 的最大值（clip 到 >=0）
    - n_delinq_months = PAY_0..PAY_6 中 >=1 的个数
    - avg_bill = BILL_AMT1..6 均值，avg_pay = PAY_AMT1..6 均值
    """
    out = df.copy()
    limit = out[COL_LIMIT_BAL]
    bill1 = out[BILL_COLS[0]]
    pay1 = out[PAY_AMT_COLS[0]]

    # 额度使用率（分母为 0 时置 0）
    out["utilization_1"] = np.where(limit != 0, bill1 / limit, 0.0)
    # 还款占账单比（分母为 0 时置 0；BILL_AMT1=-1 会导致 +1 后为 0）
    _denom = bill1 + 1.0
    out["pay_ratio_1"] = np.where(_denom != 0, pay1 / _denom, 0.0)
    # 最大逾期月数（clip >=0）
    out["max_delinq"] = out[PAY_COLS].max(axis=1).clip(lower=0)
    # 逾期月数个数
    out["n_delinq_months"] = (out[PAY_COLS] >= 1).sum(axis=1).astype(int)
    # 账单与已还金额均值
    out["avg_bill"] = out[BILL_COLS].mean(axis=1)
    out["avg_pay"] = out[PAY_AMT_COLS].mean(axis=1)
    # 兜底：清理衍生特征中可能的 inf/nan（如除法溢出）
    for c in ["utilization_1", "pay_ratio_1", "max_delinq", "n_delinq_months", "avg_bill", "avg_pay"]:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _prepare_features(df: pd.DataFrame, pay_encoding: str = "numeric+binary") -> pd.DataFrame:
    """组合清洗 + 衍生特征 + PAY 二值版本，返回建模用 DataFrame。

    pay_encoding:
        - "numeric"        : PAY_* 保留数值
        - "onehot"         : PAY_* 作为类别列 OneHot（由 build_preprocessor 处理）
        - "numeric+binary" : PAY_* 保留数值 + 额外加 is_delinq_1..6 二值列（默认）
    """
    out = clean_categorical(df)
    out = add_derived_features(out)
    if pay_encoding == "numeric+binary":
        for i, col in enumerate(PAY_COLS, 1):
            out[f"is_delinq_{i}"] = (out[col] >= 1).astype(int)
    return out


def build_preprocessor(X_train: pd.DataFrame, pay_encoding: str = "numeric+binary") -> ColumnTransformer:
    """构建预处理 ColumnTransformer（只在训练集 fit）。

    - 数值列：中位数填充 + 标准化
    - 类别列（SEX/EDUCATION/MARRIAGE）：OneHotEncoder(handle_unknown="ignore")
    - pay_encoding="onehot" 时 PAY_* 也做 OneHot
    """
    # 类别列
    cat_cols = [c for c in CATEGORICAL_COLS if c in X_train.columns]
    if pay_encoding == "onehot":
        cat_cols += [c for c in PAY_COLS if c in X_train.columns]
    # 数值列 = 其余列
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    transformers = [("num", num_pipe, num_cols)]
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    return ColumnTransformer(transformers, remainder="drop")


def train_models(train_df: pd.DataFrame, valid_df: pd.DataFrame,
                 pay_encoding: str = "numeric+binary",
                 exclude_sensitive: bool = False) -> dict:
    """训练三个模型并返回它们在训练/验证集上的预测概率。

    - baseline：全判不违约（概率恒为 0）
    - logreg：逻辑回归（class_weight="balanced"）
    - tree：HistGradientBoostingClassifier

    预处理只在训练集 fit，验证集只 transform。
    exclude_sensitive=True 时从特征中剔除 SEX/EDUCATION/MARRIAGE/AGE。
    """
    # 特征工程（不涉及拟合）
    train = _prepare_features(train_df, pay_encoding)
    valid = _prepare_features(valid_df, pay_encoding)

    # 特征矩阵：剔除标识列、划分列、标签列（可选剔除敏感属性）
    drop_cols = list(EXCLUDE_COLS)
    if exclude_sensitive:
        drop_cols += [c for c in SENSITIVE_COLS if c in train.columns]
    X_train = train.drop(columns=drop_cols)
    y_train = train[LABEL].values
    X_valid = valid.drop(columns=drop_cols)
    y_valid = valid[LABEL].values

    models: dict[str, dict] = {}

    # baseline：全判不违约
    models["baseline"] = {
        "y_prob_train": np.zeros(len(y_train)),
        "y_prob_valid": np.zeros(len(y_valid)),
        "pipe": None,
    }

    # 逻辑回归
    pre_lr = build_preprocessor(X_train, pay_encoding)
    pipe_lr = Pipeline([("pre", pre_lr), ("clf", LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))])
    pipe_lr.fit(X_train, y_train)
    models["logreg"] = {
        "y_prob_train": pipe_lr.predict_proba(X_train)[:, 1],
        "y_prob_valid": pipe_lr.predict_proba(X_valid)[:, 1],
        "pipe": pipe_lr,
    }

    # 树模型：HistGradientBoosting
    pre_tree = build_preprocessor(X_train, pay_encoding)
    pipe_tree = Pipeline([("pre", pre_tree), ("clf", HistGradientBoostingClassifier(
        random_state=RANDOM_STATE))])
    pipe_tree.fit(X_train, y_train)
    models["tree"] = {
        "y_prob_train": pipe_tree.predict_proba(X_train)[:, 1],
        "y_prob_valid": pipe_tree.predict_proba(X_valid)[:, 1],
        "pipe": pipe_tree,
    }

    return {
        "y_train": y_train,
        "y_valid": y_valid,
        "models": models,
        "feature_cols": list(X_train.columns),
        "pay_encoding": pay_encoding,
        "exclude_sensitive": exclude_sensitive,
    }


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """计算违约类（正类=1）的各项指标。

    返回 roc_auc、precision、recall、f1、accuracy、alarm_rate、
    混淆矩阵人数（TP/FP/FN/TN）、误判人数(FP)、漏判人数(FN)。
    若某指标分母为 0，返回 None 并附 reason 说明原因，不填 0。
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    # 混淆矩阵（正类=违约=1）
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    total = TP + FP + FN + TN

    # ROC-AUC（需要两类都存在）
    try:
        roc_auc = round(float(roc_auc_score(y_true, y_prob)), 4)
    except ValueError:
        roc_auc = None

    # 精确率 = TP / (TP + FP)
    precision = TP / (TP + FP) if (TP + FP) > 0 else None
    # 召回率 = TP / (TP + FN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else None
    # F1
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    # 准确率
    accuracy = (TP + TN) / total if total > 0 else None
    # 报警率 = 预测为违约的比例 = (TP + FP) / total
    alarm_rate = (TP + FP) / total if total > 0 else None

    result = {
        "roc_auc": roc_auc,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "alarm_rate": round(alarm_rate, 4) if alarm_rate is not None else None,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "误判人数": FP, "漏判人数": FN,
    }

    # 附加无法计算的原因
    reasons = []
    if roc_auc is None:
        reasons.append("ROC-AUC 无法计算（仅含单一类别）")
    if precision is None:
        reasons.append("精确率分母为 0（预测无正例）")
    if recall is None:
        reasons.append("召回率分母为 0（真实无正例）")
    if f1 is None and (precision is None or recall is None or (precision + recall) == 0):
        reasons.append("F1 因精确率或召回率为 0 无法计算")
    if reasons:
        result["reason"] = "；".join(reasons)
    return result


# ---------------------------------------------------------------------------
# Step 4：验证集阈值选择与锁定机制
# ---------------------------------------------------------------------------

def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """阈值从 0.01 到 0.99 步长 0.01，逐点计算各项指标。

    返回 DataFrame，列：threshold、precision、recall、f1、alarm_rate、误判人数、漏判人数。
    阈值只能由验证集决定，函数不接触测试集数据。
    """
    rows = []
    for t in np.arange(0.01, 1.00, 0.01):
        t = round(float(t), 2)
        m = compute_metrics(y_true, y_prob, t)
        rows.append({
            "threshold": t,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "alarm_rate": m["alarm_rate"],
            "误判人数": m["误判人数"],
            "漏判人数": m["漏判人数"],
        })
    return pd.DataFrame(rows)


def pick_threshold(sweep_df: pd.DataFrame, criterion: str = "max_f1",
                   recall_target: float = 0.60, cost_ratio: tuple = (5, 1)) -> tuple:
    """根据准则从阈值扫描表中选择阈值，返回 (threshold, 理由字符串)。

    - "max_f1"   : F1 最大（并列取较大阈值，减少误报）
    - "recall_ge": 满足 recall >= recall_target 前提下 precision 最大
    - "min_cost" : cost = FN×漏判成本 + FP×误判成本 最小，cost_ratio=(漏判成本, 误判成本)
    """
    if criterion == "max_f1":
        valid = sweep_df[sweep_df["f1"].notna()].copy()
        if valid.empty:
            return None, "max_f1 准则：无可用阈值（F1 均无法计算）"
        max_f1 = valid["f1"].max()
        best = valid[valid["f1"] == max_f1]
        chosen = float(best["threshold"].max())
        why = f"在验证集上 F1 最大（={max_f1:.4f}），并列阈值中取较大值 {chosen:.2f} 以减少误报。"

    elif criterion == "recall_ge":
        valid = sweep_df[sweep_df["recall"].notna()].copy()
        meet = valid[valid["recall"] >= recall_target]
        if meet.empty:
            best_recall = valid["recall"].max()
            chosen = float(valid.loc[valid["recall"].idxmax(), "threshold"])
            why = f"无阈值满足 recall>={recall_target}，退而取 recall 最大的阈值 {chosen:.2f}（recall={best_recall:.4f}）。"
        else:
            chosen = float(meet.loc[meet["precision"].idxmax(), "threshold"])
            why = f"在 recall>={recall_target} 前提下精确率最高，阈值 {chosen:.2f}，兼顾抓漏与误报。"

    elif criterion == "min_cost":
        fn_cost, fp_cost = cost_ratio
        df = sweep_df.copy()
        df["cost"] = df["漏判人数"] * fn_cost + df["误判人数"] * fp_cost
        chosen = float(df.loc[df["cost"].idxmin(), "threshold"])
        min_cost = int(df["cost"].min())
        why = f"漏判成本:误判成本={fn_cost}:{fp_cost}，总代价最低（cost={min_cost}）的阈值 {chosen:.2f}。"

    else:
        return None, f"未知准则：{criterion}"

    # 取选中阈值处的指标，写进理由
    row = sweep_df[sweep_df["threshold"] == chosen].iloc[0]
    reason = (
        f"【{criterion}】选中阈值 {chosen:.2f}："
        f"精确率={row['precision']}, 召回率={row['recall']}, "
        f"F1={row['f1']}, 报警率={row['alarm_rate']}；{why}"
    )
    return chosen, reason


def lock_model(model_name: str, threshold: float, valid_metrics: dict,
               results_dir: str = "results") -> dict:
    """把选定的模型与阈值写入 results/locked_model.json。"""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    info = {
        "model_name": model_name,
        "threshold": threshold,
        "locked_time": datetime.now().isoformat(),
        "valid_metrics": valid_metrics,
    }
    with open(Path(results_dir) / "locked_model.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return info


def is_locked(results_dir: str = "results") -> bool:
    """是否已锁定模型。"""
    return (Path(results_dir) / "locked_model.json").exists()


def load_locked(results_dir: str = "results") -> dict | None:
    """读取锁定信息，未锁定返回 None。"""
    path = Path(results_dir) / "locked_model.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Step 5：测试集最终评价（只跑一次，幂等保护）
# ---------------------------------------------------------------------------

def final_evaluate(locked: dict, train_df: pd.DataFrame, test_df: pd.DataFrame,
                   results_dir: str = "results") -> dict:
    """用锁定模型 + 锁定阈值在测试集上做一次性最终评价。

    - 读取 locked_model.json 里的模型名与阈值；
    - 用训练集重新 fit 该模型，在测试集上预测一次；
    - 计算 ROC-AUC、精确率、召回率、F1、混淆矩阵人数、误判/漏判人数；
    - 同时输出基线（全判不违约）在测试集上的同组指标用于对照。

    幂等保护：若 results/final_test_metrics.json 已存在，直接读取返回，不再重新计算。
    """
    # 幂等保护：已评价过则直接读缓存
    cache_path = Path(results_dir) / "final_test_metrics.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    model_name = locked["model_name"]
    threshold = locked["threshold"]
    vm = locked.get("valid_metrics", {})
    pay_encoding = vm.get("pay_encoding", "numeric+binary")
    exclude_sensitive = vm.get("exclude_sensitive", False)

    # 特征工程
    train = _prepare_features(train_df, pay_encoding)
    test = _prepare_features(test_df, pay_encoding)
    drop_cols = list(EXCLUDE_COLS)
    if exclude_sensitive:
        drop_cols += [c for c in SENSITIVE_COLS if c in train.columns]
    X_train = train.drop(columns=drop_cols)
    y_train = train[LABEL].values
    X_test = test.drop(columns=drop_cols)
    y_test = test[LABEL].values

    # 重新用训练集 fit 锁定模型，在测试集上预测一次
    if model_name == "baseline":
        y_prob_test = np.zeros(len(y_test))
    elif model_name == "logreg":
        pre = build_preprocessor(X_train, pay_encoding)
        pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))])
        pipe.fit(X_train, y_train)
        y_prob_test = pipe.predict_proba(X_test)[:, 1]
    elif model_name == "tree":
        pre = build_preprocessor(X_train, pay_encoding)
        pipe = Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(
            random_state=RANDOM_STATE))])
        pipe.fit(X_train, y_train)
        y_prob_test = pipe.predict_proba(X_test)[:, 1]
    else:
        raise ValueError(f"未知模型：{model_name}")

    # 测试集指标
    metrics = compute_metrics(y_test, y_prob_test, threshold)
    # 基线对照（全判不违约）
    baseline_metrics = compute_metrics(y_test, np.zeros(len(y_test)), threshold)

    result = {
        "evaluate_time": datetime.now().isoformat(),
        "model_name": model_name,
        "threshold": threshold,
        "n_test": len(y_test),
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "note": "测试集仅评价一次，非调参后的重复评估",
    }

    # 写入缓存
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def reset_evaluation(results_dir: str = "results") -> None:
    """重置：删除锁定与测试结果，允许重新选型与评价。"""
    for fname in ["locked_model.json", "final_test_metrics.json"]:
        p = Path(results_dir) / fname
        if p.exists():
            p.unlink()


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def _print_overview(df: pd.DataFrame) -> None:
    """打印总样本量、三组划分样本量及占比、整体违约率。"""
    total = len(df)
    print("=" * 60)
    print("数据概览")
    print("=" * 60)
    print(f"总样本量：{total}")

    # 三组划分
    split_counts = df[SPLIT_COL].value_counts()
    print("\n各组划分样本量及占比：")
    for name in ["train", "valid", "test"]:
        if name in split_counts.index:
            n = int(split_counts[name])
            ratio = n / total
            print(f"  {name:6s}：{n:>6d} 条  占比 {ratio:.4f}")
        else:
            print(f"  {name:6s}：未找到该划分")

    # 整体违约率
    default_rate = df[LABEL].mean()
    default_n = int(df[LABEL].sum())
    print(f"\n整体违约率：{default_rate:.4f}（违约 {default_n} 人 / 共 {total} 人）")
    print("=" * 60)


def _print_audit(df: pd.DataFrame) -> None:
    """打印数据核查统计表与 PAY 编码含义表。"""
    print("\n" + "=" * 60)
    print("数据核查（按 split 分组）")
    print("=" * 60)

    audit = audit_data(df)
    for name in ["train", "valid", "test"]:
        info = audit.get(name, {})
        print(f"\n【{name}】")
        if info.get("reason"):
            print(f"  原因：{info['reason']}")
            continue
        print(f"  样本量：{info['n']}")
        print(f"  违约人数：{info['default_n']}  违约率：{info['default_rate']:.4f}")
        print(f"  缺失值个数（合计）：{info['missing_total']}")
        if info["missing_cols"]:
            print(f"  有缺失的列：{info['missing_cols']}")
            print(f"  逐列缺失明细：{info['missing_by_col']}")
        else:
            print("  有缺失的列：无")
        print(f"  record_id 重复数：{info['record_id_dup']}")

    # PAY 编码含义表
    print("\n" + "=" * 60)
    print("还款状态编码含义（负值是有效编码，不作为缺失值）")
    print("=" * 60)
    pay_desc = describe_pay_codes(df)
    print(pay_desc.to_string(index=False))

    # 额度使用率分箱 vs 违约率（演示 default_rate_by_group）
    print("\n" + "-" * 60)
    print("按年龄分组的违约率（演示 default_rate_by_group）")
    print("-" * 60)
    age_rate = default_rate_by_group(df, COL_AGE)
    print(age_rate.head(10).to_string(index=False))
    print(f"...（共 {len(age_rate)} 个年龄组）")


def _print_train(df: pd.DataFrame) -> None:
    """打印三个模型在验证集上（阈值 0.5）的指标对比表。"""
    train_df = df[df[SPLIT_COL] == "train"]
    valid_df = df[df[SPLIT_COL] == "valid"]

    print("\n" + "=" * 60)
    print("模型训练（预处理只在训练集 fit，验证集只 transform）")
    print("=" * 60)
    print("训练中，请稍候 ...")

    result = train_models(train_df, valid_df)
    y_valid = result["y_valid"]
    print(f"特征数：{len(result['feature_cols'])}  pay_encoding={result['pay_encoding']}")

    # 验证集指标对比表（阈值 0.5）
    print("\n验证集指标对比（阈值 0.5，正类=违约）：")
    print(f"  {'模型':<10s} {'AUC':>7s} {'精确率':>8s} {'召回率':>8s} {'F1':>7s} {'准确率':>8s} {'报警率':>8s}")
    print("  " + "-" * 60)
    for name in ["baseline", "logreg", "tree"]:
        y_prob = result["models"][name]["y_prob_valid"]
        m = compute_metrics(y_valid, y_prob, 0.5)

        def _fmt(v):
            return f"{v:.4f}" if v is not None else "  N/A"

        print(f"  {name:<10s} {_fmt(m['roc_auc']):>7s} {_fmt(m['precision']):>8s} "
              f"{_fmt(m['recall']):>8s} {_fmt(m['f1']):>7s} {_fmt(m['accuracy']):>8s} "
              f"{_fmt(m['alarm_rate']):>8s}")
        if m.get("reason"):
            print(f"    └ 原因：{m['reason']}")

    # 训练集 AUC（用于检查过拟合）
    print("\n训练集 AUC（参考，检查过拟合）：")
    for name in ["baseline", "logreg", "tree"]:
        y_prob = result["models"][name]["y_prob_train"]
        m = compute_metrics(result["y_train"], y_prob, 0.5)
        auc = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "N/A"
        print(f"  {name:<10s} AUC={auc}")


def _print_threshold(df: pd.DataFrame) -> None:
    """打印阈值扫描表前后若干行与三种准则的选定结果。"""
    train_df = df[df[SPLIT_COL] == "train"]
    valid_df = df[df[SPLIT_COL] == "valid"]

    print("\n" + "=" * 60)
    print("阈值选择（仅用验证集，不接触测试集）")
    print("=" * 60)
    print("训练中，请稍候 ...")
    result = train_models(train_df, valid_df)
    y_valid = result["y_valid"]
    Path("results").mkdir(exist_ok=True)

    for name in ["logreg", "tree"]:
        y_prob = result["models"][name]["y_prob_valid"]
        sweep = threshold_sweep(y_valid, y_prob)
        # 保存扫描表供界面画图
        sweep.to_csv(Path("results") / f"threshold_curve_valid_{name}.csv",
                     index=False, encoding="utf-8-sig")

        print(f"\n【{name}】阈值扫描表（前 5 行 + 后 5 行）：")
        print(sweep.head(5).to_string(index=False))
        print("  ...")
        print(sweep.tail(5).to_string(index=False))

        for crit in ["max_f1", "recall_ge", "min_cost"]:
            t, reason = pick_threshold(sweep, crit)
            print(f"\n  准则 {crit} → 阈值 {t}")
            print(f"  {reason}")


def _print_evaluate(df: pd.DataFrame) -> None:
    """打印测试集最终评价表（含基线对照）。

    若未锁定模型，先自动用 tree + max_f1 锁定，以便命令行独立演示完整流程。
    """
    train_df = df[df[SPLIT_COL] == "train"]
    valid_df = df[df[SPLIT_COL] == "valid"]
    test_df = df[df[SPLIT_COL] == "test"]

    print("\n" + "=" * 60)
    print("测试集最终评价（只跑一次）")
    print("=" * 60)

    # 若未锁定，自动锁定 tree + max_f1
    locked = load_locked()
    if locked is None:
        print("未锁定模型，自动锁定：tree + max_f1 准则 ...")
        result = train_models(train_df, valid_df)
        y_valid = result["y_valid"]
        y_prob = result["models"]["tree"]["y_prob_valid"]
        sweep = threshold_sweep(y_valid, y_prob)
        t, reason = pick_threshold(sweep, "max_f1")
        valid_metrics = compute_metrics(y_valid, y_prob, t)
        valid_metrics["pay_encoding"] = result["pay_encoding"]
        locked = lock_model("tree", t, valid_metrics)
        print(f"已锁定：模型=tree, 阈值={t}, 理由：{reason}")

    print(f"锁定模型：{locked['model_name']}  阈值：{locked['threshold']}  "
          f"锁定时间：{locked.get('locked_time', '未知')}")
    print("在测试集上评价中，请稍候 ...")

    result = final_evaluate(locked, train_df, test_df)
    m = result["metrics"]
    bm = result["baseline_metrics"]

    print(f"\n测试集样本量：n={result['n_test']}")
    print(f"评价时间：{result['evaluate_time']}")
    print(f"备注：{result['note']}")

    def _fmt(v):
        return f"{v:.4f}" if v is not None else "  N/A"

    print("\n测试集最终指标（正类=违约）：")
    print(f"  {'':12s} {'AUC':>7s} {'精确率':>8s} {'召回率':>8s} {'F1':>7s} {'准确率':>8s} {'报警率':>8s} {'误判':>5s} {'漏判':>5s}")
    print("  " + "-" * 72)
    print(f"  {'锁定模型':<10s} {_fmt(m['roc_auc']):>7s} {_fmt(m['precision']):>8s} "
          f"{_fmt(m['recall']):>8s} {_fmt(m['f1']):>7s} {_fmt(m['accuracy']):>8s} "
          f"{_fmt(m['alarm_rate']):>8s} {m['误判人数']:>5d} {m['漏判人数']:>5d}")
    print(f"  {'基线(全不违约)':<10s} {_fmt(bm['roc_auc']):>7s} {_fmt(bm['precision']):>8s} "
          f"{_fmt(bm['recall']):>8s} {_fmt(bm['f1']):>7s} {_fmt(bm['accuracy']):>8s} "
          f"{_fmt(bm['alarm_rate']):>8s} {bm['误判人数']:>5d} {bm['漏判人数']:>5d}")

    # 混淆矩阵
    print(f"\n混淆矩阵（锁定模型 {result['model_name']}）：")
    print(f"  预测不违约(0)  预测违约(1)")
    print(f"  实际不违约(0)  TN={m['TN']:<6d}  FP={m['FP']:<6d}")
    print(f"  实际违约(1)    FN={m['FN']:<6d}  TP={m['TP']:<6d}")

    if m.get("reason"):
        print(f"\n  指标原因：{m['reason']}")
    if bm.get("reason"):
        print(f"  基线原因：{bm['reason']}")

    print(f"\n结果已保存至 results/final_test_metrics.json（幂等，再次运行不会重复评价）")


def main() -> None:
    """命令行入口：python credit_pipeline.py <csv路径> [--audit|--train|--threshold|--evaluate]"""
    if len(sys.argv) < 2:
        print("用法：python credit_pipeline.py <csv路径> [--audit] [--train] [--threshold] [--evaluate]")
        sys.exit(1)

    csv_path = sys.argv[1]
    flags = set(sys.argv[2:])
    df = load_data(csv_path)
    _print_overview(df)

    if "--audit" in flags:
        _print_audit(df)

    if "--train" in flags:
        _print_train(df)

    if "--threshold" in flags:
        _print_threshold(df)

    if "--evaluate" in flags:
        _print_evaluate(df)


if __name__ == "__main__":
    main()