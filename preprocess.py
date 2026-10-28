"""兼容垫片：B 同学 preprocess.py 的最小实现。

作用：C 同学交付的 best_model.pkl 里，还款状态列的字符串转换
     FunctionTransformer(func=preprocess._pay_status_to_string)
     在反序列化时必须能找到本模块，否则 pickle.load 会报
     "No module named 'preprocess'"。

说明：本文件仅补上被 pickle 引用的那一个函数；B 同学的完整
     preprocess.py（含 build_preprocessor / transform / 防泄漏检查）
     请以其正式交付为准，收到后直接覆盖本文件即可。
"""

from __future__ import annotations

import pandas as pd


def _pay_status_to_string(X):
    """还款状态列转字符串，保留负值编码（-2/-1/0 不是缺失）。

    - 只对真正的空值填 "未知"
    - 其余原样转成字符串，交给下游 OneHotEncoder
    """
    if isinstance(X, pd.DataFrame):
        out = X.copy()
        for c in out.columns:
            out[c] = out[c].map(lambda v: "未知" if pd.isna(v) else str(int(v)) if float(v).is_integer() else str(v))
        return out
    # Series / 单列
    return X.map(lambda v: "未知" if pd.isna(v) else str(int(v)) if float(v).is_integer() else str(v))


def _pay_status_to_unknown(X):  # 兼容可能的备用命名
    return _pay_status_to_string(X)
