# 信用卡客户下月违约风险分析系统

> 计算机金融学导论 · 课程综合项目 · **题目一**
> 识别潜在违约客户，辅助安排风险核查。
> ⚠️ 本项目仅作教学分析，**不用于真实授信决策**。

**架构：前后端分离**（FastAPI 后端 + Streamlit 前端，通过 HTTP/JSON 通信）

---

## 一、快速开始

### 方式 A：一键启动（推荐）

| 系统 | 操作 |
|---|---|
| Windows | 双击 **`启动.cmd`** |
| macOS / Linux | `chmod +x start.sh && ./start.sh` |

脚本会自动检查依赖 → 后台起后端 → 等后端就绪 → 前台起前端 → 打开浏览器。
**关闭窗口即同时停止前后端。**

启动后：

- 前端界面：<http://127.0.0.1:8501>
- 后端接口文档（可直接点「Try it out」试接口）：<http://127.0.0.1:8000/docs>

### 方式 B：手动分两个终端

```bash
# 1. 安装依赖（只需一次）
pip install -r requirements.txt

# 终端 1：启动后端
uvicorn backend.main:app --reload --port 8000

# 终端 2：启动前端
streamlit run frontend/app.py
```

> 前端通过环境变量 `BACKEND_URL` 找后端，默认 `http://127.0.0.1:8000`。
> 后端换了地址/端口时：`BACKEND_URL=http://127.0.0.1:9000 streamlit run frontend/app.py`

数据文件 `01_credit_default.csv` 已随仓库提供，**无需额外下载**。

---

## 二、前后端分离是怎么分的

```
浏览器
  │
  ▼
frontend/app.py          ← 纯展示：画图、排版、摆数字（625 行）
frontend/api_client.py   ← 唯一的数据通道：所有数据走 HTTP（211 行）
  │  HTTP / JSON
  ▼
backend/main.py          ← 21 个接口：算数、读文件、给 JSON（623 行）
backend/state.py         ← 服务端状态：训练结果、锁定状态
  │
  ▼
credit_pipeline.py       ← 计算层（前后端共用，里面没有一行 streamlit）
artifacts_io.py          ← 队友交付物读取
```

### 边界验收（可直接复制运行，都应当**没有任何输出**）

```bash
# 前端不许碰算法：搜不到任何算法/科学计算库的 import
grep -rnE "^\s*(import|from)\s+(credit_pipeline|artifacts_io|sklearn|joblib|preprocess|numpy|pandas|scipy)" frontend/

# 后端不许碰界面
grep -rnE "^\s*(import|from)\s+streamlit" backend/
```

前端 `import` 的全部模块只有：`streamlit`、`plotly`、`requests`、`pathlib`、`sys`、`os`、`typing`
—— 一个能算数的都没有。所有数字都来自后端返回的 JSON。

### 三个刻意的设计

1. **训练一次，前端随便切。**
   `POST /train` 一次就把三个模型在验证集上的概率、以及整条阈值扫描曲线（每个模型 99 个点）全返回。
   前端切换模型、拖阈值只从已返回的曲线里取点，**不需要重新训练** —— 否则每点一下就要等十几秒。

2. **模型对象不出后端。**
   训练好的管道（sklearn Pipeline）留在后端内存里，前端只拿到一个 `job_id`。
   预测时前端发 `{"record_id": 6}`，后端返回 `{"probability": 0.327124}`。

3. **测试集只评价一次，后端连"重新评价"的接口都不提供。**
   `/artifacts/test-result` 只读 `test_result.json`，没有能重跑测试集的端点。
   这是把课程红线写进了接口设计里。

---

## 三、接口一览（21 个）

启动后端后打开 <http://127.0.0.1:8000/docs> 可交互式查看与试用。

| 分组 | 方法与路径 | 说明 |
|---|---|---|
| 健康 | `GET /health` | 服务状态 + 交付物到达情况 + 当前锁定 |
| 数据 | `GET /data/meta` | 总样本量、列名、三组划分统计 |
| | `GET /data/distribution` | 授信额度 / 年龄 / 还款状态分布（按 split） |
| | `GET /data/pay-codes` | 还款状态编码含义表 |
| | `GET /data/default-rate` | 额度使用率分箱 vs 违约率 |
| | `POST /data/reload` | 重新读 CSV，清空缓存与训练结果 |
| 训练 | `POST /train` | 训练三模型，返回验证集指标 + 99 点阈值曲线 |
| | `POST /threshold/pick` | 按准则（max_f1 / recall_ge / min_cost）选阈值 |
| 锁定 | `GET /model/lock` | 查询锁定状态 |
| | `POST /model/lock` | 锁定模型与阈值（落盘 `results/locked_model.json`） |
| | `DELETE /model/lock` | 重置锁定与测试结果 |
| 交付物 | `GET /artifacts/status` | 各交付物是否就位 |
| | `GET /artifacts/c-model` | C 的模型信息与阈值 |
| | `GET /artifacts/comparison` | 模型对比表 / 阈值曲线 / 特征重要性 |
| | `GET /artifacts/test-result` | D 的测试集结果（只读，不重算） |
| | `GET /artifacts/field-spec` | B 的字段口径原文 |
| 预测 | `POST /predict/record` | 按 `record_id` 预测，附记录详情与衍生特征 |
| | `GET /records/random` | 随机抽一条测试集记录并预测 |
| | `POST /records/get` | 按 `record_id` 取记录详情（不预测） |
| | `POST /predict/custom` | 手工输入一条新客户并预测 |
| 字典 | `GET /schema/fields` | 可输入字段的取值范围（给前端做控件） |

---

## 四、界面结构（四个页面）

| 页面 | 内容 | 数据来源 |
|---|---|---|
| **数据概览** | 统计卡片（三组样本量/违约率）、按划分统计表、授信额度分布、年龄分布、还款状态分布、额度使用率分箱 vs 违约率、还款状态编码含义 | `/data/meta` `/data/distribution` `/data/pay-codes` `/data/default-rate` |
| **模型比较** | C 同学交付的验证集模型对比表、阈值—指标曲线、特征重要性 Top20；下方为后端现场训练的交互式选型与阈值选择 | `/artifacts/comparison` `/train` `/threshold/pick` `/model/lock` |
| **最终评价（测试集）** | D 同学锁定的测试集最终结果（读文件不重跑）、混淆矩阵、指标卡 | `/artifacts/test-result` `/model/lock` |
| **客户风险演示** | 选一条客户 → 风险概率、所用阈值、风险分档、主要输入信息、衍生特征；测试集记录单独框出事后真实标签 | `/predict/record` `/records/random` `/health` |

---

## 五、数据与方法（严格遵循课程要求）

- **数据**：`01_credit_default.csv`，30,000 条客户历史记录；train / valid / test = 20,999 / 4,500 / 4,501 条。
- **标签**：`default_next_month`，1 表示下月违约。
- **划分**：分层随机划分（数据无逐笔申请日期）。
- **预处理**：数字列缺失填补 + 标准化；类别列字符串化 + 缺失填补 + 独热编码；
  还款状态列**先转字符串再独热编码，负值 -2 / -1 是有效编码，不作为缺失处理**。
  **预处理仅在训练集上拟合**（`/train` 里 `fit` 只吃 train，valid/test 只 `transform`）。
- **模型**：全判不违约基线、逻辑回归、随机森林（C 同学交付口径：
  `n_estimators=300, max_depth=12, min_samples_leaf=5, class_weight='balanced'`）。
  后端现场训练用的是 `HistGradientBoostingClassifier`，两者参数不同，指标会有小差异，
  **团队正式口径以 C 同学的交付物为准**。
- **阈值**：仅用**验证集**确定，最终选定 **0.5**（Recall≥0.6 约束下 Precision 最高）。
- **测试集**：只评价一次。

---

## 六、目录结构

```
.
├─ 启动.cmd                    # Windows 一键启动
├─ start.sh                    # macOS / Linux 一键启动
│
├─ frontend/                   # ★ 前端（不 import 任何算法）
│  ├─ app.py                   #   四个页面
│  ├─ api_client.py            #   唯一的数据通道（HTTP 封装）
│  └─ ui_const.py              #   展示常量（列名映射、页面名）
│
├─ backend/                    # ★ 后端（不 import streamlit）
│  ├─ main.py                  #   21 个接口
│  └─ state.py                 #   服务端状态（训练结果 / 锁定）
│
├─ credit_pipeline.py          # 计算层：数据核查 / 预处理 / 训练 / 阈值 / 评价
├─ artifacts_io.py             # 队友交付物接入层（只读不重算）
├─ preprocess.py               # 特征工程模块（被 best_model.pkl 反序列化时引用）
│
├─ smoke_test_api.py           # 后端接口冒烟测试（46 项）
├─ smoke_test_frontend.py      # 前端四页冒烟测试（14 项）
├─ smoke_test.py               # 归档版（单体）回归测试
├─ requirements.txt
├─ 01_credit_default.csv
├─ results/                    # 运行时产物：locked_model.json 等
├─ archive/
│  └─ app_monolith_v1.py       # 分离前的单体版本（留档对照）
└─ artifacts/                  # 各成员交付物
   ├─ best_model.pkl                 (C) 锁定的随机森林模型
   ├─ threshold.json                 (C) 选定阈值与选择依据
   ├─ model_comparison.csv           (C) 三模型 train/valid 对比
   ├─ threshold_comparison.csv       (C) 验证集阈值扫描表
   ├─ feature_importance.csv         (C) 特征重要性排名
   ├─ test_result.json               (D) 测试集最终评价（只跑一次，锁定）
   └─ 字段口径.md                     (B) 字段表与编码说明
```

---

## 七、最终测试集结果（锁定，只评价一次）

| 指标 | 值 |
|---|---|
| ROC-AUC | **0.782545** |
| 违约类精确率 | 0.495840 |
| 召回率 | 0.598394 |
| 混淆矩阵 (TN/FP/FN/TP) | 2899 / 606 / 400 / 596 |
| 测试集样本量 | 4,501（违约 996 条，违约率 0.2213） |

结果来源：`artifacts/test_result.json`（由 D 同学在测试集上只运行一次后锁定）。
界面「最终评价」页通过 `GET /artifacts/test-result` 读取，**不重新计算**。
用 C 同学的 `best_model.pkl` 在线复算，结果与上述逐项一致。

---

## 八、成员分工

| 角色 | 姓名 | 负责内容 | 交付文件 |
|---|---|---|---|
| A | 刘思雨 | 数据核查 + 数据概览页 | `src/data_io.py`、`src/data_check.py` |
| B | 赵梓伶 | 特征预处理 + 防泄漏 | `src/preprocess.py`、`docs/字段口径.md` |
| C | 冯荟如 | 模型训练 + 阈值选择 | `src/models.py`、`artifacts/*.pkl/csv/json` |
| D | 吴姿仪 | 测试集评价（只跑一次） | `src/evaluate.py`、`artifacts/test_result.json` |
| E | 陈韬伊 | 前后端分离 + 界面 + 演示 + PPT（组长） | `frontend/`、`backend/`、`README.md`、`slides/` |

### E（陈韬伊）本人实现说明

**前后端分离**是本轮主要工作。拆分方式不是"重写一遍"，而是**按原有代码的天然边界切开** ——
`credit_pipeline.py` 本来就没有一行 streamlit，它天然就是后端；`app.py` 本来就只调用
`cp.*` 和 `ai.*`，它天然就是前端。

- **`backend/main.py`**：把 `app.py` 里每一处 `cp.xxx(...)` / `ai.xxx(...)` 调用整理成 21 个接口。
  关键处理：numpy/pandas/NaN 统一递归转 JSON（`_jsonable`）；CSV 进程内缓存，
  避免每次请求都重读 3 MB 文件。
- **`frontend/api_client.py`**：唯一的数据通道。连不上后端时抛 `BackendError`，
  界面据此显示"请先启动后端"的引导，而不是抛一堆红字。
- **`backend/state.py`**：替代 `st.session_state`。训练结果存在服务端内存里按 `job_id` 引用；
  锁定状态落盘到 `results/locked_model.json`，后端重启后锁定依然有效。
- **界面不参与任何建模**：C 的模型、D 的测试结果、B 的字段口径仍然以文件形式接入
  （由后端读取后转发），页面只读取与展示，符合"测试集只评价一次""页面上的数必须由代码算出"的红线。

---

## 九、测试记录

### 后端接口（46 项，全绿）

```bash
# 先启动后端
uvicorn backend.main:app --port 8000
python smoke_test_api.py
```

覆盖：健康检查、4 个数据接口、训练、阈值选择（含基线边界情况）、锁定/重置、
5 个交付物接口、3 种预测（含 404 边界）、字段字典。关键断言：

- D 的 `TN/FP/FN/TP = 2899/606/400/596`、`ROC-AUC = 0.782545` ✔
- `record_id=6` → 概率 `0.327124`（与分离前的界面数字完全一致）✔
- 高风险客户（连续逾期）→ 风险分档「高」、判为报警 ✔
- 基线模型选阈值 → `valid=false` 而不是崩溃 ✔

### 前端四页（14 项，全绿）

```bash
python smoke_test_frontend.py
```

```
[OK] 数据概览：能渲染 / 5 个统计卡片 / 4 张图
[OK] 模型比较：能渲染（含后端训练）/ 3 张图 / 2 张表
[OK] 最终评价：能渲染 / 读到了 D 的 test_result.json / 8 个指标卡片
[OK] 客户风险演示：能渲染 / 点击查询 / 出现风险概率 / 4 个衍生特征卡片
```

### 一致性核验

用 C 同学的 `best_model.pkl` 在测试集复算，得到 ROC-AUC 0.782545、
精确率 0.495840、召回率 0.598394、混淆矩阵 2899/606/400/596，
与 D 同学的 `artifacts/test_result.json` **逐项完全一致**。
分离后 `POST /predict/record {"record_id": 6}` 返回 `0.327124`，
与分离前 Streamlit 现场计算的数字**完全相同** —— 说明这次重构没有动到任何计算逻辑。

---

## 十、免责与说明

- 本项目**仅作教学分析**，不构成授信决定，不得用于真实风险核查。
- 模型使用了性别、婚姻、教育、年龄等**敏感属性**；**相关性不等于因果关系**，
  输出不得作为对任何个人或群体差别对待的依据。
- 数据无逐笔申请日期，划分为分层随机划分，结论不可外推到未来时段。
- 数据来源：UCI *Default of Credit Card Clients* 数据集的课程改写版，`record_id` 仅用于追溯。
