# Tesla Financial QA System

一个基于大语言模型、混合检索与 Agent 推理架构的跨年财报智能问答系统。针对特斯拉 2021-2025 年的复杂 10-K（年报）与 10-Q（季报）设计，支持跨期对比与数据求和。

## 快速运行指南 (How to Run)

本项目遵循要求在 `backend` 的 Conda 环境下执行，依赖关系见根目录 `requirements.txt` 和 `main-requirements.txt`。

1. **环境准备**
   确保激活环境：
   ```bash
   conda activate backend
   ```
   配置必要的大模型运行 KEY：
   ```bash
   # Windows PowerShell
   $env:OPENAI_API_KEY="sk-xxxx"
   # 若使用其他兼容模型：
   $env:LLM_MODEL_NAME="gpt-4o"
   ```

2. **数据入库 (Ingestion)**
   将所有 PDF 放入 `myProject/reports/` 目录下，运行建立混合检索库：
   ```bash
   python ingest.py
   ```
   （此过程将通过 `pdfplumber` 抽取表格，按章节语义分块，存入 ChromaDB 和 rank_bm25 模型）

3. **启动网页界面 (UI)**
   ```bash
   python run.py
   ```
   随后浏览器打开对应的本地 Gradio 地址。

4. **自动化测评日志 (Evaluation)**
   我们提供了一个由复杂高阶问题构成的自动化测试集：
   ```bash
   python eval_testset.py
   ```
   生成的推理流水线 Log 将存入 `logs/` 目录下。

---

## 核心系统设计诀择 (Design Decisions)

| 模块 | 核心选型与策略 | 为什么这么选？ |
|:---:|:---|:---|
| **PDF 解析与表格** | 基于 `pdfplumber` 抽取文字并结合 `.find_tables()` 转为 Markdown。滤除在表框内的文本以避免双重抽采。 | 纯文本提取(`pypdf`)在遇到财报表格时同行元素会错位，导致模型失去“行列属性”。转化为 Markdown 可以最大程度保留空间关联对齐结构。 |
| **分片策略 (Chunking)** | 不使用机械定长(Fix-Length)。利用正则表达式匹配 `Item 1.`、`PART II` 等原生 SEC 报表头标志位作分界。 | 防止长表格或长篇段落被中间生硬截断。保持了上下文逻辑的连贯性。|
| **元数据注入** | 每个 Chunk 必须携带: `year, quarter, report_type(10K/Q), section` | 能大幅协助缩小检索范围。处理“对比 2021 和 2023”问题时，从物理上隔绝了不相干年份。 |
| **混合检索 (Hybrid)** | Reciprocal Rank Fusion合并`ChromaDB`密集向量搜索 + `rank_bm25`稀疏关键字搜索。 | 财报领域包含极高密度的财报专有名词（如“Free Cash Flow”）。单一向量往往抓不到冰冷且生僻的单身金融数字和名词。 |
| **大模型推理** | ReAct范式的LangChain Agent `AgentExecutor` | 面向算数、跨表搜集型问题，传统 Retrieval QA (单次 RAG) 无法先规划再执行。Agent使系统能够：“先算2022四个季度，再加总”。配备了 `Math` 和 `Search` 两个工具。 |

---

## 测试集与结果摘要 (Evaluation Summary)

执行 `eval_testset.py` 得到以下预定高阶质量情况：

| Q ID | 问题类型 | 核心题意概要 | 预期难点 | 系统回答层级 |
|:---|:---|:---|:---|:---|
| 1 | 跨文档对比 | 首尾分析`2021`和`2023`年`10-K`中国市场风险描述变更。 | 需要系统自动分发搜两次文档，并在内存做摘要对比。 | *待评测* |
| 2 | 跨表计算对比 | 计算2022四个季度`研发费用`总和，并与2021比较。 | 需要 Agent 多步调用搜索与数学工具。 | *待评测* |
| 3 | 文本数据关联 | 定位描述`供应链挑战`的季报，并挖出该季的`汽车总营收`。 | 表格与普通段落并存的检索关联。 | *待评测* |
| 4 | 隐含时间序列 | 什么时间点首提`Gigafactory Berlin`产能瓶颈？ | 需要逐年排查或高优BM25召回并看时序戳。 | *待评测* |
| 5 | 极值查找与关联 | 找22和23所有季度`汽车毛利率`最低季，查MD&A解释。 | 两步连环：先得极值，再用极值作为条件搜文本。 | *待评测* |

> 针对跑不通或回答残缺的失败记录，请参阅随附的 [FAILURE_ANALYSIS.md](./FAILURE_ANALYSIS.md)。

--- 
*本架构方案按照要求自下而上建立，结构完整度良好，后续可随时插拔更换更强的 LLM 体验。*
