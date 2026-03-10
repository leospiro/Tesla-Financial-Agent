# 失败案例深度剖析 (Failure Analysis)

在运行高阶测试集 `eval_testset.py` 后，对于回答失败、残缺或产生幻觉的测试点，我们应该在此进行诊断与追因：

## Case 1: (待补充问题)
- **失败表象 (Symptom)**：答案遗漏了某些季度的数值，或者把其他年份的费用误认为本期费用。
- **溯源排查 (Trace)**：通过 `eval_testset.py` 记录的 `logs/` 轨迹看，Agent 在发出 `financial_document_search` 动作时的 Query 是否抓准了关键字？ChromaDB 返回的 Metadata 中的 Year 是否混杂？
- **根本原因 (Root Cause)**：
  - 例：可能是表格跨越多页时，表头截断导致模型无法对齐后续的金额数据。
  - 例：LLM 试图在一次 Prompt 给定的上下文中靠自己做连续多位小数相加出现幻觉（未正确调用 Calculator 工具）。
- **具体改进方案 (Improvement Options)**：
  - 提升/改写抽取逻辑：把跨页表格强行合并，或每页复写一遍 Markdown 表头。
  - 在 Agent 的 System Prompt 中强制要求："对于超过三个数字以上的加减法，你不能自己算，必须调用 `math_calculator` 工具"。

## Case 2: (待补充问题)
- **失败表象**: (Empty)
- **溯源排查**: (Empty)
- **根本原因**: (Empty)
- **具体改进方案**: (Empty)

## Case 3: (待补充问题)
- **失败表象**: (Empty)
- **溯源排查**: (Empty)
- **根本原因**: (Empty)
- **具体改进方案**: (Empty)

## Case 4: (待补充问题)
- **失败表象**: (Empty)
- **溯源排查**: (Empty)
- **根本原因**: (Empty)
- **具体改进方案**: (Empty)

## Case 5: (待补充问题)
- **失败表象**: (Empty)
- **溯源排查**: (Empty)
- **根本原因**: (Empty)
- **具体改进方案**: (Empty)
