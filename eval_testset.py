import os
import time
import logging
import inspect
from pathlib import Path
from src.agent.react_agent import FinancialAgent

# Create log directory if it doesn't exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure comprehensive logging to file
log_file = LOG_DIR / f"testset_{int(time.time())}.log"

# Define logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# File handler for detailed tracking
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Console handler for cleaner output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

TEST_QUERIES = [
    {
        "id": 1,
        "type": "跨文档对比",
        "question": "比较2021年10-K和2023年10-K中关于“中国市场风险 (China market risks)”的描述有什么显著的变化？",
    },
    {
        "id": 2,
        "type": "跨表计算对比",
        "question": "计算2022年四个季度（Q1到Q4）的“研发费用 (Research and development)”的总和，并与2021年10-K中汇报的全年研发费用对比，差额是多少？",
    },
    {
        "id": 3,
        "type": "文本数据关联",
        "question": "查找2022年中哪份季报（10-Q）提到了“供应链挑战 (supply chain challenges)”并且该季度的汽车总营收 (Total automotive revenues) 是多少？",
    },
    {
        "id": 4,
        "type": "隐含时间序列",
        "question": "从2021年到2023年，哪一份特定文件（指明年份和季度）首次提到了柏林超级工厂 (Gigafactory Berlin) 的产能瓶颈或者相关挑战？详细引述它。并且从那之后柏林工厂的说法有什么演变？",
    },
    {
        "id": 5,
        "type": "极值查找与关联",
        "question": "在2022和2023所有的10-Q季度报告中，找到“汽车毛利率 (Automotive gross margin)”最低的季度。提取出该季度中『管理层讨论与分析』对于毛利率下降的解释句子。",
    }
]

def format_intermediate_steps(steps) -> str:
    """Formats the array of Action/Observation pairs for readability"""
    formatted = []
    for step in steps:
        action = step[0] # AgentAction
        observation = step[1] # Observation result string
        
        formatted.append(f"-> TOOL TRIGGERED: {action.tool}\n   Input: {action.tool_input}\n<- RESULT:\n{observation}\n")
    return "\n" + "-"*40 + "\n" + "\n".join(formatted) + "\n" + "-"*40

def run_evaluation():
    logger.info("="*50)
    logger.info("🚀 Starting High-Order Financial Agent Evaluation")
    logger.info("="*50)
    
    agent = FinancialAgent()
    
    results = []
    
    for item in TEST_QUERIES:
        q_id = item['id']
        q_type = item['type']
        q_text = item['question']
        
        logger.info(f"\n[Test Case {q_id}] ({q_type})")
        logger.info(f"Question: {q_text}")
        
        try:
            start_time = time.time()
            answer, steps = agent.ask(q_text)
            elapsed = time.time() - start_time
            
            logger.info(f"✅ Final Answer ({elapsed:.2f}s):\n{answer}")
            logger.debug(f"🔍 Agent Reasoning Trace for Q{q_id}:\n{format_intermediate_steps(steps)}")
            results.append({"id": q_id, "status": "SUCCESS", "answer": answer})
            
        except Exception as e:
            logger.error(f"❌ Failed to answer Q{q_id}. Error: {e}", exc_info=True)
            results.append({"id": q_id, "status": "FAIL", "error": str(e)})
            
        logger.info("*"*50)
        
    logger.info("\n📊 === Summary ===")
    for res in results:
        stat = res['status']
        icon = '✅' if stat == 'SUCCESS' else '❌'
        logger.info(f"{icon} Q{res['id']}: [{stat}]")
        
    logger.info(f"\nDetailed tool reasoning logs saved to {log_file}")

if __name__ == "__main__":
    # Usually requires API keys to run successfully.
    run_evaluation()
