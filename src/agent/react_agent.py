import os
import logging
from typing import List, Dict, Any, Tuple

from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.agent.tools import get_agent_tools
from src.config import LLM_MODEL_NAME

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a highly capable Financial Expert Assistant specializing in analyzing Tesla's SEC filings (10-K and 10-Q).
You have access to a suite of tools:
1. `financial_document_search`: Retrieves parts of financial reports based on keywords and metadata filters (year, quarter, report_type).
2. `math_calculator`: Performs numerical calculations on financial figures you extract from the text.

YOUR OBJECTIVE: 
Answer complex quantitative and qualitative questions about Tesla's financials logically and precisely.

INSTRUCTIONS:
- Whenever answering requires specific data, ALWAYS use the `financial_document_search` tool first.
- If the question asks for multiple periods (like four quarters of 2022), you can do separate searches or one broad search. If you do multiple searches, aggregate the data in your thought process.
- If calculations are needed (e.g. sum, difference, ratio), ALWAYS extract the specific figures logically, then use the `math_calculator` tool. Do not attempt mental math on float numbers.
- Cite your sources where possible based on the "Source" and "Section" provided by the search tool.
- Provide a clear, step-by-step logical explanation of your conclusion.
- Be robust to tabular data structures: numbers in tables are separated by '|' in markdown. Interpret row and column headers accurately.

Begin!"""

def create_financial_agent():
    """Initializes and returns the LangChain Agent Executor."""
    # Ensure OPENAI_API_KEY is properly set in the environment or DashScope equivalents.
    # To use OpenAI tools feature, the model needs to be a late generation GPT (e.g., gpt-4o), 
    # or another tool-capable model (if using Qwen/Dashscope, ensure tool calling is supported).
    
    # Initialize LLM
    llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0, model_kwargs={"top_p": 0.8})
    
    # Load tools
    tools = get_agent_tools()
    
    # Construct prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Bind tools to the model and create agent pipeline
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create the AgentExecutor
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        return_intermediate_steps=True, # Important for debugging and logs!
        handle_parsing_errors=True,
        max_iterations=10
    )
    
    return agent_executor

class FinancialAgent:
    def __init__(self):
        self.agent_executor = create_financial_agent()
        
    def ask(self, query: str) -> Tuple[str, List[Any]]:
        """
        Sends a query to the agent.
        Returns a tuple: (Final Answer, List of intermediate steps for debugging)
        """
        logger.info(f"Agent received query: {query}")
        try:
            response = self.agent_executor.invoke({"input": query})
            answer = response.get("output", "No output generated.")
            steps = response.get("intermediate_steps", [])
            return answer, steps
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return f"Agent Error: {str(e)}", []

if __name__ == "__main__":
    pass
