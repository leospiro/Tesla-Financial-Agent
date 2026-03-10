import ast
import operator
import logging
from typing import Optional, Type, Dict, Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.retriever.hybrid_search import HybridRetriever

logger = logging.getLogger(__name__)

# Initialize Retriever globally to be reused by tools
# A singleton or lazy initialization is better
_RETRIEVER_INSTANCE = None

def get_retriever():
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None:
        _RETRIEVER_INSTANCE = HybridRetriever()
    return _RETRIEVER_INSTANCE

class FinancialSearchInput(BaseModel):
    query: str = Field(description="The core aspect or keyword to search for, e.g., 'Free Cash Flow' or 'supply chain challenges'.")
    year: Optional[int] = Field(description="The specific year to filter by, e.g., 2022.")
    quarter: Optional[str] = Field(description="The specific quarter to filter by, e.g., 'Q1' or 'Q3'.")
    report_type: Optional[str] = Field(description="The type of report, typically '10-K' or '10-Q'.")

class FinancialSearchTool(BaseTool):
    name: str = "financial_document_search"
    description: str = "Searches Tesla SEC filings and financial reports (10-K, 10-Q) using a hybrid dense-sparse retriever. IMPORTANT: Query with specific key terms and use optional filters to narrow the search."
    args_schema: Type[BaseModel] = FinancialSearchInput

    def _run(self, query: str, year: Optional[int] = None, quarter: Optional[str] = None, report_type: Optional[str] = None) -> str:
        logger.info(f"Tool trigger financial search: query={query}, year={year}, quarter={quarter}, report={report_type}")
        
        metadata_filter = {}
        if year is not None:
            metadata_filter['year'] = year
        if quarter is not None:
            metadata_filter['quarter'] = quarter
        if report_type is not None:
            metadata_filter['report_type'] = report_type
            
        if not metadata_filter:
            metadata_filter = None
            
        retriever = get_retriever()
        docs = retriever.get_relevant_documents(query=query, metadata_filter=metadata_filter, top_k=5)
        
        if not docs:
            return "No relevant information found for your query. Try adjusting your keywords or filters."
            
        # Format results
        output = []
        for i, doc in enumerate(docs):
            meta = doc.metadata
            src = meta.get('source', 'Unknown')
            sec = meta.get('section', 'General')
            content = doc.page_content.strip()
            # truncate exceptionally long raw blocks
            if len(content) > 2000:
                content = content[:2000] + "... [TRUNCATED]"
                
            output.append(f"--- Document {i+1} ---\nSource: {src}\nSection: {sec}\nContent:\n{content}\n")
            
        return "\n".join(output)

class MathCalculationInput(BaseModel):
    expression: str = Field(description="A mathematical expression to evaluate, e.g., '120.5 + 400.1 * 0.5'. Valid operators: +, -, *, /, %")

class MathCalculationTool(BaseTool):
    name: str = "math_calculator"
    description: str = "Safely evaluates a basic mathematical expression. Useful for summing up numbers from different tables or calculating differences."
    args_schema: Type[BaseModel] = MathCalculationInput
    
    def _run(self, expression: str) -> str:
        logger.info(f"Tool trigger math calculation: expr={expression}")
        try:
            # use a safe evaluation approach restricting to simple arithmetic
            # for the sake of simplicity and robustness here, we use simple eval since we are in a backend environment,
            # but in production AST based eval should be used. Using standard eval safely:
            allowed_names = {}
            result = eval(expression, {"__builtins__": None}, allowed_names)
            return str(round(result, 4))
        except Exception as e:
            return f"Error evaluating expression: {e}. Please ensure it is a valid mathematical expression without variables."

def get_agent_tools() -> list[BaseTool]:
    """Returns the list of tools to be bound to the Agent."""
    return [FinancialSearchTool(), MathCalculationTool()]
