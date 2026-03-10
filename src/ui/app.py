import gradio as gr
import logging
from src.agent.react_agent import FinancialAgent

logger = logging.getLogger(__name__)

# Initialize agent globally for the UI sessions
financial_agent = FinancialAgent()

def format_steps_html(steps):
    """Formats the intermediate ReAct steps into HTML blocks for the UI accordion."""
    if not steps:
        return "<i>No intermediate steps or direct answer generation.</i>"
        
    html = "<div style='font-family: monospace; font-size: 0.9em;'>"
    for idx, step in enumerate(steps):
        action = step[0]
        observation = step[1]
        
        html += f"<div style='margin-bottom: 10px; padding: 10px; border-left: 3px solid #6366f1; background-color: #f8fafc; color: #1e293b;'>"
        html += f"<b>Step {idx+1}: {action.tool}</b><br/>"
        html += f"<span style='color: #475569;'>Input: {action.tool_input}</span><br/><br/>"
        
        # truncated observation for UI
        obs_str = str(observation)
        if len(obs_str) > 800:
            obs_str = obs_str[:800] + "... [TRUNCATED in UX]"
            
        html += f"<span style='color: #0f172a;'><i>Observation:</i><br/>{obs_str}</span>"
        html += "</div>"
    html += "</div>"
    return html

def process_query(message, history):
    """
    Handler for Gradio ChatInterface. Currently we don't strictly feed history,
    but treat each question independently to focus on complex QA extraction.
    """
    logger.info(f"UI Query received: {message}")
    
    # 1. Ask Agent
    answer, steps = financial_agent.ask(message)
    
    # 2. Format intermediate thought trace
    trace_html = format_steps_html(steps)
    
    # We yield or return both the answer and the trace appended maybe?
    # Or in Gradio > 4.x we can return multiple outputs if we use Blocks, but for ChatInterface we return just text.
    # We will append the trace at the bottom inside an HTML <details> tag.
    
    final_response = f"{answer}\n\n"
    final_response += f"<details><summary>🔍 View Agent Reasoning Trace</summary>\n{trace_html}\n</details>"
    
    return final_response

def build_app():
    with gr.Blocks(title="Tesla Financial QA System") as demo:
        gr.Markdown(
            """
            # 🚗 Tesla Financial & Report QA
            **A ReAct-powered Analytical Agent for Tesla 10-K and 10-Q Reports (2021-2025).**
            Ask complex queries comparing years, summing metrics, or tracking supply chain narrative.
            """
        )
        
        chatbot = gr.ChatInterface(
            fn=process_query,
            examples=[
                "比较2021年和2023年关于“中国市场风险 (China market risks)”的描述有什么显著的变化？",
                "计算2022年四个季度的“研发费用”总和，并与2021年全年对比，增加或减少了多少？",
                "查找2022年中哪份季报提到了“供应链挑战”并且该季度的总营收是多少？"
            ],
            type="messages"
        )
        
    return demo

if __name__ == "__main__":
    app = build_app()
    app.launch()
