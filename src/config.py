import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv()

# 如果用户在环境变量里使用了 ALI_API_KEY，我们自动将它映射为 LangChain 内部死认的 OPENAI_API_KEY 和 DASHSCOPE_API_KEY
if "ALI_API_KEY" in os.environ:
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["ALI_API_KEY"]
    if "DASHSCOPE_API_KEY" not in os.environ:
        os.environ["DASHSCOPE_API_KEY"] = os.environ["ALI_API_KEY"]

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "reports"
DB_DIR = BASE_DIR / "chroma_db"

# LLM Configuration
# We can use dashscope or openai, default to OpenAI compatible interface.
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")

# Ensure required directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)
