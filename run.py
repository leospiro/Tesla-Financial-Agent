import logging
import sys
import os

# Ensure the root is in PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ui.app import build_app

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
