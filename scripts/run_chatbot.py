# scripts/run_chatbot.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.chatbot.bot import run_chatbot

if __name__ == "__main__":
    run_chatbot()