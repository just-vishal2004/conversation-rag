# config.py
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
FAISS_DIR = STORAGE_DIR / "faiss_index"

RAW_CSV = DATA_DIR / "conversations.csv"
CHECKPOINTS_FILE = STORAGE_DIR / "checkpoints.json"
PERSONA_FILE = STORAGE_DIR / "persona.json"

# ── Embedding Model ────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Topic Detection ────────────────────────────────────────────────────────────
TOPIC_WINDOW_SIZE = 5
TOPIC_SIMILARITY_THRESHOLD = 0.90   # updated based on actual score distribution
TOPIC_MIN_SEGMENT_LENGTH = 10

# ── Checkpoints ────────────────────────────────────────────────────────────────
MESSAGE_CHECKPOINT_INTERVAL = 100

# ── Retrieval ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = 10
CHUNK_OVERLAP = 2
TOP_K_CHUNKS = 5
TOP_K_SUMMARIES = 3

# ── LLM ───────────────────────────────────────────────────────────────────────
ANTHROPIC_MODEL = "claude-sonnet-4-6"   # kept for chatbot only
MAX_TOKENS_SUMMARY = 130                # used by local summarizer
MAX_TOKENS_ANSWER = 600
MAX_TOKENS_PERSONA = 1000
