# scripts/build_index.py
"""
Run this once to build the full index from scratch.
Processes all 191k messages — takes 10-20 minutes.
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    RAW_CSV, EMBEDDING_MODEL, FAISS_DIR,
    CHECKPOINTS_FILE, PERSONA_FILE,
    CHUNK_SIZE, CHUNK_OVERLAP,
    TOPIC_WINDOW_SIZE, TOPIC_SIMILARITY_THRESHOLD,
    TOPIC_MIN_SEGMENT_LENGTH, MESSAGE_CHECKPOINT_INTERVAL
)
from src.ingestion.parser import load_messages
from src.topic_detection.detector import TopicDetector
from src.checkpoints.engine import CheckpointEngine
from src.vectorstore.store import VectorStore
from src.persona.extractor import PersonaExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    print("\n" + "="*60)
    print("  Building Full Index")
    print("="*60)

    # Step 1: Load messages
    print("\n[1/5] Loading messages...")
    messages = load_messages(RAW_CSV)
    print(f"  Loaded {len(messages)} messages")

    # Step 2: Detect topics
    print("\n[2/5] Detecting topics...")
    detector = TopicDetector(
        model_name=EMBEDDING_MODEL,
        window_size=TOPIC_WINDOW_SIZE,
        similarity_threshold=TOPIC_SIMILARITY_THRESHOLD,
        min_segment_length=TOPIC_MIN_SEGMENT_LENGTH,
    )
    segments = detector.detect(messages)
    print(f"  Found {len(segments)} topic segments")

    # Step 3: Generate checkpoints
    print("\n[3/5] Generating checkpoints...")
    engine = CheckpointEngine(checkpoints_file=CHECKPOINTS_FILE)

    if CHECKPOINTS_FILE.exists():
        print("  Checkpoints already exist, loading from disk...")
        engine.load()
    else:
        engine.generate_topic_checkpoints(segments)
        engine.generate_message_checkpoints(messages, MESSAGE_CHECKPOINT_INTERVAL)
        engine.save()

    # Step 4: Build vector store
    print("\n[4/5] Building vector store...")
    store = VectorStore(model_name=EMBEDDING_MODEL, faiss_dir=FAISS_DIR)

    with open(CHECKPOINTS_FILE) as f:
        checkpoints = json.load(f)

    store.build_chunks_index(messages, CHUNK_SIZE, CHUNK_OVERLAP)
    store.build_summaries_index(checkpoints)
    store.save()

    # Step 5: Extract persona
    print("\n[5/5] Extracting persona...")
    extractor = PersonaExtractor()
    extractor.extract(messages)
    extractor.save(PERSONA_FILE)

    print("\n" + "="*60)
    print("  ✅ Build complete!")
    print("  Run: python scripts/run_chatbot.py")
    print("="*60)


if __name__ == "__main__":
    main()