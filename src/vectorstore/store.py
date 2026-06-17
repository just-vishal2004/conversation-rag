# src/vectorstore/store.py

import json
import logging
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Stores and retrieves conversation chunks and checkpoint summaries
    using FAISS for fast similarity search.

    Two separate FAISS indexes:
        1. chunks_index    — overlapping windows of raw messages
        2. summaries_index — topic + message checkpoint summaries

    Why two indexes?
        Chunks give exact message context.
        Summaries give high-level topic context.
        Retrieval combines both for better answers.

    Storage layout (all inside faiss_dir/):
        chunks.index       — FAISS binary index for chunks
        chunks_meta.json   — metadata for each chunk (text, indices, etc.)
        summaries.index    — FAISS binary index for summaries
        summaries_meta.json— metadata for each summary
    """

    def __init__(self, model_name: str, faiss_dir: Path):
        """
        Args:
            model_name: SentenceTransformer model (same one used in detector)
            faiss_dir : Directory to save/load FAISS index files
        """
        self.faiss_dir = faiss_dir
        self.faiss_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # these get populated by build() or load()
        self.chunks_index = None
        self.chunks_meta = []       # list of dicts
        self.summaries_index = None
        self.summaries_meta = []    # list of dicts

    # ── Building ──────────────────────────────────────────────────────────────

    def _create_chunks(
        self, messages: list, chunk_size: int, overlap: int
    ) -> list[dict]:
        """
        Slice the message list into overlapping chunks.

        Example with chunk_size=10, overlap=2:
            chunk 0: messages[0:10]
            chunk 1: messages[8:18]   ← 2 message overlap
            chunk 2: messages[16:26]
            ...

        Each chunk is stored as a dict:
            text        : joined text of all messages in chunk
            start_index : global_index of first message
            end_index   : global_index of last message
            message_count: number of messages

        Args:
            messages  : Full list of Message objects
            chunk_size: Messages per chunk
            overlap   : Overlapping messages between consecutive chunks

        Returns:
            List of chunk dicts
        """
        chunks = []
        step = chunk_size - overlap

        for start in range(0, len(messages), step):
            chunk_messages = messages[start : start + chunk_size]

            if not chunk_messages:
                continue

            # join all messages into one readable text block
            text = "\n".join(
                f"{m.speaker}: {m.text}" for m in chunk_messages
            )

            chunks.append({
                "text": text,
                "start_index": chunk_messages[0].global_index,
                "end_index": chunk_messages[-1].global_index,
                "message_count": len(chunk_messages),
            })

        return chunks

    def build_chunks_index(
        self, messages: list, chunk_size: int, overlap: int
    ) -> None:
        """
        Create chunks, embed them, and build the FAISS index.

        Steps:
            1. Call _create_chunks()
            2. Extract the 'text' field from each chunk
            3. Embed all texts using self.model.encode()
            4. Create a FAISS IndexFlatIP index (inner product = cosine on normalized vectors)
            5. Normalize embeddings with faiss.normalize_L2()
            6. Add to index with index.add()
            7. Store in self.chunks_index and self.chunks_meta
        """
        logger.info("Creating message chunks...")
        self.chunks_meta = self._create_chunks(messages, chunk_size, overlap)
        logger.info(f"Created {len(self.chunks_meta)} chunks")

        logger.info("Embedding chunks...")
        texts = [c["text"] for c in self.chunks_meta]
        embeddings = self.model.encode(
            texts, batch_size=64, show_progress_bar=True
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        # normalize so inner product == cosine similarity
        faiss.normalize_L2(embeddings)

        logger.info("Building FAISS chunks index...")
        self.chunks_index = faiss.IndexFlatIP(self.embedding_dim)
        self.chunks_index.add(embeddings)
        logger.info(f"Chunks index contains {self.chunks_index.ntotal} vectors")

    def build_summaries_index(self, checkpoints: dict) -> None:
        """
        Embed all checkpoint summaries and build the FAISS summaries index.

        Args:
            checkpoints: dict loaded from checkpoints.json with keys:
                         'topic_checkpoints' and 'message_checkpoints'

        Steps:
            1. Combine topic_checkpoints + message_checkpoints into one list
            2. Extract the 'summary' field from each
            3. Embed all summaries
            4. Normalize and add to a new IndexFlatIP
            5. Store in self.summaries_index and self.summaries_meta
        """
        all_checkpoints = (
            checkpoints.get("topic_checkpoints", []) +
            checkpoints.get("message_checkpoints", [])
        )

        logger.info(f"Embedding {len(all_checkpoints)} summaries...")
        texts = [cp["summary"] for cp in all_checkpoints]
        embeddings = self.model.encode(
            texts, batch_size=64, show_progress_bar=True
        )
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)

        logger.info("Building FAISS summaries index...")
        self.summaries_index = faiss.IndexFlatIP(self.embedding_dim)
        self.summaries_index.add(embeddings)
        self.summaries_meta = all_checkpoints
        logger.info(
            f"Summaries index contains {self.summaries_index.ntotal} vectors"
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string and return a normalized float32 array
        ready for FAISS search.

        Returns shape: (1, embedding_dim)
        """
        embedding = self.model.encode([query])
        embedding = np.array(embedding, dtype=np.float32)
        faiss.normalize_L2(embedding)
        return embedding

    def search_chunks(self, query: str, top_k: int) -> list[dict]:
        """
        Find the top_k most relevant message chunks for a query.

        Args:
            query : Natural language question
            top_k : Number of results to return

        Returns:
            List of chunk dicts (from self.chunks_meta) sorted by relevance
        """
        query_vec = self._embed_query(query)
        scores, indices = self.chunks_index.search(query_vec, top_k)

        results = []
        for i in indices[0]:
            if i != -1:   # FAISS returns -1 for empty slots
                results.append(self.chunks_meta[i])

        return results

    def search_summaries(self, query: str, top_k: int) -> list[dict]:
        """
        Find the top_k most relevant summaries for a query.
        Same pattern as search_chunks but uses summaries_index.
        """
        query_vec = self._embed_query(query)
        scores, indices = self.summaries_index.search(query_vec, top_k)

        results = []
        for i in indices[0]:
            if i != -1:
                results.append(self.summaries_meta[i])

        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """
        Save both FAISS indexes and their metadata to faiss_dir.
        """
        logger.info(f"Saving vector store to {self.faiss_dir}...")

        faiss.write_index(
            self.chunks_index,
            str(self.faiss_dir / "chunks.index")
        )
        faiss.write_index(
            self.summaries_index,
            str(self.faiss_dir / "summaries.index")
        )

        with open(self.faiss_dir / "chunks_meta.json", "w") as f:
            json.dump(self.chunks_meta, f, indent=2)

        with open(self.faiss_dir / "summaries_meta.json", "w") as f:
            json.dump(self.summaries_meta, f, indent=2)

        logger.info("Vector store saved.")

    def load(self) -> bool:
        """
        Load indexes and metadata from disk.
        Returns True if successful, False if files don't exist.
        """
        chunks_path = self.faiss_dir / "chunks.index"
        summaries_path = self.faiss_dir / "summaries.index"

        if not chunks_path.exists() or not summaries_path.exists():
            logger.info("No existing vector store found")
            return False

        logger.info("Loading vector store from disk...")
        self.chunks_index = faiss.read_index(str(chunks_path))
        self.summaries_index = faiss.read_index(str(summaries_path))

        with open(self.faiss_dir / "chunks_meta.json") as f:
            self.chunks_meta = json.load(f)

        with open(self.faiss_dir / "summaries_meta.json") as f:
            self.summaries_meta = json.load(f)

        logger.info(
            f"Loaded chunks index ({self.chunks_index.ntotal} vectors) "
            f"and summaries index ({self.summaries_index.ntotal} vectors)"
        )
        return True


# ── Manual test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from config import (
        RAW_CSV, EMBEDDING_MODEL, FAISS_DIR,
        CHECKPOINTS_FILE, CHUNK_SIZE, CHUNK_OVERLAP,
        TOP_K_CHUNKS, TOP_K_SUMMARIES
    )
    from src.ingestion.parser import load_messages

    logging.basicConfig(level=logging.INFO)

    print("Loading messages...")
    messages = load_messages(RAW_CSV)
    # use first 500 messages for quick test
    sample = messages[:500]

    print("Loading checkpoints...")
    with open(CHECKPOINTS_FILE) as f:
        checkpoints = json.load(f)

    store = VectorStore(model_name=EMBEDDING_MODEL, faiss_dir=FAISS_DIR)

    print("Building indexes...")
    store.build_chunks_index(sample, CHUNK_SIZE, CHUNK_OVERLAP)
    store.build_summaries_index(checkpoints)
    store.save()

    print("\n--- Test Query ---")
    query = "What do they talk about regarding pets?"

    print(f"\nQuery: {query}")
    print("\nTop chunks:")
    for chunk in store.search_chunks(query, TOP_K_CHUNKS):
        print(f"  [{chunk['start_index']}–{chunk['end_index']}] "
              f"{chunk['text'][:100]}...")

    print("\nTop summaries:")
    for summary in store.search_summaries(query, TOP_K_SUMMARIES):
        print(f"  [{summary.get('start_index')}–{summary.get('end_index')}] "
              f"{summary['summary'][:100]}...")