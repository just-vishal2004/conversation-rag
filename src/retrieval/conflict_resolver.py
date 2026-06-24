# src/retrieval/conflict_resolver.py

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

EMOTIONAL_KEYWORDS = [
    "love", "hate", "miss", "angry", "sad", "happy", "scared",
    "hurt", "cry", "tears", "angry", "fight", "argue", "worried",
    "excited", "devastated", "grateful", "lonely", "afraid", "proud",
]

CONTRADICTION_PAIRS = [
    ({"love", "like", "enjoy", "close", "best"}, {"hate", "dislike", "avoid", "distant", "worst"}),
    ({"alive", "healthy", "well", "fine"}, {"died", "dead", "passed", "sick", "ill"}),
    ({"together", "married", "dating"}, {"broke up", "separated", "divorced", "single"}),
    ({"happy", "excited", "great"}, {"sad", "depressed", "terrible", "awful"}),
]


@dataclass
class ScoredChunk:
    """A retrieved chunk with conflict resolution scores."""
    text: str
    start_index: int
    end_index: int
    recency_score: float
    emotional_score: float
    combined_score: float
    emotional_keywords_found: list = field(default_factory=list)


@dataclass
class ConflictReport:
    """Result of conflict resolution for a query."""
    query: str
    chunks: list[ScoredChunk]
    has_contradiction: bool
    contradiction_detail: str
    merged_answer: str
    resolution_method: str


class ConflictResolver:
    """
    Resolves contradictory information across multiple retrieved chunks.

    Strategy:
    1. Retrieve all relevant chunks for a query
    2. Score each chunk by recency (later = more recent) + emotional weight
    3. Detect contradictions by checking opposing sentiment pairs
    4. Rank by combined score
    5. Return merged coherent answer with contradiction flag

    Why recency + emotional weight?
    - More recent messages reflect current state of affairs
    - Emotionally weighted messages are more memorable and significant
    - Together they approximate "what matters most right now"
    """

    def __init__(self, vector_store, total_messages: int):
        """
        Args:
            vector_store  : VectorStore instance for retrieval
            total_messages: Total message count (for recency normalization)
        """
        self.store = vector_store
        self.total_messages = total_messages

    def _recency_score(self, end_index: int) -> float:
        """
        Normalize message position to 0-1 recency score.
        Later messages = higher score.
        """
        return end_index / max(self.total_messages, 1)

    def _emotional_score(self, text: str) -> tuple[float, list]:
        """
        Score emotional weight of a chunk.
        Returns score and list of found emotional keywords.
        """
        text_lower = text.lower()
        found = [kw for kw in EMOTIONAL_KEYWORDS if kw in text_lower]
        score = min(1.0, len(found) * 0.15)
        return score, found

    def _detect_contradiction(self, chunks: list[ScoredChunk]) -> tuple[bool, str]:
        """
        Check if any two chunks contain contradictory sentiment about the same entity.
        """
        all_texts = [c.text.lower() for c in chunks]

        for pos_set, neg_set in CONTRADICTION_PAIRS:
            chunks_with_pos = [i for i, t in enumerate(all_texts)
                               if any(w in t for w in pos_set)]
            chunks_with_neg = [i for i, t in enumerate(all_texts)
                               if any(w in t for w in neg_set)]

            if chunks_with_pos and chunks_with_neg:
                pos_word = next(w for w in pos_set
                               if any(w in all_texts[i] for i in chunks_with_pos))
                neg_word = next(w for w in neg_set
                               if any(w in all_texts[i] for i in chunks_with_neg))
                detail = (
                    f"Contradiction detected: chunk(s) {chunks_with_pos} contain "
                    f"'{pos_word}' while chunk(s) {chunks_with_neg} contain '{neg_word}'"
                )
                return True, detail

        return False, ""

    def _merge_answer(
        self,
        query: str,
        chunks: list[ScoredChunk],
        has_contradiction: bool,
        contradiction_detail: str,
    ) -> tuple[str, str]:
        """
        Build a coherent merged answer from ranked chunks.
        Returns (answer, resolution_method).
        """
        if not chunks:
            return "No relevant information found.", "no_results"

        top_chunks = chunks[:3]

        if has_contradiction:
            method = "contradiction_flagged_recency_ranked"
            answer = f"⚠️ Note: Contradictory information found across sources.\n\n"
            answer += f"Based on most recent and emotionally significant mentions:\n\n"
            for i, chunk in enumerate(top_chunks):
                answer += f"[Source {i+1} | msgs {chunk.start_index}–{chunk.end_index}]:\n"
                answer += f"{chunk.text[:200]}...\n\n"
            answer += f"Conflict detail: {contradiction_detail}"
        else:
            method = "recency_emotion_ranked"
            answer = f"Based on conversation history (ranked by recency + emotional weight):\n\n"
            for i, chunk in enumerate(top_chunks):
                answer += f"[Source {i+1} | msgs {chunk.start_index}–{chunk.end_index}]:\n"
                answer += f"{chunk.text[:200]}...\n\n"

        return answer, method

    def resolve(self, query: str, top_k: int = 5) -> ConflictReport:
        """
        Main entry point. Retrieve, score, detect conflicts, merge.

        Args:
            query : User's question
            top_k : Number of chunks to retrieve

        Returns:
            ConflictReport with ranked chunks and merged answer
        """
        # retrieve raw chunks
        raw_chunks = self.store.search_chunks(query, top_k=top_k * 2)

        if not raw_chunks:
            return ConflictReport(
                query=query,
                chunks=[],
                has_contradiction=False,
                contradiction_detail="",
                merged_answer="No relevant information found.",
                resolution_method="no_results",
            )

        # score each chunk
        scored = []
        for chunk in raw_chunks:
            recency = self._recency_score(chunk["end_index"])
            emotional, keywords = self._emotional_score(chunk["text"])
            combined = (recency * 0.6) + (emotional * 0.4)

            scored.append(ScoredChunk(
                text=chunk["text"],
                start_index=chunk["start_index"],
                end_index=chunk["end_index"],
                recency_score=round(recency, 3),
                emotional_score=round(emotional, 3),
                combined_score=round(combined, 3),
                emotional_keywords_found=keywords,
            ))

        # sort by combined score descending
        scored.sort(key=lambda x: x.combined_score, reverse=True)
        top_scored = scored[:top_k]

        # detect contradictions
        has_contradiction, contradiction_detail = self._detect_contradiction(top_scored)

        # merge answer
        merged, method = self._merge_answer(
            query, top_scored, has_contradiction, contradiction_detail
        )

        return ConflictReport(
            query=query,
            chunks=top_scored,
            has_contradiction=has_contradiction,
            contradiction_detail=contradiction_detail,
            merged_answer=merged,
            resolution_method=method,
        )


# ── Manual test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config import EMBEDDING_MODEL, FAISS_DIR
    from src.vectorstore.store import VectorStore

    logging.basicConfig(level=logging.INFO)

    store = VectorStore(model_name=EMBEDDING_MODEL, faiss_dir=FAISS_DIR)
    store.load()

    resolver = ConflictResolver(store, total_messages=191592)

    queries = [
        "Did I mention anything about my sister?",
        "What did they say about their job?",
        "What do they feel about their family?",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        report = resolver.resolve(query)
        print(f"Contradiction: {report.has_contradiction}")
        if report.contradiction_detail:
            print(f"Detail: {report.contradiction_detail}")
        print(f"Method: {report.resolution_method}")
        print(f"\nTop chunks by recency+emotion score:")
        for c in report.chunks[:3]:
            print(f"  msgs {c.start_index:6d}–{c.end_index:6d} | "
                  f"recency={c.recency_score:.3f} | "
                  f"emotion={c.emotional_score:.3f} | "
                  f"combined={c.combined_score:.3f}")
        print(f"\nMerged Answer:\n{report.merged_answer[:300]}")