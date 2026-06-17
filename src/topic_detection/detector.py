# src/topic_detection/detector.py

import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TopicSegment:
    """
    Represents one detected topic segment.

    Attributes:
        topic_index  : Which topic this is (0-based)
        start_index  : global_index of the first message in this segment
        end_index    : global_index of the last message in this segment
        messages     : The actual Message objects in this segment
        summary      : Generated later by the checkpoint engine (empty for now)
    """
    topic_index: int
    start_index: int
    end_index: int
    messages: list        # list[Message] — avoid circular import
    summary: str = ""

    @property
    def length(self) -> int:
        return len(self.messages)

    def __repr__(self) -> str:
        return (
            f"Topic {self.topic_index}: "
            f"messages {self.start_index}–{self.end_index} "
            f"({self.length} messages)"
        )


class TopicDetector:
    """
    Detects topic boundaries in a chronological stream of messages
    using sliding window cosine similarity on sentence embeddings.

    How it works:
        1. Embed all messages in batches (for speed)
        2. For each position i, compute the average embedding of
           messages [i : i+window_size]  → called the "window vector"
        3. Compare consecutive window vectors using cosine similarity
        4. If similarity < threshold → topic boundary detected
        5. Group messages between boundaries into TopicSegment objects
    """

    def __init__(
        self,
        model_name: str,
        window_size: int,
        similarity_threshold: float,
        min_segment_length: int,
    ):
        """
        Args:
            model_name           : HuggingFace model name for embeddings
            window_size          : Number of messages to average per window
            similarity_threshold : Cosine similarity below this = new topic
            min_segment_length   : Minimum messages before a new topic is allowed
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.min_segment_length = min_segment_length

    def embed_messages(self, messages: list, batch_size: int = 64) -> np.ndarray:
        """
        Embed all messages and return a 2D numpy array.

        Args:
            messages  : List of Message objects
            batch_size: How many to embed at once (higher = faster but more RAM)

        Returns:
            np.ndarray of shape (len(messages), embedding_dim)
            e.g. (191592, 384) for all-MiniLM-L6-v2
        """
        texts = [m.text for m in messages]
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        return np.array(embeddings)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        Formula: dot(a, b) / (norm(a) * norm(b))

        Returns a float between -1 and 1.
        1.0  = identical direction (same topic)
        0.0  = orthogonal (unrelated)
        -1.0 = opposite (rare in practice)
        """
        dot_product = np.dot(a, b)
        denominator = np.linalg.norm(a) * np.linalg.norm(b) + 1e-10
        return dot_product / denominator

    def compute_window_vectors(self, embeddings: np.ndarray) -> np.ndarray:
        """
        For each position i, compute the average embedding of the window
        [i : i + window_size].

        Args:
            embeddings: shape (n_messages, embedding_dim)

        Returns:
            np.ndarray of shape (n_windows, embedding_dim)
            where n_windows = n_messages - window_size + 1
        """
        n_messages = len(embeddings)
        window_vectors = []

        for i in range(n_messages - self.window_size + 1):
            window = embeddings[i : i + self.window_size]
            window_vector = np.mean(window, axis=0)
            window_vectors.append(window_vector)

        return np.array(window_vectors)

    def find_boundaries(self, window_vectors: np.ndarray) -> list[int]:
        """
        Compare consecutive window vectors and return indices where
        a topic boundary occurs.

        Uses two complementary strategies:
        1. Absolute threshold: similarity drops below self.similarity_threshold
        2. Local minima: similarity is lower than both its neighbors by a
           meaningful margin (detects relative drops even in high-similarity text)

        A boundary at index i means: after message i, a new topic starts.
        min_segment_length prevents boundaries from being too close together.
        """
        # compute all similarities first
        similarities = []
        for i in range(len(window_vectors) - 1):
            sim = self.cosine_similarity(window_vectors[i], window_vectors[i + 1])
            similarities.append(sim)

        similarities = np.array(similarities)
        boundaries = []
        last_boundary = 0

        for i in range(1, len(similarities) - 1):

            # enforce minimum segment length
            if i - last_boundary < self.min_segment_length:
                continue

            sim = similarities[i]
            prev_sim = similarities[i - 1]
            next_sim = similarities[i + 1]

            # strategy 1: absolute threshold
            is_below_threshold = sim < self.similarity_threshold

            # strategy 2: local minimum — dips relative to neighbors
            is_local_min = (
                sim < prev_sim - 0.005 and
                sim < next_sim - 0.005
            )

            if is_below_threshold or is_local_min:
                boundaries.append(i)
                last_boundary = i

        return boundaries

    def segment_messages(
        self, messages: list, boundaries: list[int]
    ) -> list[TopicSegment]:
        """
        Given a list of messages and boundary indices, group messages
        into TopicSegment objects.

        Args:
            messages  : Full list of Message objects
            boundaries: Indices where new topics start

        Returns:
            List of TopicSegment objects
        """
        segments = []
        prev = 0

        # add a final boundary at the end so the last segment is included
        all_boundaries = boundaries + [len(messages) - 1]

        for topic_index, boundary in enumerate(all_boundaries):
            segment_messages = messages[prev : boundary + 1]

            if not segment_messages:
                continue

            segment = TopicSegment(
                topic_index=topic_index,
                start_index=segment_messages[0].global_index,
                end_index=segment_messages[-1].global_index,
                messages=segment_messages,
            )
            segments.append(segment)
            prev = boundary + 1

        return segments

    def detect(self, messages: list) -> list[TopicSegment]:
        """
        Main entry point. Run the full pipeline:
        embed → window vectors → find boundaries → segment

        Args:
            messages: Full chronological list of Message objects

        Returns:
            List of TopicSegment objects
        """
        logger.info(f"Embedding {len(messages)} messages...")
        embeddings = self.embed_messages(messages)

        logger.info("Computing window vectors...")
        window_vectors = self.compute_window_vectors(embeddings)

        logger.info("Finding topic boundaries...")
        boundaries = self.find_boundaries(window_vectors)
        logger.info(f"Found {len(boundaries)} topic boundaries")

        logger.info("Segmenting messages...")
        segments = self.segment_messages(messages, boundaries)
        logger.info(f"Created {len(segments)} topic segments")

        return segments


# ── Manual test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config import RAW_CSV, EMBEDDING_MODEL, TOPIC_WINDOW_SIZE
    from config import TOPIC_SIMILARITY_THRESHOLD, TOPIC_MIN_SEGMENT_LENGTH
    from src.ingestion.parser import load_messages

    logging.basicConfig(level=logging.INFO)

    # load only first 500 messages for quick testing
    print("Loading messages...")
    all_messages = load_messages(RAW_CSV)
    sample = all_messages[:500]

    detector = TopicDetector(
        model_name=EMBEDDING_MODEL,
        window_size=TOPIC_WINDOW_SIZE,
        similarity_threshold=TOPIC_SIMILARITY_THRESHOLD,
        min_segment_length=TOPIC_MIN_SEGMENT_LENGTH,
    )

    segments = detector.detect(sample)

    print(f"\nFound {len(segments)} topic segments in first 500 messages:\n")
    for seg in segments:
        print(f"  {seg}")
        print(f"    First message: {seg.messages[0].text[:80]}")
        print(f"    Last message : {seg.messages[-1].text[:80]}")
        print()