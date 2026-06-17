# src/checkpoints/engine.py

import json
import logging
from pathlib import Path
from transformers import BartForConditionalGeneration, BartTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CheckpointEngine:
    """
    Generates and stores two types of checkpoints:

    1. Topic Checkpoints — one summary per detected topic segment
    2. Message Checkpoints — one summary every 100 messages

    Uses a local HuggingFace summarization model (free, no API needed).
    All checkpoints saved to storage/checkpoints.json.
    """

    def __init__(self, checkpoints_file: Path, max_tokens: int = 130):
        """
        Args:
            checkpoints_file: Path to save/load checkpoints JSON
            max_tokens      : Max tokens for each summary (keep under 130)
        """
        self.checkpoints_file = checkpoints_file
        self.max_tokens = max_tokens

        logger.info("Loading local summarization model (first run downloads ~1.6GB)...")
        from transformers import BartForConditionalGeneration, BartTokenizer
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        logger.info("Summarization model loaded.")

        self.topic_checkpoints = []
        self.message_checkpoints = []

    def _format_transcript(self, messages: list) -> str:
        """
        Format a list of Message objects into a plain transcript string.
        BART works best with clean paragraph-style input under ~1024 tokens.
        We truncate to 800 words to stay within model limits.
        """
        lines = [f"{m.speaker}: {m.text}" for m in messages]
        transcript = " ".join(lines)

        # truncate to 800 words to avoid exceeding model input limit
        words = transcript.split()
        if len(words) > 800:
            transcript = " ".join(words[:800])

        return transcript

    def _summarize(self, messages: list, context: str) -> str:
        """
        Summarize a list of messages using the local BART model.

        Args:
            messages: List of Message objects
            context : Label string for logging only

        Returns:
            Summary string
        """
        transcript = self._format_transcript(messages)

        # need at least 50 words for summarization to make sense
        if len(transcript.split()) < 50:
            return " ".join(transcript.split()[:30]) + "..."

        try:
            inputs = self.tokenizer(
                transcript,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
            )
            summary_ids = self.bart_model.generate(
                inputs["input_ids"],
                max_length=self.max_tokens,
                min_length=30,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
            )
            return self.tokenizer.decode(
                summary_ids[0], skip_special_tokens=True
            ).strip()
        except Exception as e:
            logger.warning(f"Summarization failed for {context}: {e}")
            fallback = " | ".join(
                f"{m.speaker}: {m.text}" for m in messages[:2]
            )
            return fallback[:300]

    def generate_topic_checkpoints(self, segments: list) -> list[dict]:
        """
        Generate one summary per topic segment.

        Returns list of dicts with keys:
            type, topic_index, start_index, end_index,
            message_count, summary
        """
        logger.info(f"Generating topic checkpoints for {len(segments)} segments...")
        checkpoints = []

        for seg in tqdm(segments, desc="Topic checkpoints"):
            context = (
                f"Topic {seg.topic_index} "
                f"(messages {seg.start_index}–{seg.end_index})"
            )
            summary = self._summarize(seg.messages, context)

            checkpoint = {
                "type": "topic",
                "topic_index": seg.topic_index,
                "start_index": seg.start_index,
                "end_index": seg.end_index,
                "message_count": seg.length,
                "summary": summary,
            }
            checkpoints.append(checkpoint)
            seg.summary = summary

        self.topic_checkpoints = checkpoints
        logger.info(f"Generated {len(checkpoints)} topic checkpoints")
        return checkpoints

    def generate_message_checkpoints(
        self, messages: list, interval: int
    ) -> list[dict]:
        """
        Generate one summary for every `interval` messages.

        Returns list of dicts with keys:
            type, batch_number, start_index, end_index,
            message_count, summary
        """
        logger.info(
            f"Generating message checkpoints every {interval} messages..."
        )
        checkpoints = []
        batches = range(0, len(messages), interval)

        for batch_number, start in enumerate(tqdm(batches, desc="Message checkpoints")):
            batch = messages[start : start + interval]

            if not batch:
                continue

            start_index = batch[0].global_index
            end_index = batch[-1].global_index
            context = f"Messages {start_index}–{end_index} (batch {batch_number})"

            summary = self._summarize(batch, context)

            checkpoint = {
                "type": "message_interval",
                "batch_number": batch_number,
                "start_index": start_index,
                "end_index": end_index,
                "message_count": len(batch),
                "summary": summary,
            }
            checkpoints.append(checkpoint)

        self.message_checkpoints = checkpoints
        logger.info(f"Generated {len(checkpoints)} message checkpoints")
        return checkpoints

    def save(self) -> None:
        """Save all checkpoints to JSON file."""
        self.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "topic_checkpoints": self.topic_checkpoints,
            "message_checkpoints": self.message_checkpoints,
        }

        with open(self.checkpoints_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved checkpoints to {self.checkpoints_file}")

    def load(self) -> bool:
        """
        Load checkpoints from JSON file if it exists.
        Returns True if loaded, False if file not found.
        """
        if not self.checkpoints_file.exists():
            logger.info("No existing checkpoints file found")
            return False

        with open(self.checkpoints_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.topic_checkpoints = data.get("topic_checkpoints", [])
        self.message_checkpoints = data.get("message_checkpoints", [])

        logger.info(
            f"Loaded {len(self.topic_checkpoints)} topic checkpoints "
            f"and {len(self.message_checkpoints)} message checkpoints"
        )
        return True


# ── Manual test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from config import (
        RAW_CSV, EMBEDDING_MODEL, TOPIC_WINDOW_SIZE,
        TOPIC_SIMILARITY_THRESHOLD, TOPIC_MIN_SEGMENT_LENGTH,
        CHECKPOINTS_FILE, MESSAGE_CHECKPOINT_INTERVAL
    )
    from src.ingestion.parser import load_messages
    from src.topic_detection.detector import TopicDetector

    logging.basicConfig(level=logging.INFO)

    # use only first 200 messages to test quickly
    print("Loading messages...")
    all_messages = load_messages(RAW_CSV)
    sample = all_messages[:200]

    print("Detecting topics...")
    detector = TopicDetector(
        model_name=EMBEDDING_MODEL,
        window_size=TOPIC_WINDOW_SIZE,
        similarity_threshold=TOPIC_SIMILARITY_THRESHOLD,
        min_segment_length=TOPIC_MIN_SEGMENT_LENGTH,
    )
    segments = detector.detect(sample)
    print(f"Found {len(segments)} topic segments")

    print("Generating checkpoints...")
    engine = CheckpointEngine(
        checkpoints_file=CHECKPOINTS_FILE,
    )

    engine.generate_topic_checkpoints(segments)
    engine.generate_message_checkpoints(sample, MESSAGE_CHECKPOINT_INTERVAL)
    engine.save()

    print("\n--- Topic Checkpoints (first 3) ---")
    for cp in engine.topic_checkpoints[:3]:
        print(f"\nTopic {cp['topic_index']} "
              f"(msgs {cp['start_index']}–{cp['end_index']}):")
        print(f"  {cp['summary']}")

    print("\n--- Message Checkpoints ---")
    for cp in engine.message_checkpoints:
        print(f"\nBatch {cp['batch_number']} "
              f"(msgs {cp['start_index']}–{cp['end_index']}):")
        print(f"  {cp['summary']}")