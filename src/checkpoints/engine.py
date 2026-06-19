# src/checkpoints/engine.py

import json
import logging
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CheckpointEngine:
    """
    Generates and stores two types of checkpoints:
    1. Topic Checkpoints — one summary per detected topic segment
    2. Message Checkpoints — one summary every 100 messages
    Uses fast extractive summarization (no model needed).
    """

    def __init__(self, checkpoints_file: Path, max_tokens: int = 130):
        self.checkpoints_file = checkpoints_file
        self.max_tokens = max_tokens
        logger.info("Checkpoint engine ready.")
        self.topic_checkpoints = []
        self.message_checkpoints = []

    def _summarize(self, messages: list, context: str) -> str:
        """Fast extractive summary — no model, runs instantly."""
        lines = []
        for m in messages[:3]:
            if len(m.text.split()) > 5:
                lines.append(f"{m.speaker}: {m.text[:150]}")
            if len(lines) == 2:
                break
        if not lines:
            lines = [f"{m.speaker}: {m.text[:150]}" for m in messages[:2]]
        return " | ".join(lines)

    def generate_topic_checkpoints(self, segments: list) -> list[dict]:
        logger.info(f"Generating topic checkpoints for {len(segments)} segments...")
        checkpoints = []
        for seg in tqdm(segments, desc="Topic checkpoints"):
            context = f"Topic {seg.topic_index} (messages {seg.start_index}–{seg.end_index})"
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

    def generate_message_checkpoints(self, messages: list, interval: int) -> list[dict]:
        logger.info(f"Generating message checkpoints every {interval} messages...")
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
        self.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "topic_checkpoints": self.topic_checkpoints,
            "message_checkpoints": self.message_checkpoints,
        }
        with open(self.checkpoints_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved checkpoints to {self.checkpoints_file}")

    def load(self) -> bool:
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