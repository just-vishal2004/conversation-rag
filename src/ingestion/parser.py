# src/ingestion/parser.py

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """
    Represents a single utterance in the conversation corpus.

    Attributes:
        global_index      : Absolute position across ALL conversations (0-based)
                            e.g. message 0 is the very first message ever spoken
        conversation_index: Which CSV row this came from (0-based)
                            e.g. row 0 = day 1, row 1 = day 2
        turn_index        : Position within that single conversation (0-based)
                            e.g. turn 0 = first message of that day
        speaker           : "User 1" or "User 2"
        text              : The actual message content (stripped of whitespace)
    """
    global_index: int
    conversation_index: int
    turn_index: int
    speaker: str
    text: str

    def __repr__(self) -> str:
        return (
            f"[global={self.global_index}] "
            f"conv={self.conversation_index} turn={self.turn_index} "
            f"{self.speaker}: {self.text[:60]}"
        )


def parse_conversation(raw: str, conversation_index: int) -> list[Message]:
    """
    Parse one raw conversation string into an ordered list of Message objects.

    The raw string looks like this:
        User 1: Hi! How are you?
        User 2: Good, thanks! How about you?
        User 1: I'm doing well, just tired from work.

    Rules:
        - Split by newline
        - Skip blank lines
        - Skip lines that don't start with "User 1:" or "User 2:"
        - Split each valid line on the FIRST colon only
          (because message text can contain colons e.g. "time: 3pm")
        - Strip whitespace from both speaker and text
        - global_index is NOT set here — caller sets it later

    Args:
        raw               : Full conversation text from one CSV cell
        conversation_index: Which row this came from

    Returns:
        List of Message objects with global_index=0 (placeholder, set by caller)
    """
    messages = []
    turn_index = 0

    for line in raw.split("\n"):
        line = line.strip()

        # skip blank lines
        if not line:
            continue

        # split on first colon only
        parts = line.split(":", 1)

        # skip lines that don't have a colon
        if len(parts) != 2:
            continue

        speaker = parts[0].strip()
        text = parts[1].strip()

        # skip lines that aren't from User 1 or User 2
        if speaker not in ("User 1", "User 2"):
            continue

        # skip empty messages
        if not text:
            continue

        messages.append(Message(
            global_index=0,          # placeholder — caller sets this
            conversation_index=conversation_index,
            turn_index=turn_index,
            speaker=speaker,
            text=text
        ))
        turn_index += 1

    return messages


def load_messages(csv_path: Path) -> list[Message]:
    """
    Load ALL conversations from the CSV and return one flat list of Messages
    ordered chronologically (row 0 first, then row 1, etc.)

    The CSV has NO header row. Each row has one column: the raw conversation.

    Steps:
        1. Open the CSV
        2. For each row, call parse_conversation()
        3. Assign global_index sequentially across all conversations
        4. Extend the master list

    Args:
        csv_path: Path to conversations.csv

    Returns:
        List[Message] — flat, ordered from first to last message ever
    """
    all_messages = []
    global_counter = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        for conversation_index, row in enumerate(reader):
            # each row has one column — the raw conversation string
            if not row:
                continue

            raw = row[0]
            messages = parse_conversation(raw, conversation_index)

            # assign global_index sequentially
            for message in messages:
                message.global_index = global_counter
                global_counter += 1

            all_messages.extend(messages)
            
    logger.info(f"Loaded {len(all_messages)} messages from {csv_path}")
    return all_messages


def message_stats(messages: list[Message]) -> dict:
    """
    Return basic statistics about the loaded corpus.
    Used to sanity-check the data before building the index.

    Returns a dict with:
        total_messages         : int
        total_conversations    : int
        avg_messages_per_conv  : float (rounded to 2 decimal places)
        unique_speakers        : list of unique speaker names found
    """
    if not messages:
        return {}

    total_messages = len(messages)
    total_conversations = messages[-1].conversation_index + 1
    avg_messages_per_conv = round(total_messages / total_conversations, 2)
    unique_speakers = list(set(m.speaker for m in messages))

    return {
        "total_messages": total_messages,
        "total_conversations": total_conversations,
        "avg_messages_per_conv": avg_messages_per_conv,
        "unique_speakers": unique_speakers,
    }


# ── Manual test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config import RAW_CSV

    logging.basicConfig(level=logging.INFO)

    print("Loading messages...")
    messages = load_messages(RAW_CSV)

    print("\nFirst 3 messages:")
    for m in messages[:3]:
        print(" ", m)

    print("\nLast 3 messages:")
    for m in messages[-3:]:
        print(" ", m)

    print("\nStats:")
    stats = message_stats(messages)
    for k, v in stats.items():
        print(f"  {k}: {v}")