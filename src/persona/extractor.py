# src/persona/extractor.py

import json
import logging
import re
from pathlib import Path
from transformers import pipeline

logger = logging.getLogger(__name__)


class PersonaExtractor:
    """
    Extracts structured user persona from conversation data.
    Uses keyword/pattern matching on actual conversation text.
    All traits are evidence-based — no hallucination.
    """

    def __init__(self):
        self.persona = {
            "habits": [],
            "personal_facts": [],
            "personality_traits": [],
            "communication_style": {},
            "interests": [],
            "relationships": [],
        }
        self._seen = set()  # deduplication

    def _add(self, category: str, value: str, evidence: str):
        """Add a trait only if not already seen."""
        key = f"{category}:{value.lower().strip()}"
        if key not in self._seen:
            self._seen.add(key)
            entry = {"value": value, "evidence": evidence[:120]}
            if isinstance(self.persona[category], list):
                self.persona[category].append(entry)

    def _extract_from_message(self, text: str, speaker: str):
        """Run all pattern checks on a single message."""
        t = text.lower()

        # ── Pets ──────────────────────────────────────────────────────────────
        if any(w in t for w in ["my dog", "my cat", "my pet", "my horse", "my bird"]):
            self._add("personal_facts", f"Has a pet", text)

        # ── Family ────────────────────────────────────────────────────────────
        if any(w in t for w in ["my wife", "my husband", "my girlfriend", "my boyfriend"]):
            self._add("relationships", "In a relationship", text)
        if any(w in t for w in ["my kids", "my son", "my daughter", "my children"]):
            self._add("relationships", "Has children", text)
        if any(w in t for w in ["my mom", "my dad", "my parents", "my sister", "my brother"]):
            self._add("relationships", "Close with family", text)

        # ── Work ──────────────────────────────────────────────────────────────
        for job in ["teacher", "nurse", "doctor", "engineer", "developer",
                    "designer", "manager", "chef", "firefighter", "trainer",
                    "artist", "writer", "student", "musician", "programmer"]:
            if f"i'm a {job}" in t or f"i am a {job}" in t or f"i work as a {job}" in t:
                self._add("personal_facts", f"Occupation: {job}", text)

        # ── Hobbies ───────────────────────────────────────────────────────────
        for hobby in ["hiking", "fishing", "cooking", "reading", "gaming",
                      "gardening", "painting", "yoga", "running", "cycling",
                      "swimming", "travelling", "photography", "music", "dancing"]:
            if f"love {hobby}" in t or f"enjoy {hobby}" in t or f"like {hobby}" in t:
                self._add("interests", f"Enjoys {hobby}", text)

        # ── Sleep habits ──────────────────────────────────────────────────────
        if any(w in t for w in ["night owl", "stay up late", "up late", "can't sleep"]):
            self._add("habits", "Late sleeper / night owl", text)
        if any(w in t for w in ["early riser", "wake up early", "morning person"]):
            self._add("habits", "Early riser", text)

        # ── Food ──────────────────────────────────────────────────────────────
        if any(w in t for w in ["vegetarian", "vegan", "gluten free"]):
            self._add("habits", f"Dietary preference mentioned", text)
        if "love to cook" in t or "enjoy cooking" in t:
            self._add("habits", "Enjoys cooking", text)

        # ── Personality ───────────────────────────────────────────────────────
        if any(w in t for w in ["haha", "lol", "lmao", "😂", "funny", "joke"]):
            self._add("personality_traits", "Has a sense of humor", text)
        if any(w in t for w in ["i'm nervous", "i'm anxious", "i worry", "i'm scared"]):
            self._add("personality_traits", "Shows anxiety or nervousness", text)
        if any(w in t for w in ["i'm grateful", "i'm thankful", "i appreciate", "blessed"]):
            self._add("personality_traits", "Expressive and grateful", text)
        if any(w in t for w in ["i love learning", "i like to learn", "curious about"]):
            self._add("personality_traits", "Curious and loves learning", text)

    def _analyze_communication_style(self, messages: list):
        """Analyze message length, tone, emoji usage."""
        lengths = [len(m.text.split()) for m in messages]
        avg_length = sum(lengths) / len(lengths) if lengths else 0

        emoji_count = sum(
            1 for m in messages
            if any(c in m.text for c in ["😂", "❤️", "😊", "🙏", "😍", "👍"])
        )

        question_count = sum(1 for m in messages if "?" in m.text)

        self.persona["communication_style"] = {
            "avg_message_length_words": round(avg_length, 1),
            "style": (
                "concise" if avg_length < 10
                else "moderate" if avg_length < 20
                else "detailed"
            ),
            "uses_emojis": emoji_count > 5,
            "asks_many_questions": question_count > len(messages) * 0.2,
            "total_messages_analyzed": len(messages),
        }

    def extract(self, messages: list) -> dict:
        """
        Main entry point. Process all messages and return persona dict.

        Args:
            messages: Full list of Message objects

        Returns:
            Structured persona dict
        """
        logger.info(f"Extracting persona from {len(messages)} messages...")

        # focus on User 1 as the primary user
        user1_messages = [m for m in messages if m.speaker == "User 1"]

        for m in user1_messages:
            self._extract_from_message(m.text, m.speaker)

        self._analyze_communication_style(user1_messages)

        logger.info(
            f"Extracted: {len(self.persona['habits'])} habits, "
            f"{len(self.persona['personal_facts'])} facts, "
            f"{len(self.persona['personality_traits'])} traits, "
            f"{len(self.persona['interests'])} interests"
        )
        return self.persona

    def save(self, output_path: Path):
        """Save persona to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.persona, f, indent=2, ensure_ascii=False)
        logger.info(f"Persona saved to {output_path}")

    def load(self, input_path: Path) -> dict:
        """Load persona from JSON file."""
        with open(input_path, "r", encoding="utf-8") as f:
            self.persona = json.load(f)
        return self.persona


# ── Manual test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config import RAW_CSV, PERSONA_FILE
    from src.ingestion.parser import load_messages

    logging.basicConfig(level=logging.INFO)

    print("Loading messages...")
    messages = load_messages(RAW_CSV)

    extractor = PersonaExtractor()
    persona = extractor.extract(messages)
    extractor.save(PERSONA_FILE)

    print("\n--- Persona ---")
    print(json.dumps(persona, indent=2))