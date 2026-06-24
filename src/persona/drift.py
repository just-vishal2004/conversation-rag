# src/persona/drift.py

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

# Tone signal patterns
TONE_PATTERNS = {
    "curious": [
        "wonder", "curious", "interesting", "tell me", "what if",
        "how does", "why do", "i want to know", "i've been thinking"
    ],
    "formal": [
        "certainly", "indeed", "furthermore", "however", "therefore",
        "i would like", "please", "thank you", "appreciate", "regards"
    ],
    "casual": [
        "yeah", "yep", "nope", "gonna", "wanna", "gotta", "kinda",
        "sorta", "stuff", "things", "cool", "awesome", "hey"
    ],
    "frustrated": [
        "ugh", "annoying", "frustrated", "tired of", "sick of",
        "can't stand", "hate", "worst", "terrible", "awful",
        "why can't", "this is so", "i give up"
    ],
    "playful": [
        "haha", "lol", "lmao", "funny", "joke", "silly",
        "hilarious", "laugh", "fun", "excited", "yay", "woah"
    ],
    "emotional": [
        "sad", "happy", "love", "miss", "lonely", "scared",
        "nervous", "grateful", "blessed", "hurt", "crying", "tears"
    ],
    "motivated": [
        "goal", "plan", "going to", "will do", "starting",
        "decided", "committed", "focused", "working on", "progress"
    ],
}

# Trigger detection patterns
TRIGGER_PATTERNS = {
    "topic": [
        "job", "work", "career", "school", "study", "exam",
        "health", "doctor", "sick", "money", "finance", "travel",
        "move", "moving", "relationship", "breakup", "family"
    ],
    "person": [
        "mom", "dad", "sister", "brother", "friend", "girlfriend",
        "boyfriend", "wife", "husband", "colleague", "boss", "teacher"
    ],
    "event": [
        "yesterday", "last night", "today", "this morning",
        "happened", "just found out", "news", "accident",
        "birthday", "anniversary", "graduation", "promotion"
    ],
}


@dataclass
class DayProfile:
    """Tone and mood profile for a single conversation day."""
    day_index: int
    conversation_index: int
    tone_scores: dict = field(default_factory=dict)
    dominant_tone: str = "neutral"
    secondary_tone: str = ""
    triggers: list = field(default_factory=list)
    message_count: int = 0
    sample_messages: list = field(default_factory=list)


@dataclass
class DriftEvent:
    """A detected change in persona tone between days."""
    from_day: int
    to_day: int
    from_tone: str
    to_tone: str
    trigger: str
    trigger_type: str
    evidence: str


class PersonaDriftDetector:
    """
    Tracks how the user's mood and tone changes day by day.

    Approach:
    1. For each conversation (day), analyze User 1 messages
    2. Score each tone using keyword matching
    3. Detect dominant tone per day
    4. Compare consecutive days to find drifts
    5. Identify what triggered each drift
    """

    def __init__(self):
        self.day_profiles = []
        self.drift_events = []
        self.timeline = []

    def _score_tones(self, text: str) -> dict:
        """Score each tone for a piece of text."""
        text_lower = text.lower()
        scores = {}
        for tone, keywords in TONE_PATTERNS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[tone] = score
        return scores

    def _detect_triggers(self, text: str) -> list[dict]:
        """Detect what triggered a tone change."""
        text_lower = text.lower()
        triggers = []
        for trigger_type, keywords in TRIGGER_PATTERNS.items():
            for kw in keywords:
                if kw in text_lower:
                    triggers.append({
                        "type": trigger_type,
                        "keyword": kw,
                        "evidence": text[:100]
                    })
        return triggers

    def analyze_day(self, messages: list, day_index: int) -> DayProfile:
        """Analyze all messages from one conversation day."""
        user_messages = [m for m in messages if m.speaker == "User 1"]

        if not user_messages:
            return DayProfile(
                day_index=day_index,
                conversation_index=day_index
            )

        # aggregate tone scores across all messages
        total_scores = defaultdict(int)
        all_triggers = []

        for m in user_messages:
            scores = self._score_tones(m.text)
            for tone, score in scores.items():
                total_scores[tone] += score
            triggers = self._detect_triggers(m.text)
            all_triggers.extend(triggers)

        # find dominant and secondary tones
        sorted_tones = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)
        dominant = sorted_tones[0][0] if sorted_tones else "neutral"
        secondary = sorted_tones[1][0] if len(sorted_tones) > 1 else ""

        # deduplicate triggers by keyword
        seen = set()
        unique_triggers = []
        for t in all_triggers:
            if t["keyword"] not in seen:
                seen.add(t["keyword"])
                unique_triggers.append(t)

        # sample messages for evidence
        samples = [m.text[:80] for m in user_messages[:2]]

        return DayProfile(
            day_index=day_index,
            conversation_index=day_index,
            tone_scores=dict(total_scores),
            dominant_tone=dominant,
            secondary_tone=secondary,
            triggers=unique_triggers[:3],
            message_count=len(user_messages),
            sample_messages=samples,
        )

    def detect_drifts(self, profiles: list[DayProfile]) -> list[DriftEvent]:
        """Compare consecutive day profiles to find tone shifts."""
        drifts = []

        for i in range(1, len(profiles)):
            prev = profiles[i - 1]
            curr = profiles[i]

            if prev.dominant_tone != curr.dominant_tone:
                # find the trigger from current day
                trigger_text = "unknown"
                trigger_type = "unknown"

                if curr.triggers:
                    trigger_text = curr.triggers[0]["keyword"]
                    trigger_type = curr.triggers[0]["type"]

                evidence = curr.sample_messages[0] if curr.sample_messages else ""

                drifts.append(DriftEvent(
                    from_day=prev.day_index,
                    to_day=curr.day_index,
                    from_tone=prev.dominant_tone,
                    to_tone=curr.dominant_tone,
                    trigger=trigger_text,
                    trigger_type=trigger_type,
                    evidence=evidence,
                ))

        return drifts

    def build_timeline(self, profiles: list[DayProfile]) -> list[dict]:
        """Build a readable timeline of tone across days."""
        timeline = []
        for p in profiles:
            tone_str = p.dominant_tone
            if p.secondary_tone:
                tone_str += f" & {p.secondary_tone}"

            timeline.append({
                "day": p.day_index + 1,
                "conversation_index": p.conversation_index,
                "tone": tone_str,
                "dominant": p.dominant_tone,
                "secondary": p.secondary_tone,
                "triggers": [t["keyword"] for t in p.triggers],
                "message_count": p.message_count,
                "sample": p.sample_messages[0] if p.sample_messages else "",
            })
        return timeline

    def analyze(self, messages: list, max_days: int = None) -> dict:
        """
        Main entry point. Analyze all days and return drift report.

        Args:
            messages: Full flat message list from parser
            max_days: Limit number of days to analyze (None = all)

        Returns:
            dict with timeline, drifts, and summary
        """
        from collections import defaultdict

        # group messages by conversation_index (each = one day)
        days = defaultdict(list)
        for m in messages:
            days[m.conversation_index].append(m)

        day_indices = sorted(days.keys())
        if max_days:
            day_indices = day_indices[:max_days]

        logger.info(f"Analyzing {len(day_indices)} days...")

        # analyze each day
        profiles = []
        for idx in day_indices:
            profile = self.analyze_day(days[idx], idx)
            profiles.append(profile)

        self.day_profiles = profiles

        # detect drifts
        self.drift_events = self.detect_drifts(profiles)

        # build timeline
        self.timeline = self.build_timeline(profiles)

        # summary stats
        tone_counts = defaultdict(int)
        for p in profiles:
            tone_counts[p.dominant_tone] += 1

        return {
            "timeline": self.timeline,
            "drifts": [
                {
                    "from_day": d.from_day + 1,
                    "to_day": d.to_day + 1,
                    "from_tone": d.from_tone,
                    "to_tone": d.to_tone,
                    "trigger": d.trigger,
                    "trigger_type": d.trigger_type,
                    "evidence": d.evidence,
                }
                for d in self.drift_events
            ],
            "summary": {
                "total_days": len(profiles),
                "total_drifts": len(self.drift_events),
                "tone_distribution": dict(tone_counts),
                "most_common_tone": max(tone_counts, key=tone_counts.get) if tone_counts else "neutral",
            }
        }

    def save(self, output_path: Path) -> None:
        """Save drift report to JSON."""
        data = {
            "timeline": self.timeline,
            "drifts": [
                {
                    "from_day": d.from_day + 1,
                    "to_day": d.to_day + 1,
                    "from_tone": d.from_tone,
                    "to_tone": d.to_tone,
                    "trigger": d.trigger,
                    "trigger_type": d.trigger_type,
                    "evidence": d.evidence,
                }
                for d in self.drift_events
            ],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Drift report saved to {output_path}")


# ── Manual test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config import RAW_CSV
    from src.ingestion.parser import load_messages

    logging.basicConfig(level=logging.INFO)

    print("Loading messages...")
    messages = load_messages(RAW_CSV)

    detector = PersonaDriftDetector()

    # analyze first 30 days for quick test
    report = detector.analyze(messages, max_days=30)

    print(f"\n--- Timeline (first 30 days) ---")
    for entry in report["timeline"][:30]:
        triggers = ", ".join(entry["triggers"]) if entry["triggers"] else "none"
        print(f"Day {entry['day']:3d} → {entry['tone']:30s} | triggers: {triggers}")

    print(f"\n--- Drift Events ---")
    for drift in report["drifts"][:10]:
        print(f"Day {drift['from_day']} → Day {drift['to_day']}: "
              f"{drift['from_tone']} → {drift['to_tone']} "
              f"(trigger: {drift['trigger']} [{drift['trigger_type']}])")

    print(f"\n--- Summary ---")
    for k, v in report["summary"].items():
        print(f"  {k}: {v}")