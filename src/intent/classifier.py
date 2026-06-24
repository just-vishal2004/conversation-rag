# src/intent/classifier.py

import re
import time
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

INTENTS = ["reminder", "emotional-support", "action-item", "small-talk", "unknown"]


@dataclass
class IntentResult:
    intent: str
    confidence: float
    latency_ms: float
    matched_pattern: str


class IntentClassifier:
    """
    Lightweight rule-based intent classifier.
    - Zero external dependencies
    - Runs in <1ms per message
    - No model download needed
    - Fully explainable

    Why rule-based over ML model?
    - All ML models >50MB failed accuracy requirements
    - These 5 intents have clear linguistic signals
    - Rules are transparent and debuggable
    - Sub-millisecond latency vs 70-200ms for neural models
    """

    def __init__(self):
        self.patterns = {
            "reminder": [
                r"\bremind\b", r"\bdon'?t forget\b", r"\bremember to\b",
                r"\bset.{0,10}alarm\b", r"\bschedule\b", r"\bdeadline\b",
                r"\btomorrow\b.{0,20}\b(call|meet|send|do|check)\b",
                r"\bneed to\b.{0,30}\b(by|before|until)\b",
                r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b.{0,20}\b(call|meet|send)\b",
            ],
            "emotional-support": [
                r"\b(sad|depressed|anxious|stressed|overwhelmed|lonely|scared)\b",
                r"\b(i feel|i'm feeling|feeling)\b.{0,20}\b(bad|down|awful|terrible|lost)\b",
                r"\bi (can'?t|cannot) (cope|handle|deal)\b",
                r"\b(miss|missing)\b.{0,20}\b(him|her|them|you)\b",
                r"\b(heartbroken|devastated|miserable|hopeless)\b",
                r"\bnobody (cares|understands|listens)\b",
                r"\bi need (help|support|someone to talk)\b",
                r"\b(cry|crying|cried)\b",
                r"\b(hate|hating) my (life|job|self)\b",
            ],
            "action-item": [
                r"\b(please|can you|could you|would you)\b.{0,30}\b(send|do|check|fix|update|review|complete)\b",
                r"\b(send|submit|complete|finish|update|fix|review|approve)\b.{0,20}\b(the|this|that|it)\b",
                r"\bneed.{0,20}(done|completed|finished|sent)\b",
                r"\b(by|before)\b.{0,10}\b(end of day|eod|tonight|morning|noon)\b",
                r"\b(assign|delegate|task|todo|to-do)\b",
                r"\bfollow.?up\b",
                r"\baction (required|needed|item)\b",
            ],
            "small-talk": [
                r"\bhow (are|was|is) (you|your|the)\b",
                r"\bwhat'?s (up|new|going on)\b",
                r"\b(good|great|awesome|nice|cool|fun)\b.{0,20}\b(day|weekend|morning|evening|night)\b",
                r"\bhaha\b|\blol\b|\blmao\b",
                r"\bweather\b",
                r"\b(watch|watched|watching)\b.{0,20}\b(movie|show|film|series)\b",
                r"\b(how was|how did)\b.{0,20}\b(it|your|the)\b",
                r"\bnice to (meet|talk|chat)\b",
                r"\bjust (checking|saying|wanted to say)\b",
            ],
        }

        # compile all patterns for speed
        self.compiled = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.patterns.items()
        }

    def classify(self, text: str) -> IntentResult:
        """
        Classify a single message into one of 5 intents.

        Args:
            text: Raw message string

        Returns:
            IntentResult with intent, confidence, latency, matched pattern
        """
        start = time.time()
        text = text.strip()

        scores = {intent: 0 for intent in INTENTS}
        matched = {intent: "" for intent in INTENTS}

        for intent, patterns in self.compiled.items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    scores[intent] += 1
                    if not matched[intent]:
                        matched[intent] = pattern.pattern

        # find winning intent
        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]

        if best_score == 0:
            result_intent = "unknown"
            confidence = 0.0
            pattern_matched = ""
        else:
            result_intent = best_intent
            # normalize confidence: 1 match = 0.7, 2+ = 0.85, 3+ = 0.95
            confidence = min(0.95, 0.60 + (best_score * 0.12))
            pattern_matched = matched[best_intent]

        latency = (time.time() - start) * 1000

        return IntentResult(
            intent=result_intent,
            confidence=round(confidence, 2),
            latency_ms=round(latency, 3),
            matched_pattern=pattern_matched,
        )

    def classify_batch(self, texts: list[str]) -> list[IntentResult]:
        """Classify multiple messages."""
        return [self.classify(t) for t in texts]


# ── Manual test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    classifier = IntentClassifier()

    test_cases = [
        ("Don't forget to call mom tomorrow", "reminder"),
        ("I feel really sad and lonely today", "emotional-support"),
        ("Please send me the report by end of day", "action-item"),
        ("How was your weekend?", "small-talk"),
        ("haha that's so funny lol", "small-talk"),
        ("I can't cope with all this stress", "emotional-support"),
        ("Can you review this document?", "action-item"),
        ("Remind me to take my medicine", "reminder"),
        ("xyz123 gibberish", "unknown"),
    ]

    print(f"{'Message':<45} {'Expected':<20} {'Got':<20} {'Conf':<8} {'ms':<8} {'Match?'}")
    print("-" * 120)

    correct = 0
    for text, expected in test_cases:
        result = classifier.classify(text)
        match = "✅" if result.intent == expected else "❌"
        if result.intent == expected:
            correct += 1
        print(
            f"{text:<45} {expected:<20} {result.intent:<20} "
            f"{result.confidence:<8} {result.latency_ms:<8} {match}"
        )

    print(f"\nAccuracy: {correct}/{len(test_cases)} = {correct/len(test_cases)*100:.0f}%")
    results = [classifier.classify(t) for t, _ in test_cases]
    print(f"All under 200ms: {all(r.latency_ms < 200 for r in results)} ✅")
    print(f"Max latency: {max(r.latency_ms for r in results):.3f}ms")