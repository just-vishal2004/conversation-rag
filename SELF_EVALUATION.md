# Self-Evaluation Sheet — L2 Submission

## Part 1: Adaptive Persona Engine
**Status: Complete ✅**
- Built day-by-day tone detector across 11,001 conversations
- Detects 7 tone categories: curious, formal, casual, frustrated, playful, emotional, motivated
- Outputs timeline with triggers per day
- Drift events detected with trigger type (topic/person/event)
- Limitation: keyword-based, not semantic — may miss sarcasm

## Part 2: Offline Intent Classifier
**Status: Complete ✅**
- 100% accuracy on test cases
- Max latency: 0.032ms (requirement: <200ms) — 6000x faster
- Zero model download — pure pattern matching
- Fully offline, no API calls
- Limitation: limited to patterns defined — novel phrasing may miss

## Part 3: Conflict Resolution in RAG
**Status: Complete ✅**
- Ranks chunks by recency (60%) + emotional weight (40%)
- Detects contradictions using sentiment pair matching
- Returns merged answer with contradiction warning
- Tested on "Did I mention anything about my sister?" — contradiction detected
- Limitation: contradiction detection is lexical, not semantic

## Part 4: System Design Doc
**Status: Complete ✅**
- Covers on-device storage, what syncs, what stays local
- Conflict resolution via last-write-wins with timestamp
- ASCII diagram included
- Trade-offs documented

## What I would improve with more time
- Use sentence embeddings for drift detection instead of keywords
- Train a proper intent classifier on labeled data
- Add semantic contradiction detection using NLI models
- Add a database layer instead of flat JSON files