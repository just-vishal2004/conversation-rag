# Conversation RAG System

A RAG (Retrieval-Augmented Generation) system built on 191,000+ conversation messages with topic detection, checkpointing, persona extraction, and a chatbot interface.

## Architecture

```
CSV Data → Parser → Topic Detector → Checkpoint Engine → Vector Store → Chatbot
                                                        → Persona Extractor ↗
```

## How Topic Detection Works

Messages are embedded using `sentence-transformers/all-MiniLM-L6-v2`. A sliding window of 5 messages is averaged into a context vector. Consecutive window vectors are compared using cosine similarity. When similarity drops below threshold OR a local minimum is detected, a topic boundary is created. Minimum segment length of 10 messages prevents over-segmentation.

## How Retrieval Works

Two separate FAISS indexes are maintained:
1. **Chunks index** — overlapping windows of 10 messages with 2 message overlap
2. **Summaries index** — topic checkpoint summaries + 100-message interval summaries

On every query, both indexes are searched and results are combined for richer context.

## How Persona is Built

The persona extractor scans all User 1 messages using pattern matching on actual conversation text. Every trait is backed by a direct evidence quote from the conversation. No guessing or hallucination. Categories extracted: habits, personal facts, personality traits, communication style, interests, relationships.

## Setup Instructions

```bash
git clone https://github.com/just-vishal2004/conversation-rag
cd conversation-rag
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place `conversations.csv` in the `data/` folder.

## Build the Index (run once)

```bash
python scripts/build_index.py
```

Runs all 5 pipeline stages and saves indexes to `storage/`. Takes 15-20 minutes on first run.

## Run the Chatbot

```bash
python scripts/run_chatbot.py
```

## Example Questions to Ask

- `What kind of person is this user?`
- `What are their habits?`
- `Do they have any pets?`
- `What topics did they discuss?`
- `How do they communicate?`

## Tech Stack

| Component | Technology | Reason |
|---|---|---|
| Embeddings | all-MiniLM-L6-v2 | Fast, free, 384-dim |
| Vector DB | FAISS IndexFlatIP | In-process, millisecond search |
| Summarization | facebook/bart-large-cnn | Free, runs locally |
| Persona | Pattern matching + evidence | No hallucination |
| Interface | CLI chatbot | Simple, works everywhere |

## Project Structure

```
src/
├── ingestion/       # CSV parser → Message dataclass
├── topic_detection/ # Sliding window cosine similarity topic detector
├── checkpoints/     # Topic + 100-msg summaries using local BART model
├── vectorstore/     # FAISS index build and similarity search
├── retrieval/       # RAG retriever combining chunks + summaries + persona
├── persona/         # Evidence-based persona extraction from conversations
└── chatbot/         # CLI chatbot interface

scripts/
├── build_index.py   # Runs full 5-stage pipeline
└── run_chatbot.py   # Starts the chatbot
```