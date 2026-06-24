# System Design: Sync Architecture

## Overview
This document describes how the conversation RAG system handles on-device 
storage, what data syncs to the cloud, and how conflicts resolve.

---

## Architecture Diagram
┌─────────────────────────────────────────────────────────┐

│                     USER DEVICE                         │

│                                                         │

│  ┌─────────────┐    ┌──────────────┐    ┌───────────┐  │

│  │ Raw CSV     │───▶│ Build Index  │───▶│  storage/ │  │

│  │ (local only)│    │ (local only) │    │  (local)  │  │

│  └─────────────┘    └──────────────┘    └─────┬─────┘  │

│                                               │         │

│  storage/ contains:                           │         │

│  ├── faiss_index/   (FAISS binary indexes)   │         │

│  ├── checkpoints.json (topic summaries)      │         │

│  ├── persona.json   (extracted profile)      │         │

│  └── drift_report.json (tone timeline)       │         │

└───────────────────────────────────────────────┼─────────┘

│ sync

▼

┌─────────────────────────────────────────────────────────┐

│                     CLOUD (GitHub)                      │

│                                                         │

│  ✅ syncs:  checkpoints.json, persona.json              │

│  ✅ syncs:  faiss_index/ (small, <10MB)                 │

│  ❌ stays local: raw CSV (privacy, 12MB)                │

│  ❌ stays local: venv/, pycache/                    │

└─────────────────────────────────────────────────────────┘

│

▼

┌─────────────────────────────────────────────────────────┐

│                  STREAMLIT CLOUD                        │

│                                                         │

│  Reads from GitHub on deploy                            │

│  Serves chatbot UI to users                             │

│  Stateless — no writes back to storage                  │

└─────────────────────────────────────────────────────────┘

---

## On-Device Storage

| File | Location | Size | Purpose |
|---|---|---|---|
| conversations.csv | data/ | 12MB | Raw input — never leaves device |
| faiss_index/ | storage/ | ~10MB | Vector indexes for retrieval |
| checkpoints.json | storage/ | ~5MB | Topic + interval summaries |
| persona.json | storage/ | ~50KB | Extracted user profile |
| drift_report.json | storage/ | ~100KB | Day-by-day tone timeline |

---

## What Syncs vs What Stays Local

**Stays local (privacy + size):**
- `conversations.csv` — raw personal conversation data, never uploaded
- `venv/` — Python environment, machine-specific
- `__pycache__/` — compiled bytecode, auto-generated

**Syncs to cloud:**
- `storage/faiss_index/` — pre-built indexes, needed for deployment
- `storage/checkpoints.json` — summaries, needed for RAG
- `storage/persona.json` — persona, needed for chatbot
- All source code — needed for reproducibility

---

## Conflict Resolution

**Scenario:** Two devices build indexes from the same CSV at different times.

**Resolution strategy — Last Write Wins with version tagging:**

1. Each build stamps a `build_timestamp` into `checkpoints.json`
2. On sync, compare timestamps — newer file wins
3. FAISS indexes are deterministic — same input always produces same index
4. Persona extraction is also deterministic — no merge needed

**Why last-write-wins works here:**
- The source data (CSV) is append-only — new days are added, old ones never change
- A newer index always contains a superset of the older index's data
- No partial updates possible — the full index is always replaced atomically

---

## Trade-offs

| Decision | Pro | Con |
|---|---|---|
| Store indexes in Git | Simple deployment, no extra infrastructure | Git not designed for binary files |
| Stateless cloud app | No server-side storage needed | Can't save user queries |
| Local-only raw data | Privacy preserved | Can't rebuild index in cloud |
| Deterministic builds | No merge conflicts | Must rebuild fully on new data |