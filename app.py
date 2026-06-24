# app.py
import json
import logging
import sys
from pathlib import Path
from collections import defaultdict

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    EMBEDDING_MODEL, FAISS_DIR, PERSONA_FILE,
    TOP_K_CHUNKS, TOP_K_SUMMARIES, RAW_CSV
)
from src.vectorstore.store import VectorStore
from src.retrieval.retriever import RAGRetriever
from src.retrieval.conflict_resolver import ConflictResolver
from src.intent.classifier import IntentClassifier
from src.persona.drift import PersonaDriftDetector
from src.chatbot.bot import handle_query, format_persona_answer

logging.basicConfig(level=logging.WARNING)

st.set_page_config(
    page_title="Conversation RAG Chatbot — L2",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Conversation RAG Chatbot — L2")
st.caption("RAG + Persona Drift + Intent Classification + Conflict Resolution")


@st.cache_resource
def load_system():
    store = VectorStore(model_name=EMBEDDING_MODEL, faiss_dir=FAISS_DIR)
    store.load()
    with open(PERSONA_FILE) as f:
        persona = json.load(f)
    retriever = RAGRetriever(store, persona)
    resolver = ConflictResolver(store, total_messages=191592)
    classifier = IntentClassifier()
    return retriever, resolver, classifier


@st.cache_data
def load_drift():
    from src.ingestion.parser import load_messages
    messages = load_messages(RAW_CSV)
    detector = PersonaDriftDetector()
    report = detector.analyze(messages, max_days=50)
    return report


def show_sidebar(retriever):
    st.sidebar.title("👤 User Persona")
    persona = retriever.persona

    if persona.get("personal_facts"):
        st.sidebar.subheader("📋 Personal Facts")
        for item in persona["personal_facts"][:5]:
            st.sidebar.write(f"• {item['value']}")

    if persona.get("habits"):
        st.sidebar.subheader("⏰ Habits")
        for item in persona["habits"][:5]:
            st.sidebar.write(f"• {item['value']}")

    if persona.get("interests"):
        st.sidebar.subheader("🎯 Interests")
        for item in persona["interests"][:5]:
            st.sidebar.write(f"• {item['value']}")

    if persona.get("personality_traits"):
        st.sidebar.subheader("🧠 Personality")
        for item in persona["personality_traits"][:5]:
            st.sidebar.write(f"• {item['value']}")


try:
    retriever, resolver, classifier = load_system()
    show_sidebar(retriever)

    tab1, tab2, tab3 = st.tabs(["💬 Chatbot", "📈 Persona Drift", "🔍 Intent Classifier"])

    # ── Tab 1: Chatbot ────────────────────────────────────────────────────────
    with tab1:
        st.subheader("💡 Try asking:")
        cols = st.columns(3)
        suggestions = [
            "What kind of person is this user?",
            "What are their habits?",
            "Did I mention anything about my sister?",
            "What topics did they discuss?",
            "How do they communicate?",
            "What do they feel about family?",
        ]
        for i, suggestion in enumerate(suggestions):
            if cols[i % 3].button(suggestion, key=f"btn_{i}"):
                st.session_state["query"] = suggestion

        st.divider()

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        query = st.chat_input("Ask something...")

        if "query" in st.session_state and st.session_state["query"]:
            query = st.session_state["query"]
            st.session_state["query"] = None

        if query:
            # classify intent
            intent_result = classifier.classify(query)

            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Searching..."):
                    # use conflict resolver for entity questions
                    conflict_triggers = ["sister", "brother", "family", "friend",
                                        "job", "work", "mention", "say about"]
                    use_resolver = any(t in query.lower() for t in conflict_triggers)

                    if use_resolver:
                        report = resolver.resolve(query)
                        response = report.merged_answer
                        if report.has_contradiction:
                            response = f"⚠️ **Contradictions detected across sources**\n\n{response}"
                    else:
                        response = handle_query(query, retriever)

                st.markdown(response)
                st.caption(f"🏷️ Intent: `{intent_result.intent}` "
                          f"({intent_result.confidence:.0%} confidence) | "
                          f"⚡ {intent_result.latency_ms:.2f}ms")

            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )

    # ── Tab 2: Persona Drift ──────────────────────────────────────────────────
    with tab2:
        st.subheader("📈 Persona Drift Timeline")
        st.caption("How the user's tone changes day by day")

        with st.spinner("Analyzing drift across 50 days..."):
            drift_report = load_drift()

        timeline = drift_report["timeline"]
        drifts = drift_report["drifts"]
        summary = drift_report["summary"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Days Analyzed", summary["total_days"])
        col2.metric("Tone Drifts", summary["total_drifts"])
        col3.metric("Most Common Tone", summary["most_common_tone"].title())

        st.divider()
        st.subheader("Day-by-Day Timeline")
        for entry in timeline[:30]:
            triggers = ", ".join(entry["triggers"]) if entry["triggers"] else "none"
            col_a, col_b, col_c = st.columns([1, 3, 3])
            col_a.write(f"**Day {entry['day']}**")
            col_b.write(f"🎭 {entry['tone'].title()}")
            col_c.write(f"⚡ triggers: {triggers}")

        st.divider()
        st.subheader("Drift Events")
        for drift in drifts[:10]:
            st.write(
                f"Day {drift['from_day']} → Day {drift['to_day']}: "
                f"**{drift['from_tone']}** → **{drift['to_tone']}** "
                f"_(triggered by: {drift['trigger']} [{drift['trigger_type']}])_"
            )

    # ── Tab 3: Intent Classifier ──────────────────────────────────────────────
    with tab3:
        st.subheader("🔍 Intent Classifier")
        st.caption("Classify any message into 5 intents — runs in <1ms, fully offline")

        test_input = st.text_input(
            "Type a message to classify:",
            placeholder="e.g. Don't forget to call mom tomorrow"
        )

        if test_input:
            result = classifier.classify(test_input)
            col1, col2, col3 = st.columns(3)
            col1.metric("Intent", result.intent)
            col2.metric("Confidence", f"{result.confidence:.0%}")
            col3.metric("Latency", f"{result.latency_ms:.3f}ms")

            if result.matched_pattern:
                st.caption(f"Matched pattern: `{result.matched_pattern}`")

        st.divider()
        st.subheader("Test Cases")
        examples = [
            "Don't forget to call mom tomorrow",
            "I feel really sad and lonely today",
            "Please send me the report by end of day",
            "How was your weekend?",
            "I can't cope with all this stress",
            "Set an alarm for 8am",
        ]
        for ex in examples:
            r = classifier.classify(ex)
            st.write(f"`{r.intent}` ({r.confidence:.0%}) — {ex}")

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Run: python scripts/build_index.py first")