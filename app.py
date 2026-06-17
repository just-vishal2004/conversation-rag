# app.py
import json
import logging
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    EMBEDDING_MODEL, FAISS_DIR, PERSONA_FILE,
    TOP_K_CHUNKS, TOP_K_SUMMARIES
)
from src.vectorstore.store import VectorStore
from src.retrieval.retriever import RAGRetriever
from src.chatbot.bot import handle_query, format_persona_answer

logging.basicConfig(level=logging.WARNING)

st.set_page_config(
    page_title="Conversation RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Conversation RAG Chatbot")
st.caption("Ask anything about the conversation data")


@st.cache_resource
def load_system():
    store = VectorStore(model_name=EMBEDDING_MODEL, faiss_dir=FAISS_DIR)
    store.load()
    with open(PERSONA_FILE) as f:
        persona = json.load(f)
    return RAGRetriever(store, persona)


# sidebar with persona
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

    style = persona.get("communication_style", {})
    if style:
        st.sidebar.subheader("💬 Communication")
        st.sidebar.write(f"• Style: {style.get('style')}")
        st.sidebar.write(f"• Avg length: {style.get('avg_message_length_words')} words")
        st.sidebar.write(f"• Uses emojis: {style.get('uses_emojis')}")


# main chat
try:
    retriever = load_system()
    show_sidebar(retriever)

    # suggested questions
    st.subheader("💡 Try asking:")
    cols = st.columns(3)
    suggestions = [
        "What kind of person is this user?",
        "What are their habits?",
        "Do they have any pets?",
        "What topics did they discuss?",
        "How do they communicate?",
        "What are their interests?",
    ]
    for i, suggestion in enumerate(suggestions):
        if cols[i % 3].button(suggestion, key=f"btn_{i}"):
            st.session_state["query"] = suggestion

    st.divider()

    # chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # input
    query = st.chat_input("Ask something about the user or conversations...")

    # handle suggested question click
    if "query" in st.session_state and st.session_state["query"]:
        query = st.session_state["query"]
        st.session_state["query"] = None

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                response = handle_query(query, retriever)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

except Exception as e:
    st.error(f"Error loading system: {e}")
    st.info("Make sure you have run: python scripts/build_index.py")