# src/chatbot/bot.py

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    EMBEDDING_MODEL, FAISS_DIR, PERSONA_FILE,
    TOP_K_CHUNKS, TOP_K_SUMMARIES
)
from src.vectorstore.store import VectorStore
from src.retrieval.retriever import RAGRetriever

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def load_system() -> RAGRetriever:
    """Load vector store and persona, return a ready retriever."""
    print("Loading vector store...")
    store = VectorStore(model_name=EMBEDDING_MODEL, faiss_dir=FAISS_DIR)
    loaded = store.load()

    if not loaded:
        print("ERROR: Vector store not found.")
        print("Please run: python scripts/build_index.py")
        sys.exit(1)

    print("Loading persona...")
    if not PERSONA_FILE.exists():
        print("ERROR: Persona file not found.")
        print("Please run: python scripts/build_index.py")
        sys.exit(1)

    with open(PERSONA_FILE) as f:
        persona = json.load(f)

    return RAGRetriever(store, persona)


def format_persona_answer(persona: dict) -> str:
    """Format the persona dict into a readable answer."""
    lines = ["Here's what I know about this user:\n"]

    if persona.get("personal_facts"):
        lines.append("📋 Personal Facts:")
        for item in persona["personal_facts"][:5]:
            lines.append(f"  • {item['value']}")

    if persona.get("habits"):
        lines.append("\n⏰ Habits:")
        for item in persona["habits"][:5]:
            lines.append(f"  • {item['value']}")

    if persona.get("interests"):
        lines.append("\n🎯 Interests:")
        for item in persona["interests"][:5]:
            lines.append(f"  • {item['value']}")

    if persona.get("relationships"):
        lines.append("\n👥 Relationships:")
        for item in persona["relationships"][:5]:
            lines.append(f"  • {item['value']}")

    if persona.get("personality_traits"):
        lines.append("\n🧠 Personality:")
        for item in persona["personality_traits"][:5]:
            lines.append(f"  • {item['value']}")

    style = persona.get("communication_style", {})
    if style:
        lines.append("\n💬 Communication Style:")
        lines.append(f"  • Average message length: {style.get('avg_message_length_words')} words")
        lines.append(f"  • Style: {style.get('style')}")
        lines.append(f"  • Uses emojis: {style.get('uses_emojis')}")
        lines.append(f"  • Asks many questions: {style.get('asks_many_questions')}")

    return "\n".join(lines)


def handle_query(query: str, retriever: RAGRetriever) -> str:
    """
    Route the query to the right handler.
    Persona questions get a formatted persona answer.
    Everything else goes through RAG retrieval.
    """
    q = query.lower().strip()

    persona_triggers = [
        "what kind of person", "who is this user", "describe the user",
        "what are their habits", "how do they talk", "communication style",
        "personality", "what do you know about"
    ]

    if any(trigger in q for trigger in persona_triggers):
        return format_persona_answer(retriever.persona)

    # default: RAG retrieval
    return retriever.answer(query, TOP_K_CHUNKS, TOP_K_SUMMARIES)


def run_chatbot():
    """Main chatbot loop."""
    print("\n" + "="*60)
    print("  Conversation RAG Chatbot")
    print("="*60)
    print("Ask anything about the conversations.")
    print("Type 'quit' or 'exit' to stop.\n")

    retriever = load_system()
    print("\n✅ System ready!\n")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\nBot:", handle_query(query, retriever))
        print()


if __name__ == "__main__":
    run_chatbot()