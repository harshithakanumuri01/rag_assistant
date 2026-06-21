"""
LLM answer generation via Groq's free API.
Builds a context-aware prompt from retrieved chunks + conversation history
and returns a grounded answer along with the source list used.
"""

from groq import Groq

# Groq's free-tier models as of 2026. If a model is deprecated, swap the
# string here -- nothing else in the app needs to change.
DEFAULT_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a careful research assistant. Answer the user's question \
using ONLY the information in the provided context excerpts. \
If the answer isn't in the context, say so clearly instead of guessing. \
Be concise and precise. When you use information from a specific excerpt, \
you don't need to repeat its citation inline -- citations are shown separately \
to the user already."""


def build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a labeled context block for the prompt."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        loc = f"Page {chunk['page']}" if chunk.get("page") is not None else chunk.get("section") or "N/A"
        blocks.append(f"[Excerpt {i} - {chunk['source']} ({loc})]\n{chunk['text']}")
    return "\n\n".join(blocks)


def generate_answer(
    api_key: str,
    question: str,
    retrieved_chunks: list[dict],
    chat_history: list[dict],
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Generate an answer grounded in retrieved_chunks, aware of prior chat_history.

    chat_history: list of {"role": "user"|"assistant", "content": str}
    """
    client = Groq(api_key=api_key)

    context_block = build_context_block(retrieved_chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Include recent conversation memory (last few turns) for follow-up continuity.
    # Strip any extra keys (e.g. "sources", used only for Streamlit display) --
    # Groq's API rejects messages with properties other than role/content.
    for turn in chat_history[-8:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    user_message = (
        f"Context excerpts:\n{context_block}\n\n"
        f"Question: {question}"
    )
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content
