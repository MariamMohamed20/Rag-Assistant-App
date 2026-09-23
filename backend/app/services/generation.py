"""
Builds the grounded prompt from retrieved chunks and calls the local Ollama LLM.
"""
import logging

import ollama

from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a study assistant for a Data Visualization course. "
    "Answer the user's question using ONLY the context provided below. "
    "If the context does not contain the answer, say you don't know instead of guessing. "
    "Keep answers concise and cite which lecture the information came from."
)


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        context_blocks.append(f"[{i}] Source: {chunk['source']}\n{chunk['text']}")
    context = "\n\n".join(context_blocks)

    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above. Reference the source number(s) you used, "
        "e.g. \"(see [1])\"."
    )


def generate_answer(question: str, chunks: list[dict]) -> str:
    prompt = build_prompt(question, chunks)

    response = ollama.chat(
        model=settings.OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response["message"]["content"]
