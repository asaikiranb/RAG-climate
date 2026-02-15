"""Groq client and answer generation."""

import os
from dotenv import load_dotenv
from groq import Groq

from config import SYSTEM_PROMPT, SYSTEM_MESSAGE, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TOP_P

load_dotenv()


def get_groq_client() -> Groq:
    """Return a Groq client. Raises ValueError if API key is missing."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Please add it to your .env file.")
    return Groq(api_key=api_key)


def build_context(results: list) -> str:
    """Turn search results into a numbered source string for the prompt."""
    context_parts = []
    for i, result in enumerate(results, 1):
        metadata = result['metadata']
        context_parts.append(
            f"[Source {i}] (Document: {metadata['filename']}, Page: {metadata['page_number']})\n"
            f"{result['document']}\n"
        )
    return "\n---\n".join(context_parts)


def generate_answer(query: str, context: str, groq_client: Groq) -> str:
    """Send query + context to Groq and return the answer."""
    prompt = SYSTEM_PROMPT.format(context=context, query=query)

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            top_p=LLM_TOP_P,
        )
        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"Error generating answer: {str(e)}"
