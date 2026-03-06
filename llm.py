"""Generation client using Groq with local Ollama fallback."""

import os
import requests
from groq import Groq
from config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TOP_P, SYSTEM_PROMPT, SYSTEM_MESSAGE

OLLAMA_URL = "http://localhost:11434/api/generate"

class GenerationClient:
    """Handles LLM interactions for both Groq and local Ollama fallbacks."""

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.groq = Groq(api_key=self.api_key) if self.api_key else None

    def _build_context(self, hits: list) -> str:
        """Format retrieval hits into a structured context string."""
        parts = []
        for i, hit in enumerate(hits, 1):
            meta = hit["metadata"]
            parts.append(
                f"[Source {i}] (Document: {meta['filename']}, Page: {meta['page_number']})\n"
                f"{hit['document']}\n"
            )
        return "\n---\n".join(parts)

    def generate(self, query: str, hits: list, use_fallback: bool = True) -> str:
        """Generate a safety-compliant answer from provided sources."""
        context = self._build_context(hits)
        prompt = SYSTEM_PROMPT.format(context=context, query=query)

        if self.groq:
            try:
                res = self.groq.chat.completions.create(
                    messages=[
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user", "content": prompt},
                    ],
                    model=LLM_MODEL,
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                    top_p=LLM_TOP_P,
                )
                return res.choices[0].message.content
            except Exception as e:
                print(f"Groq API error: {e}")
                if not use_fallback: return f"Error: {e}"

        fallback_answer = self._generate_ollama(prompt)
        if fallback_answer.startswith("Service unavailable:"):
            return self._deterministic_fallback(query, hits, fallback_answer)
        return fallback_answer

    def _deterministic_fallback(self, query: str, hits: list, error_message: str) -> str:
        """Graceful fallback that keeps source-grounded citations even if LLM providers fail."""
        if not hits:
            return (
                "I could not generate a model response right now, and there are no retrieved sources "
                "to answer this safely. Please retry."
            )

        lines = [
            "Model generation is temporarily unavailable; using source-grounded fallback.",
            "",
            "Most relevant evidence:",
        ]

        for idx, hit in enumerate(hits[:3], start=1):
            meta = hit.get("metadata", {})
            snippet = hit.get("document", "").replace("\n", " ").strip()
            snippet = " ".join(snippet.split())[:180]
            filename = meta.get("filename", "unknown")
            page = meta.get("page_number", "?")
            lines.append(f"- [{idx}] {filename} (p.{page}): {snippet}")

        lines.append("")
        lines.append("Please retry shortly for a full synthesized answer. [1]")
        return "\n".join(lines)

    def _generate_ollama(self, prompt: str) -> str:
        """Fallback to local Ollama if Groq fails or is unavailable."""
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
        try:
            res = requests.post(
                OLLAMA_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "system": SYSTEM_MESSAGE,
                    "stream": False,
                    "options": {"temperature": LLM_TEMPERATURE}
                },
                timeout=60
            )
            res.raise_for_status()
            return res.json().get("response", "").strip()
        except Exception as e:
            return f"Service unavailable: {e}"
