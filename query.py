"""Adaptive query expansion for version 3 experiments."""

import re
from typing import List

from groq import Groq

from config import LLM_MODEL


def _detect_intent(query: str) -> str:
    q = query.lower()
    troubleshooting_terms = ["error", "fault", "not working", "leak", "failure", "troubleshoot"]
    compliance_terms = ["code", "regulation", "compliance", "ashrae", "iec", "standard"]
    procedure_terms = ["steps", "procedure", "install", "commission", "how to"]

    if any(term in q for term in troubleshooting_terms):
        return "troubleshooting"
    if any(term in q for term in compliance_terms):
        return "compliance"
    if any(term in q for term in procedure_terms):
        return "procedure"
    return "fact"


def _target_count(intent: str) -> int:
    if intent == "fact":
        return 2
    return 3


def _sanitize_line(line: str) -> str:
    return re.sub(r"^\s*[-\d\.\)]\s*", "", line).strip()


def expand_query(query: str, groq_client: Groq) -> List[str]:
    if groq_client is None:
        return [query]

    intent = _detect_intent(query)
    count = _target_count(intent)
    expansion_prompt = f"""You are expanding an HVAC retrieval query.
Intent class: {intent}
Generate {count} alternatives that keep the same intent and constraints.

Rules:
- preserve model/part numbers and all units
- keep the query domain as HVAC technical support
- each output line must be a standalone search query
- no numbering, no explanation

Question: {query}
"""

    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": expansion_prompt}],
            model=LLM_MODEL,
            temperature=0.25,
            max_tokens=220,
        )
        raw = response.choices[0].message.content.strip()
        variants = [query]
        for line in raw.splitlines():
            clean = _sanitize_line(line)
            if clean and clean.lower() != query.lower() and clean not in variants:
                variants.append(clean)
        return variants[: count + 1]
    except Exception:
        return [query]
