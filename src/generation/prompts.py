"""
System prompts and templates for answer generation.
Optimized for technical HVAC domain with citation requirements.
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Template for answer generation prompts."""
    system: str
    user_template: str

    def format_user(self, **kwargs) -> str:
        """Format user template with provided arguments."""
        return self.user_template.format(**kwargs)


# System prompt for HVAC technical question answering
SYSTEM_PROMPT_QA = """You are a highly knowledgeable HVAC (Heating, Ventilation, and Air Conditioning) technical expert assistant. Your role is to provide accurate, detailed, and helpful answers to questions about HVAC systems, troubleshooting, installation, maintenance, and best practices.

**CRITICAL INSTRUCTIONS:**

1. **Answer ONLY using the provided context**: Base your answer strictly on the information given in the context documents. Do not use external knowledge or make assumptions beyond what is explicitly stated.

2. **Cite your sources**: For every factual claim or piece of information in your answer, include a citation in square brackets [N] where N is the document number from the context. You can cite multiple documents [1,2] if the information appears in multiple sources.

3. **Be precise and technical**: Use proper HVAC terminology and provide specific details when available. Include model numbers, specifications, or measurements when mentioned in the context.

4. **Admit uncertainty**: If the context does not contain enough information to fully answer the question, clearly state what you can answer based on the context and what information is missing. Say "Based on the provided context, I cannot determine..." for aspects not covered.

5. **Structure your answer**: Use clear paragraphs, bullet points, or numbered lists to organize complex information. Make your answer easy to read and actionable.

6. **Safety first**: When discussing repairs, maintenance, or troubleshooting, emphasize safety considerations if they are mentioned in the context.

7. **No fabrication**: Never invent information, specifications, or procedures that are not in the provided context. It's better to say "This information is not available in the provided context" than to guess.

**OUTPUT FORMAT:**
- Start with a direct answer to the question
- Provide supporting details with citations [N]
- Include step-by-step instructions if applicable
- End with any important warnings or considerations
- Each citation [N] must reference a document number from the provided context"""


# Alternative system prompt for conversational mode
SYSTEM_PROMPT_CONVERSATIONAL = """You are a friendly and knowledgeable HVAC assistant helping users understand their heating and cooling systems. Provide clear, accessible explanations while maintaining technical accuracy.

Answer based strictly on the provided context documents. Cite sources using [N] notation. If the context doesn't contain the information needed, politely explain what you can and cannot answer.

Keep your tone helpful and professional. Break down complex technical concepts into understandable terms when appropriate, but always cite your sources."""


# System prompt for citation validation
SYSTEM_PROMPT_CITATION_VALIDATOR = """You are a fact-checking assistant. Your job is to verify that claims made in an answer are actually supported by the provided context documents.

For each claim in the answer, check if:
1. The information appears in the cited document(s)
2. The citation accurately represents the source material
3. The claim is not overstated or distorted from the source

Return a JSON object with:
- "valid_citations": list of citation numbers that are correctly grounded
- "invalid_citations": list of citation numbers with problems
- "unsupported_claims": list of claims without proper support
- "overall_score": float from 0.0 to 1.0 indicating citation quality"""


# System prompt for answer verification
SYSTEM_PROMPT_ANSWER_VERIFIER = """You are a quality assurance assistant for HVAC technical answers. Evaluate the answer on these criteria:

1. **Completeness**: Does it fully address the question?
2. **Accuracy**: Are the technical details correct based on the context?
3. **Citation Quality**: Are claims properly cited?
4. **Clarity**: Is the answer well-organized and easy to understand?
5. **Safety**: Are important safety considerations mentioned when relevant?

Return a JSON object with:
- "completeness_score": 0.0 to 1.0
- "accuracy_score": 0.0 to 1.0
- "citation_score": 0.0 to 1.0
- "clarity_score": 0.0 to 1.0
- "overall_score": 0.0 to 1.0
- "issues": list of specific problems found
- "suggestions": list of improvements"""


def get_system_prompt(mode: str = "qa") -> str:
    """
    Get system prompt for specified mode.

    Args:
        mode: Prompt mode - "qa", "conversational", "citation_validator", or "verifier"

    Returns:
        System prompt string

    Raises:
        ValueError: If mode is not recognized
    """
    prompts = {
        "qa": SYSTEM_PROMPT_QA,
        "conversational": SYSTEM_PROMPT_CONVERSATIONAL,
        "citation_validator": SYSTEM_PROMPT_CITATION_VALIDATOR,
        "verifier": SYSTEM_PROMPT_ANSWER_VERIFIER,
    }

    if mode not in prompts:
        raise ValueError(
            f"Unknown prompt mode: {mode}. "
            f"Must be one of: {list(prompts.keys())}"
        )

    return prompts[mode]


def format_context_documents(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into numbered context documents.

    Args:
        chunks: List of chunk dictionaries with 'text' and 'metadata'

    Returns:
        Formatted context string with numbered documents
    """
    if not chunks:
        return "No context documents provided."

    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})

        # Build document header with metadata
        doc_header = f"[Document {i}]"
        if metadata:
            meta_info = []
            if "source" in metadata:
                meta_info.append(f"Source: {metadata['source']}")
            if "page" in metadata:
                meta_info.append(f"Page: {metadata['page']}")
            if "section" in metadata:
                meta_info.append(f"Section: {metadata['section']}")

            if meta_info:
                doc_header += " (" + ", ".join(meta_info) + ")"

        context_parts.append(f"{doc_header}\n{text}")

    return "\n\n".join(context_parts)


def create_qa_prompt(query: str, chunks: List[Dict[str, Any]]) -> PromptTemplate:
    """
    Create question-answering prompt with context.

    Args:
        query: User question
        chunks: Retrieved context chunks

    Returns:
        PromptTemplate with system and user prompts
    """
    context = format_context_documents(chunks)

    user_template = """**CONTEXT DOCUMENTS:**

{context}

**QUESTION:**
{query}

**YOUR ANSWER:**
Please provide a comprehensive answer using only the information from the context documents above. Remember to cite your sources using [N] notation."""

    return PromptTemplate(
        system=SYSTEM_PROMPT_QA,
        user_template=user_template
    )


def create_citation_validation_prompt(
    answer: str,
    chunks: List[Dict[str, Any]]
) -> PromptTemplate:
    """
    Create prompt for validating citations in an answer.

    Args:
        answer: Generated answer with citations
        chunks: Context documents that were used

    Returns:
        PromptTemplate for citation validation
    """
    context = format_context_documents(chunks)

    user_template = """**CONTEXT DOCUMENTS:**

{context}

**ANSWER TO VALIDATE:**
{answer}

**TASK:**
Verify that each claim in the answer is supported by the cited documents. Check for:
- Accuracy of citations
- Proper grounding of claims
- Absence of hallucinations

Respond in JSON format only."""

    return PromptTemplate(
        system=SYSTEM_PROMPT_CITATION_VALIDATOR,
        user_template=user_template
    )


def create_answer_verification_prompt(
    question: str,
    answer: str,
    chunks: List[Dict[str, Any]]
) -> PromptTemplate:
    """
    Create prompt for verifying answer quality.

    Args:
        question: Original question
        answer: Generated answer
        chunks: Context documents

    Returns:
        PromptTemplate for answer verification
    """
    context = format_context_documents(chunks)

    user_template = """**CONTEXT DOCUMENTS:**

{context}

**QUESTION:**
{question}

**ANSWER TO VERIFY:**
{answer}

**TASK:**
Evaluate the answer quality across all dimensions: completeness, accuracy, citations, clarity, and safety.

Respond in JSON format only."""

    return PromptTemplate(
        system=SYSTEM_PROMPT_ANSWER_VERIFIER,
        user_template=user_template
    )


def format_few_shot_examples() -> str:
    """
    Get few-shot examples for improved answer quality.

    Returns:
        Formatted few-shot examples
    """
    examples = """
**Example 1:**

Question: What is the recommended refrigerant charge for a residential heat pump?

Context: [1] The refrigerant charge for residential heat pumps should be determined by the manufacturer's specifications on the unit nameplate. Typical charges range from 6-15 lbs depending on system size. [2] Always use the subcooling or superheat method to verify proper charge rather than relying on pressure alone.

Good Answer: The refrigerant charge for a residential heat pump should be determined by the manufacturer's specifications found on the unit nameplate [1]. Typical charges generally range from 6 to 15 pounds, depending on the system size [1]. However, rather than relying solely on pressure readings, it's important to use the subcooling or superheat method to verify that the system is properly charged [2].

**Example 2:**

Question: How often should HVAC filters be replaced?

Context: [1] Filters should be checked monthly and replaced when visibly dirty or every 1-3 months depending on usage.

Good Answer: HVAC filters should be checked monthly and replaced when visibly dirty, typically every 1-3 months depending on system usage [1]. The exact replacement frequency may vary based on factors like household conditions, though these specific factors are not detailed in the provided context.
"""
    return examples.strip()
