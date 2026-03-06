"""
Shared RAG pipeline — used by both the Streamlit app and the WhatsApp bot.

Any retrieval or generation improvement made to retrieve.py / rerank.py / llm.py
automatically flows through here to all channels.
"""

from retrieve import HybridRetriever
from rerank import CrossEncoderReranker
from llm import GenerationClient
from query import expand_query

# Module-level singletons so models are loaded once per process
_retriever: HybridRetriever | None = None
_reranker: CrossEncoderReranker | None = None
_generator: GenerationClient | None = None


def _get_components():
    """Lazy-initialise and cache the heavy RAG components."""
    global _retriever, _reranker, _generator
    if _retriever is None:
        _retriever = HybridRetriever()
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    if _generator is None:
        _generator = GenerationClient()
    return _retriever, _reranker, _generator


def retrieve(query: str) -> list:
    """
    Run query expansion → hybrid retrieval → cross-encoder reranking.
    Returns the ranked list of result dicts (each has 'document', 'metadata', 'id').
    """
    retriever, reranker, generator = _get_components()

    queries = expand_query(query, generator.groq)

    seen_ids: set = set()
    candidates: list = []
    for q in queries:
        for result in retriever.search(q):
            if result["id"] not in seen_ids:
                seen_ids.add(result["id"])
                candidates.append(result)

    return reranker.rerank(query, candidates)


def answer(query: str, top_k: int = 5) -> tuple[str, list]:
    """
    Full RAG pipeline: retrieval + generation.

    Returns:
        (raw_answer_text, top_results)
        raw_answer_text — LLM output including source citations like [1][2]
        top_results     — list of the top_k retrieved chunks (for Streamlit to render)
    """
    results = retrieve(query)
    if not results:
        return "I could not find relevant information in the documentation.", []

    top_results = results[:top_k]
    _, _, generator = _get_components()
    raw_answer = generator.generate(query, top_results)

    return raw_answer, top_results
