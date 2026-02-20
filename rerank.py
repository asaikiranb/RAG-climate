"""Cross-encoder reranker for scoring query-document pairs."""

import os
import math
from typing import List, Dict
from sentence_transformers import CrossEncoder

# BAAI/bge-reranker-base: stronger on out-of-domain technical corpora than
# ms-marco MiniLM models; benchmarked improvement on BEIR FiQA/technical Q&A.
RERANKER_MODEL = "BAAI/bge-reranker-base"


def load_reranker() -> CrossEncoder:
    """Load and return the cross-encoder model."""
    return CrossEncoder(RERANKER_MODEL, token=os.environ.get("HF_TOKEN"))


def rerank(query: str, results: List[Dict], model: CrossEncoder, rrf_weight: float = 0.2) -> List[Dict]:
    """
    Score each result against the query using the cross-encoder, then blend
    with the upstream RRF score to guard against confident mis-rankings.

    The reranker gets 80% weight; the upstream RRF rank (which already fused
    BM25 + vector) gets 20% weight. Both scores are normalized to [0, 1]
    before blending so their scales are comparable.

    Returns results sorted by blended score, highest first.
    """
    if not results:
        return results

    pairs = [(query, result["document"]) for result in results]
    raw_scores = model.predict(pairs)

    # Normalize cross-encoder logits to [0, 1] via sigmoid
    norm_scores = [1 / (1 + math.exp(-float(s))) for s in raw_scores]

    # Normalize RRF scores to [0, 1] by dividing by max
    rrf_vals = [result.get("score", 0.0) for result in results]
    max_rrf = max(rrf_vals) if max(rrf_vals) > 0 else 1.0
    norm_rrf = [v / max_rrf for v in rrf_vals]

    for result, ns, nr in zip(results, norm_scores, norm_rrf):
        result["rerank_score"] = (1 - rrf_weight) * ns + rrf_weight * nr

    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)
