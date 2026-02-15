"""
Result fusion and diversification techniques.

Implements:
1. Reciprocal Rank Fusion (RRF) - Merge results from multiple retrievers
2. Maximal Marginal Relevance (MMR) - Diversify results to avoid redundancy
"""

from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.config import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReciprocalRankFusion:
    """
    Merge results from multiple retrieval methods using RRF.
    RRF formula: RRF(d) = sum(1 / (k + rank(d)))
    """

    def __init__(self, settings: Settings, k: int = 60):
        """
        Initialize RRF.

        Args:
            settings: Application settings
            k: RRF constant (default 60 from literature)
        """
        self.settings = settings
        self.k = k or settings.retrieval.rrf_k

    def fuse(self, *result_lists: List[List[Dict]]) -> List[Dict]:
        """
        Fuse multiple result lists using Reciprocal Rank Fusion.

        Args:
            *result_lists: Multiple lists of search results

        Returns:
            Fused and ranked results
        """
        # Track scores per document ID
        rrf_scores = {}

        # Process each result list
        for result_list in result_lists:
            for rank, result in enumerate(result_list, start=1):
                doc_id = result['id']

                # Initialize if first time seeing this document
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = {
                        'score': 0,
                        'document': result['document'],
                        'metadata': result['metadata'],
                        'methods': []
                    }

                # Add RRF score contribution
                rrf_scores[doc_id]['score'] += 1 / (self.k + rank)

                # Track which methods retrieved this doc
                method = result.get('method', 'unknown')
                if method not in rrf_scores[doc_id]['methods']:
                    rrf_scores[doc_id]['methods'].append(method)

        # Sort by RRF score
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )

        # Format results
        fused_results = []
        for doc_id, data in sorted_results:
            fused_results.append({
                'id': doc_id,
                'document': data['document'],
                'metadata': data['metadata'],
                'rrf_score': data['score'],
                'methods': data['methods']
            })

        logger.debug(f"RRF fused {sum(len(r) for r in result_lists)} → {len(fused_results)} unique results")

        return fused_results


class MMRDiversifier:
    """
    Maximal Marginal Relevance for result diversification.
    Balances relevance and diversity to avoid redundant results.
    """

    def __init__(self, settings: Settings, lambda_param: float = None):
        """
        Initialize MMR diversifier.

        Args:
            settings: Application settings
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
        """
        self.settings = settings
        self.lambda_param = lambda_param or settings.retrieval.mmr_lambda

    def diversify(
        self,
        results: List[Dict],
        embeddings: np.ndarray,
        top_k: int
    ) -> List[Dict]:
        """
        Diversify results using MMR.

        Args:
            results: Search results
            embeddings: Document embeddings (for similarity calculation)
            top_k: Number of diverse results to return

        Returns:
            Diversified results
        """
        if len(results) <= top_k:
            return results  # No need to diversify

        # Start with highest scoring document
        selected_indices = [0]
        remaining_indices = list(range(1, len(results)))

        # Iteratively select documents that maximize MMR
        while len(selected_indices) < top_k and remaining_indices:
            mmr_scores = []

            selected_embeddings = embeddings[selected_indices]

            for idx in remaining_indices:
                # Relevance score (from original ranking)
                relevance = results[idx].get('rrf_score', results[idx].get('score', 0))

                # Similarity to already selected documents
                candidate_embedding = embeddings[idx:idx+1]
                similarities = cosine_similarity(candidate_embedding, selected_embeddings)[0]
                max_similarity = np.max(similarities)

                # MMR score = λ * relevance - (1-λ) * max_similarity
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_similarity
                mmr_scores.append((idx, mmr_score))

            # Select document with highest MMR score
            best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        # Return diversified results
        diversified = [results[i] for i in selected_indices]

        logger.debug(f"MMR diversified {len(results)} → {len(diversified)} results")

        return diversified
