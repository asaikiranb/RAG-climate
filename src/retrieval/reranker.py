"""
Cross-encoder reranking for improved relevance.
This is THE MOST IMPORTANT component for quality improvement.

Cross-encoders take query + document pairs and score their relevance.
Much more accurate than bi-encoder cosine similarity, but slower.
We use it to rerank top candidates from fast retrieval.
"""

from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder
import numpy as np

from src.config import Settings
from src.utils.logger import get_logger
from src.utils.exceptions import RetrievalError

logger = get_logger(__name__)


class CrossEncoderReranker:
    """
    Rerank search results using a cross-encoder model.
    This provides MUCH better relevance scoring than cosine similarity.
    """

    def __init__(self, settings: Settings):
        """
        Initialize cross-encoder reranker.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.config = settings.retrieval

        if not self.config.use_reranking:
            logger.info("Reranking disabled in config")
            self.model = None
            return

        model_name = self.config.reranker_model
        logger.info(f"Loading cross-encoder model: {model_name}")

        try:
            self.model = CrossEncoder(model_name, max_length=512)
            logger.info("Cross-encoder loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
            raise RetrievalError(f"Failed to load reranker: {e}")

    def rerank(
        self,
        query: str,
        documents: List[str],
        metadata: List[Dict] = None,
        top_k: int = None
    ) -> List[Dict]:
        """
        Rerank documents using cross-encoder.

        Args:
            query: Search query
            documents: List of document texts
            metadata: Optional metadata for each document
            top_k: Number of top results to return (None = return all)

        Returns:
            List of reranked documents with scores
        """
        if not self.model:
            logger.warning("Reranking not available, returning original order")
            return [
                {"document": doc, "metadata": meta if metadata else {}, "score": 1.0 - i*0.01}
                for i, (doc, meta) in enumerate(zip(documents, metadata or [{}]*len(documents)))
            ]

        if not documents:
            return []

        try:
            # Prepare query-document pairs
            pairs = [[query, doc] for doc in documents]

            logger.debug(f"Reranking {len(documents)} documents")

            # Get relevance scores from cross-encoder
            scores = self.model.predict(
                pairs,
                batch_size=self.config.reranker_batch_size,
                show_progress_bar=False
            )

            # Create results with scores
            results = []
            for idx, (doc, score) in enumerate(zip(documents, scores)):
                meta = metadata[idx] if metadata else {}

                results.append({
                    "document": doc,
                    "metadata": meta,
                    "rerank_score": float(score),
                    "original_rank": idx
                })

            # Sort by rerank score (descending)
            results.sort(key=lambda x: x["rerank_score"], reverse=True)

            # Return top-k if specified
            if top_k:
                results = results[:top_k]

            logger.info(f"Reranked {len(documents)} → {len(results)} documents")
            logger.debug(f"Top score: {results[0]['rerank_score']:.4f}, Bottom score: {results[-1]['rerank_score']:.4f}")

            return results

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            raise RetrievalError(f"Reranking failed: {e}")

    def rerank_with_scores(
        self,
        query: str,
        results: List[Dict]
    ) -> List[Dict]:
        """
        Rerank results that already have documents and metadata.

        Args:
            query: Search query
            results: List of result dicts with 'document' and 'metadata' keys

        Returns:
            Reranked results
        """
        documents = [r['document'] for r in results]
        metadatas = [r['metadata'] for r in results]

        # Preserve original scores if they exist
        for i, result in enumerate(results):
            if 'metadata' not in result:
                result['metadata'] = {}
            result['metadata']['original_score'] = result.get('score', 0)

        return self.rerank(query, documents, metadatas)
