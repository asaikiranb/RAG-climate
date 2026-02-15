"""
Hybrid retrieval system combining all advanced techniques.

Pipeline:
1. Query Enhancement (HyDE / Query Expansion)
2. Multi-Method Retrieval (Vector + BM25)
3. Result Fusion (RRF)
4. Reranking (Cross-Encoder)
5. Diversification (MMR - optional)
"""

from typing import List, Dict, Optional
import numpy as np

from src.config import Settings
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.query_enhancer import QueryEnhancer
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.fusion import ReciprocalRankFusion, MMRDiversifier
from src.core.embeddings import EmbeddingModel
from src.utils.logger import get_logger, with_trace_id
from src.utils.exceptions import RetrievalError

logger = get_logger(__name__)


class HybridRetriever:
    """
    Advanced hybrid retrieval combining multiple strategies.
    This is the main retrieval interface used by the application.
    """

    def __init__(self, settings: Settings):
        """
        Initialize hybrid retriever.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.config = settings.retrieval

        logger.info("Initializing hybrid retriever with advanced features")

        # Initialize components
        self.vector_retriever = VectorRetriever(settings) if self.config.use_vector_search else None
        self.bm25_retriever = BM25Retriever(settings) if self.config.use_bm25_search else None
        self.query_enhancer = QueryEnhancer(settings)
        self.reranker = CrossEncoderReranker(settings)
        self.rrf_fusion = ReciprocalRankFusion(settings)
        self.mmr_diversifier = MMRDiversifier(settings) if self.config.use_mmr else None
        self.embedding_model = EmbeddingModel(settings)

        logger.info("Hybrid retriever initialized successfully")
        self._log_configuration()

    def _log_configuration(self):
        """Log the current retrieval configuration."""
        logger.info("Retrieval configuration:")
        logger.info(f"  - Vector search: {self.config.use_vector_search}")
        logger.info(f"  - BM25 search: {self.config.use_bm25_search}")
        logger.info(f"  - HyDE: {self.config.use_hyde}")
        logger.info(f"  - Query expansion: {self.config.use_query_expansion}")
        logger.info(f"  - Reranking: {self.config.use_reranking}")
        logger.info(f"  - MMR diversification: {self.config.use_mmr}")

    @with_trace_id
    def search(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict] = None,
        return_metadata: bool = True
    ) -> List[Dict]:
        """
        Perform advanced hybrid search.

        Args:
            query: User query
            top_k: Number of final results to return
            filters: Optional metadata filters
            return_metadata: Whether to include detailed metadata

        Returns:
            List of search results with documents and scores
        """
        top_k = top_k or self.config.top_k
        initial_k = self.config.initial_k

        logger.info(f"Starting hybrid search for: '{query[:100]}...'")

        try:
            # STEP 1: Query Enhancement
            enhanced_queries = [query]  # Always include original

            if self.config.use_hyde or self.config.use_query_expansion:
                logger.info("Step 1/5: Enhancing query")
                enhanced_queries = self.query_enhancer.enhance_query(query)
                logger.info(f"Enhanced into {len(enhanced_queries)} query variations")
            else:
                logger.info("Step 1/5: Using original query (enhancement disabled)")

            # STEP 2: Multi-Method Retrieval
            logger.info("Step 2/5: Retrieving candidates")
            all_results = []

            # Vector search
            if self.vector_retriever:
                logger.info("  - Running vector search")
                vector_results = self.vector_retriever.search_multiple_queries(
                    enhanced_queries,
                    top_k_per_query=initial_k // len(enhanced_queries)
                )
                all_results.append(vector_results)
                logger.info(f"    Got {len(vector_results)} vector results")

            # BM25 search
            if self.bm25_retriever:
                logger.info("  - Running BM25 search")
                # BM25 uses original query (keyword matching doesn't benefit from HyDE)
                bm25_results = self.bm25_retriever.search(
                    query,
                    top_k=initial_k,
                    filters=filters
                )
                all_results.append(bm25_results)
                logger.info(f"    Got {len(bm25_results)} BM25 results")

            if not all_results:
                logger.warning("No retrieval methods enabled!")
                return []

            # STEP 3: Fusion
            logger.info("Step 3/5: Fusing results with RRF")
            fused_results = self.rrf_fusion.fuse(*all_results)
            logger.info(f"Fused into {len(fused_results)} unique results")

            # Take top candidates for reranking
            candidates = fused_results[:initial_k]

            # STEP 4: Reranking
            if self.reranker.model:
                logger.info("Step 4/5: Reranking with cross-encoder")
                reranked_results = self.reranker.rerank_with_scores(
                    query,  # Use original query for reranking
                    candidates
                )
                logger.info(f"Reranked {len(reranked_results)} results")
            else:
                logger.info("Step 4/5: Skipping reranking (disabled)")
                reranked_results = candidates

            # STEP 5: Diversification (optional)
            final_results = reranked_results[:top_k]

            if self.mmr_diversifier and len(reranked_results) > top_k:
                logger.info("Step 5/5: Diversifying with MMR")

                # Get embeddings for MMR
                documents = [r['document'] for r in reranked_results]
                embeddings = self.embedding_model.encode_documents(
                    documents,
                    show_progress=False
                )

                final_results = self.mmr_diversifier.diversify(
                    reranked_results,
                    embeddings,
                    top_k
                )
                logger.info(f"Diversified to {len(final_results)} results")
            else:
                logger.info("Step 5/5: No diversification (disabled or not needed)")

            # Add retrieval metadata if requested
            if return_metadata:
                for result in final_results:
                    result['retrieval_metadata'] = {
                        'query_variations': len(enhanced_queries),
                        'initial_candidates': len(fused_results),
                        'reranked': self.reranker.model is not None,
                        'diversified': self.mmr_diversifier is not None
                    }

            logger.info(f"Search complete: returning {len(final_results)} results")

            return final_results

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise RetrievalError(f"Search failed: {e}")

    def search_simple(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Simplified search interface for backward compatibility.

        Args:
            query: User query
            top_k: Number of results

        Returns:
            Search results
        """
        return self.search(query, top_k=top_k, return_metadata=False)

    def get_available_filters(self) -> Dict[str, List[str]]:
        """
        Get available metadata filters from the collection.

        Returns:
            Dict mapping filter keys to possible values
        """
        # Get all documents
        results = self.vector_retriever.vector_store.get_all_documents(limit=1000)

        filters = {}
        for metadata in results['metadatas']:
            for key, value in metadata.items():
                if key not in filters:
                    filters[key] = set()
                filters[key].add(value)

        # Convert sets to sorted lists
        return {k: sorted(list(v)) for k, v in filters.items()}
