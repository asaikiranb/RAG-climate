"""Vector similarity search using embeddings."""

from typing import List, Dict, Optional
import numpy as np

from src.config import Settings
from src.core.embeddings import EmbeddingModel
from src.core.vector_store import VectorStore
from src.utils.logger import get_logger
from src.utils.exceptions import RetrievalError

logger = get_logger(__name__)


class VectorRetriever:
    """Retrieve documents using vector similarity search."""

    def __init__(self, settings: Settings):
        """
        Initialize vector retriever.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.config = settings.retrieval

        self.embedding_model = EmbeddingModel(settings)
        self.vector_store = VectorStore(settings)

        logger.info("Vector retriever initialized")

    def search(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for relevant documents using vector similarity.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of results with documents, metadata, and scores
        """
        top_k = top_k or self.config.initial_k

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode_query(query)

            # Query vector store
            results = self.vector_store.query(
                query_embedding=query_embedding.tolist(),
                top_k=top_k,
                where=filters
            )

            # Format results
            formatted_results = []
            for i in range(len(results['documents'])):
                # Convert distance to similarity score (0-1)
                distance = results['distances'][i]
                similarity = 1 - distance  # ChromaDB uses L2 distance

                formatted_results.append({
                    'id': results['ids'][i],
                    'document': results['documents'][i],
                    'metadata': results['metadatas'][i],
                    'score': max(0, similarity),  # Ensure non-negative
                    'method': 'vector'
                })

            logger.debug(f"Vector search returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise RetrievalError(f"Vector search failed: {e}")

    def search_multiple_queries(
        self,
        queries: List[str],
        top_k_per_query: int = None
    ) -> List[Dict]:
        """
        Search with multiple queries and merge results.
        Used for query expansion and HyDE.

        Args:
            queries: List of search queries
            top_k_per_query: Results per query

        Returns:
            Merged list of unique results
        """
        top_k_per_query = top_k_per_query or self.config.initial_k

        all_results = []
        seen_ids = set()

        for query in queries:
            results = self.search(query, top_k=top_k_per_query)

            for result in results:
                doc_id = result['id']
                if doc_id not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(doc_id)

        logger.debug(f"Multi-query search returned {len(all_results)} unique results")
        return all_results
