"""BM25 keyword-based retrieval."""

from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
import numpy as np

from src.config import Settings
from src.core.vector_store import VectorStore
from src.utils.logger import get_logger
from src.utils.exceptions import RetrievalError

logger = get_logger(__name__)


class BM25Retriever:
    """Retrieve documents using BM25 keyword search."""

    def __init__(self, settings: Settings):
        """
        Initialize BM25 retriever.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.config = settings.retrieval
        self.vector_store = VectorStore(settings)

        # BM25 index (loaded lazily)
        self.bm25_index = None
        self.documents = None
        self.metadatas = None
        self.ids = None

        logger.info("BM25 retriever initialized")

    def _load_index(self, filters: Optional[Dict] = None) -> None:
        """
        Load all documents and build BM25 index.

        Args:
            filters: Optional metadata filters
        """
        logger.info("Loading documents for BM25 indexing")

        try:
            # Get all documents from vector store
            results = self.vector_store.get_all_documents()

            self.documents = results['documents']
            self.metadatas = results['metadatas']
            self.ids = results['ids']

            # Apply filters if provided
            if filters:
                filtered_data = []
                for doc, meta, doc_id in zip(self.documents, self.metadatas, self.ids):
                    # Simple filter matching (can be enhanced)
                    match = True
                    for key, value in filters.items():
                        if key not in meta or meta[key] != value:
                            match = False
                            break

                    if match:
                        filtered_data.append((doc, meta, doc_id))

                if filtered_data:
                    self.documents, self.metadatas, self.ids = zip(*filtered_data)
                    self.documents = list(self.documents)
                    self.metadatas = list(self.metadatas)
                    self.ids = list(self.ids)
                else:
                    logger.warning("No documents match filters")
                    self.documents, self.metadatas, self.ids = [], [], []

            # Tokenize documents for BM25
            tokenized_docs = [doc.lower().split() for doc in self.documents]
            self.bm25_index = BM25Okapi(tokenized_docs)

            logger.info(f"BM25 index built with {len(self.documents)} documents")

        except Exception as e:
            raise RetrievalError(f"Failed to load BM25 index: {e}")

    def search(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search using BM25 keyword matching.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of results with documents, metadata, and scores
        """
        top_k = top_k or self.config.initial_k

        # Load index if not already loaded (or if filters changed)
        if self.bm25_index is None:
            self._load_index(filters)

        if not self.documents:
            logger.warning("No documents available for BM25 search")
            return []

        try:
            # Tokenize query
            tokenized_query = query.lower().split()

            # Get BM25 scores
            scores = self.bm25_index.get_scores(tokenized_query)

            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:top_k]

            # Format results
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include results with positive scores
                    results.append({
                        'id': self.ids[idx],
                        'document': self.documents[idx],
                        'metadata': self.metadatas[idx],
                        'score': float(scores[idx]),
                        'method': 'bm25'
                    })

            logger.debug(f"BM25 search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            raise RetrievalError(f"BM25 search failed: {e}")

    def clear_cache(self) -> None:
        """Clear cached BM25 index."""
        self.bm25_index = None
        self.documents = None
        self.metadatas = None
        self.ids = None
        logger.info("BM25 cache cleared")
