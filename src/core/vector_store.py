"""
Vector store abstraction for ChromaDB.
Provides clean interface for storing and querying embeddings.
"""

from typing import List, Dict, Optional, Any
import chromadb
from chromadb.config import Settings as ChromaSettings
from pathlib import Path
import uuid

from src.config import Settings
from src.utils.logger import get_logger
from src.utils.exceptions import VectorStoreError

logger = get_logger(__name__)


class VectorStore:
    """Abstraction over ChromaDB for vector storage and retrieval."""

    def __init__(self, settings: Settings):
        """
        Initialize vector store.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.config = settings.chromadb

        logger.info("Initializing ChromaDB vector store")

        try:
            if self.config.use_cloud:
                # Cloud configuration
                if not self.config.api_key or not self.config.host:
                    raise VectorStoreError("ChromaDB Cloud requires API key and host")

                self.client = chromadb.HttpClient(
                    host=self.config.host,
                    settings=ChromaSettings(
                        chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
                        chroma_client_auth_credentials=self.config.api_key
                    )
                )
                logger.info(f"Connected to ChromaDB Cloud at {self.config.host}")
            else:
                # Local persistent storage
                persist_dir = self.config.persist_directory
                persist_dir.mkdir(parents=True, exist_ok=True)

                self.client = chromadb.PersistentClient(path=str(persist_dir))
                logger.info(f"Using local ChromaDB at {persist_dir}")

            # Get or create collection
            self.collection_name = self.config.collection_name
            self._initialize_collection()

        except Exception as e:
            raise VectorStoreError(f"Failed to initialize vector store: {e}")

    def _initialize_collection(self):
        """Get or create the collection."""
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Climate and HVAC technical documents"}
            )
            logger.info(f"Created new collection: {self.collection_name}")

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: List of metadata dicts
            ids: Optional list of document IDs (generated if not provided)
        """
        if not documents:
            logger.warning("No documents to add")
            return

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]

        try:
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to vector store")

        except Exception as e:
            raise VectorStoreError(f"Failed to add documents: {e}")

    def add_documents_batch(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> None:
        """
        Add documents in batches for large ingestion.

        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: List of metadata dicts
            batch_size: Documents per batch
        """
        total = len(documents)
        logger.info(f"Adding {total} documents in batches of {batch_size}")

        for i in range(0, total, batch_size):
            end_idx = min(i + batch_size, total)
            batch_docs = documents[i:end_idx]
            batch_embeds = embeddings[i:end_idx]
            batch_metas = metadatas[i:end_idx]

            self.add_documents(batch_docs, batch_embeds, batch_metas)

            logger.info(f"Processed batch {i//batch_size + 1}/{(total-1)//batch_size + 1}")

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> Dict[str, List]:
        """
        Query the vector store.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            where: Metadata filter conditions
            where_document: Document content filter

        Returns:
            Dictionary with ids, documents, metadatas, distances
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                where_document=where_document,
                include=['documents', 'metadatas', 'distances']
            )

            # Unwrap the nested lists (ChromaDB returns lists of lists)
            return {
                'ids': results['ids'][0],
                'documents': results['documents'][0],
                'metadatas': results['metadatas'][0],
                'distances': results['distances'][0]
            }

        except Exception as e:
            raise VectorStoreError(f"Query failed: {e}")

    def get_all_documents(self, limit: Optional[int] = None) -> Dict[str, List]:
        """
        Get all documents from collection.

        Args:
            limit: Maximum number of documents to retrieve

        Returns:
            Dictionary with ids, documents, metadatas
        """
        try:
            if limit:
                results = self.collection.get(limit=limit, include=['documents', 'metadatas'])
            else:
                results = self.collection.get(include=['documents', 'metadatas'])

            return results

        except Exception as e:
            raise VectorStoreError(f"Failed to get documents: {e}")

    def count(self) -> int:
        """Get total number of documents in collection."""
        try:
            return self.collection.count()
        except Exception as e:
            raise VectorStoreError(f"Failed to count documents: {e}")

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.warning(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            raise VectorStoreError(f"Failed to delete collection: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        count = self.count()

        stats = {
            "collection_name": self.collection_name,
            "total_documents": count,
            "unique_sources": set()
        }

        if count > 0:
            results = self.get_all_documents()
            filenames = [meta.get('filename', 'unknown') for meta in results['metadatas']]
            stats["unique_sources"] = len(set(filenames))
            stats["source_list"] = sorted(set(filenames))

        return stats
