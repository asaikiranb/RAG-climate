"""
Embedding model wrapper with caching and batching.
Provides efficient embedding generation for both ingestion and retrieval.
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import hashlib
import pickle
from tqdm import tqdm

from src.config import Settings
from src.utils.logger import get_logger
from src.utils.exceptions import EmbeddingError

logger = get_logger(__name__)


class EmbeddingModel:
    """
    Wrapper around SentenceTransformer for embedding generation.
    Includes caching for efficiency.
    """

    def __init__(self, settings: Settings):
        """
        Initialize embedding model.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.config = settings.embedding

        logger.info(f"Loading embedding model: {self.config.model_name}")

        try:
            self.model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
                cache_folder=str(self.config.cache_dir) if self.config.cache_dir else None
            )
            logger.info(f"Model loaded successfully on {self.config.device}")
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model: {e}")

        # Cache setup
        self.cache_enabled = settings.caching.enable_embedding_cache
        if self.cache_enabled:
            self.cache_dir = settings.caching.cache_dir / "embeddings"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Embedding cache enabled at {self.cache_dir}")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = None,
        show_progress: bool = False,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding (uses config default if None)
            show_progress: Show progress bar
            use_cache: Use cached embeddings if available

        Returns:
            Numpy array of embeddings
        """
        # Handle single text
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        # Check cache
        if use_cache and self.cache_enabled:
            cached_embeddings = self._get_from_cache(texts)
            if cached_embeddings is not None:
                logger.debug(f"Retrieved {len(texts)} embeddings from cache")
                return cached_embeddings[0] if is_single else cached_embeddings

        # Generate embeddings
        batch_size = batch_size or self.config.batch_size

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )

            # Cache the results
            if use_cache and self.cache_enabled:
                self._save_to_cache(texts, embeddings)

            return embeddings[0] if is_single else embeddings

        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {e}")

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a search query.
        Convenience method for single query encoding.

        Args:
            query: Query text

        Returns:
            Query embedding
        """
        return self.encode(query, use_cache=True)

    def encode_documents(
        self,
        documents: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode multiple documents with progress bar.

        Args:
            documents: List of document texts
            show_progress: Show progress bar

        Returns:
            Document embeddings
        """
        return self.encode(documents, show_progress=show_progress)

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        # Include model name in hash to avoid collisions between models
        key = f"{self.config.model_name}:{text}"
        return hashlib.sha256(key.encode()).hexdigest()

    def _get_from_cache(self, texts: List[str]) -> Union[np.ndarray, None]:
        """Retrieve embeddings from cache if all texts are cached."""
        embeddings = []

        for text in texts:
            cache_key = self._get_cache_key(text)
            cache_file = self.cache_dir / f"{cache_key}.pkl"

            if not cache_file.exists():
                return None  # If any text is not cached, return None

            with open(cache_file, 'rb') as f:
                embeddings.append(pickle.load(f))

        return np.array(embeddings)

    def _save_to_cache(self, texts: List[str], embeddings: np.ndarray) -> None:
        """Save embeddings to cache."""
        for text, embedding in zip(texts, embeddings):
            cache_key = self._get_cache_key(text)
            cache_file = self.cache_dir / f"{cache_key}.pkl"

            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.model.get_sentence_embedding_dimension()

    def clear_cache(self) -> None:
        """Clear all cached embeddings."""
        if self.cache_enabled:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Embedding cache cleared")
