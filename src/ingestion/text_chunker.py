"""
Advanced text chunking strategies.

Supports multiple chunking methods:
1. Token-based chunking (baseline)
2. Semantic chunking (breaks at meaning boundaries)
3. Hierarchical chunking (parent-child relationships)
"""

from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
import re

from src.config import Settings
from src.core.tokenizer import Tokenizer
from src.utils.logger import get_logger
from src.utils.exceptions import ChunkingError

logger = get_logger(__name__)


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""

    def __init__(self, settings: Settings):
        """
        Initialize chunker.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.config = settings.chunking
        self.tokenizer = Tokenizer(settings)

    @abstractmethod
    def chunk(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Chunk text into smaller pieces.

        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of dicts with 'text' and 'metadata' keys
        """
        pass


class TextChunker(BaseChunker):
    """
    Token-based text chunking with overlap.
    This is the baseline/default method.
    """

    def chunk(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into chunks by token count with overlap.

        Args:
            text: Input text
            metadata: Metadata to include with each chunk

        Returns:
            List of chunk dicts
        """
        if not text.strip():
            return []

        try:
            # Split into token-based chunks
            chunk_texts = self.tokenizer.split_by_tokens(
                text,
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap
            )

            # Create chunk objects with metadata
            chunks = []
            for idx, chunk_text in enumerate(chunk_texts):
                # Skip chunks that are too small
                if self.tokenizer.count_tokens(chunk_text) < self.config.min_chunk_size:
                    continue

                chunk_meta = metadata.copy() if metadata else {}
                chunk_meta['chunk_index'] = idx
                chunk_meta['chunk_method'] = 'token'
                chunk_meta['token_count'] = self.tokenizer.count_tokens(chunk_text)

                chunks.append({
                    'text': chunk_text.strip(),
                    'metadata': chunk_meta
                })

            logger.debug(f"Created {len(chunks)} token-based chunks")
            return chunks

        except Exception as e:
            raise ChunkingError(f"Token chunking failed: {e}")


class SemanticChunker(BaseChunker):
    """
    Semantic chunking that respects sentence and paragraph boundaries.
    Better than pure token chunking for preserving context.
    """

    def __init__(self, settings: Settings):
        super().__init__(settings)

        # Sentence boundary detection patterns
        self.sentence_endings = re.compile(r'[.!?]\s+')
        self.paragraph_break = re.compile(r'\n\s*\n')

    def chunk(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text semantically at sentence/paragraph boundaries.

        Args:
            text: Input text
            metadata: Metadata to include

        Returns:
            List of chunk dicts
        """
        if not text.strip():
            return []

        try:
            # First split by paragraphs
            paragraphs = self.paragraph_break.split(text)

            chunks = []
            current_chunk = ""
            chunk_idx = 0

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                # Split paragraph into sentences
                sentences = self.sentence_endings.split(para)

                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    # Check if adding this sentence would exceed chunk size
                    potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
                    token_count = self.tokenizer.count_tokens(potential_chunk)

                    if token_count <= self.config.chunk_size:
                        current_chunk = potential_chunk
                    else:
                        # Save current chunk if it's big enough
                        if current_chunk and self.tokenizer.count_tokens(current_chunk) >= self.config.min_chunk_size:
                            chunk_meta = metadata.copy() if metadata else {}
                            chunk_meta['chunk_index'] = chunk_idx
                            chunk_meta['chunk_method'] = 'semantic'
                            chunk_meta['token_count'] = self.tokenizer.count_tokens(current_chunk)

                            chunks.append({
                                'text': current_chunk.strip(),
                                'metadata': chunk_meta
                            })
                            chunk_idx += 1

                        # Start new chunk with current sentence
                        current_chunk = sentence

            # Add final chunk
            if current_chunk and self.tokenizer.count_tokens(current_chunk) >= self.config.min_chunk_size:
                chunk_meta = metadata.copy() if metadata else {}
                chunk_meta['chunk_index'] = chunk_idx
                chunk_meta['chunk_method'] = 'semantic'
                chunk_meta['token_count'] = self.tokenizer.count_tokens(current_chunk)

                chunks.append({
                    'text': current_chunk.strip(),
                    'metadata': chunk_meta
                })

            logger.debug(f"Created {len(chunks)} semantic chunks")
            return chunks

        except Exception as e:
            raise ChunkingError(f"Semantic chunking failed: {e}")


class HierarchicalChunker(BaseChunker):
    """
    Hierarchical chunking creates small chunks but tracks parent context.
    Enables parent-child retrieval strategy.
    """

    def __init__(self, settings: Settings, parent_size: int = 2000, child_size: int = 400):
        """
        Initialize hierarchical chunker.

        Args:
            settings: Application settings
            parent_size: Tokens in parent chunks
            child_size: Tokens in child chunks
        """
        super().__init__(settings)
        self.parent_size = parent_size
        self.child_size = child_size

    def chunk(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Create hierarchical chunks with parent-child relationships.

        Args:
            text: Input text
            metadata: Metadata to include

        Returns:
            List of chunk dicts (both parent and child chunks)
        """
        if not text.strip():
            return []

        try:
            # First create parent chunks
            parent_chunks = self.tokenizer.split_by_tokens(
                text,
                chunk_size=self.parent_size,
                overlap=200
            )

            all_chunks = []
            parent_idx = 0

            for parent_text in parent_chunks:
                # Create parent chunk
                parent_meta = metadata.copy() if metadata else {}
                parent_meta['chunk_type'] = 'parent'
                parent_meta['parent_index'] = parent_idx
                parent_meta['chunk_method'] = 'hierarchical'

                parent_chunk = {
                    'text': parent_text.strip(),
                    'metadata': parent_meta
                }
                all_chunks.append(parent_chunk)

                # Split parent into child chunks
                child_texts = self.tokenizer.split_by_tokens(
                    parent_text,
                    chunk_size=self.child_size,
                    overlap=50
                )

                for child_idx, child_text in enumerate(child_texts):
                    child_meta = metadata.copy() if metadata else {}
                    child_meta['chunk_type'] = 'child'
                    child_meta['parent_index'] = parent_idx
                    child_meta['child_index'] = child_idx
                    child_meta['chunk_method'] = 'hierarchical'
                    child_meta['parent_text'] = parent_text[:200] + "..."  # Preview

                    child_chunk = {
                        'text': child_text.strip(),
                        'metadata': child_meta
                    }
                    all_chunks.append(child_chunk)

                parent_idx += 1

            logger.debug(f"Created {len(all_chunks)} hierarchical chunks ({len(parent_chunks)} parents)")
            return all_chunks

        except Exception as e:
            raise ChunkingError(f"Hierarchical chunking failed: {e}")


def get_chunker(settings: Settings) -> BaseChunker:
    """
    Factory function to get appropriate chunker based on config.

    Args:
        settings: Application settings

    Returns:
        Chunker instance
    """
    method = settings.chunking.method.lower()

    if method == "semantic":
        logger.info("Using semantic chunker")
        return SemanticChunker(settings)
    elif method == "hierarchical":
        logger.info("Using hierarchical chunker")
        return HierarchicalChunker(settings)
    else:  # default to token
        logger.info("Using token-based chunker")
        return TextChunker(settings)
