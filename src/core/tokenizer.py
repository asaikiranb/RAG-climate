"""
Tokenization utilities using tiktoken.
Used for chunking and token counting.
"""

from typing import List
import tiktoken

from src.config import Settings
from src.utils.logger import get_logger
from src.utils.exceptions import RAGException

logger = get_logger(__name__)


class Tokenizer:
    """Wrapper around tiktoken for consistent tokenization."""

    def __init__(self, settings: Settings = None, encoding_name: str = "cl100k_base"):
        """
        Initialize tokenizer.

        Args:
            settings: Application settings (optional)
            encoding_name: Tiktoken encoding name
        """
        self.encoding_name = encoding_name

        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
            logger.debug(f"Initialized tokenizer with encoding: {encoding_name}")
        except Exception as e:
            raise RAGException(f"Failed to load tokenizer: {e}")

    def encode(self, text: str) -> List[int]:
        """
        Encode text to tokens.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        return self.encoding.encode(text)

    def decode(self, tokens: List[int]) -> str:
        """
        Decode tokens to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text
        """
        return self.encoding.decode(tokens)

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text

        Returns:
            Token count
        """
        return len(self.encoding.encode(text))

    def truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token limit.

        Args:
            text: Input text
            max_tokens: Maximum tokens allowed

        Returns:
            Truncated text
        """
        tokens = self.encode(text)
        if len(tokens) <= max_tokens:
            return text

        truncated_tokens = tokens[:max_tokens]
        return self.decode(truncated_tokens)

    def split_by_tokens(
        self,
        text: str,
        chunk_size: int,
        overlap: int = 0
    ) -> List[str]:
        """
        Split text into chunks by token count.

        Args:
            text: Input text
            chunk_size: Tokens per chunk
            overlap: Overlapping tokens between chunks

        Returns:
            List of text chunks
        """
        tokens = self.encode(text)
        chunks = []

        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.decode(chunk_tokens)
            chunks.append(chunk_text)

            # Move start with overlap
            start = end - overlap

            if end >= len(tokens):
                break

        return chunks
