"""
Custom exceptions for the RAG system.
Provides better error handling and debugging.
"""


class RAGException(Exception):
    """Base exception for all RAG system errors."""

    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class IngestionError(RAGException):
    """Raised when document ingestion fails."""
    pass


class RetrievalError(RAGException):
    """Raised when retrieval fails."""
    pass


class GenerationError(RAGException):
    """Raised when answer generation fails."""
    pass


class EvaluationError(RAGException):
    """Raised when evaluation fails."""
    pass


class ConfigurationError(RAGException):
    """Raised when configuration is invalid."""
    pass


class VectorStoreError(RAGException):
    """Raised when vector store operations fail."""
    pass


class EmbeddingError(RAGException):
    """Raised when embedding generation fails."""
    pass


class ChunkingError(RAGException):
    """Raised when text chunking fails."""
    pass
