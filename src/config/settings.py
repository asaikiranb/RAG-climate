"""
Centralized configuration management using Pydantic for validation.
All environment variables and constants in one place.
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    model_name: str = Field(default="all-MiniLM-L6-v2", description="Sentence transformer model")
    batch_size: int = Field(default=32, description="Batch size for embedding generation")
    device: str = Field(default="cpu", description="Device to run embeddings on (cpu/cuda)")
    cache_dir: Optional[Path] = Field(default=None, description="Cache directory for model")


class ChunkingConfig(BaseModel):
    """Text chunking configuration."""
    chunk_size: int = Field(default=1000, description="Target tokens per chunk")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks in tokens")
    method: str = Field(default="token", description="Chunking method: token, semantic, or hierarchical")
    min_chunk_size: int = Field(default=100, description="Minimum chunk size in tokens")

    @validator("chunk_overlap")
    def validate_overlap(cls, v, values):
        if "chunk_size" in values and v >= values["chunk_size"]:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class ChromaDBConfig(BaseModel):
    """ChromaDB configuration."""
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("CHROMA_API_KEY"))
    host: Optional[str] = Field(default_factory=lambda: os.getenv("CHROMA_HOST"))
    collection_name: str = Field(
        default_factory=lambda: os.getenv("CHROMA_COLLECTION_NAME", "hvac_documents")
    )
    persist_directory: Path = Field(default=Path("chroma_db"))
    use_cloud: bool = Field(default=False, description="Use ChromaDB Cloud vs local")


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""
    # Search settings
    top_k: int = Field(default=5, description="Final number of chunks to return")
    initial_k: int = Field(default=20, description="Initial retrieval before reranking")

    # Retrieval methods
    use_vector_search: bool = Field(default=True)
    use_bm25_search: bool = Field(default=True)
    use_reranking: bool = Field(default=True, description="Use cross-encoder reranking")

    # Advanced retrieval
    use_hyde: bool = Field(default=True, description="Use HyDE (Hypothetical Document Embeddings)")
    use_query_expansion: bool = Field(default=True, description="Generate multiple query variations")
    num_query_variations: int = Field(default=3, description="Number of query expansions")
    use_mmr: bool = Field(default=True, description="Use MMR for diversity")
    mmr_lambda: float = Field(default=0.7, description="MMR diversity parameter (0=diverse, 1=relevant)")

    # RRF settings
    rrf_k: int = Field(default=60, description="RRF constant for rank fusion")

    # Reranker settings
    reranker_model: str = Field(
        default="ms-marco-MiniLM-L-12-v2",
        description="Cross-encoder model for reranking"
    )
    reranker_batch_size: int = Field(default=16)


class GenerationConfig(BaseModel):
    """LLM generation configuration."""
    # API settings
    groq_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))

    # Model settings
    model_name: str = Field(default="llama-3.3-70b-versatile", description="LLM model to use")
    temperature: float = Field(default=0.2, description="Generation temperature")
    max_tokens: int = Field(default=800, description="Max output tokens")
    top_p: float = Field(default=0.9, description="Nucleus sampling parameter")

    # Advanced generation
    use_citation_validation: bool = Field(default=True, description="Validate citations are grounded")
    use_answer_verification: bool = Field(default=True, description="Second pass to check quality")
    use_confidence_scoring: bool = Field(default=True, description="Compute confidence score")
    use_context_reordering: bool = Field(default=True, description="Reorder context for better performance")

    # Self-consistency
    num_generations: int = Field(default=1, description="Number of answers to generate for self-consistency")


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    # Test dataset
    test_dataset_path: Path = Field(default=Path("evals/datasets/test_set.json"))
    num_synthetic_examples: int = Field(default=100, description="Synthetic examples to generate")
    num_human_examples: int = Field(default=50, description="Target for human-labeled examples")

    # Metrics to compute
    compute_retrieval_metrics: bool = Field(default=True)
    compute_generation_metrics: bool = Field(default=True)
    compute_ragas_metrics: bool = Field(default=True)

    # LLM-as-judge
    judge_model: str = Field(default="gpt-4", description="Model for LLM-as-judge evaluation")
    judge_temperature: float = Field(default=0.0)


class CachingConfig(BaseModel):
    """Caching configuration."""
    enable_query_cache: bool = Field(default=True)
    enable_embedding_cache: bool = Field(default=True)
    enable_llm_cache: bool = Field(default=True)
    cache_dir: Path = Field(default=Path(".cache"))
    redis_url: Optional[str] = Field(default_factory=lambda: os.getenv("REDIS_URL"))


class ObservabilityConfig(BaseModel):
    """Logging and monitoring configuration."""
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[Path] = Field(default=Path("logs/rag.log"))
    enable_tracing: bool = Field(default=True, description="Enable request tracing")
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")


class Settings(BaseModel):
    """Main settings container."""
    # Project paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = Field(default=Path("data"))

    # Component configs
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    chromadb: ChromaDBConfig = Field(default_factory=ChromaDBConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    caching: CachingConfig = Field(default_factory=CachingConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    class Config:
        arbitrary_types_allowed = True

    def validate_api_keys(self) -> None:
        """Validate required API keys are present."""
        if not self.generation.groq_api_key:
            raise ValueError("GROQ_API_KEY is required in .env file")

        if self.chromadb.use_cloud and not self.chromadb.api_key:
            raise ValueError("CHROMA_API_KEY required when using ChromaDB Cloud")


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses LRU cache to avoid re-parsing environment on every call.
    """
    settings = Settings()
    settings.validate_api_keys()
    return settings


# Convenience function to load custom config from YAML
def load_config_from_yaml(yaml_path: Path) -> Settings:
    """Load configuration from YAML file (for experiments)."""
    import yaml

    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return Settings(**config_dict)
