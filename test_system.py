#!/usr/bin/env python3
"""
Quick system test to verify all components are working.
Run this after installation to check the system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_settings
from src.utils.logger import setup_logging, get_logger

# Setup logging
setup_logging(log_level='INFO')
logger = get_logger(__name__)

def test_configuration():
    """Test configuration loading."""
    logger.info("=" * 60)
    logger.info("Testing Configuration System")
    logger.info("=" * 60)

    try:
        settings = get_settings()
        logger.info("✅ Settings loaded successfully")
        logger.info(f"   Model: {settings.generation.model_name}")
        logger.info(f"   Chunking: {settings.chunking.method}")
        logger.info(f"   Vector search: {settings.retrieval.use_vector_search}")
        logger.info(f"   Reranking: {settings.retrieval.use_reranking}")
        return True
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False

def test_tokenizer():
    """Test tokenizer."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Tokenizer")
    logger.info("=" * 60)

    try:
        from src.core.tokenizer import Tokenizer
        from src.config import get_settings

        settings = get_settings()
        tokenizer = Tokenizer(settings)

        text = "The Montreal Protocol is an international environmental agreement."
        count = tokenizer.count_tokens(text)
        tokens = tokenizer.encode(text)

        logger.info(f"✅ Tokenizer working")
        logger.info(f"   Token count: {count}")
        logger.info(f"   Tokens generated: {len(tokens)}")
        return True
    except Exception as e:
        logger.error(f"❌ Tokenizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vector_store():
    """Test vector store connection."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Vector Store")
    logger.info("=" * 60)

    try:
        from src.core.vector_store import VectorStore
        from src.config import get_settings

        settings = get_settings()
        vector_store = VectorStore(settings)

        count = vector_store.count()
        logger.info(f"✅ Vector store connected")
        logger.info(f"   Collection: {vector_store.collection_name}")
        logger.info(f"   Document count: {count}")

        if count == 0:
            logger.warning("   ⚠️  Collection is empty - run ingestion first")

        return True
    except Exception as e:
        logger.error(f"❌ Vector store test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chunking():
    """Test text chunking."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Text Chunking")
    logger.info("=" * 60)

    try:
        from src.ingestion.text_chunker import TextChunker
        from src.config import get_settings

        settings = get_settings()
        # Adjust settings for test
        settings.chunking.chunk_size = 100
        settings.chunking.min_chunk_size = 10

        chunker = TextChunker(settings)

        # Create longer test text
        text = " ".join([
            "The Montreal Protocol is an international treaty designed to protect the ozone layer.",
            "It was agreed on 16 September 1987, and entered into force on 1 January 1989.",
            "The protocol regulates the production and consumption of ozone depleting substances."
        ] * 5)  # Repeat to make it longer

        metadata = {'filename': 'test.pdf', 'page_number': '1'}
        chunks = chunker.chunk(text, metadata)

        logger.info(f"✅ Chunking working")
        logger.info(f"   Input length: {len(text)} chars")
        logger.info(f"   Chunks created: {len(chunks)}")
        if chunks:
            logger.info(f"   First chunk preview: {chunks[0]['text'][:80]}...")

        return True
    except Exception as e:
        logger.error(f"❌ Chunking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_imports():
    """Test that all major modules can be imported."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Module Imports")
    logger.info("=" * 60)

    modules = [
        ("Configuration", "src.config", "get_settings"),
        ("Embeddings", "src.core.embeddings", "EmbeddingModel"),
        ("Vector Store", "src.core.vector_store", "VectorStore"),
        ("Tokenizer", "src.core.tokenizer", "Tokenizer"),
        ("PDF Extractor", "src.ingestion.pdf_extractor", "PDFExtractor"),
        ("Text Chunker", "src.ingestion.text_chunker", "TextChunker"),
        ("Ingestion Pipeline", "src.ingestion.pipeline", "IngestionPipeline"),
        ("Vector Retriever", "src.retrieval.vector_retriever", "VectorRetriever"),
        ("BM25 Retriever", "src.retrieval.bm25_retriever", "BM25Retriever"),
        ("Query Enhancer", "src.retrieval.query_enhancer", "QueryEnhancer"),
        ("Reranker", "src.retrieval.reranker", "CrossEncoderReranker"),
        ("Hybrid Retriever", "src.retrieval.hybrid_retriever", "HybridRetriever"),
        ("Answer Generator", "src.generation.answer_generator", "AnswerGenerator"),
        ("LLM Client", "src.generation.llm_client", "LLMClient"),
        ("Evaluator", "src.evaluation.evaluator", "RAGEvaluator"),
    ]

    failed = []
    for name, module, cls in modules:
        try:
            mod = __import__(module, fromlist=[cls])
            getattr(mod, cls)
            logger.info(f"✅ {name:20s} - OK")
        except Exception as e:
            logger.error(f"❌ {name:20s} - FAILED: {e}")
            failed.append(name)

    if failed:
        logger.error(f"\n❌ {len(failed)} module(s) failed to import: {', '.join(failed)}")
        return False
    else:
        logger.info(f"\n✅ All modules imported successfully")
        return True

def main():
    """Run all tests."""
    print("\n" + "🚀 " * 20)
    print("WORLD-CLASS RAG SYSTEM - SYSTEM TEST")
    print("🚀 " * 20 + "\n")

    results = []

    # Run tests
    results.append(("Configuration", test_configuration()))
    results.append(("Module Imports", test_imports()))
    results.append(("Tokenizer", test_tokenizer()))
    results.append(("Vector Store", test_vector_store()))
    results.append(("Text Chunking", test_chunking()))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status:10s} - {name}")

    logger.info("=" * 60)
    logger.info(f"Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n🎉 All tests passed! System is ready to use.")
        logger.info("\nNext steps:")
        logger.info("  1. Add PDFs to data/ folder")
        logger.info("  2. Run: python scripts/ingest.py data/")
        logger.info("  3. Run: streamlit run app.py")
        logger.info("  4. For evaluation: streamlit run eval_dashboard.py")
        return 0
    else:
        logger.error(f"\n⚠️  {total - passed} test(s) failed. Please check errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
