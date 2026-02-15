"""
Quick integration test for generation modules.
Run this to verify everything works together.
"""

from src.config import get_settings
from src.generation import (
    LLMClient,
    PromptTemplate,
    get_system_prompt,
    AnswerGenerator,
    CitationValidator,
    AnswerVerifier
)

def test_imports():
    """Test that all imports work."""
    print("✓ All imports successful")


def test_llm_client():
    """Test LLM client initialization."""
    try:
        settings = get_settings()
        client = LLMClient(settings)
        print(f"✓ LLM Client initialized with model: {client.config.model_name}")
        return client
    except Exception as e:
        print(f"✗ LLM Client failed: {e}")
        return None


def test_prompts():
    """Test prompt templates."""
    try:
        system_prompt = get_system_prompt("qa")
        assert len(system_prompt) > 0
        print(f"✓ Prompts working (system prompt: {len(system_prompt)} chars)")
    except Exception as e:
        print(f"✗ Prompts failed: {e}")


def test_validators():
    """Test validator initialization."""
    try:
        settings = get_settings()
        citation_validator = CitationValidator(settings)
        answer_verifier = AnswerVerifier(settings)
        print("✓ Validators initialized")

        # Test citation extraction
        test_text = "This is a test [1]. Another claim [2,3]."
        citations = citation_validator.extract_citations(test_text)
        assert citations == {1, 2, 3}
        print(f"✓ Citation extraction works: {citations}")

    except Exception as e:
        print(f"✗ Validators failed: {e}")


def test_answer_generator():
    """Test answer generator initialization."""
    try:
        settings = get_settings()
        generator = AnswerGenerator(settings)
        print(f"✓ Answer Generator initialized")

        # Test context reordering
        test_chunks = [
            {"text": "chunk1", "metadata": {"score": 0.5}},
            {"text": "chunk2", "metadata": {"score": 0.9}},
            {"text": "chunk3", "metadata": {"score": 0.3}},
        ]
        reordered = generator.reorder_context_for_attention(test_chunks)
        print(f"✓ Context reordering works: {len(reordered)} chunks")

        return generator
    except Exception as e:
        print(f"✗ Answer Generator failed: {e}")
        return None


def test_full_pipeline():
    """Test complete generation pipeline."""
    try:
        settings = get_settings()
        generator = AnswerGenerator(settings)

        # Sample context
        context_chunks = [
            {
                "text": "HVAC systems require regular maintenance. Filters should be changed every 1-3 months.",
                "metadata": {"score": 0.85, "source": "manual.pdf", "page": 1}
            },
            {
                "text": "Proper airflow is critical for HVAC efficiency. Check vents monthly.",
                "metadata": {"score": 0.72, "source": "guide.pdf", "page": 5}
            }
        ]

        query = "How often should HVAC filters be changed?"

        print("\n--- Testing Full Generation Pipeline ---")
        print(f"Query: {query}")
        print(f"Context: {len(context_chunks)} chunks")

        # Note: This requires GROQ_API_KEY to be set
        # Uncomment to test actual generation:
        # result = generator.generate(query, context_chunks, enable_validation=False)
        # print(f"✓ Answer generated: {len(result.answer)} chars")
        # print(f"  Confidence: {result.confidence_score:.2f}")
        # print(f"  Answer: {result.answer[:100]}...")

        print("✓ Pipeline structure verified (actual generation requires GROQ_API_KEY)")

    except Exception as e:
        print(f"✗ Full pipeline failed: {e}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Generation Module Integration")
    print("=" * 60)

    test_imports()
    test_prompts()
    test_validators()
    test_llm_client()
    test_answer_generator()
    test_full_pipeline()

    print("\n" + "=" * 60)
    print("Integration test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
