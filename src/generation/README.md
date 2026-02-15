# Generation Module

Production-ready answer generation with citation validation, quality assurance, and advanced optimizations.

## Overview

This module implements the final stage of the RAG pipeline, transforming retrieved context chunks into high-quality, cited answers. It includes multiple layers of quality assurance and optimization techniques.

## Architecture

```
generation/
├── llm_client.py         # Groq API wrapper with retry logic
├── prompts.py           # System prompts and templates
├── answer_generator.py  # Main answer generation orchestrator
└── validators.py        # Citation and answer quality validation
```

## Components

### 1. LLMClient (`llm_client.py`)

Robust API wrapper for Groq LLM inference.

**Features:**
- Exponential backoff retry logic
- Rate limiting and error handling
- Token usage tracking
- Latency monitoring
- Structured error responses

**Example Usage:**
```python
from src.config import get_settings
from src.generation import LLMClient

settings = get_settings()
client = LLMClient(settings)

# Simple generation
response = client.generate_with_system_prompt(
    system_prompt="You are a helpful assistant.",
    user_message="What is HVAC?",
    temperature=0.2
)
print(response.content)
print(f"Tokens used: {response.tokens_used}")
```

**Key Methods:**
- `generate(messages, **kwargs)` - Main generation with full control
- `generate_with_system_prompt(system, user, **kwargs)` - Convenience method
- `get_usage_stats()` - Get token usage and error metrics

### 2. Prompts (`prompts.py`)

System prompts and templates optimized for HVAC technical QA.

**Features:**
- Domain-specific instruction prompts
- Citation enforcement
- Few-shot examples
- Context formatting utilities

**Prompt Types:**
- `SYSTEM_PROMPT_QA` - Main technical QA prompt
- `SYSTEM_PROMPT_CONVERSATIONAL` - Friendlier variant
- `SYSTEM_PROMPT_CITATION_VALIDATOR` - For validation
- `SYSTEM_PROMPT_ANSWER_VERIFIER` - For quality checks

**Example Usage:**
```python
from src.generation import get_system_prompt, create_qa_prompt

# Get system prompt
system = get_system_prompt("qa")

# Create full QA prompt
chunks = [
    {"text": "HVAC filters should be changed every 1-3 months.",
     "metadata": {"source": "manual.pdf"}}
]
prompt = create_qa_prompt("How often to change filters?", chunks)

# Use with LLM
user_message = prompt.format_user(
    query="How often to change filters?",
    context=format_context_documents(chunks)
)
```

### 3. AnswerGenerator (`answer_generator.py`)

Main orchestrator for answer generation with quality assurance.

**Features:**
- **Context Reordering**: Mitigates "Lost in the Middle" problem
- **Multi-attempt Generation**: Retry with fallback strategies
- **Confidence Scoring**: Multi-factor confidence calculation
- **Validation Integration**: Citation and answer verification
- **Batch Processing**: Handle multiple queries efficiently

**Key Techniques:**

#### Context Reordering (Lost in the Middle Mitigation)
LLMs pay more attention to the start and end of context. We reorder chunks to place:
- Most relevant chunks at the **beginning** (primacy effect)
- Second-most relevant at the **end** (recency effect)
- Less relevant chunks in the **middle**

```python
# Automatic reordering
generator = AnswerGenerator(settings)
answer = generator.generate(query, chunks)  # Chunks auto-reordered
```

#### Confidence Scoring
Combines multiple signals:
- Context quality (avg relevance score): 30%
- Citation quality: 25%
- Answer verification score: 30%
- LLM completion status: 15%

**Example Usage:**
```python
from src.generation import AnswerGenerator
from src.config import get_settings

settings = get_settings()
generator = AnswerGenerator(settings)

# Generate answer
context_chunks = [
    {
        "text": "HVAC systems require regular maintenance...",
        "metadata": {"score": 0.85, "source": "manual.pdf"}
    }
]

result = generator.generate(
    query="How to maintain HVAC system?",
    context_chunks=context_chunks,
    enable_validation=True
)

print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence_score:.2f}")
print(f"High Quality: {result.is_high_quality}")
```

**Batch Generation:**
```python
queries_and_contexts = [
    ("Query 1", chunks1),
    ("Query 2", chunks2),
]

results = generator.generate_batch(queries_and_contexts)
for result in results:
    print(f"Q: {result.question}")
    print(f"A: {result.answer}")
    print(f"Confidence: {result.confidence_score}\n")
```

### 4. Validators (`validators.py`)

Citation validation and answer quality verification.

**Components:**

#### CitationValidator
Ensures claims are grounded in provided context.

**Validation Layers:**
1. **Syntactic**: Check citation format `[1]`, `[2,3]`
2. **Reference Check**: Ensure citations point to valid documents
3. **Semantic**: Use LLM to verify claims match cited sources
4. **Coverage**: Check that major claims have citations

```python
from src.generation import CitationValidator

validator = CitationValidator(settings, llm_client)

# Validate citations
result = validator.validate(
    answer="HVAC filters need changing every 1-3 months [1].",
    chunks=context_chunks,
    use_semantic=True
)

print(f"Valid: {result.is_valid}")
print(f"Citation Score: {result.citation_score:.2f}")
print(f"Invalid Citations: {result.invalid_citations}")
```

#### AnswerVerifier
Evaluates answer quality across multiple dimensions.

**Quality Dimensions:**
- **Completeness**: Does it fully address the question?
- **Accuracy**: Are technical details correct?
- **Citation Quality**: Are claims properly cited?
- **Clarity**: Is it well-organized and readable?

```python
from src.generation import AnswerVerifier

verifier = AnswerVerifier(settings, llm_client)

result = verifier.verify(
    question="How often to change HVAC filters?",
    answer="Filters should be changed every 1-3 months [1]...",
    chunks=context_chunks
)

print(f"Overall Score: {result.overall_score:.2f}")
print(f"Acceptable: {result.is_acceptable}")
print(f"Issues: {result.issues}")
print(f"Suggestions: {result.suggestions}")
```

## Configuration

All generation settings are in `src/config/settings.py`:

```python
class GenerationConfig(BaseModel):
    # API settings
    groq_api_key: str  # Set in .env

    # Model settings
    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.2
    max_tokens: int = 800
    top_p: float = 0.9

    # Features
    use_citation_validation: bool = True
    use_answer_verification: bool = True
    use_confidence_scoring: bool = True
    use_context_reordering: bool = True
```

## Complete Example

```python
from src.config import get_settings
from src.generation import AnswerGenerator

# Initialize
settings = get_settings()
generator = AnswerGenerator(settings)

# Retrieved context from retrieval module
context_chunks = [
    {
        "text": "Regular HVAC maintenance includes changing filters every 1-3 months, checking refrigerant levels annually, and cleaning coils biannually.",
        "metadata": {
            "score": 0.92,
            "source": "hvac_maintenance_guide.pdf",
            "page": 12
        }
    },
    {
        "text": "Dirty filters reduce airflow efficiency by up to 15% and increase energy costs.",
        "metadata": {
            "score": 0.78,
            "source": "energy_efficiency_report.pdf",
            "page": 5
        }
    }
]

# Generate answer
result = generator.generate(
    query="What HVAC maintenance should I perform regularly?",
    context_chunks=context_chunks,
    enable_validation=True,
    max_attempts=2
)

# Access results
print("=" * 60)
print(f"QUESTION: {result.question}")
print("=" * 60)
print(f"\nANSWER:\n{result.answer}\n")
print("=" * 60)
print(f"Confidence Score: {result.confidence_score:.2f}")
print(f"High Quality: {result.is_high_quality}")
print(f"Generation Time: {result.metadata['generation_time_ms']:.0f}ms")
print(f"Tokens Used: {result.metadata['tokens_used']}")

if result.citation_validation:
    print(f"\nCitation Validation:")
    print(f"  Valid: {result.citation_validation.is_valid}")
    print(f"  Score: {result.citation_validation.citation_score:.2f}")

if result.answer_verification:
    print(f"\nAnswer Verification:")
    print(f"  Overall Score: {result.answer_verification.overall_score:.2f}")
    print(f"  Completeness: {result.answer_verification.completeness_score:.2f}")
    print(f"  Accuracy: {result.answer_verification.accuracy_score:.2f}")

# Check usage stats
stats = generator.get_usage_stats()
print(f"\nLLM Usage Stats:")
print(f"  Total Calls: {stats['total_calls']}")
print(f"  Total Tokens: {stats['total_tokens']}")
print(f"  Error Rate: {stats['error_rate']:.1%}")
```

## Error Handling

All modules use structured exceptions from `src.utils.exceptions`:

```python
from src.utils.exceptions import GenerationError

try:
    result = generator.generate(query, chunks)
except GenerationError as e:
    print(f"Error: {e.message}")
    print(f"Details: {e.details}")
```

## Testing

Run integration tests:

```bash
python test_generation_integration.py
```

Test individual components:

```python
# Test citation extraction
from src.generation import CitationValidator
validator = CitationValidator(settings)
citations = validator.extract_citations("This is true [1,2].")
assert citations == {1, 2}

# Test context reordering
from src.generation import AnswerGenerator
generator = AnswerGenerator(settings)
chunks = [...]
reordered = generator.reorder_context_for_attention(chunks)
```

## Performance Optimization

**Tips for production:**

1. **Disable validation for speed**: Set `enable_validation=False` if you trust your context quality
2. **Adjust max_attempts**: Use 1 for speed, 2-3 for quality
3. **Batch processing**: Use `generate_batch()` for multiple queries
4. **Cache LLM responses**: Enable LLM caching in settings
5. **Monitor token usage**: Use `get_usage_stats()` to track costs

## Advanced Features

### Self-Consistency (Multi-Generation)

Generate multiple answers and select the best:

```python
# In settings
generation.num_generations = 3  # Generate 3 candidates

# The generator will automatically pick the highest quality answer
```

### Custom Prompts

Override default prompts:

```python
from src.generation.prompts import PromptTemplate

custom_prompt = PromptTemplate(
    system="You are an HVAC expert...",
    user_template="Context: {context}\n\nQuestion: {query}"
)
```

### Temperature Tuning

- **Low (0.0-0.3)**: Deterministic, factual answers
- **Medium (0.3-0.7)**: Balanced creativity
- **High (0.7-1.0)**: Creative, diverse answers

```python
# Override temperature per query
response = llm_client.generate(messages, temperature=0.1)
```

## Troubleshooting

**Issue: "GROQ_API_KEY not found"**
- Solution: Set `GROQ_API_KEY` in `.env` file

**Issue: "Citation validation fails"**
- Check that context chunks have proper text
- Verify LLM is generating citations in `[N]` format
- Try disabling semantic validation: `use_semantic=False`

**Issue: Low confidence scores**
- Improve retrieval quality (better chunks)
- Increase temperature for more detailed answers
- Check that chunks have relevance scores in metadata

**Issue: Token limit exceeded**
- Reduce `max_tokens` in settings
- Use fewer context chunks
- Implement truncation in prompts

## References

- **Lost in the Middle**: [Paper](https://arxiv.org/abs/2307.03172) on LLM attention patterns
- **Citation Generation**: Best practices for grounded QA
- **Groq API**: [Documentation](https://console.groq.com/docs)

## Next Steps

After generation, answers can be:
1. Returned to users via the Streamlit UI
2. Evaluated using the `evals/` module
3. Logged for monitoring and analysis
4. Cached for repeated queries

See `main.py` for integration with the full RAG pipeline.
