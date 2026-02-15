# RAG System Evaluation Framework

Comprehensive evaluation suite for the HVAC RAG system, supporting retrieval metrics, generation quality assessment, and end-to-end performance analysis.

## Overview

This evaluation framework provides:

- **Retrieval Metrics**: Precision@K, Recall@K, MRR, MAP, NDCG, Hit Rate
- **Generation Metrics**: Faithfulness, Relevance, Citation Accuracy, BLEU, ROUGE
- **RAGAS Integration**: LLM-based evaluation (optional)
- **End-to-End Metrics**: Latency, quality, efficiency
- **Synthetic Dataset Generation**: Automatic Q&A pair creation from PDFs
- **Ablation Studies**: Compare different configurations

## Installation

### Required Dependencies

```bash
pip install numpy loguru pydantic
```

### Optional Dependencies

For BLEU/ROUGE metrics:
```bash
pip install nltk rouge-score
```

For RAGAS evaluation:
```bash
pip install ragas datasets
```

## Quick Start

### 1. Evaluate Retrieval

```bash
cd evals/runners
python run_retrieval_eval.py \
    --dataset ../datasets/example_test_set.json \
    --output ../reports/retrieval_results.json \
    --k-values 1 3 5 10
```

### 2. Evaluate Generation

```bash
python run_generation_eval.py \
    --dataset ../datasets/example_test_set.json \
    --output ../reports/generation_results.json \
    --use-ragas  # Optional: include RAGAS metrics
```

### 3. End-to-End Evaluation

```bash
python run_e2e_eval.py \
    --dataset ../datasets/example_test_set.json \
    --output ../reports/e2e_results.json \
    --config-name "baseline"
```

### 4. Ablation Study

Create an ablation configuration file (e.g., `ablation_configs.json`):

```json
[
  {
    "name": "baseline",
    "settings": {
      "use_hyde": true,
      "use_reranking": true,
      "use_mmr": true
    }
  },
  {
    "name": "no_hyde",
    "settings": {
      "use_hyde": false,
      "use_reranking": true,
      "use_mmr": true
    }
  },
  {
    "name": "no_reranking",
    "settings": {
      "use_hyde": true,
      "use_reranking": false,
      "use_mmr": true
    }
  },
  {
    "name": "minimal",
    "settings": {
      "use_hyde": false,
      "use_reranking": false,
      "use_mmr": false
    }
  }
]
```

Run ablation study:

```bash
python run_e2e_eval.py \
    --dataset ../datasets/example_test_set.json \
    --ablation ablation_configs.json \
    --export-csv ../reports/ablation_comparison.csv
```

## Dataset Format

### Test Dataset Structure

```json
{
  "metadata": {
    "name": "Dataset Name",
    "num_examples": 10
  },
  "examples": [
    {
      "question": "What is X?",
      "answer": "X is...",
      "relevant_chunk_ids": ["chunk_001", "chunk_002"],
      "question_type": "factual",
      "difficulty": "easy",
      "reference": "Optional reference answer"
    }
  ]
}
```

### Required Fields

- `question` or `query`: The user query
- `relevant_chunk_ids`: List of relevant document chunk IDs (for retrieval eval)

### Optional Fields

- `answer`: Pre-generated answer (if not provided, will generate on-the-fly)
- `reference` or `ground_truth`: Reference answer for comparison
- `question_type`: Type of question (factual, how-to, why, comparison, calculation)
- `difficulty`: Difficulty level (easy, medium, hard)
- `context_chunks`: Pre-retrieved context chunks
- `citations`: List of citation indices

## Synthetic Dataset Generation

Generate evaluation datasets automatically from PDFs:

```python
from pathlib import Path
from src.config import get_settings
from src.evaluation import DatasetGenerator

# Initialize
settings = get_settings()
generator = DatasetGenerator(settings)

# Generate from PDF
examples = generator.generate_from_pdf(
    pdf_path=Path("data/hvac_manual.pdf"),
    num_examples=50
)

# Save dataset
generator.save_dataset(
    examples,
    output_path=Path("evals/datasets/synthetic_test_set.json")
)
```

### Control Dataset Characteristics

```python
# Generate with custom distributions
examples = generator.generate_dataset(
    chunks=chunks,
    num_examples=100,
    difficulty_distribution={
        DifficultyLevel.EASY: 0.3,
        DifficultyLevel.MEDIUM: 0.5,
        DifficultyLevel.HARD: 0.2,
    },
    question_type_distribution={
        QuestionType.FACTUAL: 0.3,
        QuestionType.HOWTO: 0.3,
        QuestionType.WHY: 0.2,
        QuestionType.COMPARISON: 0.1,
        QuestionType.CALCULATION: 0.1,
    }
)
```

## Using the Evaluation API

### Programmatic Usage

```python
from src.config import get_settings
from src.evaluation import RAGEvaluator

# Initialize
settings = get_settings()
evaluator = RAGEvaluator(settings=settings)

# Load test dataset
with open("evals/datasets/test_set.json") as f:
    test_dataset = json.load(f)["examples"]

# Run complete evaluation
result = evaluator.evaluate_complete(
    test_dataset,
    config_name="my_config",
    compute_ragas=True
)

# Print summary
print(result.summary())

# Export results
evaluator.export_to_json(result, Path("results.json"))
```

### Evaluate Individual Components

```python
# Retrieval only
retrieval_result = evaluator.evaluate_retrieval(test_cases)

# Generation only
generation_result = evaluator.evaluate_generation(test_cases)

# RAGAS only
ragas_result = evaluator.evaluate_with_ragas(test_cases)

# End-to-end
e2e_result, details = evaluator.evaluate_end_to_end(test_cases)
```

## Metrics Explained

### Retrieval Metrics

- **Precision@K**: Fraction of top-K retrieved documents that are relevant
- **Recall@K**: Fraction of all relevant documents found in top-K
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank of first relevant document
- **MAP (Mean Average Precision)**: Mean of precision values at each relevant position
- **NDCG@K**: Normalized Discounted Cumulative Gain at K
- **Hit Rate@K**: Percentage of queries with at least one relevant doc in top-K

### Generation Metrics

- **Faithfulness**: Answer is grounded in the provided context (0-1)
- **Relevance**: Answer addresses the question (0-1)
- **Citation Accuracy**: Citations are correct and complete (0-1)
- **BLEU**: N-gram overlap with reference answer (0-1)
- **ROUGE**: Recall-oriented overlap with reference (0-1)

### RAGAS Metrics (LLM-based)

- **Faithfulness**: Factual consistency with context
- **Answer Relevancy**: Relevance to the question
- **Context Relevancy**: Relevance of retrieved context
- **Context Recall**: Coverage of ground truth in context
- **Context Precision**: Precision of retrieved context

### End-to-End Metrics

- **Average Latency**: Total time per query (ms)
- **Retrieval Time**: Time spent on retrieval (ms)
- **Generation Time**: Time spent on generation (ms)
- **Confidence Score**: Average answer confidence (0-1)
- **High Quality Rate**: Percentage of high-quality answers
- **Success Rate**: Percentage of successful queries
- **Tokens Used**: Average tokens per query

## Output Format

### JSON Output

```json
{
  "config_name": "baseline",
  "timestamp": "2024-02-14T12:00:00",
  "num_queries": 100,
  "retrieval_metrics": {
    "mrr": 0.85,
    "map_score": 0.78,
    "precision_at_k": {
      "1": 0.75,
      "3": 0.68,
      "5": 0.62
    },
    "recall_at_k": {
      "1": 0.25,
      "3": 0.58,
      "5": 0.72
    },
    "ndcg_at_k": {
      "1": 0.75,
      "3": 0.71,
      "5": 0.74
    }
  },
  "generation_metrics": {
    "faithfulness_score": 0.88,
    "relevance_score": 0.92,
    "citation_accuracy": 0.85
  },
  "e2e_metrics": {
    "avg_latency_ms": 1250.5,
    "avg_confidence_score": 0.87,
    "success_rate": 0.98
  }
}
```

### CSV Output (Ablation Studies)

| config_name | mrr  | map  | faithfulness | relevance | avg_latency_ms | success_rate |
|-------------|------|------|--------------|-----------|----------------|--------------|
| baseline    | 0.85 | 0.78 | 0.88         | 0.92      | 1250.5         | 0.98         |
| no_hyde     | 0.82 | 0.75 | 0.87         | 0.91      | 1100.2         | 0.97         |
| no_rerank   | 0.79 | 0.71 | 0.85         | 0.89      | 980.3          | 0.96         |

## Best Practices

### 1. Dataset Quality

- Use diverse question types and difficulty levels
- Include edge cases and failure modes
- Ensure relevant chunk IDs are accurate
- Provide reference answers when possible

### 2. Evaluation Strategy

- Start with retrieval evaluation to tune retrieval settings
- Then evaluate generation quality
- Run end-to-end evaluation last
- Use ablation studies to understand component impact

### 3. Interpreting Results

- **High Precision, Low Recall**: Retriever is conservative, increase K
- **High Recall, Low Precision**: Retriever is too broad, improve ranking
- **Low Faithfulness**: Generation is hallucinating, improve prompts
- **Low Relevance**: Generation not addressing question, improve prompts
- **High Latency**: Optimize retrieval (caching, fewer candidates)

### 4. Continuous Evaluation

- Track metrics over time as system evolves
- Create regression tests for critical queries
- Monitor production metrics vs evaluation metrics
- Update test datasets as domain evolves

## Troubleshooting

### RAGAS Not Working

```bash
# Install dependencies
pip install ragas datasets

# Set OpenAI API key (RAGAS uses OpenAI by default)
export OPENAI_API_KEY=your_key_here
```

### BLEU/ROUGE Not Available

```bash
# Install NLTK and download data
pip install nltk rouge-score
python -c "import nltk; nltk.download('punkt')"
```

### Out of Memory

- Reduce batch size in generation
- Process dataset in chunks
- Use smaller K values for retrieval
- Disable RAGAS for large datasets

### Slow Evaluation

- Use caching for embeddings and LLM calls
- Reduce number of query variations
- Skip reranking for initial tests
- Use parallel processing (future enhancement)

## Examples

See `datasets/example_test_set.json` for a complete example with 10 test cases covering different question types and difficulty levels.

## Contributing

To add new metrics:

1. Add metric computation to appropriate class in `src/evaluation/metrics.py`
2. Update result dataclasses with new fields
3. Update evaluation orchestrator in `src/evaluation/evaluator.py`
4. Update output formatting and export functions

## License

Part of the ContextualAI-Capstone project.
