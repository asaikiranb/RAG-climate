# RAG Evaluation Framework - Quick Start Guide

## Installation

```bash
# Core framework (no additional dependencies needed)
# Optional: Install BLEU/ROUGE support
pip install nltk rouge-score

# Optional: Install RAGAS support
pip install ragas datasets
```

## 5-Minute Quick Start

### 1. Run Retrieval Evaluation

```bash
cd evals/runners
python3 run_retrieval_eval.py \
    --dataset ../datasets/example_test_set.json \
    --k-values 1 3 5 10
```

**Output**: MRR, MAP, Precision@K, Recall@K, NDCG@K, Hit Rate@K

### 2. Run Generation Evaluation

```bash
python3 run_generation_eval.py \
    --dataset ../datasets/example_test_set.json \
    --generate-answers  # Use this to generate fresh answers
```

**Output**: Faithfulness, Relevance, Citation Accuracy, BLEU, ROUGE

### 3. Run End-to-End Evaluation

```bash
python3 run_e2e_eval.py \
    --dataset ../datasets/example_test_set.json \
    --config-name "my_test"
```

**Output**: All metrics + latency + quality scores

### 4. Run Ablation Study

```bash
python3 run_e2e_eval.py \
    --dataset ../datasets/example_test_set.json \
    --ablation ../datasets/ablation_configs.json \
    --export-csv ../reports/comparison.csv
```

**Output**: Comparison of 10 different configurations

## Programmatic Usage

### Simple Retrieval Check

```python
from src.config import get_settings
from src.evaluation import RAGEvaluator

# Initialize
evaluator = RAGEvaluator(settings=get_settings())

# Prepare test cases
test_cases = [
    {
        'query': 'What is the ideal temperature?',
        'relevant_chunk_ids': ['chunk_001', 'chunk_002']
    }
]

# Evaluate
result = evaluator.evaluate_retrieval(test_cases)

# Print results
print(f"MRR: {result.mrr:.3f}")
print(f"MAP: {result.map_score:.3f}")
```

### Complete Evaluation

```python
from src.evaluation import RAGEvaluator
import json

# Load dataset
with open('evals/datasets/example_test_set.json') as f:
    test_dataset = json.load(f)['examples']

# Run evaluation
evaluator = RAGEvaluator()
result = evaluator.evaluate_complete(
    test_dataset,
    config_name="baseline",
    compute_ragas=False  # Set to True if you have RAGAS
)

# Print summary
print(result.summary())

# Save results
evaluator.export_to_json(result, 'results.json')
```

### Generate Synthetic Dataset

```python
from pathlib import Path
from src.config import get_settings
from src.evaluation import DatasetGenerator

# Initialize
generator = DatasetGenerator(get_settings())

# Generate from PDF
examples = generator.generate_from_pdf(
    pdf_path=Path("data/my_document.pdf"),
    num_examples=50
)

# Save
generator.save_dataset(
    examples,
    output_path=Path("evals/datasets/my_test_set.json")
)
```

## Common Tasks

### Task 1: Evaluate After Code Changes

```bash
# Before changes
python3 run_e2e_eval.py \
    --dataset ../datasets/example_test_set.json \
    --config-name "before" \
    --output ../reports/before.json

# Make your changes...

# After changes
python3 run_e2e_eval.py \
    --dataset ../datasets/example_test_set.json \
    --config-name "after" \
    --output ../reports/after.json

# Compare manually or use diff
```

### Task 2: Find Best Configuration

```bash
# Run ablation study
python3 run_e2e_eval.py \
    --dataset ../datasets/example_test_set.json \
    --ablation ../datasets/ablation_configs.json \
    --export-csv ../reports/comparison.csv

# Open comparison.csv to see which config performs best
```

### Task 3: Create Custom Test Set

```python
import json

# Create test cases
test_cases = {
    "metadata": {"name": "My Test Set"},
    "examples": [
        {
            "question": "How do I troubleshoot X?",
            "answer": "To troubleshoot X, follow these steps...",
            "relevant_chunk_ids": ["chunk_123", "chunk_456"],
            "question_type": "how-to",
            "difficulty": "medium"
        },
        # Add more examples...
    ]
}

# Save
with open('my_test_set.json', 'w') as f:
    json.dump(test_cases, f, indent=2)
```

### Task 4: Debug Poor Performance

```python
# Run with detailed results
evaluator = RAGEvaluator()
e2e_result, detailed_results = evaluator.evaluate_end_to_end(test_dataset)

# Examine failures
for i, detail in enumerate(detailed_results):
    if not detail.get('success'):
        print(f"Query {i} failed:")
        print(f"  Question: {detail['query']}")
        print(f"  Error: {detail.get('error')}")
    elif detail.get('confidence', 1.0) < 0.5:
        print(f"Query {i} low confidence:")
        print(f"  Question: {detail['query']}")
        print(f"  Answer: {detail['answer']}")
        print(f"  Confidence: {detail['confidence']:.2f}")
```

## Understanding Results

### Retrieval Metrics

- **MRR ≥ 0.8**: Excellent - relevant docs usually in top 1-2
- **MRR 0.6-0.8**: Good - relevant docs in top 3-5
- **MRR < 0.6**: Needs improvement

- **Recall@5 ≥ 0.8**: Excellent - finding most relevant docs
- **Recall@5 0.6-0.8**: Good - missing some relevant docs
- **Recall@5 < 0.6**: Poor - missing many relevant docs

### Generation Metrics

- **Faithfulness ≥ 0.9**: Excellent - answers well-grounded
- **Faithfulness 0.7-0.9**: Good - mostly grounded
- **Faithfulness < 0.7**: Hallucination issues

- **Relevance ≥ 0.9**: Excellent - answers address questions
- **Relevance 0.7-0.9**: Good - mostly relevant
- **Relevance < 0.7**: Not addressing questions well

### End-to-End Metrics

- **Latency < 1000ms**: Excellent for user experience
- **Latency 1000-2000ms**: Acceptable
- **Latency > 2000ms**: Slow, optimize needed

- **Success Rate ≥ 0.95**: Production-ready
- **Success Rate 0.8-0.95**: Good, but watch failures
- **Success Rate < 0.8**: Reliability issues

## Troubleshooting

### "No module named 'ragas'"
```bash
pip install ragas datasets
# Or skip RAGAS: don't use --use-ragas flag
```

### "NLTK/ROUGE not available"
```bash
pip install nltk rouge-score
python3 -c "import nltk; nltk.download('punkt')"
```

### "Vector store not found"
```bash
# You need to ingest documents first
cd /path/to/project
python3 src/ingestion/pipeline.py --input data/pdfs
```

### "Evaluation too slow"
```bash
# Use fewer test cases
python3 run_e2e_eval.py --dataset first_10_examples.json

# Or disable expensive features
# Edit ablation config to set:
# "use_hyde": false
# "use_reranking": false
```

## Next Steps

1. **Read**: `evals/README.md` for comprehensive documentation
2. **Run**: `python3 evals/example_usage.py` for working examples
3. **Explore**: Modify `ablation_configs.json` to test your hypotheses
4. **Create**: Generate your own test datasets from domain PDFs
5. **Monitor**: Track metrics over time as you improve the system

## Support

- Full Documentation: `evals/README.md`
- Implementation Details: `evals/IMPLEMENTATION_SUMMARY.md`
- Example Code: `evals/example_usage.py`
- Example Dataset: `evals/datasets/example_test_set.json`
