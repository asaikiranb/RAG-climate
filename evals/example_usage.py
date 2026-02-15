#!/usr/bin/env python3
"""
Example usage of the RAG evaluation framework.

This script demonstrates:
1. Loading a test dataset
2. Running retrieval evaluation
3. Running generation evaluation
4. Running end-to-end evaluation
5. Running an ablation study
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_settings
from src.evaluation import RAGEvaluator
from src.utils.logger import setup_logging, get_logger
import json

logger = get_logger(__name__)


def example_1_retrieval_evaluation():
    """Example: Evaluate retrieval performance."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Retrieval Evaluation")
    print("="*60)

    # Load test dataset
    dataset_path = project_root / "evals" / "datasets" / "example_test_set.json"
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    test_cases = [
        {
            'query': ex['question'],
            'relevant_chunk_ids': ex['relevant_chunk_ids']
        }
        for ex in data['examples']
    ]

    # Initialize evaluator
    settings = get_settings()
    evaluator = RAGEvaluator(settings=settings, k_values=[1, 3, 5])

    # Run evaluation
    print(f"Evaluating {len(test_cases)} queries...")
    result = evaluator.evaluate_retrieval(test_cases)

    # Print results
    print(f"\nResults:")
    print(f"  MRR:  {result.mrr:.4f}")
    print(f"  MAP:  {result.map_score:.4f}")
    print(f"  P@5:  {result.precision_at_k.get(5, 0):.4f}")
    print(f"  R@5:  {result.recall_at_k.get(5, 0):.4f}")


def example_2_generation_evaluation():
    """Example: Evaluate generation quality."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Generation Evaluation")
    print("="*60)

    # For this example, we'll use pre-existing answers from the dataset
    dataset_path = project_root / "evals" / "datasets" / "example_test_set.json"
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    # Prepare test cases (using dataset answers as both generated and reference)
    test_cases = [
        {
            'question': ex['question'],
            'answer': ex['answer'],
            'context_chunks': [{'document': ex['answer']}],  # Simplified for example
            'citations': [],
            'reference': ex.get('reference', ex['answer'])
        }
        for ex in data['examples'][:3]  # Use first 3 examples
    ]

    # Initialize evaluator
    settings = get_settings()
    evaluator = RAGEvaluator(settings=settings)

    # Run evaluation
    print(f"Evaluating {len(test_cases)} answers...")
    result = evaluator.evaluate_generation(test_cases)

    # Print results
    print(f"\nResults:")
    print(f"  Faithfulness:      {result.faithfulness_score:.4f}")
    print(f"  Relevance:         {result.relevance_score:.4f}")
    print(f"  Citation Accuracy: {result.citation_accuracy:.4f}")


def example_3_e2e_evaluation():
    """Example: End-to-end evaluation."""
    print("\n" + "="*60)
    print("EXAMPLE 3: End-to-End Evaluation")
    print("="*60)

    # Load test dataset
    dataset_path = project_root / "evals" / "datasets" / "example_test_set.json"
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    test_cases = data['examples'][:3]  # Use first 3 for demo

    # Initialize evaluator
    settings = get_settings()
    evaluator = RAGEvaluator(settings=settings)

    # Run evaluation
    print(f"Running end-to-end evaluation for {len(test_cases)} queries...")
    print("This will retrieve context and generate answers...")

    try:
        result = evaluator.evaluate_complete(
            test_cases,
            config_name="example_run",
            compute_ragas=False  # Set to True if you have RAGAS installed
        )

        # Print summary
        print(result.summary())

        # Save results
        output_path = project_root / "evals" / "reports" / "example_results.json"
        evaluator.export_to_json(result, output_path)
        print(f"\nResults saved to: {output_path}")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Make sure you have ingested documents into the vector store first!")


def example_4_ablation_study():
    """Example: Ablation study comparing configurations."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Ablation Study")
    print("="*60)

    # Define configurations to compare
    configurations = [
        {
            'name': 'full_system',
            'settings': {
                'use_hyde': True,
                'use_reranking': True,
                'use_mmr': True
            }
        },
        {
            'name': 'no_hyde',
            'settings': {
                'use_hyde': False,
                'use_reranking': True,
                'use_mmr': True
            }
        },
        {
            'name': 'no_reranking',
            'settings': {
                'use_hyde': True,
                'use_reranking': False,
                'use_mmr': True
            }
        }
    ]

    # Load test dataset
    dataset_path = project_root / "evals" / "datasets" / "example_test_set.json"
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    test_cases = data['examples'][:2]  # Use just 2 for quick demo

    # Initialize evaluator
    settings = get_settings()
    evaluator = RAGEvaluator(settings=settings)

    # Run ablation study
    print(f"Comparing {len(configurations)} configurations...")
    print(f"On {len(test_cases)} test cases...\n")

    try:
        results = evaluator.compare_configurations(test_cases, configurations)

        # Export comparison
        csv_path = project_root / "evals" / "reports" / "example_ablation.csv"
        evaluator.export_to_csv(results, csv_path)
        print(f"\nComparison saved to: {csv_path}")

    except Exception as e:
        print(f"Error during ablation study: {e}")
        print("Make sure you have ingested documents into the vector store first!")


def main():
    """Run all examples."""
    setup_logging(log_level="INFO")

    print("\n" + "="*60)
    print("RAG Evaluation Framework - Examples")
    print("="*60)

    # Example 1: Retrieval only
    try:
        example_1_retrieval_evaluation()
    except Exception as e:
        logger.error(f"Example 1 failed: {e}")

    # Example 2: Generation only
    try:
        example_2_generation_evaluation()
    except Exception as e:
        logger.error(f"Example 2 failed: {e}")

    # Example 3: End-to-end (requires ingested documents)
    print("\n" + "="*60)
    print("NOTE: Examples 3 and 4 require documents to be ingested")
    print("Run the ingestion pipeline first if these fail")
    print("="*60)

    try:
        example_3_e2e_evaluation()
    except Exception as e:
        logger.error(f"Example 3 failed: {e}")

    # Example 4: Ablation study (requires ingested documents)
    try:
        example_4_ablation_study()
    except Exception as e:
        logger.error(f"Example 4 failed: {e}")

    print("\n" + "="*60)
    print("Examples complete! Check evals/reports/ for results")
    print("="*60)


if __name__ == "__main__":
    main()
