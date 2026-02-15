#!/usr/bin/env python3
"""
CLI script to evaluate retrieval performance.

Usage:
    python run_retrieval_eval.py --dataset path/to/test_set.json --output results.json
    python run_retrieval_eval.py --dataset test_set.json --k-values 1 3 5 10 20
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_settings
from src.evaluation.evaluator import RAGEvaluator
from src.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


def load_test_dataset(dataset_path: Path) -> list:
    """Load test dataset from JSON file."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    # Handle different dataset formats
    if 'examples' in data:
        examples = data['examples']
    else:
        examples = data

    # Convert to retrieval test cases
    test_cases = []
    for example in examples:
        test_cases.append({
            'query': example.get('question', example.get('query')),
            'relevant_chunk_ids': example.get('relevant_chunk_ids', []),
        })

    return test_cases


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG system retrieval performance"
    )
    parser.add_argument(
        '--dataset',
        type=Path,
        required=True,
        help='Path to test dataset JSON file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Path to save results JSON (default: evals/reports/retrieval_results.json)'
    )
    parser.add_argument(
        '--k-values',
        type=int,
        nargs='+',
        default=[1, 3, 5, 10],
        help='K values for metrics (default: 1 3 5 10)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level=args.log_level)

    logger.info("="*60)
    logger.info("RAG Retrieval Evaluation")
    logger.info("="*60)

    # Load dataset
    logger.info(f"Loading test dataset from: {args.dataset}")
    test_cases = load_test_dataset(args.dataset)
    logger.info(f"Loaded {len(test_cases)} test cases")

    # Initialize evaluator
    settings = get_settings()
    evaluator = RAGEvaluator(settings=settings, k_values=args.k_values)

    # Run evaluation
    logger.info("Starting retrieval evaluation...")
    try:
        result = evaluator.evaluate_retrieval(test_cases)

        # Print results
        print("\n" + "="*60)
        print("RETRIEVAL EVALUATION RESULTS")
        print("="*60)
        print(f"Number of queries: {len(test_cases)}")
        print(f"\nMean Reciprocal Rank (MRR): {result.mrr:.4f}")
        print(f"Mean Average Precision (MAP): {result.map_score:.4f}")
        print(f"Avg Retrieved Docs: {result.avg_retrieved_docs:.1f}")

        print("\nPrecision@K:")
        for k, score in sorted(result.precision_at_k.items()):
            print(f"  P@{k:2d}: {score:.4f}")

        print("\nRecall@K:")
        for k, score in sorted(result.recall_at_k.items()):
            print(f"  R@{k:2d}: {score:.4f}")

        print("\nNDCG@K:")
        for k, score in sorted(result.ndcg_at_k.items()):
            print(f"  NDCG@{k:2d}: {score:.4f}")

        print("\nHit Rate@K:")
        for k, score in sorted(result.hit_rate_at_k.items()):
            print(f"  HR@{k:2d}: {score:.4f}")

        print("="*60)

        # Save results
        if args.output:
            output_path = args.output
        else:
            output_path = project_root / "evals" / "reports" / "retrieval_results.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_dict = result.to_dict()
        results_dict['num_queries'] = len(test_cases)
        results_dict['k_values'] = args.k_values

        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"Results saved to: {output_path}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
