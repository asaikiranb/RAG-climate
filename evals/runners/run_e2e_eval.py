#!/usr/bin/env python3
"""
CLI script for end-to-end RAG system evaluation.

Usage:
    python run_e2e_eval.py --dataset path/to/test_set.json
    python run_e2e_eval.py --dataset test_set.json --output results.json --use-ragas
    python run_e2e_eval.py --ablation configs.json --dataset test_set.json
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

    return examples


def load_ablation_configs(config_path: Path) -> list:
    """Load ablation study configurations."""
    with open(config_path, 'r') as f:
        configs = json.load(f)

    return configs


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end RAG system evaluation"
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
        help='Path to save results JSON (default: evals/reports/e2e_results.json)'
    )
    parser.add_argument(
        '--use-ragas',
        action='store_true',
        help='Include RAGAS metrics in evaluation'
    )
    parser.add_argument(
        '--ablation',
        type=Path,
        default=None,
        help='Path to ablation study configuration JSON'
    )
    parser.add_argument(
        '--config-name',
        type=str,
        default='default',
        help='Name for this evaluation configuration'
    )
    parser.add_argument(
        '--export-csv',
        type=Path,
        default=None,
        help='Export comparison results to CSV (for ablation studies)'
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
    logger.info("RAG End-to-End Evaluation")
    logger.info("="*60)

    # Load dataset
    logger.info(f"Loading test dataset from: {args.dataset}")
    test_dataset = load_test_dataset(args.dataset)
    logger.info(f"Loaded {len(test_dataset)} test cases")

    try:
        if args.ablation:
            # Run ablation study
            logger.info(f"Loading ablation configurations from: {args.ablation}")
            configurations = load_ablation_configs(args.ablation)
            logger.info(f"Loaded {len(configurations)} configurations")

            # Initialize evaluator
            settings = get_settings()
            evaluator = RAGEvaluator(settings=settings)

            # Run comparison
            logger.info("Starting ablation study...")
            results = evaluator.compare_configurations(test_dataset, configurations)

            # Print all results
            for result in results:
                print(result.summary())

            # Save individual results
            output_dir = args.output.parent if args.output else (project_root / "evals" / "reports")
            output_dir.mkdir(parents=True, exist_ok=True)

            for result in results:
                result_path = output_dir / f"{result.config_name}_results.json"
                evaluator.export_to_json(result, result_path)

            # Export CSV comparison if requested
            if args.export_csv:
                evaluator.export_to_csv(results, args.export_csv)
                logger.info(f"Comparison CSV saved to: {args.export_csv}")
            else:
                csv_path = output_dir / "ablation_comparison.csv"
                evaluator.export_to_csv(results, csv_path)
                logger.info(f"Comparison CSV saved to: {csv_path}")

        else:
            # Single configuration evaluation
            logger.info(f"Running evaluation: {args.config_name}")

            # Initialize evaluator
            settings = get_settings()
            evaluator = RAGEvaluator(settings=settings)

            # Run complete evaluation
            result = evaluator.evaluate_complete(
                test_dataset,
                config_name=args.config_name,
                compute_ragas=args.use_ragas
            )

            # Print summary
            print(result.summary())

            # Save results
            if args.output:
                output_path = args.output
            else:
                output_path = project_root / "evals" / "reports" / "e2e_results.json"

            evaluator.export_to_json(result, output_path)

            logger.info(f"\nEvaluation complete!")
            logger.info(f"Results saved to: {output_path}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
