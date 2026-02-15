#!/usr/bin/env python3
"""
CLI script to evaluate generation quality.

Usage:
    python run_generation_eval.py --dataset path/to/test_set.json --output results.json
    python run_generation_eval.py --dataset test_set.json --use-ragas
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG system generation quality"
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
        help='Path to save results JSON (default: evals/reports/generation_results.json)'
    )
    parser.add_argument(
        '--use-ragas',
        action='store_true',
        help='Use RAGAS framework for evaluation (requires ragas package)'
    )
    parser.add_argument(
        '--generate-answers',
        action='store_true',
        help='Generate answers on-the-fly instead of using pre-generated answers'
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
    logger.info("RAG Generation Evaluation")
    logger.info("="*60)

    # Load dataset
    logger.info(f"Loading test dataset from: {args.dataset}")
    examples = load_test_dataset(args.dataset)
    logger.info(f"Loaded {len(examples)} examples")

    # Initialize evaluator
    settings = get_settings()
    evaluator = RAGEvaluator(settings=settings)

    try:
        if args.generate_answers:
            # Generate answers on-the-fly
            logger.info("Generating answers for evaluation...")

            test_cases = []
            for i, example in enumerate(examples, 1):
                query = example.get('question', example.get('query'))
                logger.info(f"Generating answer {i}/{len(examples)}: {query[:50]}...")

                # Retrieve context
                retrieved_docs = evaluator.retriever.search(query, top_k=5)

                # Generate answer
                generated_answer = evaluator.generator.generate(
                    query,
                    retrieved_docs,
                    enable_validation=True
                )

                test_cases.append({
                    'question': query,
                    'answer': generated_answer.answer,
                    'context_chunks': retrieved_docs,
                    'citations': [],  # Extract from answer if available
                    'reference': example.get('answer', example.get('reference')),
                })

        else:
            # Use pre-generated answers from dataset
            logger.info("Using pre-generated answers from dataset...")

            test_cases = []
            for example in examples:
                # Check if answer exists in dataset
                if 'answer' not in example:
                    logger.warning(f"No answer found for question: {example.get('question', 'unknown')}")
                    continue

                test_cases.append({
                    'question': example.get('question', example.get('query')),
                    'answer': example['answer'],
                    'context_chunks': example.get('context_chunks', []),
                    'citations': example.get('citations', []),
                    'reference': example.get('reference'),
                })

        logger.info(f"Evaluating {len(test_cases)} test cases")

        # Run generation evaluation
        logger.info("Evaluating generation quality...")
        gen_result = evaluator.evaluate_generation(test_cases)

        # Print results
        print("\n" + "="*60)
        print("GENERATION EVALUATION RESULTS")
        print("="*60)
        print(f"Number of examples: {len(test_cases)}")
        print(f"\nFaithfulness Score:      {gen_result.faithfulness_score:.4f}")
        print(f"Relevance Score:         {gen_result.relevance_score:.4f}")
        print(f"Citation Accuracy:       {gen_result.citation_accuracy:.4f}")

        if gen_result.bleu_score is not None:
            print(f"BLEU Score:              {gen_result.bleu_score:.4f}")

        if gen_result.rouge_scores:
            print("\nROUGE Scores:")
            for metric, score in gen_result.rouge_scores.items():
                print(f"  {metric:8s}: {score:.4f}")

        print("="*60)

        # RAGAS evaluation (optional)
        ragas_result = None
        if args.use_ragas:
            logger.info("Running RAGAS evaluation...")

            ragas_cases = []
            for case in test_cases:
                ragas_cases.append({
                    'question': case['question'],
                    'answer': case['answer'],
                    'contexts': [c.get('document', '') for c in case['context_chunks']],
                    'ground_truth': case.get('reference'),
                })

            ragas_result = evaluator.evaluate_with_ragas(ragas_cases)

            if ragas_result:
                print("\n" + "="*60)
                print("RAGAS EVALUATION RESULTS")
                print("="*60)
                print(f"Faithfulness:            {ragas_result.faithfulness:.4f}")
                print(f"Answer Relevancy:        {ragas_result.answer_relevancy:.4f}")
                print(f"Context Relevancy:       {ragas_result.context_relevancy:.4f}")
                print(f"Context Recall:          {ragas_result.context_recall:.4f}")
                print(f"Context Precision:       {ragas_result.context_precision:.4f}")
                print("="*60)

        # Save results
        if args.output:
            output_path = args.output
        else:
            output_path = project_root / "evals" / "reports" / "generation_results.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_dict = {
            'num_examples': len(test_cases),
            'generation_metrics': gen_result.to_dict(),
        }

        if ragas_result:
            results_dict['ragas_metrics'] = ragas_result.to_dict()

        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"Results saved to: {output_path}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
