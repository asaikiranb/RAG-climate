"""
Main evaluation orchestrator for RAG system.

This module coordinates all evaluation metrics and provides:
- Complete RAG system evaluation
- Ablation studies (compare different configurations)
- Result aggregation and reporting
- Export to JSON/CSV
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import csv
import time
from datetime import datetime

from src.config import Settings, get_settings
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.answer_generator import AnswerGenerator
from src.evaluation.metrics import (
    RetrievalMetrics,
    GenerationMetrics,
    RAGASMetrics,
    EndToEndMetrics,
    RetrievalMetricsResult,
    GenerationMetricsResult,
    RAGASMetricsResult,
    EndToEndMetricsResult,
)
from src.utils.logger import get_logger
from src.utils.exceptions import EvaluationError

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Complete evaluation results."""
    config_name: str
    timestamp: str
    retrieval_metrics: Optional[RetrievalMetricsResult]
    generation_metrics: Optional[GenerationMetricsResult]
    ragas_metrics: Optional[RAGASMetricsResult]
    e2e_metrics: Optional[EndToEndMetricsResult]
    num_queries: int
    total_time_seconds: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert nested dataclass results to dicts
        if self.retrieval_metrics:
            data['retrieval_metrics'] = self.retrieval_metrics.to_dict()
        if self.generation_metrics:
            data['generation_metrics'] = self.generation_metrics.to_dict()
        if self.ragas_metrics:
            data['ragas_metrics'] = self.ragas_metrics.to_dict()
        if self.e2e_metrics:
            data['e2e_metrics'] = self.e2e_metrics.to_dict()
        return data

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"\n{'='*60}",
            f"Evaluation Results: {self.config_name}",
            f"{'='*60}",
            f"Timestamp: {self.timestamp}",
            f"Queries: {self.num_queries}",
            f"Total Time: {self.total_time_seconds:.2f}s",
            ""
        ]

        if self.retrieval_metrics:
            lines.extend([
                "RETRIEVAL METRICS:",
                f"  MRR:        {self.retrieval_metrics.mrr:.4f}",
                f"  MAP:        {self.retrieval_metrics.map_score:.4f}",
                f"  P@5:        {self.retrieval_metrics.precision_at_k.get(5, 0):.4f}",
                f"  R@5:        {self.retrieval_metrics.recall_at_k.get(5, 0):.4f}",
                f"  NDCG@5:     {self.retrieval_metrics.ndcg_at_k.get(5, 0):.4f}",
                f"  Hit Rate@5: {self.retrieval_metrics.hit_rate_at_k.get(5, 0):.4f}",
                ""
            ])

        if self.generation_metrics:
            lines.extend([
                "GENERATION METRICS:",
                f"  Faithfulness:      {self.generation_metrics.faithfulness_score:.4f}",
                f"  Relevance:         {self.generation_metrics.relevance_score:.4f}",
                f"  Citation Accuracy: {self.generation_metrics.citation_accuracy:.4f}",
            ])
            if self.generation_metrics.bleu_score:
                lines.append(f"  BLEU:             {self.generation_metrics.bleu_score:.4f}")
            if self.generation_metrics.rouge_scores:
                lines.append(
                    f"  ROUGE-L:          {self.generation_metrics.rouge_scores.get('rougeL', 0):.4f}"
                )
            lines.append("")

        if self.ragas_metrics:
            lines.extend([
                "RAGAS METRICS:",
                f"  Faithfulness:       {self.ragas_metrics.faithfulness:.4f}",
                f"  Answer Relevancy:   {self.ragas_metrics.answer_relevancy:.4f}",
                f"  Context Relevancy:  {self.ragas_metrics.context_relevancy:.4f}",
                f"  Context Recall:     {self.ragas_metrics.context_recall:.4f}",
                f"  Context Precision:  {self.ragas_metrics.context_precision:.4f}",
                ""
            ])

        if self.e2e_metrics:
            lines.extend([
                "END-TO-END METRICS:",
                f"  Avg Latency:     {self.e2e_metrics.avg_latency_ms:.1f}ms",
                f"  Avg Retrieval:   {self.e2e_metrics.avg_retrieval_time_ms:.1f}ms",
                f"  Avg Generation:  {self.e2e_metrics.avg_generation_time_ms:.1f}ms",
                f"  Avg Confidence:  {self.e2e_metrics.avg_confidence_score:.4f}",
                f"  High Quality %:  {self.e2e_metrics.high_quality_rate*100:.1f}%",
                f"  Success Rate:    {self.e2e_metrics.success_rate*100:.1f}%",
                f"  Avg Tokens:      {self.e2e_metrics.avg_tokens_used:.0f}",
                ""
            ])

        lines.append("="*60)

        return "\n".join(lines)


class RAGEvaluator:
    """
    Main evaluator for RAG system.

    Supports:
    - Complete system evaluation
    - Ablation studies
    - Custom metrics
    - Result export
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        k_values: List[int] = None
    ):
        """
        Initialize RAG evaluator.

        Args:
            settings: Optional custom settings (uses default if None)
            k_values: List of K values for retrieval metrics
        """
        self.settings = settings or get_settings()
        self.k_values = k_values or [1, 3, 5, 10]

        # Initialize metrics
        self.retrieval_metrics = RetrievalMetrics(k_values=self.k_values)
        self.generation_metrics = GenerationMetrics(
            use_llm_judge=False  # Can be enabled if needed
        )
        self.ragas_metrics = RAGASMetrics()
        self.e2e_metrics = EndToEndMetrics()

        # Initialize RAG components (lazy loaded)
        self._retriever = None
        self._generator = None

        logger.info("Initialized RAGEvaluator")

    @property
    def retriever(self) -> HybridRetriever:
        """Lazy-load retriever."""
        if self._retriever is None:
            self._retriever = HybridRetriever(self.settings)
        return self._retriever

    @property
    def generator(self) -> AnswerGenerator:
        """Lazy-load generator."""
        if self._generator is None:
            self._generator = AnswerGenerator(self.settings)
        return self._generator

    def evaluate_retrieval(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> RetrievalMetricsResult:
        """
        Evaluate retrieval performance.

        Args:
            test_cases: List of dicts with keys:
                - 'query': User query
                - 'relevant_chunk_ids': List of relevant chunk IDs

        Returns:
            RetrievalMetricsResult
        """
        logger.info(f"Evaluating retrieval for {len(test_cases)} queries")

        queries_results = []

        for i, test_case in enumerate(test_cases, 1):
            query = test_case['query']
            relevant_ids = test_case['relevant_chunk_ids']

            # Retrieve
            try:
                retrieved_docs = self.retriever.search(query, top_k=max(self.k_values))
                retrieved_ids = [doc.get('id', f'doc_{i}') for i, doc in enumerate(retrieved_docs)]

                queries_results.append({
                    'retrieved_ids': retrieved_ids,
                    'relevant_ids': relevant_ids,
                    'relevance_scores': None  # Can add if available
                })

                if i % 10 == 0:
                    logger.info(f"Processed {i}/{len(test_cases)} retrieval queries")

            except Exception as e:
                logger.error(f"Retrieval failed for query {i}: {e}")
                # Add empty result
                queries_results.append({
                    'retrieved_ids': [],
                    'relevant_ids': relevant_ids,
                })

        # Compute metrics
        result = self.retrieval_metrics.evaluate(queries_results)

        logger.info(
            f"Retrieval evaluation complete: MRR={result.mrr:.3f}, MAP={result.map_score:.3f}"
        )

        return result

    def evaluate_generation(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> GenerationMetricsResult:
        """
        Evaluate generation quality.

        Args:
            test_cases: List of dicts with keys:
                - 'question': User question
                - 'answer': Generated answer
                - 'context_chunks': List of context chunks
                - 'citations': Optional list of citation indices
                - 'reference': Optional reference answer

        Returns:
            GenerationMetricsResult
        """
        logger.info(f"Evaluating generation for {len(test_cases)} examples")

        result = self.generation_metrics.evaluate(test_cases)

        logger.info(
            f"Generation evaluation complete: "
            f"Faithfulness={result.faithfulness_score:.3f}, "
            f"Relevance={result.relevance_score:.3f}"
        )

        return result

    def evaluate_with_ragas(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> Optional[RAGASMetricsResult]:
        """
        Evaluate using RAGAS framework.

        Args:
            test_cases: List of dicts with keys:
                - 'question': User question
                - 'answer': Generated answer
                - 'contexts': List of context strings
                - 'ground_truth': Optional reference answer

        Returns:
            RAGASMetricsResult or None if RAGAS not available
        """
        logger.info(f"Evaluating with RAGAS for {len(test_cases)} examples")

        # Convert test cases to RAGAS format
        ragas_data = []
        for test_case in test_cases:
            ragas_data.append({
                'question': test_case['question'],
                'answer': test_case['answer'],
                'contexts': test_case['contexts'],
                'ground_truth': test_case.get('ground_truth', test_case.get('reference')),
            })

        result = self.ragas_metrics.evaluate(ragas_data)

        if result:
            logger.info(
                f"RAGAS evaluation complete: Faithfulness={result.faithfulness:.3f}"
            )

        return result

    def evaluate_end_to_end(
        self,
        test_cases: List[Dict[str, Any]],
        retrieve_and_generate: bool = True
    ) -> Tuple[EndToEndMetricsResult, List[Dict[str, Any]]]:
        """
        Evaluate end-to-end RAG system.

        Args:
            test_cases: List of dicts with keys:
                - 'query' or 'question': User query
                - 'relevant_chunk_ids': List of relevant chunk IDs (for retrieval eval)
                - 'reference': Optional reference answer
            retrieve_and_generate: Whether to run retrieval+generation (vs use provided results)

        Returns:
            Tuple of (EndToEndMetricsResult, list of detailed results per query)
        """
        logger.info(f"End-to-end evaluation for {len(test_cases)} queries")

        e2e_results = []
        detailed_results = []

        for i, test_case in enumerate(test_cases, 1):
            query = test_case.get('query') or test_case.get('question')

            try:
                total_start = time.time()

                # Retrieval
                retrieval_start = time.time()
                retrieved_docs = self.retriever.search(query, top_k=5)
                retrieval_time = (time.time() - retrieval_start) * 1000

                # Generation
                generation_start = time.time()
                generated_answer = self.generator.generate(
                    query,
                    retrieved_docs,
                    enable_validation=True
                )
                generation_time = (time.time() - generation_start) * 1000

                total_time = (time.time() - total_start) * 1000

                # Collect metrics
                e2e_results.append({
                    'latency_ms': total_time,
                    'retrieval_time_ms': retrieval_time,
                    'generation_time_ms': generation_time,
                    'confidence_score': generated_answer.confidence_score,
                    'is_high_quality': generated_answer.is_high_quality,
                    'success': True,
                    'tokens_used': generated_answer.metadata.get('tokens_used', 0),
                })

                # Store detailed result
                detailed_results.append({
                    'query': query,
                    'answer': generated_answer.answer,
                    'retrieved_docs': retrieved_docs,
                    'confidence': generated_answer.confidence_score,
                    'latency_ms': total_time,
                    'success': True,
                })

                if i % 10 == 0:
                    logger.info(f"Processed {i}/{len(test_cases)} end-to-end queries")

            except Exception as e:
                logger.error(f"End-to-end evaluation failed for query {i}: {e}")

                e2e_results.append({
                    'latency_ms': 0,
                    'retrieval_time_ms': 0,
                    'generation_time_ms': 0,
                    'confidence_score': 0,
                    'is_high_quality': False,
                    'success': False,
                    'tokens_used': 0,
                })

                detailed_results.append({
                    'query': query,
                    'answer': '',
                    'error': str(e),
                    'success': False,
                })

        # Compute metrics
        result = self.e2e_metrics.evaluate(e2e_results)

        logger.info(
            f"End-to-end evaluation complete: "
            f"Avg latency={result.avg_latency_ms:.1f}ms, "
            f"Success rate={result.success_rate*100:.1f}%"
        )

        return result, detailed_results

    def evaluate_complete(
        self,
        test_dataset: List[Dict[str, Any]],
        config_name: str = "default",
        compute_ragas: bool = True
    ) -> EvaluationResult:
        """
        Run complete evaluation with all metrics.

        Args:
            test_dataset: List of test cases with all required fields
            config_name: Name for this configuration
            compute_ragas: Whether to compute RAGAS metrics

        Returns:
            EvaluationResult with all metrics
        """
        logger.info(f"Starting complete evaluation: {config_name}")
        start_time = time.time()

        # Run end-to-end evaluation first (gets retrieval + generation)
        e2e_result, detailed_results = self.evaluate_end_to_end(test_dataset)

        # Prepare data for retrieval metrics
        retrieval_cases = []
        for test_case, detail in zip(test_dataset, detailed_results):
            if detail.get('success'):
                retrieval_cases.append({
                    'query': test_case.get('query') or test_case.get('question'),
                    'relevant_chunk_ids': test_case.get('relevant_chunk_ids', []),
                })

        # Evaluate retrieval
        retrieval_result = None
        if retrieval_cases:
            # Re-run retrieval to get IDs
            retrieval_data = []
            for case in retrieval_cases:
                retrieved_docs = self.retriever.search(case['query'], top_k=max(self.k_values))
                retrieved_ids = [doc.get('id', f'doc_{i}') for i, doc in enumerate(retrieved_docs)]
                retrieval_data.append({
                    'retrieved_ids': retrieved_ids,
                    'relevant_ids': case['relevant_chunk_ids'],
                })
            retrieval_result = self.retrieval_metrics.evaluate(retrieval_data)

        # Prepare data for generation metrics
        generation_cases = []
        for test_case, detail in zip(test_dataset, detailed_results):
            if detail.get('success') and detail.get('answer'):
                generation_cases.append({
                    'question': test_case.get('query') or test_case.get('question'),
                    'answer': detail['answer'],
                    'context_chunks': detail.get('retrieved_docs', []),
                    'citations': [],  # Extract if available
                    'reference': test_case.get('reference'),
                })

        # Evaluate generation
        generation_result = None
        if generation_cases:
            generation_result = self.generation_metrics.evaluate(generation_cases)

        # RAGAS evaluation
        ragas_result = None
        if compute_ragas and generation_cases:
            ragas_cases = []
            for case in generation_cases:
                ragas_cases.append({
                    'question': case['question'],
                    'answer': case['answer'],
                    'contexts': [c.get('document', '') for c in case['context_chunks']],
                    'ground_truth': case.get('reference'),
                })
            ragas_result = self.evaluate_with_ragas(ragas_cases)

        total_time = time.time() - start_time

        # Create result object
        result = EvaluationResult(
            config_name=config_name,
            timestamp=datetime.now().isoformat(),
            retrieval_metrics=retrieval_result,
            generation_metrics=generation_result,
            ragas_metrics=ragas_result,
            e2e_metrics=e2e_result,
            num_queries=len(test_dataset),
            total_time_seconds=total_time,
            metadata={
                'settings': {
                    'retrieval': {
                        'use_hyde': self.settings.retrieval.use_hyde,
                        'use_reranking': self.settings.retrieval.use_reranking,
                        'use_mmr': self.settings.retrieval.use_mmr,
                    },
                    'generation': {
                        'model': self.settings.generation.model_name,
                        'temperature': self.settings.generation.temperature,
                    }
                }
            }
        )

        logger.info(f"Complete evaluation finished in {total_time:.1f}s")

        return result

    def compare_configurations(
        self,
        test_dataset: List[Dict[str, Any]],
        configurations: List[Dict[str, Any]]
    ) -> List[EvaluationResult]:
        """
        Run ablation study comparing different configurations.

        Args:
            test_dataset: Test dataset to evaluate on
            configurations: List of dicts with 'name' and 'settings' keys

        Returns:
            List of EvaluationResult objects
        """
        logger.info(f"Running ablation study with {len(configurations)} configurations")

        results = []

        for config in configurations:
            config_name = config['name']
            config_settings = config['settings']

            logger.info(f"\nEvaluating configuration: {config_name}")

            # Update settings
            # Create new settings object with overrides
            from copy import deepcopy
            temp_settings = deepcopy(self.settings)

            # Apply overrides
            for key, value in config_settings.items():
                if hasattr(temp_settings.retrieval, key):
                    setattr(temp_settings.retrieval, key, value)
                elif hasattr(temp_settings.generation, key):
                    setattr(temp_settings.generation, key, value)

            # Create new evaluator with updated settings
            evaluator = RAGEvaluator(settings=temp_settings)

            # Run evaluation
            result = evaluator.evaluate_complete(test_dataset, config_name)
            results.append(result)

            # Print summary
            print(result.summary())

        logger.info(f"Ablation study complete: {len(results)} configurations evaluated")

        return results

    def export_to_json(
        self,
        result: EvaluationResult,
        output_path: Path
    ) -> None:
        """
        Export evaluation results to JSON.

        Args:
            result: EvaluationResult to export
            output_path: Output file path
        """
        logger.info(f"Exporting results to JSON: {output_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Results exported to {output_path}")

    def export_to_csv(
        self,
        results: List[EvaluationResult],
        output_path: Path
    ) -> None:
        """
        Export multiple evaluation results to CSV (for comparison).

        Args:
            results: List of EvaluationResult objects
            output_path: Output file path
        """
        logger.info(f"Exporting {len(results)} results to CSV: {output_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Flatten results for CSV
        rows = []
        for result in results:
            row = {
                'config_name': result.config_name,
                'timestamp': result.timestamp,
                'num_queries': result.num_queries,
                'total_time_seconds': result.total_time_seconds,
            }

            # Add retrieval metrics
            if result.retrieval_metrics:
                row['mrr'] = result.retrieval_metrics.mrr
                row['map'] = result.retrieval_metrics.map_score
                for k in [1, 3, 5, 10]:
                    row[f'precision_at_{k}'] = result.retrieval_metrics.precision_at_k.get(k, 0)
                    row[f'recall_at_{k}'] = result.retrieval_metrics.recall_at_k.get(k, 0)
                    row[f'ndcg_at_{k}'] = result.retrieval_metrics.ndcg_at_k.get(k, 0)

            # Add generation metrics
            if result.generation_metrics:
                row['faithfulness'] = result.generation_metrics.faithfulness_score
                row['relevance'] = result.generation_metrics.relevance_score
                row['citation_accuracy'] = result.generation_metrics.citation_accuracy

            # Add e2e metrics
            if result.e2e_metrics:
                row['avg_latency_ms'] = result.e2e_metrics.avg_latency_ms
                row['success_rate'] = result.e2e_metrics.success_rate
                row['high_quality_rate'] = result.e2e_metrics.high_quality_rate

            rows.append(row)

        # Write CSV
        if rows:
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        logger.info(f"Results exported to {output_path}")
