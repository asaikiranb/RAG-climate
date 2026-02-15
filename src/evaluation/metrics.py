"""
Comprehensive metrics for RAG system evaluation.

This module implements:
1. Retrieval Metrics: Precision@K, Recall@K, MRR, MAP, NDCG, Hit Rate
2. Generation Metrics: Faithfulness, Relevance, Citation Accuracy, BLEU, ROUGE
3. RAGAS Metrics: Integration with ragas library (optional)
4. End-to-End Metrics: Overall system performance
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict
import warnings

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Optional imports
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK/ROUGE not available. Install with: pip install nltk rouge-score")

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_relevancy,
        context_recall,
        context_precision,
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("RAGAS not available. Install with: pip install ragas datasets")


@dataclass
class RetrievalMetricsResult:
    """Results from retrieval evaluation."""
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    mrr: float
    map_score: float
    ndcg_at_k: Dict[int, float]
    hit_rate_at_k: Dict[int, float]
    avg_retrieved_docs: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class GenerationMetricsResult:
    """Results from generation evaluation."""
    faithfulness_score: float
    relevance_score: float
    citation_accuracy: float
    bleu_score: Optional[float] = None
    rouge_scores: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class RAGASMetricsResult:
    """Results from RAGAS evaluation."""
    faithfulness: float
    answer_relevancy: float
    context_relevancy: float
    context_recall: float
    context_precision: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class EndToEndMetricsResult:
    """Overall system performance metrics."""
    avg_latency_ms: float
    avg_retrieval_time_ms: float
    avg_generation_time_ms: float
    avg_confidence_score: float
    high_quality_rate: float
    success_rate: float
    avg_tokens_used: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class RetrievalMetrics:
    """
    Compute retrieval metrics for RAG system.

    Metrics:
    - Precision@K: Fraction of retrieved docs that are relevant
    - Recall@K: Fraction of relevant docs that are retrieved
    - MRR (Mean Reciprocal Rank): Average 1/rank of first relevant doc
    - MAP (Mean Average Precision): Mean of precision values at relevant positions
    - NDCG@K: Normalized Discounted Cumulative Gain
    - Hit Rate@K: Fraction of queries with at least one relevant doc in top K
    """

    def __init__(self, k_values: List[int] = None):
        """
        Initialize retrieval metrics.

        Args:
            k_values: List of K values to compute metrics at (e.g., [1, 3, 5, 10])
        """
        self.k_values = k_values or [1, 3, 5, 10]
        logger.info(f"Initialized RetrievalMetrics with K values: {self.k_values}")

    def compute_precision_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int
    ) -> float:
        """
        Compute Precision@K.

        Args:
            retrieved_ids: List of retrieved document IDs (ordered by rank)
            relevant_ids: List of relevant document IDs
            k: Number of top documents to consider

        Returns:
            Precision@K score
        """
        if not retrieved_ids or k == 0:
            return 0.0

        retrieved_at_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)

        relevant_retrieved = retrieved_at_k & relevant_set

        return len(relevant_retrieved) / k

    def compute_recall_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int
    ) -> float:
        """
        Compute Recall@K.

        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs
            k: Number of top documents to consider

        Returns:
            Recall@K score
        """
        if not relevant_ids:
            return 0.0

        retrieved_at_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)

        relevant_retrieved = retrieved_at_k & relevant_set

        return len(relevant_retrieved) / len(relevant_set)

    def compute_mrr(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """
        Compute Mean Reciprocal Rank for a single query.

        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs

        Returns:
            Reciprocal rank (1/rank of first relevant doc)
        """
        relevant_set = set(relevant_ids)

        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_set:
                return 1.0 / rank

        return 0.0

    def compute_average_precision(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """
        Compute Average Precision for a single query.

        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs

        Returns:
            Average Precision score
        """
        if not relevant_ids:
            return 0.0

        relevant_set = set(relevant_ids)
        precision_sum = 0.0
        relevant_count = 0

        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_set:
                relevant_count += 1
                precision_at_rank = relevant_count / rank
                precision_sum += precision_at_rank

        if relevant_count == 0:
            return 0.0

        return precision_sum / len(relevant_set)

    def compute_ndcg_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int,
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain@K.

        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs
            k: Number of top documents to consider
            relevance_scores: Optional dict mapping doc_id to relevance score (0-1)

        Returns:
            NDCG@K score
        """
        if not relevant_ids or k == 0:
            return 0.0

        # Default relevance score is 1.0 for relevant docs
        if relevance_scores is None:
            relevance_scores = {doc_id: 1.0 for doc_id in relevant_ids}

        # Compute DCG
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
            rel = relevance_scores.get(doc_id, 0.0)
            dcg += rel / np.log2(rank + 1)

        # Compute ideal DCG
        ideal_scores = sorted(
            [relevance_scores.get(doc_id, 0.0) for doc_id in relevant_ids],
            reverse=True
        )
        idcg = sum(
            score / np.log2(rank + 1)
            for rank, score in enumerate(ideal_scores[:k], start=1)
        )

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def compute_hit_rate_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int
    ) -> float:
        """
        Compute Hit Rate@K (1 if any relevant doc in top K, 0 otherwise).

        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs
            k: Number of top documents to consider

        Returns:
            1.0 if hit, 0.0 otherwise
        """
        retrieved_at_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)

        return 1.0 if (retrieved_at_k & relevant_set) else 0.0

    def evaluate(
        self,
        queries_results: List[Dict[str, Any]]
    ) -> RetrievalMetricsResult:
        """
        Evaluate retrieval metrics across multiple queries.

        Args:
            queries_results: List of dicts with keys:
                - 'retrieved_ids': List of retrieved doc IDs
                - 'relevant_ids': List of relevant doc IDs
                - 'relevance_scores': Optional dict of doc_id -> score

        Returns:
            RetrievalMetricsResult with all metrics
        """
        logger.info(f"Evaluating retrieval metrics for {len(queries_results)} queries")

        precision_scores = defaultdict(list)
        recall_scores = defaultdict(list)
        mrr_scores = []
        map_scores = []
        ndcg_scores = defaultdict(list)
        hit_rate_scores = defaultdict(list)
        retrieved_counts = []

        for query_result in queries_results:
            retrieved_ids = query_result['retrieved_ids']
            relevant_ids = query_result['relevant_ids']
            relevance_scores = query_result.get('relevance_scores')

            retrieved_counts.append(len(retrieved_ids))

            # Compute metrics at each K
            for k in self.k_values:
                precision_scores[k].append(
                    self.compute_precision_at_k(retrieved_ids, relevant_ids, k)
                )
                recall_scores[k].append(
                    self.compute_recall_at_k(retrieved_ids, relevant_ids, k)
                )
                ndcg_scores[k].append(
                    self.compute_ndcg_at_k(retrieved_ids, relevant_ids, k, relevance_scores)
                )
                hit_rate_scores[k].append(
                    self.compute_hit_rate_at_k(retrieved_ids, relevant_ids, k)
                )

            # Compute MRR and MAP
            mrr_scores.append(self.compute_mrr(retrieved_ids, relevant_ids))
            map_scores.append(self.compute_average_precision(retrieved_ids, relevant_ids))

        # Aggregate results
        result = RetrievalMetricsResult(
            precision_at_k={k: np.mean(scores) for k, scores in precision_scores.items()},
            recall_at_k={k: np.mean(scores) for k, scores in recall_scores.items()},
            mrr=np.mean(mrr_scores),
            map_score=np.mean(map_scores),
            ndcg_at_k={k: np.mean(scores) for k, scores in ndcg_scores.items()},
            hit_rate_at_k={k: np.mean(scores) for k, scores in hit_rate_scores.items()},
            avg_retrieved_docs=np.mean(retrieved_counts)
        )

        logger.info(f"Retrieval metrics: MRR={result.mrr:.3f}, MAP={result.map_score:.3f}")

        return result


class GenerationMetrics:
    """
    Compute generation quality metrics for RAG system.

    Metrics:
    - Faithfulness: Answer is grounded in context
    - Relevance: Answer addresses the question
    - Citation Accuracy: Citations are correct and complete
    - BLEU: N-gram overlap with reference (if available)
    - ROUGE: Recall-oriented overlap with reference (if available)
    """

    def __init__(self, use_llm_judge: bool = True, judge_model: str = "gpt-4"):
        """
        Initialize generation metrics.

        Args:
            use_llm_judge: Whether to use LLM-as-judge for subjective metrics
            judge_model: Model to use for LLM-as-judge
        """
        self.use_llm_judge = use_llm_judge
        self.judge_model = judge_model

        if NLTK_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )

        logger.info(f"Initialized GenerationMetrics (LLM judge: {use_llm_judge})")

    def compute_faithfulness(
        self,
        answer: str,
        context_chunks: List[str],
        use_llm: bool = True
    ) -> float:
        """
        Compute faithfulness score (grounding in context).

        Args:
            answer: Generated answer
            context_chunks: List of context documents
            use_llm: Whether to use LLM for evaluation

        Returns:
            Faithfulness score (0-1)
        """
        if not use_llm or not self.use_llm_judge:
            # Simple heuristic: check if key answer phrases appear in context
            answer_words = set(answer.lower().split())
            context_text = " ".join(context_chunks).lower()
            context_words = set(context_text.split())

            if not answer_words:
                return 0.0

            overlap = len(answer_words & context_words) / len(answer_words)
            return min(overlap * 1.2, 1.0)  # Boost slightly

        # TODO: Implement LLM-as-judge for faithfulness
        # For now, use heuristic
        return self.compute_faithfulness(answer, context_chunks, use_llm=False)

    def compute_relevance(
        self,
        question: str,
        answer: str,
        use_llm: bool = True
    ) -> float:
        """
        Compute answer relevance to question.

        Args:
            question: User question
            answer: Generated answer
            use_llm: Whether to use LLM for evaluation

        Returns:
            Relevance score (0-1)
        """
        if not use_llm or not self.use_llm_judge:
            # Simple heuristic: check if question keywords appear in answer
            question_words = set(question.lower().split())
            answer_words = set(answer.lower().split())

            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
            question_keywords = question_words - stop_words

            if not question_keywords:
                return 0.5

            overlap = len(question_keywords & answer_words) / len(question_keywords)
            return min(overlap * 1.5, 1.0)

        # TODO: Implement LLM-as-judge for relevance
        return self.compute_relevance(question, answer, use_llm=False)

    def compute_citation_accuracy(
        self,
        answer: str,
        context_chunks: List[Dict[str, Any]],
        citations: List[int]
    ) -> float:
        """
        Compute citation accuracy.

        Args:
            answer: Generated answer
            context_chunks: Context documents with IDs
            citations: List of cited chunk indices

        Returns:
            Citation accuracy score (0-1)
        """
        if not citations:
            # No citations - check if answer needs them
            # If answer is short or generic, no citations needed
            if len(answer.split()) < 20:
                return 1.0
            return 0.0

        # Check if cited chunks are actually relevant
        # Simple heuristic: verify cited text appears in answer
        correct_citations = 0
        for citation_idx in citations:
            if citation_idx < len(context_chunks):
                chunk_text = context_chunks[citation_idx].get('document', '')
                # Check if any substantial phrase from chunk appears in answer
                chunk_words = set(chunk_text.lower().split())
                answer_words = set(answer.lower().split())

                # At least 3 words should overlap
                if len(chunk_words & answer_words) >= 3:
                    correct_citations += 1

        return correct_citations / len(citations) if citations else 0.0

    def compute_bleu(
        self,
        answer: str,
        reference: str,
        max_n: int = 4
    ) -> Optional[float]:
        """
        Compute BLEU score against reference answer.

        Args:
            answer: Generated answer
            reference: Reference answer
            max_n: Maximum n-gram size

        Returns:
            BLEU score or None if NLTK not available
        """
        if not NLTK_AVAILABLE:
            return None

        # Tokenize
        reference_tokens = reference.lower().split()
        answer_tokens = answer.lower().split()

        # Compute BLEU with smoothing
        smoothing = SmoothingFunction()
        score = sentence_bleu(
            [reference_tokens],
            answer_tokens,
            smoothing_function=smoothing.method1
        )

        return score

    def compute_rouge(
        self,
        answer: str,
        reference: str
    ) -> Optional[Dict[str, float]]:
        """
        Compute ROUGE scores against reference answer.

        Args:
            answer: Generated answer
            reference: Reference answer

        Returns:
            Dict of ROUGE scores or None if rouge_score not available
        """
        if not NLTK_AVAILABLE:
            return None

        scores = self.rouge_scorer.score(reference, answer)

        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure,
        }

    def evaluate(
        self,
        generation_results: List[Dict[str, Any]]
    ) -> GenerationMetricsResult:
        """
        Evaluate generation metrics across multiple examples.

        Args:
            generation_results: List of dicts with keys:
                - 'question': User question
                - 'answer': Generated answer
                - 'context_chunks': List of context documents
                - 'citations': List of citation indices
                - 'reference': Optional reference answer

        Returns:
            GenerationMetricsResult with all metrics
        """
        logger.info(f"Evaluating generation metrics for {len(generation_results)} examples")

        faithfulness_scores = []
        relevance_scores = []
        citation_scores = []
        bleu_scores = []
        rouge_scores = defaultdict(list)

        for result in generation_results:
            question = result['question']
            answer = result['answer']
            context_chunks = result['context_chunks']
            citations = result.get('citations', [])
            reference = result.get('reference')

            # Compute core metrics
            faithfulness_scores.append(
                self.compute_faithfulness(answer, [c.get('document', '') for c in context_chunks])
            )
            relevance_scores.append(
                self.compute_relevance(question, answer)
            )
            citation_scores.append(
                self.compute_citation_accuracy(answer, context_chunks, citations)
            )

            # Compute reference-based metrics if available
            if reference:
                bleu = self.compute_bleu(answer, reference)
                if bleu is not None:
                    bleu_scores.append(bleu)

                rouge = self.compute_rouge(answer, reference)
                if rouge:
                    for key, value in rouge.items():
                        rouge_scores[key].append(value)

        # Aggregate results
        result = GenerationMetricsResult(
            faithfulness_score=np.mean(faithfulness_scores),
            relevance_score=np.mean(relevance_scores),
            citation_accuracy=np.mean(citation_scores),
            bleu_score=np.mean(bleu_scores) if bleu_scores else None,
            rouge_scores={k: np.mean(v) for k, v in rouge_scores.items()} if rouge_scores else None
        )

        logger.info(
            f"Generation metrics: "
            f"Faithfulness={result.faithfulness_score:.3f}, "
            f"Relevance={result.relevance_score:.3f}"
        )

        return result


class RAGASMetrics:
    """
    Integration with RAGAS framework for comprehensive RAG evaluation.

    RAGAS provides LLM-based evaluation metrics:
    - Faithfulness: Factual consistency with context
    - Answer Relevancy: Relevance to question
    - Context Relevancy: Relevance of retrieved context
    - Context Recall: Coverage of ground truth
    - Context Precision: Precision of retrieved context
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize RAGAS metrics.

        Args:
            openai_api_key: OpenAI API key for RAGAS evaluation
        """
        if not RAGAS_AVAILABLE:
            logger.warning("RAGAS not available. Metrics will return None.")
            self.available = False
        else:
            self.available = True
            logger.info("Initialized RAGASMetrics")

    def evaluate(
        self,
        ragas_data: List[Dict[str, Any]]
    ) -> Optional[RAGASMetricsResult]:
        """
        Evaluate using RAGAS framework.

        Args:
            ragas_data: List of dicts with keys:
                - 'question': User question
                - 'answer': Generated answer
                - 'contexts': List of retrieved context strings
                - 'ground_truth': Optional ground truth answer

        Returns:
            RAGASMetricsResult or None if RAGAS not available
        """
        if not self.available:
            logger.warning("RAGAS not available, skipping evaluation")
            return None

        logger.info(f"Evaluating with RAGAS framework for {len(ragas_data)} examples")

        try:
            # Convert to RAGAS dataset format
            dataset_dict = {
                'question': [d['question'] for d in ragas_data],
                'answer': [d['answer'] for d in ragas_data],
                'contexts': [d['contexts'] for d in ragas_data],
            }

            # Add ground truth if available
            if all('ground_truth' in d for d in ragas_data):
                dataset_dict['ground_truth'] = [d['ground_truth'] for d in ragas_data]

            dataset = Dataset.from_dict(dataset_dict)

            # Run evaluation
            metrics = [
                faithfulness,
                answer_relevancy,
                context_relevancy,
            ]

            # Add recall/precision if ground truth available
            if 'ground_truth' in dataset_dict:
                metrics.extend([context_recall, context_precision])

            result = evaluate(dataset, metrics=metrics)

            # Extract scores
            ragas_result = RAGASMetricsResult(
                faithfulness=result['faithfulness'],
                answer_relevancy=result['answer_relevancy'],
                context_relevancy=result['context_relevancy'],
                context_recall=result.get('context_recall', 0.0),
                context_precision=result.get('context_precision', 0.0),
            )

            logger.info(f"RAGAS evaluation complete: faithfulness={ragas_result.faithfulness:.3f}")

            return ragas_result

        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return None


class EndToEndMetrics:
    """
    Compute end-to-end system performance metrics.

    Metrics:
    - Latency (total, retrieval, generation)
    - Quality (confidence, accuracy)
    - Efficiency (tokens, success rate)
    """

    def __init__(self):
        """Initialize end-to-end metrics."""
        logger.info("Initialized EndToEndMetrics")

    def evaluate(
        self,
        e2e_results: List[Dict[str, Any]]
    ) -> EndToEndMetricsResult:
        """
        Evaluate end-to-end metrics.

        Args:
            e2e_results: List of dicts with keys:
                - 'latency_ms': Total latency
                - 'retrieval_time_ms': Retrieval time
                - 'generation_time_ms': Generation time
                - 'confidence_score': Answer confidence
                - 'is_high_quality': Whether answer is high quality
                - 'success': Whether query succeeded
                - 'tokens_used': Number of tokens used

        Returns:
            EndToEndMetricsResult with all metrics
        """
        logger.info(f"Evaluating end-to-end metrics for {len(e2e_results)} queries")

        latencies = []
        retrieval_times = []
        generation_times = []
        confidence_scores = []
        high_quality_count = 0
        success_count = 0
        tokens_used = []

        for result in e2e_results:
            latencies.append(result.get('latency_ms', 0))
            retrieval_times.append(result.get('retrieval_time_ms', 0))
            generation_times.append(result.get('generation_time_ms', 0))
            confidence_scores.append(result.get('confidence_score', 0))
            tokens_used.append(result.get('tokens_used', 0))

            if result.get('is_high_quality', False):
                high_quality_count += 1

            if result.get('success', False):
                success_count += 1

        total_queries = len(e2e_results)

        return EndToEndMetricsResult(
            avg_latency_ms=np.mean(latencies) if latencies else 0.0,
            avg_retrieval_time_ms=np.mean(retrieval_times) if retrieval_times else 0.0,
            avg_generation_time_ms=np.mean(generation_times) if generation_times else 0.0,
            avg_confidence_score=np.mean(confidence_scores) if confidence_scores else 0.0,
            high_quality_rate=high_quality_count / total_queries if total_queries > 0 else 0.0,
            success_rate=success_count / total_queries if total_queries > 0 else 0.0,
            avg_tokens_used=np.mean(tokens_used) if tokens_used else 0.0,
        )
