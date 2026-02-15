"""
Main answer generation module with advanced features.

Features:
- Context reordering (Lost in the Middle mitigation)
- Citation-based answer generation
- Confidence scoring
- Answer validation
- Multi-attempt generation with fallback
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
import numpy as np

from src.config import Settings
from src.generation.llm_client import LLMClient, LLMResponse
from src.generation.prompts import create_qa_prompt, get_system_prompt
from src.generation.validators import (
    CitationValidator,
    AnswerVerifier,
    CitationValidationResult,
    AnswerVerificationResult
)
from src.utils.logger import get_logger, with_trace_id
from src.utils.exceptions import GenerationError

logger = get_logger(__name__)


@dataclass
class GeneratedAnswer:
    """Complete answer with metadata and quality scores."""
    answer: str
    question: str
    context_chunks: List[Dict[str, Any]]
    confidence_score: float
    citation_validation: Optional[CitationValidationResult]
    answer_verification: Optional[AnswerVerificationResult]
    metadata: Dict[str, Any]

    @property
    def is_high_quality(self) -> bool:
        """Check if answer meets quality thresholds."""
        # High quality requires good confidence and validation
        confidence_ok = self.confidence_score >= 0.7

        citation_ok = True
        if self.citation_validation:
            citation_ok = self.citation_validation.is_valid

        verification_ok = True
        if self.answer_verification:
            verification_ok = self.answer_verification.is_acceptable

        return confidence_ok and citation_ok and verification_ok


class AnswerGenerator:
    """
    Production-ready answer generator with quality assurance.

    Implements:
    - Lost in the Middle mitigation via context reordering
    - Citation-grounded answer generation
    - Multi-pass validation
    - Confidence scoring
    - Fallback strategies
    """

    def __init__(self, settings: Settings):
        """
        Initialize answer generator.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.config = settings.generation

        # Initialize components
        self.llm_client = LLMClient(settings)
        self.citation_validator = CitationValidator(settings, self.llm_client)
        self.answer_verifier = AnswerVerifier(settings, self.llm_client)

        logger.info(
            f"Initialized answer generator with model: {self.config.model_name}"
        )

    def reorder_context_for_attention(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Reorder context chunks to mitigate "Lost in the Middle" problem.

        Strategy: Place most relevant chunks at beginning and end,
        less relevant in the middle. This aligns with LLM attention patterns.

        Research shows LLMs pay more attention to:
        1. Start of context (primacy effect)
        2. End of context (recency effect)
        3. Less attention to middle chunks

        Args:
            chunks: Retrieved chunks with 'score' in metadata

        Returns:
            Reordered chunks for better LLM performance
        """
        if not self.config.use_context_reordering or len(chunks) <= 2:
            return chunks

        logger.debug(f"Reordering {len(chunks)} context chunks")

        # Sort by relevance score (highest first)
        sorted_chunks = sorted(
            chunks,
            key=lambda x: x.get('metadata', {}).get('score', 0.0),
            reverse=True
        )

        # Interleave: best chunks at start/end, worst in middle
        reordered = []
        left_idx = 0
        right_idx = len(sorted_chunks) - 1
        place_at_start = True

        while left_idx <= right_idx:
            if place_at_start:
                # Take from high-relevance end, place at start
                reordered.append(sorted_chunks[left_idx])
                left_idx += 1
            else:
                # Take from low-relevance end, place at end
                reordered.insert(len(reordered) // 2, sorted_chunks[right_idx])
                right_idx -= 1

            place_at_start = not place_at_start

        logger.debug(
            f"Reordered context: highest score={sorted_chunks[0].get('metadata', {}).get('score', 0):.3f}, "
            f"lowest score={sorted_chunks[-1].get('metadata', {}).get('score', 0):.3f}"
        )

        return reordered

    def compute_confidence_score(
        self,
        llm_response: LLMResponse,
        chunks: List[Dict[str, Any]],
        citation_validation: Optional[CitationValidationResult] = None,
        answer_verification: Optional[AnswerVerificationResult] = None
    ) -> float:
        """
        Compute confidence score for the generated answer.

        Factors considered:
        1. Context relevance (average chunk scores)
        2. Citation quality
        3. Answer verification scores
        4. LLM finish reason

        Args:
            llm_response: LLM response
            chunks: Context chunks
            citation_validation: Optional citation validation result
            answer_verification: Optional answer verification result

        Returns:
            Confidence score from 0.0 to 1.0
        """
        scores = []

        # 1. Context quality (avg relevance of chunks)
        if chunks:
            relevance_scores = [
                c.get('metadata', {}).get('score', 0.5)
                for c in chunks
            ]
            context_score = np.mean(relevance_scores)
            scores.append(context_score * 0.3)  # 30% weight

        # 2. Citation quality
        if citation_validation:
            scores.append(citation_validation.citation_score * 0.25)  # 25% weight
        else:
            scores.append(0.5 * 0.25)

        # 3. Answer verification
        if answer_verification:
            scores.append(answer_verification.overall_score * 0.3)  # 30% weight
        else:
            scores.append(0.7 * 0.3)

        # 4. LLM completion quality
        completion_score = 1.0
        if llm_response.finish_reason != "stop":
            completion_score = 0.5  # Incomplete/truncated
        scores.append(completion_score * 0.15)  # 15% weight

        # Compute weighted average
        confidence = sum(scores)

        logger.debug(f"Computed confidence score: {confidence:.3f}")

        return min(max(confidence, 0.0), 1.0)  # Clamp to [0, 1]

    @with_trace_id
    def generate(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        enable_validation: bool = True,
        max_attempts: int = 2
    ) -> GeneratedAnswer:
        """
        Generate answer with context, citations, and validation.

        Args:
            query: User question
            context_chunks: Retrieved context chunks
            enable_validation: Whether to validate citations and answer
            max_attempts: Max generation attempts if validation fails

        Returns:
            GeneratedAnswer with answer, scores, and metadata

        Raises:
            GenerationError: If generation fails
        """
        if not context_chunks:
            raise GenerationError(
                "Cannot generate answer without context chunks",
                details={"query": query}
            )

        logger.info(
            f"Generating answer for query: '{query[:100]}...' "
            f"with {len(context_chunks)} context chunks"
        )

        start_time = time.time()
        attempt = 0
        best_answer = None
        best_score = 0.0

        # Reorder context for better LLM performance
        reordered_chunks = self.reorder_context_for_attention(context_chunks)

        while attempt < max_attempts:
            attempt += 1
            logger.debug(f"Generation attempt {attempt}/{max_attempts}")

            try:
                # Generate answer
                answer_text, llm_response = self._generate_answer(
                    query,
                    reordered_chunks,
                    temperature=self.config.temperature if attempt == 1 else self.config.temperature + 0.1
                )

                # Validate if enabled
                citation_validation = None
                answer_verification = None

                if enable_validation:
                    # Validate citations
                    if self.config.use_citation_validation:
                        citation_validation = self.citation_validator.validate(
                            answer_text,
                            reordered_chunks,
                            use_semantic=True
                        )

                    # Verify answer quality
                    if self.config.use_answer_verification:
                        answer_verification = self.answer_verifier.verify(
                            query,
                            answer_text,
                            reordered_chunks
                        )

                # Compute confidence
                confidence = self.compute_confidence_score(
                    llm_response,
                    reordered_chunks,
                    citation_validation,
                    answer_verification
                )

                # Track best answer
                if confidence > best_score:
                    best_score = confidence
                    best_answer = (
                        answer_text,
                        llm_response,
                        citation_validation,
                        answer_verification,
                        confidence
                    )

                # If answer is high quality, stop early
                if citation_validation and citation_validation.is_valid:
                    if answer_verification and answer_verification.is_acceptable:
                        logger.info(f"High quality answer on attempt {attempt}")
                        break

                logger.debug(
                    f"Attempt {attempt} confidence: {confidence:.3f}, "
                    f"valid: {citation_validation.is_valid if citation_validation else 'N/A'}"
                )

            except Exception as e:
                logger.error(f"Attempt {attempt} failed: {e}")
                if attempt == max_attempts:
                    raise GenerationError(
                        f"All {max_attempts} generation attempts failed",
                        details={"last_error": str(e), "query": query}
                    ) from e

        # Use best answer found
        if not best_answer:
            raise GenerationError("No valid answer generated")

        answer_text, llm_response, citation_val, answer_ver, confidence = best_answer

        # Build metadata
        total_time = (time.time() - start_time) * 1000
        metadata = {
            "generation_time_ms": total_time,
            "num_context_chunks": len(context_chunks),
            "model": llm_response.model,
            "tokens_used": llm_response.tokens_used,
            "attempts": attempt,
            "reordered_context": self.config.use_context_reordering,
        }

        logger.info(
            f"Generated answer: confidence={confidence:.3f}, "
            f"time={total_time:.0f}ms, tokens={llm_response.tokens_used}"
        )

        return GeneratedAnswer(
            answer=answer_text,
            question=query,
            context_chunks=reordered_chunks,
            confidence_score=confidence,
            citation_validation=citation_val,
            answer_verification=answer_ver,
            metadata=metadata
        )

    def _generate_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        temperature: Optional[float] = None
    ) -> Tuple[str, LLMResponse]:
        """
        Internal method to generate answer.

        Args:
            query: User question
            chunks: Context chunks
            temperature: Optional temperature override

        Returns:
            Tuple of (answer_text, llm_response)
        """
        # Create prompt
        prompt = create_qa_prompt(query, chunks)

        # Format user message
        from src.generation.prompts import format_context_documents
        context_text = format_context_documents(chunks)

        user_message = prompt.format_user(
            query=query,
            context=context_text
        )

        # Generate with LLM
        response = self.llm_client.generate_with_system_prompt(
            system_prompt=prompt.system,
            user_message=user_message,
            temperature=temperature
        )

        return response.content, response

    def generate_batch(
        self,
        queries_and_contexts: List[Tuple[str, List[Dict[str, Any]]]],
        enable_validation: bool = True
    ) -> List[GeneratedAnswer]:
        """
        Generate answers for multiple queries (sequential processing).

        Args:
            queries_and_contexts: List of (query, chunks) tuples
            enable_validation: Whether to validate answers

        Returns:
            List of GeneratedAnswer objects
        """
        logger.info(f"Generating answers for {len(queries_and_contexts)} queries")

        results = []
        for i, (query, chunks) in enumerate(queries_and_contexts, 1):
            logger.debug(f"Processing query {i}/{len(queries_and_contexts)}")

            try:
                answer = self.generate(
                    query,
                    chunks,
                    enable_validation=enable_validation
                )
                results.append(answer)
            except Exception as e:
                logger.error(f"Failed to generate answer for query {i}: {e}")
                # Add placeholder with error
                results.append(GeneratedAnswer(
                    answer=f"Error generating answer: {e}",
                    question=query,
                    context_chunks=chunks,
                    confidence_score=0.0,
                    citation_validation=None,
                    answer_verification=None,
                    metadata={"error": str(e)}
                ))

        logger.info(
            f"Batch generation complete: {len(results)} answers, "
            f"avg confidence: {np.mean([r.confidence_score for r in results]):.3f}"
        )

        return results

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics from LLM client.

        Returns:
            Dictionary with usage metrics
        """
        return self.llm_client.get_usage_stats()
