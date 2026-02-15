"""
Citation validation and answer verification for quality assurance.
Ensures generated answers are grounded in context and meet quality standards.
"""

import re
import json
from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass

from src.config import Settings
from src.generation.llm_client import LLMClient
from src.generation.prompts import (
    create_citation_validation_prompt,
    create_answer_verification_prompt
)
from src.utils.logger import get_logger
from src.utils.exceptions import GenerationError

logger = get_logger(__name__)


@dataclass
class CitationValidationResult:
    """Results from citation validation."""
    is_valid: bool
    valid_citations: List[int]
    invalid_citations: List[int]
    unsupported_claims: List[str]
    citation_score: float
    details: Dict[str, Any]


@dataclass
class AnswerVerificationResult:
    """Results from answer verification."""
    completeness_score: float
    accuracy_score: float
    citation_score: float
    clarity_score: float
    overall_score: float
    issues: List[str]
    suggestions: List[str]
    is_acceptable: bool  # True if overall_score >= threshold


class CitationValidator:
    """
    Validates that citations in answers are properly grounded in context.

    Uses multiple validation approaches:
    1. Syntactic: Check citation format and references
    2. Semantic: Verify claims match cited documents (LLM-based)
    3. Coverage: Ensure major claims have citations
    """

    def __init__(self, settings: Settings, llm_client: Optional[LLMClient] = None):
        """
        Initialize citation validator.

        Args:
            settings: Application settings
            llm_client: Optional LLM client for semantic validation
        """
        self.settings = settings
        self.config = settings.generation
        self.llm_client = llm_client

        logger.info("Initialized citation validator")

    def extract_citations(self, text: str) -> Set[int]:
        """
        Extract citation numbers from text.

        Args:
            text: Text containing citations like [1], [2,3], etc.

        Returns:
            Set of cited document numbers
        """
        # Match patterns like [1], [2,3], [1, 2, 3]
        pattern = r'\[(\d+(?:\s*,\s*\d+)*)\]'
        matches = re.findall(pattern, text)

        citations = set()
        for match in matches:
            # Split by comma and extract numbers
            numbers = [int(n.strip()) for n in match.split(',')]
            citations.update(numbers)

        return citations

    def validate_citation_format(
        self,
        answer: str,
        num_documents: int
    ) -> Tuple[bool, List[str]]:
        """
        Validate citation format and references.

        Args:
            answer: Generated answer with citations
            num_documents: Number of context documents available

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        citations = self.extract_citations(answer)

        if not citations:
            issues.append("No citations found in answer")
            return False, issues

        # Check for out-of-range citations
        invalid_refs = [c for c in citations if c < 1 or c > num_documents]
        if invalid_refs:
            issues.append(
                f"Citations reference non-existent documents: {invalid_refs}. "
                f"Only {num_documents} documents available."
            )

        # Check if answer has substantive content without citations
        # (potential hallucination)
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        uncited_sentences = []
        for sentence in sentences:
            if not re.search(r'\[\d+(?:\s*,\s*\d+)*\]', sentence):
                # Skip meta-statements like "Based on the context..."
                if not any(phrase in sentence.lower() for phrase in [
                    "based on", "according to", "the context", "not available",
                    "cannot determine", "not mentioned"
                ]):
                    uncited_sentences.append(sentence[:100])

        if len(uncited_sentences) > len(sentences) * 0.3:  # >30% uncited
            issues.append(
                f"Many substantive claims lack citations. "
                f"{len(uncited_sentences)}/{len(sentences)} sentences uncited."
            )

        is_valid = len(issues) == 0
        return is_valid, issues

    def validate_semantic_grounding(
        self,
        answer: str,
        chunks: List[Dict[str, Any]]
    ) -> CitationValidationResult:
        """
        Use LLM to verify claims are grounded in cited documents.

        Args:
            answer: Generated answer
            chunks: Context documents

        Returns:
            CitationValidationResult
        """
        if not self.llm_client:
            logger.warning("No LLM client provided, skipping semantic validation")
            return CitationValidationResult(
                is_valid=True,
                valid_citations=[],
                invalid_citations=[],
                unsupported_claims=[],
                citation_score=1.0,
                details={"skipped": "No LLM client"}
            )

        try:
            # Create validation prompt
            prompt = create_citation_validation_prompt(answer, chunks)
            user_message = prompt.format_user(
                answer=answer,
                context=answer  # Will be formatted in prompt
            )

            # Generate validation
            response = self.llm_client.generate_with_system_prompt(
                system_prompt=prompt.system,
                user_message=user_message,
                temperature=0.0,  # Deterministic
                max_tokens=500
            )

            # Parse JSON response
            try:
                result_json = json.loads(response.content)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    result_json = json.loads(json_match.group())
                else:
                    raise GenerationError("Failed to parse validation response as JSON")

            valid_cites = result_json.get("valid_citations", [])
            invalid_cites = result_json.get("invalid_citations", [])
            unsupported = result_json.get("unsupported_claims", [])
            score = float(result_json.get("overall_score", 0.5))

            is_valid = score >= 0.7 and len(invalid_cites) == 0

            logger.info(
                f"Semantic validation: score={score:.2f}, "
                f"valid={len(valid_cites)}, invalid={len(invalid_cites)}"
            )

            return CitationValidationResult(
                is_valid=is_valid,
                valid_citations=valid_cites,
                invalid_citations=invalid_cites,
                unsupported_claims=unsupported,
                citation_score=score,
                details=result_json
            )

        except Exception as e:
            logger.error(f"Semantic validation failed: {e}")
            # Return conservative result on error
            return CitationValidationResult(
                is_valid=False,
                valid_citations=[],
                invalid_citations=[],
                unsupported_claims=[],
                citation_score=0.0,
                details={"error": str(e)}
            )

    def validate(
        self,
        answer: str,
        chunks: List[Dict[str, Any]],
        use_semantic: bool = True
    ) -> CitationValidationResult:
        """
        Perform complete citation validation.

        Args:
            answer: Generated answer
            chunks: Context documents
            use_semantic: Whether to use LLM-based semantic validation

        Returns:
            CitationValidationResult
        """
        logger.info("Validating citations in answer")

        # First check format
        format_valid, issues = self.validate_citation_format(answer, len(chunks))

        if not format_valid:
            logger.warning(f"Citation format issues: {issues}")
            return CitationValidationResult(
                is_valid=False,
                valid_citations=[],
                invalid_citations=list(self.extract_citations(answer)),
                unsupported_claims=issues,
                citation_score=0.3,
                details={"format_issues": issues}
            )

        # Then semantic validation if enabled
        if use_semantic and self.config.use_citation_validation:
            return self.validate_semantic_grounding(answer, chunks)
        else:
            # Just return format validation
            citations = list(self.extract_citations(answer))
            return CitationValidationResult(
                is_valid=format_valid,
                valid_citations=citations,
                invalid_citations=[],
                unsupported_claims=[],
                citation_score=1.0 if format_valid else 0.5,
                details={"format_only": True}
            )


class AnswerVerifier:
    """
    Verifies answer quality across multiple dimensions.

    Checks:
    - Completeness: Does it address the question?
    - Accuracy: Are technical details correct?
    - Citations: Are claims properly cited?
    - Clarity: Is it well-organized?
    """

    def __init__(self, settings: Settings, llm_client: Optional[LLMClient] = None):
        """
        Initialize answer verifier.

        Args:
            settings: Application settings
            llm_client: Optional LLM client for verification
        """
        self.settings = settings
        self.config = settings.generation
        self.llm_client = llm_client
        self.acceptance_threshold = 0.7

        logger.info("Initialized answer verifier")

    def verify_with_llm(
        self,
        question: str,
        answer: str,
        chunks: List[Dict[str, Any]]
    ) -> AnswerVerificationResult:
        """
        Use LLM to verify answer quality.

        Args:
            question: Original question
            answer: Generated answer
            chunks: Context documents

        Returns:
            AnswerVerificationResult
        """
        if not self.llm_client:
            logger.warning("No LLM client, skipping verification")
            return AnswerVerificationResult(
                completeness_score=1.0,
                accuracy_score=1.0,
                citation_score=1.0,
                clarity_score=1.0,
                overall_score=1.0,
                issues=[],
                suggestions=[],
                is_acceptable=True
            )

        try:
            # Create verification prompt
            prompt = create_answer_verification_prompt(question, answer, chunks)
            user_message = prompt.format_user(
                question=question,
                answer=answer,
                context=answer  # Will be formatted in prompt
            )

            # Generate verification
            response = self.llm_client.generate_with_system_prompt(
                system_prompt=prompt.system,
                user_message=user_message,
                temperature=0.0,
                max_tokens=600
            )

            # Parse JSON response
            try:
                result_json = json.loads(response.content)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    result_json = json.loads(json_match.group())
                else:
                    raise GenerationError("Failed to parse verification response as JSON")

            # Extract scores
            completeness = float(result_json.get("completeness_score", 0.5))
            accuracy = float(result_json.get("accuracy_score", 0.5))
            citation = float(result_json.get("citation_score", 0.5))
            clarity = float(result_json.get("clarity_score", 0.5))
            overall = float(result_json.get("overall_score", 0.5))
            issues = result_json.get("issues", [])
            suggestions = result_json.get("suggestions", [])

            is_acceptable = overall >= self.acceptance_threshold

            logger.info(
                f"Answer verification: overall={overall:.2f}, "
                f"acceptable={is_acceptable}"
            )

            return AnswerVerificationResult(
                completeness_score=completeness,
                accuracy_score=accuracy,
                citation_score=citation,
                clarity_score=clarity,
                overall_score=overall,
                issues=issues,
                suggestions=suggestions,
                is_acceptable=is_acceptable
            )

        except Exception as e:
            logger.error(f"Answer verification failed: {e}")
            # Conservative default
            return AnswerVerificationResult(
                completeness_score=0.5,
                accuracy_score=0.5,
                citation_score=0.5,
                clarity_score=0.5,
                overall_score=0.5,
                issues=[f"Verification error: {e}"],
                suggestions=[],
                is_acceptable=False
            )

    def verify(
        self,
        question: str,
        answer: str,
        chunks: List[Dict[str, Any]]
    ) -> AnswerVerificationResult:
        """
        Perform complete answer verification.

        Args:
            question: Original question
            answer: Generated answer
            chunks: Context documents

        Returns:
            AnswerVerificationResult
        """
        logger.info("Verifying answer quality")

        if self.config.use_answer_verification:
            return self.verify_with_llm(question, answer, chunks)
        else:
            # Skip verification, return optimistic scores
            return AnswerVerificationResult(
                completeness_score=1.0,
                accuracy_score=1.0,
                citation_score=1.0,
                clarity_score=1.0,
                overall_score=1.0,
                issues=[],
                suggestions=[],
                is_acceptable=True
            )
