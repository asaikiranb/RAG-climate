"""
Synthetic dataset generation for RAG evaluation.

This module generates high-quality Q&A pairs from PDF documents:
- Uses LLM to generate diverse questions
- Automatically labels relevant chunks
- Creates difficulty levels (easy, medium, hard)
- Supports multiple question types
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import random
from enum import Enum

from src.config import Settings
from src.generation.llm_client import LLMClient
from src.core.embeddings import EmbeddingModel
from src.ingestion.text_chunker import TextChunker
from src.ingestion.pdf_extractor import PDFExtractor
from src.utils.logger import get_logger
from src.utils.exceptions import EvaluationError

logger = get_logger(__name__)


class QuestionType(str, Enum):
    """Types of questions to generate."""
    FACTUAL = "factual"  # What/When/Where questions
    HOWTO = "how-to"  # How to do something
    WHY = "why"  # Why/Explain questions
    COMPARISON = "comparison"  # Compare/contrast
    CALCULATION = "calculation"  # Numerical/calculation questions


class DifficultyLevel(str, Enum):
    """Difficulty levels for questions."""
    EASY = "easy"  # Single chunk, direct answer
    MEDIUM = "medium"  # 2-3 chunks, some reasoning
    HARD = "hard"  # Multiple chunks, complex reasoning


@dataclass
class SyntheticExample:
    """A single synthetic Q&A example."""
    question: str
    answer: str
    relevant_chunk_ids: List[str]
    question_type: QuestionType
    difficulty: DifficultyLevel
    source_document: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['question_type'] = self.question_type.value
        data['difficulty'] = self.difficulty.value
        return data


class DatasetGenerator:
    """
    Generate synthetic evaluation datasets from documents.

    Uses LLM to create diverse, high-quality Q&A pairs with:
    - Multiple question types
    - Varying difficulty levels
    - Automatic relevance labeling
    - Rich metadata for evaluation
    """

    def __init__(self, settings: Settings):
        """
        Initialize dataset generator.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.llm_client = LLMClient(settings)
        self.embedding_model = EmbeddingModel(settings)
        self.chunker = TextChunker(settings)
        self.pdf_extractor = PDFExtractor(settings)

        logger.info("Initialized DatasetGenerator")

    def _generate_question_prompt(
        self,
        chunk_text: str,
        question_type: QuestionType,
        difficulty: DifficultyLevel,
        existing_questions: List[str] = None
    ) -> str:
        """
        Create prompt for question generation.

        Args:
            chunk_text: Text chunk to generate question from
            question_type: Type of question to generate
            difficulty: Difficulty level
            existing_questions: Existing questions to avoid duplication

        Returns:
            Prompt string
        """
        existing_qs = "\n".join(f"- {q}" for q in (existing_questions or []))

        type_instructions = {
            QuestionType.FACTUAL: "Ask a factual question (What/When/Where/Who) that can be answered directly from the text.",
            QuestionType.HOWTO: "Ask a 'how-to' question about a process or procedure described in the text.",
            QuestionType.WHY: "Ask a 'why' or explanation question that requires understanding the reasoning in the text.",
            QuestionType.COMPARISON: "Ask a comparison question about different concepts, methods, or components mentioned in the text.",
            QuestionType.CALCULATION: "Ask a numerical or calculation question based on specifications, values, or measurements in the text.",
        }

        difficulty_instructions = {
            DifficultyLevel.EASY: "The answer should be directly stated in the text with minimal inference needed.",
            DifficultyLevel.MEDIUM: "The answer should require connecting 2-3 pieces of information or some reasoning.",
            DifficultyLevel.HARD: "The answer should require deep understanding and synthesis of multiple concepts.",
        }

        prompt = f"""You are an expert at creating evaluation questions for a RAG system.

Generate ONE high-quality question based on the following text chunk.

TEXT CHUNK:
{chunk_text}

REQUIREMENTS:
- Question Type: {question_type.value}
  {type_instructions[question_type]}
- Difficulty: {difficulty.value}
  {difficulty_instructions[difficulty]}
- The question must be answerable using the provided text
- Make the question specific and clear
- Avoid yes/no questions
- Use technical terminology appropriately"""

        if existing_questions:
            prompt += f"""

AVOID DUPLICATING THESE EXISTING QUESTIONS:
{existing_qs}"""

        prompt += """

Output ONLY the question text, nothing else."""

        return prompt

    def _generate_answer_prompt(
        self,
        question: str,
        context_chunks: List[str]
    ) -> str:
        """
        Create prompt for answer generation.

        Args:
            question: Question to answer
            context_chunks: Relevant context chunks

        Returns:
            Prompt string
        """
        context_text = "\n\n".join(
            f"[CHUNK {i+1}]\n{chunk}"
            for i, chunk in enumerate(context_chunks)
        )

        prompt = f"""You are an expert at answering technical questions based on provided documentation.

Answer the following question using ONLY the information in the provided context chunks.

QUESTION:
{question}

CONTEXT:
{context_text}

REQUIREMENTS:
- Answer must be factual and grounded in the context
- Include specific details from the context
- If the question asks for a list, provide a complete list
- Be concise but complete
- Use professional technical language

Output ONLY the answer, nothing else."""

        return prompt

    def generate_from_chunk(
        self,
        chunk: Dict[str, Any],
        question_type: Optional[QuestionType] = None,
        difficulty: Optional[DifficultyLevel] = None
    ) -> Optional[SyntheticExample]:
        """
        Generate a Q&A pair from a single chunk.

        Args:
            chunk: Document chunk with 'text' or 'document' and 'metadata' keys
            question_type: Optional question type (random if None)
            difficulty: Optional difficulty level (random if None)

        Returns:
            SyntheticExample or None if generation fails
        """
        chunk_text = chunk.get('text', chunk.get('document', ''))
        chunk_id = chunk.get('id', chunk.get('metadata', {}).get('chunk_id', 'unknown'))

        # Skip very short chunks
        if len(chunk_text.split()) < 50:
            logger.debug(f"Skipping short chunk {chunk_id}")
            return None

        # Random selection if not specified
        if question_type is None:
            question_type = random.choice(list(QuestionType))

        if difficulty is None:
            difficulty = random.choice(list(DifficultyLevel))

        try:
            # Generate question
            question_prompt = self._generate_question_prompt(
                chunk_text,
                question_type,
                difficulty
            )

            question_response = self.llm_client.generate(
                question_prompt,
                temperature=0.7,
                max_tokens=200
            )

            question = question_response.content.strip()

            # Generate answer
            answer_prompt = self._generate_answer_prompt(question, [chunk_text])

            answer_response = self.llm_client.generate(
                answer_prompt,
                temperature=0.3,
                max_tokens=400
            )

            answer = answer_response.content.strip()

            # Create example
            example = SyntheticExample(
                question=question,
                answer=answer,
                relevant_chunk_ids=[chunk_id],
                question_type=question_type,
                difficulty=difficulty,
                source_document=chunk.get('metadata', {}).get('source', 'unknown'),
                metadata={
                    'chunk_id': chunk_id,
                    'generation_model': self.settings.generation.model_name,
                    'question_tokens': question_response.tokens_used,
                    'answer_tokens': answer_response.tokens_used,
                }
            )

            logger.debug(
                f"Generated {difficulty.value} {question_type.value} question from chunk {chunk_id}"
            )

            return example

        except Exception as e:
            logger.error(f"Failed to generate from chunk {chunk_id}: {e}")
            return None

    def generate_from_multiple_chunks(
        self,
        chunks: List[Dict[str, Any]],
        num_chunks: int = 2,
        question_type: Optional[QuestionType] = None
    ) -> Optional[SyntheticExample]:
        """
        Generate a Q&A pair requiring multiple chunks.

        Args:
            chunks: List of document chunks
            num_chunks: Number of chunks to use (2-4)
            question_type: Optional question type

        Returns:
            SyntheticExample or None if generation fails
        """
        if len(chunks) < num_chunks:
            return None

        # Select random chunks
        selected_chunks = random.sample(chunks, min(num_chunks, len(chunks)))
        chunk_ids = [c.get('id', c.get('metadata', {}).get('chunk_id', f'chunk_{i}')) for i, c in enumerate(selected_chunks)]
        chunk_texts = [c.get('text', c.get('document', '')) for c in selected_chunks]

        # Difficulty based on number of chunks
        if num_chunks == 2:
            difficulty = DifficultyLevel.MEDIUM
        else:
            difficulty = DifficultyLevel.HARD

        if question_type is None:
            # Prefer comparison/why for multi-chunk questions
            question_type = random.choice([
                QuestionType.COMPARISON,
                QuestionType.WHY,
                QuestionType.HOWTO
            ])

        try:
            # Combine chunks for question generation
            combined_text = "\n\n".join(chunk_texts)

            # Generate question
            question_prompt = self._generate_question_prompt(
                combined_text,
                question_type,
                difficulty
            )

            question_response = self.llm_client.generate(
                question_prompt,
                temperature=0.7,
                max_tokens=200
            )

            question = question_response.content.strip()

            # Generate answer
            answer_prompt = self._generate_answer_prompt(question, chunk_texts)

            answer_response = self.llm_client.generate(
                answer_prompt,
                temperature=0.3,
                max_tokens=500
            )

            answer = answer_response.content.strip()

            # Create example
            example = SyntheticExample(
                question=question,
                answer=answer,
                relevant_chunk_ids=chunk_ids,
                question_type=question_type,
                difficulty=difficulty,
                source_document=selected_chunks[0].get('metadata', {}).get('source', 'unknown'),
                metadata={
                    'num_chunks': num_chunks,
                    'chunk_ids': chunk_ids,
                    'generation_model': self.settings.generation.model_name,
                }
            )

            logger.debug(
                f"Generated {difficulty.value} multi-chunk question using {num_chunks} chunks"
            )

            return example

        except Exception as e:
            logger.error(f"Failed to generate from multiple chunks: {e}")
            return None

    def generate_dataset(
        self,
        chunks: List[Dict[str, Any]],
        num_examples: int = 100,
        difficulty_distribution: Dict[DifficultyLevel, float] = None,
        question_type_distribution: Dict[QuestionType, float] = None
    ) -> List[SyntheticExample]:
        """
        Generate a complete synthetic dataset.

        Args:
            chunks: List of document chunks
            num_examples: Number of examples to generate
            difficulty_distribution: Distribution of difficulty levels
            question_type_distribution: Distribution of question types

        Returns:
            List of SyntheticExample objects
        """
        logger.info(f"Generating {num_examples} synthetic examples from {len(chunks)} chunks")

        # Default distributions
        if difficulty_distribution is None:
            difficulty_distribution = {
                DifficultyLevel.EASY: 0.4,
                DifficultyLevel.MEDIUM: 0.4,
                DifficultyLevel.HARD: 0.2,
            }

        if question_type_distribution is None:
            question_type_distribution = {
                QuestionType.FACTUAL: 0.3,
                QuestionType.HOWTO: 0.25,
                QuestionType.WHY: 0.2,
                QuestionType.COMPARISON: 0.15,
                QuestionType.CALCULATION: 0.1,
            }

        # Calculate target counts
        difficulty_targets = {
            level: int(num_examples * ratio)
            for level, ratio in difficulty_distribution.items()
        }

        question_type_targets = {
            qtype: int(num_examples * ratio)
            for qtype, ratio in question_type_distribution.items()
        }

        examples = []
        attempts = 0
        max_attempts = num_examples * 3

        while len(examples) < num_examples and attempts < max_attempts:
            attempts += 1

            # Select difficulty and question type based on current needs
            current_difficulty_counts = {level: 0 for level in DifficultyLevel}
            current_qtype_counts = {qtype: 0 for qtype in QuestionType}

            for ex in examples:
                current_difficulty_counts[ex.difficulty] += 1
                current_qtype_counts[ex.question_type] += 1

            # Find most needed difficulty
            difficulty = max(
                DifficultyLevel,
                key=lambda d: difficulty_targets[d] - current_difficulty_counts[d]
            )

            # Find most needed question type
            question_type = max(
                QuestionType,
                key=lambda q: question_type_targets[q] - current_qtype_counts[q]
            )

            # Generate example
            if difficulty == DifficultyLevel.EASY:
                # Single chunk
                chunk = random.choice(chunks)
                example = self.generate_from_chunk(chunk, question_type, difficulty)
            else:
                # Multiple chunks for medium/hard
                num_chunks = 2 if difficulty == DifficultyLevel.MEDIUM else 3
                example = self.generate_from_multiple_chunks(
                    chunks,
                    num_chunks,
                    question_type
                )

            if example:
                examples.append(example)

                if len(examples) % 10 == 0:
                    logger.info(f"Generated {len(examples)}/{num_examples} examples")

        logger.info(
            f"Dataset generation complete: {len(examples)} examples "
            f"({attempts} attempts)"
        )

        # Log distribution
        difficulty_counts = {level: 0 for level in DifficultyLevel}
        qtype_counts = {qtype: 0 for qtype in QuestionType}

        for ex in examples:
            difficulty_counts[ex.difficulty] += 1
            qtype_counts[ex.question_type] += 1

        logger.info(f"Difficulty distribution: {difficulty_counts}")
        logger.info(f"Question type distribution: {qtype_counts}")

        return examples

    def generate_from_pdf(
        self,
        pdf_path: Path,
        num_examples: int = 50
    ) -> List[SyntheticExample]:
        """
        Generate synthetic dataset from a PDF file.

        Args:
            pdf_path: Path to PDF file
            num_examples: Number of examples to generate

        Returns:
            List of SyntheticExample objects
        """
        logger.info(f"Generating dataset from PDF: {pdf_path}")

        # Extract PDF
        try:
            pages = self.pdf_extractor.extract(pdf_path)
            logger.info(f"Extracted {len(pages)} pages from PDF")

            # Chunk documents
            chunks = []
            for page in pages:
                page_chunks = self.chunker.chunk(
                    page['text'],
                    metadata=page['metadata']
                )
                chunks.extend(page_chunks)

            logger.info(f"Created {len(chunks)} chunks from PDF")

            # Generate dataset
            examples = self.generate_dataset(chunks, num_examples)

            return examples

        except Exception as e:
            raise EvaluationError(
                f"Failed to generate dataset from PDF: {e}",
                details={'pdf_path': str(pdf_path)}
            ) from e

    def save_dataset(
        self,
        examples: List[SyntheticExample],
        output_path: Path
    ) -> None:
        """
        Save dataset to JSON file.

        Args:
            examples: List of SyntheticExample objects
            output_path: Output file path
        """
        logger.info(f"Saving {len(examples)} examples to {output_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        dataset = {
            'examples': [ex.to_dict() for ex in examples],
            'metadata': {
                'num_examples': len(examples),
                'generation_model': self.settings.generation.model_name,
                'difficulty_distribution': {
                    level.value: sum(1 for ex in examples if ex.difficulty == level)
                    for level in DifficultyLevel
                },
                'question_type_distribution': {
                    qtype.value: sum(1 for ex in examples if ex.question_type == qtype)
                    for qtype in QuestionType
                },
            }
        }

        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)

        logger.info(f"Dataset saved to {output_path}")

    def load_dataset(self, dataset_path: Path) -> List[SyntheticExample]:
        """
        Load dataset from JSON file.

        Args:
            dataset_path: Path to dataset file

        Returns:
            List of SyntheticExample objects
        """
        logger.info(f"Loading dataset from {dataset_path}")

        with open(dataset_path, 'r') as f:
            data = json.load(f)

        examples = []
        for ex_data in data['examples']:
            # Convert string enums back to enum types
            ex_data['question_type'] = QuestionType(ex_data['question_type'])
            ex_data['difficulty'] = DifficultyLevel(ex_data['difficulty'])

            examples.append(SyntheticExample(**ex_data))

        logger.info(f"Loaded {len(examples)} examples")

        return examples
