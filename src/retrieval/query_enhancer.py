"""
Query enhancement techniques for better retrieval.

Implements:
1. HyDE (Hypothetical Document Embeddings) - Generate hypothetical answer and search with it
2. Query Expansion - Generate multiple query variations
3. Query Decomposition - Break complex queries into sub-queries
"""

from typing import List, Optional
from groq import Groq
import os

from src.config import Settings
from src.utils.logger import get_logger
from src.utils.exceptions import RetrievalError

logger = get_logger(__name__)


class QueryEnhancer:
    """Enhance queries for better retrieval using LLM-based techniques."""

    def __init__(self, settings: Settings):
        """
        Initialize query enhancer.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.config = settings.retrieval
        self.gen_config = settings.generation

        # Initialize Groq client for query enhancement
        api_key = self.gen_config.groq_api_key
        if not api_key:
            logger.warning("No Groq API key - query enhancement disabled")
            self.client = None
        else:
            self.client = Groq(api_key=api_key)

    def generate_hyde_document(self, query: str) -> str:
        """
        Generate a hypothetical document that would answer the query.
        This document embedding often matches better than the query embedding.

        Args:
            query: User query

        Returns:
            Hypothetical answer text
        """
        if not self.client or not self.config.use_hyde:
            return query  # Fallback to original query

        prompt = f"""Given this question, write a detailed answer that would appear in a technical document.
Be specific and technical. Do not include the question in your answer.

Question: {query}

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Use fast model for HyDE
                messages=[
                    {
                        "role": "system",
                        "content": "You are a technical writer creating documentation snippets."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=200
            )

            hyde_doc = response.choices[0].message.content.strip()
            logger.debug(f"Generated HyDE document ({len(hyde_doc)} chars)")
            return hyde_doc

        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}, using original query")
            return query

    def expand_query(self, query: str, num_variations: int = None) -> List[str]:
        """
        Generate multiple variations of the query for multi-query retrieval.

        Args:
            query: Original query
            num_variations: Number of variations (uses config default if None)

        Returns:
            List of query variations (includes original)
        """
        if not self.client or not self.config.use_query_expansion:
            return [query]

        num_variations = num_variations or self.config.num_query_variations

        prompt = f"""Generate {num_variations - 1} alternative phrasings of this question.
Each variation should ask the same thing but use different words.
Return ONLY the questions, one per line, without numbering.

Original question: {query}

Alternative phrasings:"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "You rephrase questions while keeping the same meaning."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,  # Higher temperature for diversity
                max_tokens=150
            )

            variations_text = response.choices[0].message.content.strip()
            variations = [v.strip() for v in variations_text.split('\n') if v.strip()]

            # Add original query
            all_queries = [query] + variations[:num_variations - 1]

            logger.debug(f"Expanded query into {len(all_queries)} variations")
            return all_queries

        except Exception as e:
            logger.warning(f"Query expansion failed: {e}, using original query")
            return [query]

    def decompose_query(self, query: str) -> List[str]:
        """
        Decompose complex query into simpler sub-queries.
        Useful for multi-hop reasoning.

        Args:
            query: Complex query

        Returns:
            List of sub-queries
        """
        if not self.client:
            return [query]

        prompt = f"""Break down this complex question into 2-3 simpler sub-questions that would help answer it.
Return ONLY the sub-questions, one per line.

Complex question: {query}

Sub-questions:"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "You decompose complex questions into simpler ones."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=150
            )

            sub_queries_text = response.choices[0].message.content.strip()
            sub_queries = [q.strip() for q in sub_queries_text.split('\n') if q.strip()]

            if sub_queries:
                logger.debug(f"Decomposed query into {len(sub_queries)} sub-queries")
                return sub_queries
            else:
                return [query]

        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            return [query]

    def enhance_query(self, query: str, method: str = "auto") -> List[str]:
        """
        Enhance query using configured method.

        Args:
            query: Original query
            method: Enhancement method ('hyde', 'expansion', 'decomposition', 'auto')

        Returns:
            List of enhanced queries
        """
        if method == "hyde":
            hyde_doc = self.generate_hyde_document(query)
            return [hyde_doc]
        elif method == "expansion":
            return self.expand_query(query)
        elif method == "decomposition":
            return self.decompose_query(query)
        else:  # auto - use config settings
            if self.config.use_hyde:
                # For HyDE, we search with the hypothetical document
                return [self.generate_hyde_document(query)]
            elif self.config.use_query_expansion:
                return self.expand_query(query)
            else:
                return [query]
