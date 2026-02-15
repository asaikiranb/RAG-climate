"""
Groq API wrapper with retry logic, rate limiting, and error handling.
Provides robust LLM inference for answer generation.
"""

import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import asyncio
from functools import wraps

try:
    from groq import Groq
    from groq.types.chat import ChatCompletion
except ImportError:
    raise ImportError("Groq SDK not installed. Run: pip install groq")

from src.config import Settings
from src.utils.logger import get_logger
from src.utils.exceptions import GenerationError

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """Structured LLM response."""
    content: str
    model: str
    tokens_used: int
    latency_ms: float
    finish_reason: str
    metadata: Dict[str, Any]


def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0
):
    """
    Decorator for exponential backoff retry logic.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        max_delay: Maximum delay in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # Don't retry on certain errors
                    error_msg = str(e).lower()
                    if any(x in error_msg for x in ["invalid api key", "authentication", "not found"]):
                        logger.error(f"Non-retryable error: {e}")
                        raise

                    if attempt < max_retries:
                        sleep_time = min(delay, max_delay)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                            f"Retrying in {sleep_time:.2f}s..."
                        )
                        time.sleep(sleep_time)
                        delay *= backoff_factor
                    else:
                        logger.error(f"All {max_retries} retry attempts failed")
                        raise GenerationError(
                            f"LLM API call failed after {max_retries} retries",
                            details={"last_error": str(last_exception)}
                        ) from last_exception

            raise last_exception

        return wrapper
    return decorator


class LLMClient:
    """
    Production-ready Groq API client with retry logic and error handling.

    Features:
    - Exponential backoff retry
    - Rate limiting
    - Token usage tracking
    - Latency monitoring
    - Structured error handling
    """

    def __init__(self, settings: Settings):
        """
        Initialize Groq LLM client.

        Args:
            settings: Application settings

        Raises:
            GenerationError: If API key is missing
        """
        self.settings = settings
        self.config = settings.generation

        if not self.config.groq_api_key:
            raise GenerationError(
                "GROQ_API_KEY not found in settings",
                details={"hint": "Set GROQ_API_KEY in .env file"}
            )

        try:
            self.client = Groq(api_key=self.config.groq_api_key)
        except Exception as e:
            raise GenerationError(
                f"Failed to initialize Groq client: {e}",
                details={"error": str(e)}
            )

        logger.info(
            f"Initialized Groq LLM client with model: {self.config.model_name}"
        )

        # Track usage statistics
        self.total_tokens = 0
        self.total_calls = 0
        self.total_errors = 0

    @retry_with_exponential_backoff(max_retries=3, initial_delay=1.0)
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> LLMResponse:
        """
        Generate completion from Groq API with retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (overrides config)
            max_tokens: Max tokens to generate (overrides config)
            top_p: Nucleus sampling parameter (overrides config)
            stop: Stop sequences
            stream: Whether to stream response (not implemented yet)

        Returns:
            LLMResponse with generated text and metadata

        Raises:
            GenerationError: If generation fails
        """
        start_time = time.time()

        # Use config defaults if not specified
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        top_p = top_p if top_p is not None else self.config.top_p

        try:
            logger.debug(
                f"Calling Groq API with {len(messages)} messages, "
                f"temp={temperature}, max_tokens={max_tokens}"
            )

            # Make API call
            response: ChatCompletion = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                stream=stream,
            )

            # Extract response data
            completion = response.choices[0]
            content = completion.message.content
            finish_reason = completion.finish_reason

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Extract token usage
            usage = response.usage
            tokens_used = usage.total_tokens if usage else 0

            # Update statistics
            self.total_tokens += tokens_used
            self.total_calls += 1

            logger.info(
                f"Generated response: {tokens_used} tokens, "
                f"{latency_ms:.0f}ms, finish_reason={finish_reason}"
            )

            return LLMResponse(
                content=content,
                model=response.model,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
                metadata={
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )

        except Exception as e:
            self.total_errors += 1
            logger.error(f"LLM generation failed: {e}")
            raise GenerationError(
                f"Failed to generate LLM response: {e}",
                details={
                    "model": self.config.model_name,
                    "num_messages": len(messages),
                    "error": str(e)
                }
            ) from e

    def generate_with_system_prompt(
        self,
        system_prompt: str,
        user_message: str,
        **kwargs
    ) -> LLMResponse:
        """
        Convenience method to generate with system + user message.

        Args:
            system_prompt: System instruction
            user_message: User query/prompt
            **kwargs: Additional arguments for generate()

        Returns:
            LLMResponse
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        return self.generate(messages, **kwargs)

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics for monitoring.

        Returns:
            Dictionary with usage metrics
        """
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_errors": self.total_errors,
            "avg_tokens_per_call": self.total_tokens / max(self.total_calls, 1),
            "error_rate": self.total_errors / max(self.total_calls, 1),
        }

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self.total_tokens = 0
        self.total_calls = 0
        self.total_errors = 0
        logger.info("Reset LLM usage statistics")
