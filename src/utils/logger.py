"""
Structured logging setup using loguru.
Provides better logging than print statements with:
- Structured JSON logs
- Log levels
- File rotation
- Trace IDs for request tracking
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger
import contextvars

# Context variable for request tracing
trace_id_var = contextvars.ContextVar("trace_id", default=None)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    enable_json: bool = False
) -> None:
    """
    Configure logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging
        enable_json: Whether to use JSON format for logs
    """
    # Remove default handler
    logger.remove()

    # Format for console logging
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Add console handler
    logger.add(
        sys.stderr,
        format=console_format,
        level=log_level,
        colorize=True,
    )

    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        if enable_json:
            logger.add(
                log_file,
                format="{time} {level} {message}",
                level=log_level,
                rotation="100 MB",
                retention="10 days",
                compression="zip",
                serialize=True,  # JSON format
            )
        else:
            logger.add(
                log_file,
                format=console_format,
                level=log_level,
                rotation="100 MB",
                retention="10 days",
                compression="zip",
            )

    logger.info(f"Logging initialized at {log_level} level")


def get_logger(name: str):
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logger.bind(module=name)


def set_trace_id(trace_id: str) -> None:
    """Set trace ID for request tracking."""
    trace_id_var.set(trace_id)


def get_trace_id() -> Optional[str]:
    """Get current trace ID."""
    return trace_id_var.get()


# Decorator for adding trace ID to logs
def with_trace_id(func):
    """Decorator to add trace ID to function logs."""
    import functools
    import uuid

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        trace_id = str(uuid.uuid4())[:8]
        set_trace_id(trace_id)

        logger_instance = get_logger(func.__module__)
        logger_instance.info(f"[{trace_id}] Starting {func.__name__}")

        try:
            result = func(*args, **kwargs)
            logger_instance.info(f"[{trace_id}] Completed {func.__name__}")
            return result
        except Exception as e:
            logger_instance.error(f"[{trace_id}] Error in {func.__name__}: {e}")
            raise

    return wrapper
