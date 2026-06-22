"""
KnowledgeBrain API — Structured Logging
"""

import logging
import sys
from app.config import get_settings


def setup_logging() -> logging.Logger:
    """Configure structured logging for the application."""
    settings = get_settings()

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Root logger
    root_logger = logging.getLogger("knowledgebrain")
    root_logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    root_logger.addHandler(console_handler)

    # Suppress noisy third-party loggers
    for noisy in ["httpx", "httpcore", "urllib3", "neo4j", "sentence_transformers"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a named logger under the knowledgebrain namespace."""
    return logging.getLogger(f"knowledgebrain.{name}")
