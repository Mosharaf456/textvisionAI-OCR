"""Structured logging with structlog for CPU OCR pipeline.

Provides JSON formatted logs in production and pretty-printed logs in console.
Every OCR request is tagged with request_id for end-to-end tracing.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from typing import Any, Dict, Optional
import structlog


def setup_logger(
    log_level: str = "INFO",
    json_format: bool = False,
) -> structlog.stdlib.BoundLogger:
    """Configure structlog for application-wide logging."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_format or os.getenv("LOG_FORMAT", "").lower() == "json":
        processors = shared_processors + [
            structlog.processors.JSONRenderer()
        ]
    else:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    return structlog.get_logger("ocr_pipeline")


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """Get a named structured logger."""
    return structlog.get_logger(name or "ocr_pipeline")


def create_request_id() -> str:
    """Generate a unique request ID for tracing."""
    return str(uuid.uuid4())
