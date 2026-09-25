"""Logging configuration utility for Academic Engagement Risk Dashboard."""

import logging
import sys
from typing import Optional

_DEFAULT_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "academic_risk", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with standard stream formatting.

    Args:
        name: Name for the logger hierarchy.
        level: Logging level (e.g. logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger instance configured with standard stream handler.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger was already created
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(fmt=_DEFAULT_FORMAT, datefmt=_DEFAULT_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
