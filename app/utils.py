"""Shared utility helpers for the Assort app."""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def env_int(name: str, default: int) -> int:
    """Read an integer from an environment variable.

    Falls back to *default* and logs a warning if the value is missing or
    cannot be parsed as an integer.  Never raises.

    Usage:
        RETRAIN_EVERY_N_JOBS = env_int("RETRAIN_EVERY_N_JOBS", 20)
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "Invalid value for env var %s=%r — expected an integer, using default %d",
            name,
            raw,
            default,
        )
        return default


def env_float(name: str, default: float) -> float:
    """Read a float from an environment variable.

    Falls back to *default* and logs a warning if the value is missing or
    cannot be parsed as a float.  Never raises.

    Usage:
        TIMEOUT = env_float("OPENAI_TIMEOUT_SEC", 30.0)
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "Invalid value for env var %s=%r — expected a float, using default %g",
            name,
            raw,
            default,
        )
        return default
