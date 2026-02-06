from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI


def has_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def get_client() -> Optional[OpenAI]:
    if not has_api_key():
        return None
    return OpenAI()


def is_mock_mode() -> bool:
    return not has_api_key()
