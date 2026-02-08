from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def has_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def is_mock_mode() -> bool:
    return not has_api_key()


def is_langsmith_tracing_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").strip().lower() in ("1", "true", "yes", "on")

# Small factory/helper with type dict that returns an OpenAI client only if it can be used, and optionally wraps it for tracing.
def get_client() -> Optional[OpenAI]:
    if not has_api_key():
        return None
    client = OpenAI()
    if is_langsmith_tracing_enabled():
        try:
            from langsmith.wrappers import wrap_openai

            client = wrap_openai(client)
        except Exception:
            pass
    return client


def route_audience(router_prompt: str, document_text: str) -> Dict[str, Any]:
    if is_mock_mode():
        return _mock_route(document_text)
    user_prompt = f"Document:\n{document_text}"
    return _call_llm_json(router_prompt, user_prompt)

# LLM-backed content generator with mock-mode fallback.
def generate_content(
    system_prompt: str,
    generation_prompt: str,
    document_text: str,
    required_sections: List[str],
    max_words: int,
    fix_instructions: List[str] | None = None,
) -> Dict[str, Any]:
    if is_mock_mode():
        return _mock_generate(document_text, required_sections, max_words)

    rendered_prompt = _render_prompt(
        generation_prompt,
        required_sections=required_sections,
        max_words=max_words,
        fix_instructions=fix_instructions or [],
    )
    user_prompt = f"{rendered_prompt}\n\nDocument:\n{document_text}"
    return _call_llm_json(system_prompt, user_prompt)

# LLM-based evaluator for generated content (returns a JSON verdict).
# Checks required sections and max word constraint.
# Uses a mock fallback when demo/test mode is enabled.

def evaluate_content(
    evaluator_prompt: str,
    one_line_summary: str,
    decision_bullets: List[str],
    required_sections: List[str],
    max_words: int,
) -> Dict[str, Any]:
    if is_mock_mode():
        return _mock_evaluate(one_line_summary, decision_bullets, required_sections, max_words)

    required_str = ", ".join(required_sections)
    bullets_text = "\n".join([f"- {b}" for b in decision_bullets])
    user_prompt = (
        "One-line summary:\n"
        f"{one_line_summary}\n\n"
        "Decision bullets:\n"
        f"{bullets_text}\n\n"
        f"Required sections: {required_str}\n"
        f"Max words: {max_words}\n"
    )
    return _call_llm_json(evaluator_prompt, user_prompt)

# LLM helper: run a Responses API call and normalize the output.
# Extracts text from the response payload and parses it into a dict.
# Fails fast if the OpenAI client isn't configured.

def _call_llm_json(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    client = get_client()
    if not client:
        raise RuntimeError("OpenAI client not available.")

    response = client.responses.create(
        model=DEFAULT_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = _extract_text(response)
    return _parse_json(text)


def _extract_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text

    try:
        parts: List[str] = []
        for item in response.output:
            for content in item.content:
                if hasattr(content, "text"):
                    parts.append(content.text)
        if parts:
            return "\n".join(parts)
    except Exception:
        pass

    return str(response)


def _parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                return {}
    return {}


def _render_prompt(
    template: str,
    required_sections: List[str],
    max_words: int,
    fix_instructions: List[str],
) -> str:
    required_str = ", ".join(required_sections)
    fixes = "; ".join([f for f in fix_instructions if f]) or "None"
    return (
        template.replace("{{required_sections}}", required_str)
        .replace("{{max_words}}", str(max_words))
        .replace("{{fix_instructions}}", fixes)
    )


def _mock_route(document_text: str) -> Dict[str, Any]:
    text = document_text.lower()
    if any(k in text for k in ["market", "sales", "commercial", "pricing"]):
        return {
            "audience": "commercial",
            "confidence": 0.78,
            "reasons": ["market"],
            "candidates": ["commercial", "r_and_d"],
        }
    if any(k in text for k in ["clinical", "patient", "safety", "evidence"]):
        return {
            "audience": "medical_affairs",
            "confidence": 0.8,
            "reasons": ["clinical"],
            "candidates": ["medical_affairs", "r_and_d"],
        }
    if any(k in text for k in ["experiment", "assay", "protocol", "method"]):
        return {
            "audience": "r_and_d",
            "confidence": 0.76,
            "reasons": ["research"],
            "candidates": ["r_and_d", "medical_affairs"],
        }
    return {
        "audience": "cross_functional",
        "confidence": 0.4,
        "reasons": ["low_signal"],
        "candidates": ["commercial", "medical_affairs"],
    }


def _mock_generate(
    document_text: str, required_sections: List[str], max_words: int
) -> Dict[str, Any]:
    words = document_text.split()
    snippet = " ".join(words[:40]) if words else "No source text provided."
    one_line_summary = f"{snippet[:max_words]}."

    decision_bullets = []
    for section in required_sections:
        decision_bullets.append(f"{section}: {snippet}")
    if len(decision_bullets) > 5:
        decision_bullets = decision_bullets[:5]

    tags = ["mock", "summary", "analysis"]
    key_clues = [
        "Focus on the core message.",
        "Note the intended audience.",
        "Identify key risks or opportunities.",
    ]
    mind_map = _mock_mind_map(required_sections)

    return {
        "one_line_summary": one_line_summary,
        "tags": tags,
        "key_clues": key_clues,
        "decision_bullets": decision_bullets,
        "mind_map": mind_map,
    }


def _mock_evaluate(
    one_line_summary: str,
    decision_bullets: List[str],
    required_sections: List[str],
    max_words: int,
) -> Dict[str, Any]:
    combined_text = " ".join([one_line_summary] + decision_bullets)
    word_count = len(combined_text.split())
    missing = [
        section
        for section in required_sections
        if not any(section.lower() in bullet.lower() for bullet in decision_bullets)
    ]
    bullet_count_ok = 3 <= len(decision_bullets) <= 5
    passed = word_count <= max_words and not missing and bullet_count_ok
    fail_reasons = []
    if word_count > max_words:
        fail_reasons.append("Summary exceeds max words.")
    if missing:
        fail_reasons.append("Missing required sections.")
    if not bullet_count_ok:
        fail_reasons.append("Decision bullets must be 3 to 5.")

    return {
        "pass": passed,
        "word_count": word_count,
        "missing_sections": missing,
        "fail_reasons": fail_reasons,
        "fix_instructions": [
            "Add missing sections." if missing else "Shorten summary to max words."
        ]
        if not passed
        else [],
    }


def _mock_mind_map(required_sections: List[str]) -> str:
    lines = ["mindmap", "  root((Summary))"]
    for section in required_sections[:5]:
        lines.append(f"    {section}")
    return "\n".join(lines)
