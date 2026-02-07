from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

_PROFILES: Dict[str, Any] | None = None


def load_profiles() -> Dict[str, Any]:
    """Load and cache agent profiles from YAML."""
    global _PROFILES
    if _PROFILES is None:
        path = Path(__file__).resolve().parent / "agent_profiles.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"agent_profiles.yaml not found at {path}."
            )
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("agent_profiles.yaml must contain a top-level mapping.")
        _validate_profiles(data)
        _PROFILES = data
    return _PROFILES


def get_audience_profile(audience: str) -> Dict[str, Any]:
    profiles = load_profiles()
    audiences = profiles.get("audiences", {})
    if audience not in audiences:
        available = ", ".join(sorted(audiences.keys()))
        raise ValueError(
            f"Unknown audience '{audience}'. Available: {available}."
        )
    return audiences[audience]


def get_routing_config() -> Dict[str, Any]:
    profiles = load_profiles()
    routing = profiles.get("routing")
    if not isinstance(routing, dict):
        raise ValueError("routing config must be a mapping in agent_profiles.yaml.")
    return routing


def get_evaluation_config() -> Dict[str, Any]:
    profiles = load_profiles()
    evaluation = profiles.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation config must be a mapping in agent_profiles.yaml.")
    return evaluation


def get_generation_config() -> Dict[str, Any]:
    profiles = load_profiles()
    generation = profiles.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("generation config must be a mapping in agent_profiles.yaml.")
    return generation


def _validate_profiles(data: Dict[str, Any]) -> None:
    audiences = data.get("audiences")
    if not isinstance(audiences, dict) or not audiences:
        raise ValueError("'audiences' must be a non-empty mapping.")

    for name, profile in audiences.items():
        if not isinstance(profile, dict):
            raise ValueError(f"audiences.{name} must be a mapping.")

        required_keys = [
            "display_name",
            "system_prompt",
            "required_sections",
            "default_max_words",
        ]
        for key in required_keys:
            if key not in profile:
                raise ValueError(f"audiences.{name}.{key} is required.")

        if not isinstance(profile["display_name"], str) or not profile[
            "display_name"
        ].strip():
            raise ValueError(f"audiences.{name}.display_name must be a non-empty string.")

        if not isinstance(profile["system_prompt"], str) or not profile[
            "system_prompt"
        ].strip():
            raise ValueError(f"audiences.{name}.system_prompt must be a non-empty string.")

        sections = profile["required_sections"]
        if not isinstance(sections, list) or not sections:
            raise ValueError(
                f"audiences.{name}.required_sections must be a non-empty list of strings."
            )
        if not all(isinstance(s, str) and s.strip() for s in sections):
            raise ValueError(
                f"audiences.{name}.required_sections must contain only non-empty strings."
            )

        max_words = profile["default_max_words"]
        if not isinstance(max_words, int) or max_words <= 0:
            raise ValueError(
                f"audiences.{name}.default_max_words must be a positive integer."
            )

    routing = data.get("routing")
    if not isinstance(routing, dict):
        raise ValueError("'routing' must be a mapping.")
    if not isinstance(routing.get("auto_router_prompt"), str) or not routing[
        "auto_router_prompt"
    ].strip():
        raise ValueError("routing.auto_router_prompt must be a non-empty string.")
    threshold = routing.get("low_confidence_threshold")
    if not isinstance(threshold, (int, float)):
        raise ValueError("routing.low_confidence_threshold must be a number.")
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("routing.low_confidence_threshold must be between 0.0 and 1.0.")

    evaluation = data.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("'evaluation' must be a mapping.")
    if not isinstance(evaluation.get("evaluator_prompt"), str) or not evaluation[
        "evaluator_prompt"
    ].strip():
        raise ValueError("evaluation.evaluator_prompt must be a non-empty string.")

    generation = data.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("'generation' must be a mapping.")
    if not isinstance(generation.get("prompt"), str) or not generation[
        "prompt"
    ].strip():
        raise ValueError("generation.prompt must be a non-empty string.")
