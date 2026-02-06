1) app/agent_profiles.yaml
Include:
- audiences: commercial, medical_affairs, r_and_d, cross_functional
  For each audience define:
    - display_name
    - system_prompt (multi-line)
    - required_sections (list of strings)
    - default_max_words (int)
- routing:
    - auto_router_prompt (multi-line) that instructs LLM to output strict JSON:
      {"audience":"commercial|medical_affairs|r_and_d|cross_functional","confidence":0.0-1.0,"reasons":[...]}
    - low_confidence_threshold (float)
- evaluation:
    - evaluator_prompt (multi-line) that instructs LLM to output strict JSON:
      {
        "pass": true/false,
        "word_count": int,
        "missing_sections": [...],
        "fail_reasons": [...],
        "fix_instructions": [...]
      }

2) app/config.py
Implement a small YAML loader that:
- Loads app/agent_profiles.yaml once (cache in module global).
- Provides functions:
  - load_profiles() -> dict
  - get_audience_profile(audience: str) -> dict (raise ValueError if unknown)
  - get_routing_config() -> dict
  - get_evaluation_config() -> dict
- Use safe YAML parsing (PyYAML safe_load).
- Add minimal type checks and helpful error messages.

3) requirements.txt
Add needed dependency:
- pyyaml