# Interview Demo Code Review Prompt

Review this repository as an demo app, not as a production system.

Score it from 1 to 10 for:
- `code_clarity`
- `architecture_explainability`
- `technical_depth`
- `maintainability`
- `correctness_reliability`
- `testing_signal`
- `interview_readiness`
- `overall_score`

Guidance:
- Missing enterprise hardening should not dominate the evaluation unless it reflects a deeper engineering misunderstanding.
- Judge it primarily as an interview artifact meant to showcase technical competence, architecture thinking, and engineering skill.
- Also provide a probabilistic human-vs-AI effort assessment.
- Estimate how many hours a 4-person senior engineering team would need to build a similar demo:
  - without AI coding help
  - with AI coding help

Return **strict JSON only** in this format:

```json
{
  "summary": "1-2 lines only",
  "scores": {
    "code_clarity": 0,
    "architecture_explainability": 0,
    "technical_depth": 0,
    "maintainability": 0,
    "correctness_reliability": 0,
    "testing_signal": 0,
    "interview_readiness": 0,
    "overall_score": 0
  },
  "human_effort_assessment": {
    "likely_human_led_percent": 0,
    "likely_ai_assisted_percent": 0,
    "confidence": "low|medium|high"
  },
  "effort_estimate_4_senior_engineers": {
    "without_ai_hours": { "min": 0, "max": 0 },
    "with_ai_hours": { "min": 0, "max": 0 }
  }
}