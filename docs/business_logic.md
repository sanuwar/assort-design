# Business Logic - Asort Design

## Scope
This document defines the business logic for the Asort Design app using FastAPI, Jinja2, SQLite, SQLModel, LangGraph, and OpenAI Responses API with a mock fallback.

## Core Entities
- Document: user-provided content from pasted text, URL extraction, or sample text.
- Job: processing request tied to a Document, with selected audience, routed audience, status, attempts, and constraints.
- Job Attempt: one pipeline run containing one-line summary, key clues, decision bullets, mind map, tags, evaluator verdict, and pass/fail.
- Tag: unique label assigned to a Document.
- DocumentTag: join table between Document and Tag.
- DocumentClue: key clue tied to a Document.

## Inputs
- Text input (paste).
- URL input (fetch and extract readable text).
- Sample content (built-in text in code).

Only one input type is required per submission. Text is persisted as a Document.

## Audience Selection
User selects an audience:
- `auto` (default)
- `commercial`
- `medical_affairs`
- `r_and_d`
- `cross_functional`

If `auto`, the routing step classifies the audience. If confidence is below threshold, use `cross_functional`.

## Pipeline Behavior
The LangGraph pipeline runs per Job:
1. `route_audience`
2. `specialist_generate`
3. `evaluate`
4. `revise` (max retries)
5. `persist`

## Step Rules
### route_audience
- If `job.selected_audience != "auto"`, set `job.audience = job.selected_audience`.
- Else, call the router prompt and parse strict JSON:
  `{"audience":"commercial|medical_affairs|r_and_d|cross_functional","confidence":0.0-1.0,"reasons":[...]}`
- If confidence < threshold, set `job.audience = "cross_functional"`.

### specialist_generate
- Use the audience profile system prompt from `app/agent_profiles.yaml`.
- Generate:
  - one-line summary
  - key clues list
  - decision bullets
  - mind map
  - tags list
- Enforce `default_max_words` for the selected audience unless overridden by Job.

### evaluate
- Use evaluator prompt from `app/agent_profiles.yaml`.
- Enforce constraints:
  - max words
  - required sections
  - audience-specific rubric
- Return strict JSON verdict:
  `{"pass":true/false,"word_count":int,"missing_sections":[...],"fail_reasons":[...],"fix_instructions":[...]}`

### revise
- If evaluation fails and attempts < max retries:
  - Use evaluator feedback to regenerate one-line summary, clues, bullets, and mind map.
- Max retries = 2 by default.

### persist
- Persist attempts, final accepted one-line summary, decision bullets, mind map, tags, and clues.
- If all attempts fail, persist final attempt and mark job as failed.

## Status Rules
- Job status transitions: `pending` -> `running` -> `completed` or `failed`.
- Attempt count increments per run.

## Mock LLM Mode
If `OPENAI_API_KEY` is missing:
- Use deterministic mock responses for routing, generation, and evaluation.
- Ensure valid JSON for router and evaluator outputs.

## URL Extraction
When input is a URL:
- Fetch with `httpx`.
- Extract readable text using `BeautifulSoup` and `lxml`.
- Best-effort parsing; fall back to raw text if extraction is weak.

## UI Flow Summary
- `/web` shows the home form.
- POST `/web/documents` creates Document and Job.
- POST `/web/jobs/{job_id}/run` runs pipeline.
- GET `/web/jobs/{job_id}` shows job details, attempts, and results.
- GET `/web/documents/{doc_id}` shows document details.
