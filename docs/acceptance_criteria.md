# Acceptance Criteria - Asort Design

## General
- The app runs with FastAPI, Jinja2, SQLite, SQLModel, and LangGraph.
- The app uses OpenAI Responses API when `OPENAI_API_KEY` is set.
- The app uses mock LLM mode when `OPENAI_API_KEY` is missing.
- All prompts and rubrics are in `app/agent_profiles.yaml`.

## Home Page
- GET `/web` renders a page where users can:
  - Paste text
  - Enter a URL
  - Use sample content
- The page includes an Audience dropdown:
  - `auto` (default)
  - `commercial`
  - `medical_affairs`
  - `r_and_d`
  - `cross_functional`

## Document Creation
- POST `/web/documents` creates a Document from:
  - pasted text, or
  - URL extraction, or
  - sample content.
- The request creates a Job tied to the Document.
- The Job stores `selected_audience` and can be `auto`.

## Pipeline Execution
- POST `/web/jobs/{job_id}/run` runs the LangGraph pipeline.
- The pipeline performs:
  - audience routing
  - specialist generation
  - evaluation
  - revise up to max retries
  - persistence

## Routing
- If `selected_audience` is not `auto`, the pipeline uses it as final audience.
- If `selected_audience` is `auto`, the router returns strict JSON with audience and confidence.
- If confidence is below threshold, audience is set to `cross_functional`.

## Generation
- The specialist generation produces:
  - one-line summary
  - key clues list
  - decision bullets (3 to 5)
  - mind map (Mermaid)
  - tags list
- Output follows audience-specific prompts, required sections, and max words.

## Evaluation
- The evaluator returns strict JSON verdict with pass/fail and fix instructions.
- The pipeline retries up to `max_retries`.

## Persistence
- Each attempt is stored in `job_attempts`.
- Final accepted one-line summary, decision bullets, and mind map are stored on success.
- If all attempts fail, job is marked failed and the last attempt is persisted.
- Tags are stored and linked to the Document.
- Key clues are stored for the Document.

## Job Detail Page
- GET `/web/jobs/{job_id}` shows:
  - audience badge and status
  - attempt history with pass/fail and reasons
  - one-line summary preview for each attempt
  - final accepted one-line summary
  - decision bullets
  - mind map
  - tags list
  - key clues list

## Document Detail Page
- GET `/web/documents/{doc_id}` shows:
  - document content
  - tags
  - key clues

## Test Notes (Non-Binding)
The following notes align acceptance testing with current business logic and
implementation details. These are informational and should not be treated as
requirements if implementation evolves.

- Default `max_retries` is 2 unless overridden.
- Job status values are `pending`, `running`, `completed`, and `failed`.
- Evaluator JSON schema includes:
  `{"pass":true/false,"word_count":int,"missing_sections":[...],"fail_reasons":[...],"fix_instructions":[...]}`
- URL extraction currently uses `httpx` with `BeautifulSoup` + `lxml` for
  best-effort readable text parsing.
