# Changelog

All notable changes to this project will be documented in this file.

The format is based on **Keep a Changelog**, and this project aims to follow **Semantic Versioning**.

---

## [2.0.0] — Unreleased

### Added
- **Tag Intelligence layer (domain-agnostic)**
  - Canonical tag normalization (case/spacing/punctuation consistency)
  - Alias/synonym mapping support (e.g., `genai` → `generative ai`)
  - Deterministic domain/topic-lane inference from canonical tags
  - Persisted “tag summary” output to support fast UI aggregation

- **Insights UI (“wow” widgets)**
  - **Topic lanes (domain buckets)** over the most recent documents
  - **Top tags** across the last N records
  - **Rising tags** (last N vs previous N) for trend detection
  - **Co-occurring tag pairs** to surface recurring theme combinations
  - **Bridge tags** (tags appearing across multiple domains) to show connectors

- **Related documents**
  - “Related docs” view powered by tag-set similarity (Jaccard)
  - Displays top matches with overlap-based scoring and overlapping tags

- **Tag alias management (admin utility)**
  - Simple UI to add/update alias → canonical mappings
  - Alias updates immediately influence canonicalization and insights

- **Post-generation deterministic tools (traceability & safety)**
  - Citation Finder: claim → supporting quote with offsets
  - Structured Extractor: normalized JSON entity summary
  - Risk/Compliance Checker: flags overconfident/unsupported phrasing
  - Tool outputs persisted so job detail pages show traceability

### Changed
- **LangGraph pipeline** now runs additional deterministic steps after generation:
  - Tag canonicalization + domain inference + persistence
  - Tool nodes for citations/entities/risk checks (when enabled)
- **Job/Document detail UI** updated to display:
  - Canonical tags + inferred domain
  - Related-document recommendations
  - Tool outputs (citations, extracted JSON, risk flags) when available

### Database
- Added new tables to support v2 features (minimal schema change approach):
  - `tag_aliases` — alias → canonical mapping
  - `document_tag_summaries` — per-document/per-job canonical tags + inferred domain
  - `document_claims` — citation finder results (claim ↔ quote + offsets)
  - `document_entity_summaries` — structured extractor JSON output
  - `document_risk_flags` — risk checker flags
- Added indexes on key lookup fields (e.g., `document_id`, `job_id`, `domain`, `created_at`).

### Notes
- All “wow” widgets are **pre-defined UI features**; they are populated dynamically from persisted canonical tags (and tool outputs if enabled).
- Mock mode remains supported:
  - Tag intelligence runs deterministically without external services
  - Tool steps can return realistic mock outputs for demos

---

## [1.0.0] — Current Stable
- Initial release focused on document ingestion + summarization/sorting.
- Generates and displays outputs such as decision bullets, tags/clues, and mind map.
- Provides the core FastAPI + Jinja2 + SQLite (SQLModel) web UI for documents and jobs.
