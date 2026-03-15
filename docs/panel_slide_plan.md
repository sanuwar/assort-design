# Assort Design Panel Deck (14-Slide Cut)

Use this with the existing deck: `docs/demo_slides.html`.

## Keep/Skip Map

Keep these slides for the panel run:
- 1 `TITLE`
- 2 `AGENDA`
- 3 `PROBLEM`
- 4 `SOLUTION`
- 5 `LIVE DEMO`
- 6 `ARCHITECTURE`
- 7 `LANGGRAPH PIPELINE`
- 8 `ROUTING`
- 10 `CITATION FINDER`
- 11 `RISK CHECKER`
- 12 `DATA MODEL`
- 14 `AUTO-RETRAIN`
- 15 `WHY MADRIGAL`
- 17 `CLOSING`

Skip in main run (use only if asked):
- 9 `ML ROUTER INTERNALS`
- 13 `TAG INTELLIGENCE`
- 16 `CODEBASE MAP`
- Appendix A/B

## Timing (Target: 19-20 min)

1. Slide 1: 0:45
2. Slide 2: 0:45
3. Slide 3: 1:00
4. Slide 4: 1:00
5. Slide 5: 5:00
6. Slide 6: 1:15
7. Slide 7: 1:45
8. Slide 8: 1:30
9. Slide 10: 1:15
10. Slide 11: 1:15
11. Slide 12: 1:30
12. Slide 14: 1:30
13. Slide 15: 1:30
14. Slide 17: 1:00

## Speaker Notes + Transitions

1. Slide 1: Open with role relevance.
`Line:` "I built Assort Design to convert unstructured pharma content into decision-ready, audience-specific outputs with traceability."
`Transition:` "Let me show the flow first, then prove the mechanism in code."

2. Slide 2: Set interview contract.
`Line:` "I will cover business value, live run, pipeline mechanics, and controls for compliance and reliability."
`Transition:` "The need becomes obvious when we look at current document handling pain."

3. Slide 3: Problem framing.
`Line:` "Teams see the same document differently, and manual triage creates delay, inconsistency, and audit risk."
`Transition:` "This system standardizes that path."

4. Slide 4: Solution framing.
`Line:` "One ingest path, auto-routing, structured generation, deterministic checks, then persistence with job history."
`Transition:` "Now I will run the product live."

5. Slide 5: Live demo sequence.
- Start hosted: `https://assortdemo.duckdns.org`
- Submit sample or pasted text.
- Run job and show audience route, summary, clues, bullets, and risk/citation outputs.
- Pivot to localhost and repeat quickly:
  - `.\.venv\Scripts\Activate.ps1`
  - `python -m uvicorn app.main:app --reload --env-file .env`
  - Open `http://127.0.0.1:8000/web`
`Transition:` "Now that you have seen behavior, here is the execution design behind it."

6. Slide 6: Architecture.
`Line:` "FastAPI orchestrates the user flow, LangGraph handles stateful execution, tools enforce grounding and risk checks, SQLite stores full history."
`Transition:` "The core is a 9-node graph."

7. Slide 7: LangGraph pipeline.
`Line:` "Pipeline order is route, generate, evaluate, citation tool, risk tool, gate, persist attempt, then either revise or finalize."
`Code anchors:` `app/graph.py:109`, `app/graph.py:150`, `app/graph.py:340`
`Transition:` "Routing is hybrid to balance speed and reasoning quality."

8. Slide 8: Routing intelligence.
`Line:` "ML handles clear cases; uncertain cases fall back to LLM. This keeps latency/cost down while preserving decision quality."
`Code anchors:` `app/agent_profiles.yaml`, `app/graph.py`
`Transition:` "After generation, deterministic tools add auditability."

9. Slide 10: Citation finder.
`Line:` "Claims are checked against source text spans with offsets and confidence so reviewers can verify grounding quickly."
`Code anchor:` `app/tools.py:75`
`Transition:` "Grounding alone is not enough in pharma; safety language must be screened."

10. Slide 11: Risk checker.
`Line:` "Regex-based risk rules flag absolute efficacy, off-label cues, and safety minimization with severity and suggested fixes."
`Code anchor:` `app/tools.py:154`
`Transition:` "All of this is persisted as queryable records."

11. Slide 12: Data model.
`Line:` "Core entities are Document, Job, JobAttempt, DocumentClaim, and DocumentRiskFlag, which gives full trace from input to decision."
`Code anchors:` `app/models.py`, `app/db.py`
`Transition:` "The routing layer also improves over time."

12. Slide 14: Self-improving loop.
`Line:` "After enough completed jobs, the ML router retrains and hot-reloads with locking to avoid duplicate retrain races."
`Code anchor:` `app/graph.py:44`
`Transition:` "This is the practical impact for Madrigal."

13. Slide 15: Why Madrigal.
`Line:` "This is useful where scientific, commercial, and compliance stakeholders need a consistent decision artifact from the same source document."
`Transition:` "I will close with fit and open Q&A."

14. Slide 17: Closing.
`Line:` "This demonstrates end-to-end ownership: product UX, orchestration, controls, and data model with measurable reliability."

## Panel Safety Plan (if demo risk appears)

1. If hosted site is slow, switch to localhost within 20 seconds.
2. If API key is unavailable, state mock mode behavior is deterministic and still exercises pipeline structure.
3. If generation is delayed, show last completed job detail page and then jump to code anchors.

