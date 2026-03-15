# Assort Design 20-Minute Talk Track (Panel Version)

Use with `docs/demo_slides.html` and the 14-slide cut in `docs/panel_slide_plan.md`.

## 0:00-0:45 | Slide 1 (Title)

"Thank you for the opportunity. I will demonstrate Assort Design, an AI document intelligence workflow I built for life sciences use cases.  
The goal is to turn unstructured content into audience-specific, decision-ready output, while preserving traceability and compliance-focused checks."

## 0:45-1:30 | Slide 2 (Agenda)

"I will cover four parts:  
first the business problem and what the product does,  
second a live walkthrough from hosted app to localhost,  
third the technical mechanism including LangGraph, tools, and SQL model,  
and fourth why this pattern is relevant to cross-functional pharma operations."

## 1:30-2:30 | Slide 3 (Problem)

"Pharma teams process high volumes of mixed-quality content across medical, commercial, and R&D stakeholders.  
Manual triage is slow and inconsistent, and each team may interpret the same source differently.  
The practical result is slower decisions and weaker traceability from source text to stakeholder output."

## 2:30-3:30 | Slide 4 (Solution)

"Assort Design provides one intake path and one orchestrated pipeline.  
It routes the document to the right audience profile, generates structured outputs, evaluates quality, runs deterministic citation and risk tools, and persists all artifacts by job and attempt.  
So instead of ad hoc summaries, teams get consistent, auditable outputs."

## 3:30-8:30 | Slide 5 (Live Demo)

"I will start on the hosted demo at `assortdemo.duckdns.org`, then repeat the same flow on localhost."

Live narration sequence:
1. "Here is the intake form. I can paste text, use a URL, or sample content."
2. "I choose `auto` routing so the system decides the best audience."
3. "Now I run analysis."
4. "This output shows routed audience, one-line summary, decision bullets, clues, and tool badges."
5. "In job details, you can see attempt history and evaluation behavior."
6. "Now I will switch to localhost to show this is reproducible from my project root."
7. "I run `python -m uvicorn app.main:app --reload --env-file .env`, open `/web`, and execute the same job flow."

Close the demo segment:
"So the product behavior is consistent between hosted and local environments.  
Next I will map that behavior directly to code."

## 8:30-9:45 | Slide 6 (Architecture)

"At the web layer, FastAPI handles routes and page rendering.  
At orchestration, LangGraph manages stateful node execution and retry branching.  
At controls, deterministic tools run for citations and risk checks.  
At persistence, SQLModel/SQLite store documents, jobs, attempts, claims, and risk flags."

## 9:45-11:30 | Slide 7 (LangGraph Pipeline)

"The core function is `run_job_pipeline` in `app/graph.py`.  
The graph is built with 9 nodes: route, generate, evaluate, citation tool, risk tool, gate, persist attempt, revise, and persist results."

"The critical behavior is conditional progression:  
after each attempt, the graph either revises and loops or persists final results.  
That gives reliability without manual operator intervention."

Code pointers to mention while showing editor:
1. `app/graph.py:109` `_build_graph`
2. `app/graph.py:150` `run_job_pipeline`
3. `app/graph.py:340` `tool_citation_node`
4. `app/graph.py:348` `tool_risk_node`
5. `app/graph.py:359` `tool_gate_node`

## 11:30-13:00 | Slide 8 (Routing Intelligence)

"Routing is hybrid.  
If ML confidence is high, ML decides directly.  
If confidence is low or ambiguous, the pipeline falls back to the LLM route decision."

"This gives a practical balance:  
fast and low-cost for clear cases, reasoning fallback for edge cases."

Mention config:
1. `app/agent_profiles.yaml` for thresholds and prompts
2. `low_confidence_threshold` and `ml_router_threshold` currently `0.58`

## 13:00-14:15 | Slide 10 (Citation Finder)

"The citation tool checks generated claims against source text and stores offsets plus confidence.  
This turns output from 'trust me' into 'verify me'."

"In code, `citation_finder` is deterministic and intentionally bounded to keep outputs reliable and inspectable."

Code pointer:
1. `app/tools.py:75`

## 14:15-15:30 | Slide 11 (Risk Checker)

"The risk checker scans for compliance-sensitive language patterns, including absolute efficacy claims, off-label cues, and safety minimization.  
It produces severity, category, text span, and suggested fix."

"This does not replace medical-legal-regulatory review, but it catches high-risk language early and consistently."

Code pointer:
1. `app/tools.py:154`

## 15:30-17:00 | Slide 12 (Data Model)

"The SQL model captures full lineage from source to outcome.  
`Document` stores input, `Job` stores run context, `JobAttempt` stores each generation/evaluation cycle, and tool outputs persist into `DocumentClaim` and `DocumentRiskFlag`."

"That model supports auditability, troubleshooting, and future analytics."

Code pointers:
1. `app/models.py`
2. `app/db.py`
3. `app/main.py:921` run endpoint
4. `app/main.py:932` job detail endpoint

## 17:00-18:30 | Slide 14 (Self-Improving Loop)

"After enough completed jobs, the router can auto-retrain.  
The pipeline checks whether new labeled examples exceed the retrain interval, retrains, and hot-reloads the model."

"A lock guards against duplicate concurrent retraining, so this remains stable under parallel usage."

Code pointer:
1. `app/graph.py:44` `_maybe_retrain`

## 18:30-19:30 | Slide 15 (Why Madrigal)

"For Madrigal, the fit is cross-functional decision support: one source document can yield structured outputs aligned for medical, commercial, and R&D audiences while enforcing repeatable quality controls."

"The immediate value is faster triage, more consistent messaging, and better traceability between source evidence and downstream narrative."

## 19:30-20:00 | Slide 17 (Closing)

"To summarize: this demo shows full-stack ownership from user workflow to orchestration internals, deterministic controls, and data persistence.  
I am happy to go deeper into pipeline reliability, compliance controls, or deployment strategy based on your priorities."

## Time-Cut Fallback (if panel time drops to 15 min)

1. Skip Slide 6 and go directly from demo to Slide 7.
2. Merge Slides 10 and 11 into one 90-second tool explanation.
3. Keep Slides 12, 15, and 17 intact.

