# Mock Q&A Pack (Madrigal Hiring Committee)

## 1) "How do you reduce hallucination risk?"

Answer:
"I combine LLM generation with deterministic post-checks.  
Citations are matched back to source spans, and a risk checker flags problematic language.  
I also persist attempts and evaluator feedback, so we can inspect and improve behavior over time."

Proof anchors:
- `app/tools.py:75`
- `app/tools.py:154`
- `app/models.py` (`JobAttempt`, `DocumentClaim`, `DocumentRiskFlag`)

## 2) "Why LangGraph instead of a simple function pipeline?"

Answer:
"LangGraph makes the workflow explicit and stateful.  
I can define retry logic, conditional branching, and node-level separation cleanly.  
That gives better reliability and easier debugging than ad hoc control flow."

Proof anchors:
- `app/graph.py:109`
- `app/graph.py:150`

## 3) "How do you decide audience routing?"

Answer:
"It is hybrid. ML handles clear cases for speed and cost, while uncertain or ambiguous cases fall back to LLM reasoning.  
Threshold and margin are config-driven so behavior is tunable without code edits."

Proof anchors:
- `app/graph.py` (`route_audience_node`)
- `app/agent_profiles.yaml` (`ml_router_threshold`, `ml_router_margin`)

## 4) "What happens if the model artifacts are missing?"

Answer:
"The app degrades gracefully to LLM-only routing.  
So routing still works even before ML artifacts exist or if artifact loading fails."

Proof anchors:
- `app/graph.py` (fallback branches in `route_audience_node`)

## 5) "How do you handle compliance-sensitive language?"

Answer:
"I use deterministic risk patterns for categories like absolute efficacy, off-label indicators, and safety minimization.  
Each flag includes severity and suggested fix so reviewers can act quickly."

Proof anchors:
- `app/tools.py:154`

## 6) "Is this replacing MLR review?"

Answer:
"No. It is a pre-review quality and triage layer, not a replacement for medical-legal-regulatory review.  
The value is earlier detection, consistency, and traceability."

## 7) "How is quality evaluated before persistence?"

Answer:
"Each attempt is evaluated for section coverage and word constraints.  
If it fails, the graph revises and retries up to the configured limit, then persists outcome and status."

Proof anchors:
- `app/agent_profiles.yaml` (`evaluation`)
- `app/models.py` (`max_retries`, `attempt_count`)
- `app/graph.py` (conditional revise/persist flow)

## 8) "What is persisted for auditability?"

Answer:
"Input document, job context, each attempt, tool outputs, and final artifacts.  
That gives full lineage from source text to generated decision outputs."

Proof anchors:
- `app/models.py`
- `app/db.py`

## 9) "How do you ensure reproducibility between environments?"

Answer:
"I demonstrate the same workflow on hosted and localhost.  
The same FastAPI routes and pipeline code are used in both."

Proof anchors:
- `app/main.py:297`
- `app/main.py:807`
- `app/main.py:921`

## 10) "How does retraining work?"

Answer:
"After enough completed jobs, the system checks if new labeled volume crosses the retrain threshold.  
If yes, it retrains the ML router and hot-reloads artifacts."

Proof anchors:
- `app/graph.py:44`
- `app/train_router.py`

## 11) "How do you prevent concurrent retrain conflicts?"

Answer:
"There is an in-process lock around retraining.  
If one retrain is running, others skip, so I avoid duplicate overlapping training jobs."

Proof anchors:
- `app/graph.py:44`

## 12) "What are your main failure modes?"

Answer:
"Three main modes: input quality issues, LLM variance, and routing uncertainty.  
Mitigations are input validation, evaluator + retries, and ML/LLM fallback routing."

## 13) "How do you scale this design?"

Answer:
"Scale path is straightforward: move SQLite to managed relational storage, externalize queue/worker execution, and keep deterministic tools as stateless services.  
The node structure already separates concerns for that migration."

## 14) "What would you improve first?"

Answer:
"First, add richer observability on node latency and failure reasons.  
Second, expand quality metrics by audience.  
Third, formalize validation test sets for compliance-heavy scenarios."

## 15) "How do you test this?"

Answer:
"I test at multiple levels: route and page behavior, pipeline behavior with mocks, and deterministic tool behavior with known edge cases.  
Because tool nodes are deterministic, they are easy to validate with fixture-based tests."

## 16) "How do you protect sensitive data?"

Answer:
"Current demo scope is local/dev.  
In production I would enforce encryption in transit and at rest, strict access control, secret management, and data retention policies aligned to policy requirements."

## 17) "Why not use only deterministic NLP, no LLM?"

Answer:
"Deterministic rules are strong for controls but weak for nuanced summarization across mixed scientific/business context.  
Hybrid architecture gives expressive generation while preserving deterministic guardrails."

## 18) "Why should we trust this output?"

Answer:
"Trust comes from transparency, not blind confidence: source-linked citations, explicit risk flags, persisted attempt history, and reproducible pipeline behavior."

## 19) "What is your personal contribution here?"

Answer:
"I built this end-to-end: FastAPI app flow, LangGraph orchestration, ML router integration, deterministic tools, schema design, and deployment-ready structure."

## 20) "What is the most important technical decision you made?"

Answer:
"Separating creative generation from deterministic controls.  
That separation keeps the system practical in a regulated communication context."

## Strong Closing Answer (if asked "Anything else?")

"If selected, I can operationalize this pattern into production-grade workflows with stronger governance, monitoring, and validation while keeping delivery speed high for cross-functional teams."

