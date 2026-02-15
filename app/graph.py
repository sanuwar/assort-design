from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List

from langgraph.graph import END, StateGraph
from sqlalchemy import text
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session, delete, select

from app.config import (
    get_audience_profile,
    get_evaluation_config,
    get_generation_config,
    get_ml_router_config,
    get_routing_config,
)
from app.ml_router import _ml_router
from app.llm import evaluate_content, generate_content, route_audience
from app.models import (
    Document,
    DocumentClaim,
    DocumentClue,
    DocumentRiskFlag,
    DocumentTag,
    Job,
    JobAttempt,
    Tag,
)
from app.tag_intel import persist_tag_summary
from app.tools import citation_finder, count_supported_citations, risk_checker

logger = logging.getLogger(__name__)

# ── Auto-retrain settings ──────────────────────────────────────────────────
_RETRAIN_EVERY_N_JOBS = int(os.getenv("RETRAIN_EVERY_N_JOBS", "20"))
_retrain_lock = threading.Lock()


def _maybe_retrain(session: Session) -> None:
    """Retrain the ML router if enough new completed jobs have accumulated
    since the last training run.  Runs in-process but is guarded by a lock
    so concurrent requests don't trigger duplicate training."""
    if _RETRAIN_EVERY_N_JOBS <= 0:
        return

    # How many completed specialist jobs exist in total?
    total_completed = session.exec(
        select(Job)
        .where(Job.status == "completed")
        .where(Job.audience.in_(["commercial", "medical_affairs", "r_and_d"]))  # type: ignore[union-attr]
    ).all()
    n_completed = len(total_completed)

    # Read the last training size from metadata.json
    from app.ml_router import ARTIFACTS_DIR

    metadata_file = ARTIFACTS_DIR / "metadata.json"
    last_n = 0
    if metadata_file.exists():
        try:
            with metadata_file.open("r", encoding="utf-8") as f:
                last_n = json.load(f).get("n_docs", 0)
        except Exception:
            pass

    new_since_training = n_completed - last_n
    if new_since_training < _RETRAIN_EVERY_N_JOBS:
        return

    # Guard against concurrent retrains
    if not _retrain_lock.acquire(blocking=False):
        return
    try:
        logger.info(
            "Auto-retrain triggered: %d new completed jobs since last training (%d total).",
            new_since_training,
            n_completed,
        )
        from app.train_router import train_model

        result = train_model(min_samples=10)
        if result:
            _ml_router.reload()
            logger.info("Auto-retrain complete. Model reloaded. n_docs=%d", result["n_docs"])
        else:
            logger.info("Auto-retrain skipped (not enough labelled samples).")
    except Exception:
        logger.exception("Auto-retrain failed — app continues with previous model.")
    finally:
        _retrain_lock.release()


def _get_pipeline_timeout() -> int:
    value = os.getenv("PIPELINE_TIMEOUT_SEC", "120")
    try:
        return int(value)
    except ValueError:
        return 120


PIPELINE_TIMEOUT_SEC = _get_pipeline_timeout()


def _build_graph(
    route_audience_node,
    specialist_generate_node,
    evaluate_node,
    tool_citation_node,
    tool_risk_node,
    tool_gate_node,
    persist_attempt_node,
    revise_node,
    persist_results_node,
    next_step,
) -> Any:
    graph = StateGraph(dict)
    graph.add_node("route_audience", route_audience_node)
    graph.add_node("specialist_generate", specialist_generate_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("tool_citation", tool_citation_node)
    graph.add_node("tool_risk", tool_risk_node)
    graph.add_node("tool_gate", tool_gate_node)
    graph.add_node("persist_attempt", persist_attempt_node)
    graph.add_node("revise", revise_node)
    graph.add_node("persist_results", persist_results_node)

    graph.set_entry_point("route_audience")
    graph.add_edge("route_audience", "specialist_generate")
    graph.add_edge("specialist_generate", "evaluate")
    graph.add_edge("evaluate", "tool_citation")
    graph.add_edge("tool_citation", "tool_risk")
    graph.add_edge("tool_risk", "tool_gate")
    graph.add_edge("tool_gate", "persist_attempt")
    graph.add_conditional_edges(
        "persist_attempt",
        next_step,
        {"revise": "revise", "persist_results": "persist_results"},
    )
    graph.add_edge("revise", "specialist_generate")
    graph.add_edge("persist_results", END)

    return graph.compile()


def run_job_pipeline(session: Session, job: Job) -> Job:
    """Run routing -> generate -> evaluate -> revise -> persist via LangGraph."""
    document = session.get(Document, job.document_id)
    if not document:
        raise ValueError("Document not found for job.")

    job.status = "running"
    session.add(job)
    session.commit()
    session.refresh(job)

    routing_config = get_routing_config()
    generation_prompt = get_generation_config()["prompt"]
    evaluator_prompt = get_evaluation_config()["evaluator_prompt"]

    def _ensure_not_timed_out(state: Dict[str, Any]) -> None:
        deadline = state.get("deadline")
        if deadline and time.monotonic() > deadline:
            raise TimeoutError("Job pipeline timed out.")

    def route_audience_node(state: Dict[str, Any]) -> Dict[str, Any]:
        _ensure_not_timed_out(state)
        new_state = dict(state)
        if job.selected_audience != "auto":
            audience = job.selected_audience
            job.routing_confidence = 1.0
            if audience == "cross_functional":
                job.routing_candidates_json = "[]"
            else:
                job.routing_candidates_json = json.dumps([audience])
            job.routing_reasons_json = json.dumps(["Manual override"])
            job.routing_source = "manual"
            job.router_version = None
            flag_modified(job, "routing_source")
            flag_modified(job, "router_version")
        else:
            ml_cfg = get_ml_router_config()
            threshold = ml_cfg["ml_router_threshold"]
            margin = ml_cfg["ml_router_margin"]

            ml_result = None
            routing_source = "llm"
            audience = "cross_functional"
            confidence = 0.0
            candidates: List[str] = []
            reasons: List[str] = []

            if _ml_router.load():
                try:
                    ml_result = _ml_router.predict(document.content, threshold, margin)
                    ml_audience = ml_result["audience"]

                    if ml_audience != "cross_functional":
                        # ML succeeded — use it directly, skip LLM
                        audience = ml_audience
                        routing_source = "ml"
                        confidence = ml_result["confidence"]
                        candidates = ml_result["candidates"]
                        signals_str = ", ".join(ml_result["top_signals"])
                        reasons = [
                            f"ML router {int(confidence * 100)}% — top signals: {signals_str}"
                        ]
                    else:
                        # ML uncertain → fallback to LLM
                        routing_source = "ml+llm_fallback"
                        fallback_note = ml_result.get("fallback_reason") or "ML uncertain"
                        existing_reasons = [f"ML fallback: {fallback_note}"]
                        audience, confidence, candidates, reasons = _parse_llm_routing(
                            existing_reasons, [], routing_config, document.content
                        )
                except Exception:
                    routing_source = "llm"
                    audience, confidence, candidates, reasons = _parse_llm_routing(
                        [], [], routing_config, document.content
                    )
            else:
                # No model artifacts — LLM only
                routing_source = "llm"
                audience, confidence, candidates, reasons = _parse_llm_routing(
                    [], [], routing_config, document.content
                )

            job.routing_confidence = confidence
            job.routing_candidates_json = json.dumps(candidates)
            job.routing_reasons_json = json.dumps(reasons[:5])
            job.routing_source = routing_source
            job.router_version = ml_result["router_version"] if ml_result else None
            flag_modified(job, "routing_source")
            flag_modified(job, "router_version")

        try:
            profile = get_audience_profile(audience)
        except ValueError:
            audience = "cross_functional"
            profile = get_audience_profile(audience)

        job.audience = audience
        session.add(job)
        session.commit()

        new_state.update(
            {
                "audience": audience,
                "required_sections": profile["required_sections"],
                "max_words": job.max_words or profile["default_max_words"],
                "system_prompt": profile["system_prompt"],
            }
        )
        return new_state

    def _parse_llm_routing(
        existing_reasons: List[str],
        existing_candidates: List[str],
        routing_cfg: Dict[str, Any],
        content: str,
    ) -> tuple:
        result = route_audience(routing_cfg["auto_router_prompt"], content)
        confidence = float(result.get("confidence") or 0.0)
        audience = result.get("audience", "cross_functional")
        raw_reasons = result.get("reasons")
        if isinstance(raw_reasons, list):
            reasons = [str(r).strip() for r in raw_reasons if str(r).strip()]
        else:
            reasons = []
        reasons = existing_reasons + reasons
        raw_candidates = result.get("candidates")
        allowed = {"commercial", "medical_affairs", "r_and_d"}
        if isinstance(raw_candidates, list):
            candidates = [c for c in raw_candidates if c in allowed]
        else:
            candidates = existing_candidates
        if not candidates and audience in allowed:
            candidates = [audience]
        llm_threshold = float(routing_cfg.get("low_confidence_threshold", 0.5))
        if confidence < llm_threshold:
            audience = "cross_functional"
            if not reasons:
                reasons = ["Low confidence: routed to cross-functional."]
        return audience, confidence, candidates, reasons

    def specialist_generate_node(state: Dict[str, Any]) -> Dict[str, Any]:
        _ensure_not_timed_out(state)
        generated = generate_content(
            system_prompt=state["system_prompt"],
            generation_prompt=generation_prompt,
            document_text=document.content,
            required_sections=state["required_sections"],
            max_words=state["max_words"],
            fix_instructions=state.get("fix_instructions", []),
        )
        new_state = dict(state)
        new_state["generated"] = generated
        return new_state

    def evaluate_node(state: Dict[str, Any]) -> Dict[str, Any]:
        _ensure_not_timed_out(state)
        generated = state.get("generated", {})
        one_line_summary = str(generated.get("one_line_summary", "")).strip()
        tags = _normalize_list(generated.get("tags"))
        clues = _normalize_list(generated.get("key_clues"))
        bullets = _normalize_list(generated.get("decision_bullets"))
        mind_map = str(generated.get("mind_map", "")).strip()
        if not mind_map:
            mind_map = _fallback_mind_map(state["required_sections"], bullets)

        evaluation = evaluate_content(
            evaluator_prompt=evaluator_prompt,
            one_line_summary=one_line_summary,
            decision_bullets=bullets,
            required_sections=state["required_sections"],
            max_words=state["max_words"],
        )
        passed = bool(evaluation.get("pass"))
        fix_instructions = _normalize_list(evaluation.get("fix_instructions"))

        new_state = dict(state)
        new_state.update(
            {
                "one_line_summary": one_line_summary,
                "tags": tags,
                "clues": clues,
                "bullets": bullets,
                "mind_map": mind_map,
                "evaluation": evaluation,
                "passed": passed,
                "fix_instructions": fix_instructions,
            }
        )
        return new_state

    def tool_citation_node(state: Dict[str, Any]) -> Dict[str, Any]:
        _ensure_not_timed_out(state)
        bullets = state.get("bullets", [])
        citations = citation_finder(document.content, bullets)
        new_state = dict(state)
        new_state["citations"] = citations
        return new_state

    def tool_risk_node(state: Dict[str, Any]) -> Dict[str, Any]:
        _ensure_not_timed_out(state)
        bullets = state.get("bullets", [])
        bullets_text = "\n".join([str(b) for b in bullets if str(b).strip()])
        # Run on source document AND generated bullets for richer detection.
        combined_text = "\n\n".join(filter(None, [document.content or "", bullets_text]))
        risks = risk_checker(combined_text)
        new_state = dict(state)
        new_state["risk_flags"] = risks
        return new_state

    def tool_gate_node(state: Dict[str, Any]) -> Dict[str, Any]:
        _ensure_not_timed_out(state)
        citations = state.get("citations", [])
        risks = state.get("risk_flags", [])
        supported_count = count_supported_citations(citations)
        high_risks = [flag for flag in risks if flag.severity == "high"]

        evaluation = dict(state.get("evaluation", {}))
        fix_instructions = list(state.get("fix_instructions", []))

        support_warning = supported_count < 3
        evaluation["support_warning"] = support_warning

        hard_fail = False
        if high_risks and supported_count == 0:
            hard_fail = True
            evaluation["pass"] = False
            evaluation["fail_reasons"] = list(
                evaluation.get("fail_reasons", [])
            ) + ["High-risk claims without supporting citations."]
            fix_instructions.append(
                "Add citations for strong claims or soften the language."
            )

        new_state = dict(state)
        new_state["evaluation"] = evaluation
        new_state["passed"] = bool(evaluation.get("pass")) and not hard_fail
        new_state["fix_instructions"] = fix_instructions
        new_state["support_warning"] = support_warning
        return new_state

    def persist_attempt_node(state: Dict[str, Any]) -> Dict[str, Any]:
        _ensure_not_timed_out(state)
        attempt_no = state["attempt_no"]
        attempt = JobAttempt(
            job_id=job.id,
            attempt_no=attempt_no,
            audience=state["audience"],
            agent_used=state["audience"],
            generated_one_line_summary=state["one_line_summary"],
            generated_tags_json=json.dumps(state["tags"]),
            generated_clues_json=json.dumps(state["clues"]),
            generated_bullets_json=json.dumps(state["bullets"]),
            generated_mindmap=state["mind_map"],
            evaluator_json=json.dumps(_ensure_json(state["evaluation"])),
            passed=state["passed"],
        )
        session.add(attempt)
        session.commit()
        return dict(state)

    def revise_node(state: Dict[str, Any]) -> Dict[str, Any]:
        _ensure_not_timed_out(state)
        new_state = dict(state)
        new_state["attempt_no"] = state["attempt_no"] + 1
        return new_state

    def persist_results_node(state: Dict[str, Any]) -> Dict[str, Any]:
        _ensure_not_timed_out(state)
        job.attempt_count = state["attempt_no"]
        job.status = "completed" if state["passed"] else "failed"
        session.add(job)
        session.commit()

        _replace_tags(session, document.id, state["tags"])
        _replace_clues(session, document.id, state["clues"])
        session.exec(
            delete(DocumentClaim).where(
                DocumentClaim.document_id == document.id,
                DocumentClaim.job_id == job.id,
            )
        )
        session.exec(
            delete(DocumentRiskFlag).where(
                DocumentRiskFlag.document_id == document.id,
                DocumentRiskFlag.job_id == job.id,
            )
        )
        for item in state.get("citations", []):
            session.add(
                DocumentClaim(
                    document_id=document.id,
                    job_id=job.id,
                    claim_text=item.claim_text,
                    quote_text=item.quote_text,
                    source_start=item.source_start,
                    source_end=item.source_end,
                    confidence=item.confidence,
                )
            )
        for item in state.get("risk_flags", []):
            session.add(
                DocumentRiskFlag(
                    document_id=document.id,
                    job_id=job.id,
                    severity=item.severity,
                    category=item.category,
                    text_span=item.text_span,
                    suggested_fix=item.suggested_fix,
                )
            )
        tag_summary = persist_tag_summary(
            session,
            document_id=document.id,
            job_id=job.id,
            raw_tags=state["tags"],
        )
        session.commit()
        new_state = dict(state)
        new_state.update(tag_summary)
        return new_state

    def next_step(state: Dict[str, Any]) -> str:
        _ensure_not_timed_out(state)
        if state.get("passed"):
            return "persist_results"
        if state["attempt_no"] <= state["max_retries"]:
            return "revise"
        return "persist_results"

    app = _build_graph(
        route_audience_node,
        specialist_generate_node,
        evaluate_node,
        tool_citation_node,
        tool_risk_node,
        tool_gate_node,
        persist_attempt_node,
        revise_node,
        persist_results_node,
        next_step,
    )
    try:
        app.invoke(
            {
                "attempt_no": 1,
                "max_retries": job.max_retries,
                "fix_instructions": [],
                "deadline": time.monotonic() + PIPELINE_TIMEOUT_SEC,
            }
        )
    except Exception as exc:
        reason = _format_failure_reason(exc)
        job.status = "failed"
        session.add(job)
        session.commit()

        failure_attempt = JobAttempt(
            job_id=job.id,
            attempt_no=job.attempt_count + 1,
            audience=job.audience or job.selected_audience or "auto",
            agent_used="system_error",
            generated_one_line_summary="",
            generated_tags_json="[]",
            generated_clues_json="[]",
            generated_bullets_json="[]",
            generated_mindmap="",
            evaluator_json=json.dumps(
                {
                    "pass": False,
                    "word_count": 0,
                    "missing_sections": [],
                    "fail_reasons": [reason],
                    "fix_instructions": [],
                }
            ),
            passed=False,
        )
        session.add(failure_attempt)
        session.commit()
        raise

    session.refresh(job)

    # Check if we should retrain the ML router (every N completed jobs)
    if job.status == "completed":
        try:
            _maybe_retrain(session)
        except Exception:
            logger.exception("Auto-retrain check failed — non-fatal, continuing.")

    return job


def build_graph_for_visualization() -> Any:
    """Build a no-op graph that mirrors the pipeline shape for visualization."""
    def _noop(state: Dict[str, Any]) -> Dict[str, Any]:
        return dict(state)

    def _next_step(_: Dict[str, Any]) -> str:
        return "persist_results"

    return _build_graph(
        _noop,
        _noop,
        _noop,
        _noop,
        _noop,
        _noop,
        _noop,
        _noop,
        _noop,
        _next_step,
    )


def _get_graph_object(compiled_graph: Any) -> Any:
    if hasattr(compiled_graph, "get_graph"):
        try:
            return compiled_graph.get_graph()
        except Exception:
            return compiled_graph
    return compiled_graph


def render_graph_mermaid() -> str:
    graph = build_graph_for_visualization()
    graph_obj = _get_graph_object(graph)
    if hasattr(graph_obj, "draw_mermaid"):
        try:
            return graph_obj.draw_mermaid()
        except Exception:
            pass
    return _fallback_mermaid()


def _fallback_mermaid() -> str:
    return "\n".join(
        [
            "%%{init: {\"flowchart\": {\"curve\": \"basis\", \"nodeSpacing\": 24, \"rankSpacing\": 30}, \"themeVariables\": {\"fontFamily\": \"Inter, ui-sans-serif, system-ui\", \"fontSize\": \"12px\"}}}%%",
            "flowchart TD",
            "  subgraph Routing[Routing]",
            "    route_source{[?] ML model available?}",
            "    ml_router[[ML Router]]",
            "    llm_router[[LLM Router]]",
            "    route_audience[[Audience Router]]",
            "  end",
            "  subgraph Generation[Generation]",
            "    specialist_generate[[Generate Artifacts]]",
            "    evaluate[[Evaluate Output]]",
            "  end",
            "  subgraph Tools[Tools]",
            "    tool_citation[[[Citations Tool]]]",
            "    tool_risk[[[Risk Checker]]]",
            "    tool_gate[[[Quality Gate]]]",
            "  end",
            "  subgraph Persistence[Persistence]",
            "    persist_attempt[[Persist Attempt]]",
            "    decision{[?] pass?}",
            "    revise[[Revise]]",
            "    persist_results[[Persist Results]]",
            "  end",
            "  route_source -->|yes| ml_router",
            "  route_source -->|no| llm_router",
            "  ml_router --> route_audience",
            "  llm_router --> route_audience",
            "  route_audience --> specialist_generate",
            "  specialist_generate --> evaluate",
            "  evaluate --> tool_citation",
            "  tool_citation --> tool_risk",
            "  tool_risk --> tool_gate",
            "  tool_gate --> persist_attempt",
            "  persist_attempt --> decision",
            "  decision -->|revise| revise",
            "  decision -->|persist_results| persist_results",
            "  revise --> specialist_generate",
            "  persist_results --> END",
            "  classDef llm fill:#E8F1FF,stroke:#4C6FFF,stroke-width:1px,color:#1E2A5A;",
            "  classDef ml fill:#E7FAF7,stroke:#20B2AA,stroke-width:1px,color:#0D3B3A;",
            "  classDef tool fill:#FFF6E5,stroke:#F4A340,stroke-width:1px,color:#6A3B00;",
            "  classDef persist fill:#E9F9EE,stroke:#3CB371,stroke-width:1px,color:#114B2F;",
            "  classDef control fill:#F2F2F2,stroke:#888,stroke-width:1px,color:#333;",
            "  classDef group fill:#F8FAFC,stroke:#CBD5E1,stroke-width:1px,color:#1F2937;",
            "  class route_audience,specialist_generate,evaluate llm;",
            "  class tool_citation,tool_risk,tool_gate tool;",
            "  class persist_attempt,persist_results persist;",
            "  class ml_router ml;",
            "  class llm_router llm;",
            "  class route_source,decision,revise control;",
            "  class Routing,Generation,Tools,Persistence group;",
            "  subgraph Legend[Legend]",
            "    legend_ml[[ML step]]",
            "    legend_llm[[LLM step]]",
            "    legend_tool[[Tool step]]",
            "    legend_persist[[Persistence]]",
            "    legend_control[[Decision/control]]",
            "  end",
            "  class legend_ml ml;",
            "  class legend_llm llm;",
            "  class legend_tool tool;",
            "  class legend_persist persist;",
            "  class legend_control control;",
        ]
    )


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Render the Assort Design LangGraph pipeline."
    )
    parser.add_argument(
        "--mermaid",
        action="store_true",
        help="Print Mermaid diagram text to stdout.",
    )
    parser.add_argument(
        "--png",
        metavar="PATH",
        help="Write a Mermaid PNG to the given path.",
    )
    args = parser.parse_args()

    if not args.mermaid and not args.png:
        args.mermaid = True

    graph = build_graph_for_visualization()
    graph_obj = _get_graph_object(graph)

    if args.png:
        png_bytes = None
        if hasattr(graph_obj, "draw_mermaid_png"):
            try:
                png_bytes = graph_obj.draw_mermaid_png()
            except Exception:
                png_bytes = None
        if png_bytes is None:
            raise SystemExit(
                "PNG rendering is unavailable here. Use --mermaid and preview with a Mermaid viewer."
            )
        with open(args.png, "wb") as handle:
            handle.write(png_bytes)

    if args.mermaid:
        sys.stdout.write(render_graph_mermaid())


def _replace_tags(session: Session, document_id: int, tags: List[str]) -> None:
    session.exec(delete(DocumentTag).where(DocumentTag.document_id == document_id))

    names: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        name = tag.strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)

    if not names:
        return

    existing_tags = session.exec(select(Tag).where(Tag.name.in_(names))).all()
    tags_by_name = {tag.name: tag for tag in existing_tags}
    missing = [name for name in names if name not in tags_by_name]

    for name in missing:
        session.exec(
            text("INSERT OR IGNORE INTO tag (name) VALUES (:name)").bindparams(
                name=name
            )
        )

    if missing:
        existing_tags = session.exec(select(Tag).where(Tag.name.in_(names))).all()
        tags_by_name = {tag.name: tag for tag in existing_tags}

    for name in names:
        tag_obj = tags_by_name.get(name)
        if tag_obj:
            session.add(DocumentTag(document_id=document_id, tag_id=tag_obj.id))


def _replace_clues(session: Session, document_id: int, clues: List[str]) -> None:
    session.exec(delete(DocumentClue).where(DocumentClue.document_id == document_id))

    for clue in clues:
        text = clue.strip()
        if not text:
            continue
        session.add(DocumentClue(document_id=document_id, clue_text=text))


def _normalize_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _ensure_json(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {
        "pass": False,
        "word_count": 0,
        "missing_sections": [],
        "fail_reasons": ["Invalid evaluator output."],
        "fix_instructions": [],
    }


def _fallback_mind_map(required_sections: List[str], bullets: List[str]) -> str:
    lines = ["mindmap", "  root((Summary))"]
    sources = bullets or required_sections
    for item in sources[:5]:
        title = str(item).split(":", 1)[0].strip()
        if not title:
            title = str(item).strip()[:40]
        if title:
            lines.append(f"    {title}")
    return "\n".join(lines)


def _format_failure_reason(exc: Exception) -> str:
    message = str(exc)
    lower = message.lower()

    if "insufficient_quota" in lower or "quota" in lower or "rate limit" in lower:
        return "OpenAI quota or rate limit exceeded."
    if "invalid_api_key" in lower or "api key" in lower or "authentication" in lower:
        return "OpenAI API key invalid or missing."
    if "model_not_found" in lower or "model" in lower and "not found" in lower:
        return "Requested model is unavailable."
    if "timeout" in lower or "timed out" in lower:
        return "LLM request timed out."
    return f"LLM request failed: {message}"

# keywords are huristic sanity check. 
def _filter_candidates_by_keywords(text: str, candidates: List[str]) -> List[str]:
    if not text or not candidates:
        return []

    keywords = {
        "commercial": [
            "market",
            "pricing",
            "revenue",
            "sales",
            "commercial",
            "launch",
            "positioning",
            "brand",
            "customer",
            "segment",
            "demand",
            "competition",
        ],
        "medical_affairs": [
            "clinical",
            "patient",
            "safety",
            "efficacy",
            "trial",
            "adverse",
            "label",
            "regulatory",
            "outcome",
            "physician",
            "kol",
            "publication",
        ],
        "r_and_d": [
            "experiment",
            "assay",
            "protocol",
            "method",
            "preclinical",
            "hypothesis",
            "mechanism",
            "in vivo",
            "in vitro",
            "model",
            "dataset",
            "lab",
        ],
    }

    haystack = text.lower()
    filtered = []
    for candidate in candidates:
        terms = keywords.get(candidate, [])
        if any(term in haystack for term in terms):
            filtered.append(candidate)
    return filtered
