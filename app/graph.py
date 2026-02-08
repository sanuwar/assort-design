from __future__ import annotations

import json
from typing import Any, Dict, List

from langgraph.graph import END, StateGraph
from sqlmodel import Session, delete, select

from app.config import (
    get_audience_profile,
    get_evaluation_config,
    get_generation_config,
    get_routing_config,
)
from app.llm import evaluate_content, generate_content, route_audience
from app.models import Document, DocumentClue, DocumentTag, Job, JobAttempt, Tag


def _build_graph(
    route_audience_node,
    specialist_generate_node,
    evaluate_node,
    persist_attempt_node,
    revise_node,
    persist_results_node,
    next_step,
) -> Any:
    graph = StateGraph(dict)
    graph.add_node("route_audience", route_audience_node)
    graph.add_node("specialist_generate", specialist_generate_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("persist_attempt", persist_attempt_node)
    graph.add_node("revise", revise_node)
    graph.add_node("persist_results", persist_results_node)

    graph.set_entry_point("route_audience")
    graph.add_edge("route_audience", "specialist_generate")
    graph.add_edge("specialist_generate", "evaluate")
    graph.add_edge("evaluate", "persist_attempt")
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

    def route_audience_node(state: Dict[str, Any]) -> Dict[str, Any]:
        new_state = dict(state)
        if job.selected_audience != "auto":
            audience = job.selected_audience
            job.routing_confidence = 1.0
            if audience == "cross_functional":
                job.routing_candidates_json = "[]"
            else:
                job.routing_candidates_json = json.dumps([audience])
            job.routing_reasons_json = json.dumps(["Manual override"])
        else:
            result = route_audience(routing_config["auto_router_prompt"], document.content)
            confidence = float(result.get("confidence") or 0.0)
            audience = result.get("audience", "cross_functional")
            reasons = result.get("reasons")
            if isinstance(reasons, list):
                reasons = [str(r).strip() for r in reasons if str(r).strip()]
            else:
                reasons = []
            candidates = result.get("candidates")
            allowed = {"commercial", "medical_affairs", "r_and_d"}
            if isinstance(candidates, list):
                candidates = [c for c in candidates if c in allowed]
            else:
                candidates = []
            candidates = _filter_candidates_by_keywords(document.content, candidates)
            if not candidates and audience in allowed:
                candidates = _filter_candidates_by_keywords(document.content, [audience])
            threshold = float(routing_config.get("low_confidence_threshold", 0.5))
            if confidence < threshold:
                audience = "cross_functional"
                if not reasons:
                    reasons = ["Low confidence: routed to cross-functional."]
            job.routing_confidence = confidence
            job.routing_candidates_json = json.dumps(candidates)
            job.routing_reasons_json = json.dumps(reasons[:5])
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

    def specialist_generate_node(state: Dict[str, Any]) -> Dict[str, Any]:
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

    def persist_attempt_node(state: Dict[str, Any]) -> Dict[str, Any]:
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
        new_state = dict(state)
        new_state["attempt_no"] = state["attempt_no"] + 1
        return new_state

    def persist_results_node(state: Dict[str, Any]) -> Dict[str, Any]:
        job.attempt_count = state["attempt_no"]
        job.status = "completed" if state["passed"] else "failed"
        session.add(job)
        session.commit()

        _replace_tags(session, document.id, state["tags"])
        _replace_clues(session, document.id, state["clues"])
        session.commit()
        return dict(state)

    def next_step(state: Dict[str, Any]) -> str:
        if state.get("passed"):
            return "persist_results"
        if state["attempt_no"] <= state["max_retries"]:
            return "revise"
        return "persist_results"

    app = _build_graph(
        route_audience_node,
        specialist_generate_node,
        evaluate_node,
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
            "flowchart TD",
            "  route_audience[route_audience] --> specialist_generate[specialist_generate]",
            "  specialist_generate --> evaluate",
            "  evaluate --> persist_attempt",
            "  persist_attempt -->|revise| revise",
            "  persist_attempt -->|persist_results| persist_results",
            "  revise --> specialist_generate",
            "  persist_results --> END",
        ]
    )


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Render the Asort Design LangGraph pipeline."
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

    for tag in tags:
        name = tag.strip().lower()
        if not name:
            continue

        existing = session.exec(select(Tag).where(Tag.name == name)).first()
        if not existing:
            existing = Tag(name=name)
            session.add(existing)
            session.commit()
            session.refresh(existing)

        session.add(DocumentTag(document_id=document_id, tag_id=existing.id))


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
