from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

from sqlmodel import Session, delete, select

from app.models import DocumentTagSummary, TagAlias

_TRAILING_PUNCT_RE = re.compile(r"[ \t\r\n\.,;:!\?\)\]\}]+$")
_WHITESPACE_RE = re.compile(r"\s+")


BUILTIN_ALIAS_MAP = {
    # General
    "genai": "generative ai",
    "ai in health care": "ai in healthcare",
    "clinical decision-making support": "clinical decision support",
    "rct": "clinical trial",
    "randomized trial": "clinical trial",
    "non-pharmacological intervention": "nonpharmacological intervention",
    "peri-op outcomes": "perioperative outcomes",
    "colorectal cancer": "colon cancer",
    "gc b cells": "germinal center b cells",
    "u.s. fda approval": "fda approval",
    # Company / product
    "madrigal": "madrigal pharmaceuticals",
    "rezdiffra": "rezdiffra (resmetirom)",
    "resmetirom": "rezdiffra (resmetirom)",
    # Regulatory / region
    "ema approval": "regulatory approval (ema)",
    "eu approval": "regulatory approval (eu)",
    "european commission approval": "regulatory approval (eu)",
    # Trial synonyms
    "clinical trials": "clinical trial",
    "phase 3 trial": "clinical trial (phase 3)",
    "maestro-nafld-1": "maestro-nafld-1 trial",
    # Disease / outcomes
    "liver fibrosis": "fibrosis",
    "fibrosis treatment": "fibrosis",
    # Modality / mechanism
    "dgat-2 inhibitor": "dgat2 inhibitor",
    "pharmaceutical licensing": "licensing",
    "pharmaceutical launch": "launch",
}

# ── Domain lanes (department-aligned) ──────────────────────────────────────
# Tie-break priority order: Corporate > Regulatory > Clinical > R&D > General
DOMAIN_ORDER = [
    "Corporate & Investor Updates",
    "Regulatory, Launch & Market Strategy",
    "Clinical & Medical Strategy",
    "Translational Science & Drug R&D",
]

STRONG_DOMAIN_INDICATORS = {
    "Clinical & Medical Strategy": {
        "clinical trial",
        "clinical trial (phase 3)",
        "maestro-nafld-1 trial",
        "portal hypertension",
        "cirrhosis",
        "quality of life",
        "fibrosis",
        "endpoint",
        "safety",
        "efficacy",
    },
    "Translational Science & Drug R&D": {
        "sirna",
        "gene silencing",
        "pharmacology",
        "drug development",
        "dgat2 inhibitor",
        "ervogastat",
        "syh2086",
        "combination therapy",
    },
    "Regulatory, Launch & Market Strategy": {
        "regulatory approval (ema)",
        "regulatory approval (eu)",
        "fda approval",
        "launch",
        "licensing",
    },
    "Corporate & Investor Updates": {
        "nasdaq",
        "equity awards",
        "employee inducement",
    },
}

WEAK_DOMAIN_INDICATORS = {
    "Clinical & Medical Strategy": {
        "mash",
        "liver disease",
        "rezdiffra (resmetirom)",
        "patient",
        "outcomes",
    },
    "Translational Science & Drug R&D": {
        "mash",
        "liver disease",
        "rezdiffra (resmetirom)",
    },
    "Regulatory, Launch & Market Strategy": {
        "madrigal pharmaceuticals",
        "rezdiffra (resmetirom)",
        "biopharmaceuticals",
    },
    "Corporate & Investor Updates": {
        "madrigal pharmaceuticals",
        "biopharmaceuticals",
    },
}

FALLBACK_DOMAIN = "General / Other"


def normalize_tag(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = text.replace("’", "'")
    text = _WHITESPACE_RE.sub(" ", text)
    text = _TRAILING_PUNCT_RE.sub("", text)
    return text.strip()


def _normalized_alias_map(raw_map: Dict[str, str]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for alias, canonical in raw_map.items():
        alias_norm = normalize_tag(alias)
        canonical_norm = normalize_tag(canonical)
        if alias_norm and canonical_norm:
            normalized[alias_norm] = canonical_norm
    return normalized


def load_alias_map(session: Session) -> Dict[str, str]:
    alias_map = _normalized_alias_map(BUILTIN_ALIAS_MAP)
    rows = session.exec(select(TagAlias)).all()
    for row in rows:
        alias_norm = normalize_tag(row.alias)
        canonical_norm = normalize_tag(row.canonical)
        if alias_norm and canonical_norm:
            alias_map[alias_norm] = canonical_norm
    return alias_map


def canonicalize_tags(raw_tags: Iterable[str], alias_map: Dict[str, str]) -> List[str]:
    canonical: List[str] = []
    seen = set()
    for tag in raw_tags:
        normalized = normalize_tag(tag)
        if not normalized:
            continue
        canonical_value = alias_map.get(normalized, normalized)
        if canonical_value in seen:
            continue
        seen.add(canonical_value)
        canonical.append(canonical_value)
    return canonical


def infer_domain(canonical_tags: Iterable[str]) -> str:
    tags = set(canonical_tags)
    if not tags:
        return FALLBACK_DOMAIN

    # Score each lane: strong × 3, weak × 1
    scores: Dict[str, int] = {}
    strong_hits: Dict[str, int] = {}
    for domain in DOMAIN_ORDER:
        s_indicators = STRONG_DOMAIN_INDICATORS.get(domain, set())
        w_indicators = WEAK_DOMAIN_INDICATORS.get(domain, set())
        s_count = len(tags.intersection(s_indicators))
        w_count = len(tags.intersection(w_indicators))
        strong_hits[domain] = s_count
        scores[domain] = s_count * 3 + w_count

    # Combination therapy context bonus
    if "combination therapy" in tags:
        if tags.intersection({"licensing", "launch"}):
            scores["Regulatory, Launch & Market Strategy"] += 3
        if tags.intersection({"dgat2 inhibitor", "sirna", "gene silencing", "pharmacology"}):
            scores["Translational Science & Drug R&D"] += 3

    # Pick best lane; DOMAIN_ORDER encodes tie-break priority
    best_domain = FALLBACK_DOMAIN
    best_score = 0
    for domain in DOMAIN_ORDER:
        if scores[domain] > best_score:
            best_score = scores[domain]
            best_domain = domain

    # Madrigal anchor rule: mash + rezdiffra present but score is low/tied
    if best_score <= 1 and {"mash", "rezdiffra (resmetirom)"}.issubset(tags):
        if strong_hits.get("Clinical & Medical Strategy", 0) > 0:
            return "Clinical & Medical Strategy"
        if strong_hits.get("Regulatory, Launch & Market Strategy", 0) > 0:
            return "Regulatory, Launch & Market Strategy"
        return "Regulatory, Launch & Market Strategy"

    return best_domain if best_score > 0 else FALLBACK_DOMAIN


def persist_tag_summary(
    session: Session,
    document_id: int,
    job_id: int,
    raw_tags: Iterable[str],
) -> Dict[str, object]:
    if job_id is None:
        raise ValueError(
            "persist_tag_summary requires a non-None job_id to prevent "
            "an unscoped DELETE that would wipe all summaries for the document."
        )
    raw_tags = list(raw_tags or [])
    if not raw_tags:
        raw_tags = ["general"]

    alias_map = load_alias_map(session)
    canonical_tags = canonicalize_tags(raw_tags, alias_map)
    domain = infer_domain(canonical_tags)

    session.exec(
        delete(DocumentTagSummary).where(
            DocumentTagSummary.document_id == document_id,
            DocumentTagSummary.job_id == job_id,
        )
    )

    summary = DocumentTagSummary(
        document_id=document_id,
        job_id=job_id,
        domain=domain,
        canonical_tags_json=json.dumps(canonical_tags),
    )
    session.add(summary)
    return {"domain": domain, "canonical_tags": canonical_tags}


def parse_summary_tags(summary: DocumentTagSummary) -> List[str]:
    try:
        data = json.loads(summary.canonical_tags_json or "[]")
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [normalize_tag(tag) for tag in data if normalize_tag(tag)]
    return []


def compute_jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 0.0
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    return len(intersection) / max(1, len(union))


def count_domains(summaries: Iterable[DocumentTagSummary]) -> List[Tuple[str, int]]:
    counts = Counter(summary.domain for summary in summaries if summary.domain)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def count_tags(summaries: Iterable[DocumentTagSummary]) -> Counter:
    counter: Counter = Counter()
    for summary in summaries:
        counter.update(parse_summary_tags(summary))
    return counter


def compute_cooccurrence_pairs(
    summaries: Iterable[DocumentTagSummary],
) -> List[Tuple[Tuple[str, str], int]]:
    pair_counts: Counter = Counter()
    for summary in summaries:
        tags = parse_summary_tags(summary)
        unique_tags = sorted(set(tags))
        for i in range(len(unique_tags)):
            for j in range(i + 1, len(unique_tags)):
                pair_counts[(unique_tags[i], unique_tags[j])] += 1
    return pair_counts.most_common(15)


def compute_bridge_tags(
    summaries: Iterable[DocumentTagSummary],
) -> List[Tuple[str, int]]:
    tag_domains: dict[str, set[str]] = defaultdict(set)
    for summary in summaries:
        tags = parse_summary_tags(summary)
        for tag in tags:
            if summary.domain:
                tag_domains[tag].add(summary.domain)

    bridge = [
        (tag, len(domains))
        for tag, domains in tag_domains.items()
        if len(domains) >= 2
    ]
    return sorted(bridge, key=lambda item: (-item[1], item[0]))[:15]
