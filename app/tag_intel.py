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
    "genai": "generative ai",
    "ai in health care": "ai in healthcare",
    "clinical decision-making support": "clinical decision support",
    "rct": "clinical trial",
    "randomized trial": "clinical trial",
    "non-pharmacological intervention": "nonpharmacological intervention",
    "peri-op outcomes": "perioperative outcomes",
    "colorectal cancer": "colon cancer",
    "gc b cells": "germinal center b cells",
    "madrigal": "madrigal pharmaceuticals",
    "u.s. fda approval": "fda approval",
}

DOMAIN_INDICATORS = {
    "Healthcare Delivery & Equity": {
        "healthcare access",
        "health disparities",
        "rural health",
        "health equity",
        "patient access",
        "care pathway",
        "quality improvement",
        "population health",
        "healthcare delivery",
    },
    "Clinical Oncology": {
        "oncology",
        "cancer",
        "tumor",
        "cancer surgery",
        "chemotherapy",
        "immunotherapy",
        "colon cancer",
    },
    "Biomedical Research & Vaccines": {
        "vaccine",
        "hiv vaccine",
        "immunology",
        "germinal center b cells",
        "preclinical",
        "assay",
        "dna origami",
    },
    "Biomedical Systems & Physiology": {
        "wearable device",
        "medical device",
        "human physiology",
        "gut microbiome",
        "microbial metabolism",
        "microbiome",
        "physiology",
        "metabolism",
        "flatulence",
    },
    "Aging & Brain Health": {
        "alzheimer's",
        "dementia",
        "aging",
        "brain health",
        "neurodegeneration",
    },
    "Pharma & Regulatory": {
        "fda approval",
        "regulatory",
        "label",
        "madrigal pharmaceuticals",
        "safety",
        "mash",
    },
    "AI/Computing & Platforms": {
        "generative ai",
        "machine learning",
        "ai computing",
        "data center",
        "data platform",
        "ai platform",
        "model",
        "inference",
    },
    "AI in Healthcare": {
        "ai in healthcare",
        "clinical decision support",
        "diagnostics",
        "health ai",
    },
}

DOMAIN_ORDER = list(DOMAIN_INDICATORS.keys())

STRONG_DOMAIN_INDICATORS = {
    "Healthcare Delivery & Equity": {
        "healthcare access",
        "health disparities",
        "rural health",
        "health equity",
    },
    "Clinical Oncology": {
        "oncology",
        "cancer surgery",
        "colon cancer",
    },
    "Biomedical Research & Vaccines": {
        "vaccine",
        "hiv vaccine",
        "dna origami",
        "germinal center b cells",
    },
    "Biomedical Systems & Physiology": {
        "medical device",
        "wearable device",
        "gut microbiome",
        "human physiology",
    },
    "Aging & Brain Health": {
        "alzheimer's",
        "dementia",
        "brain health",
    },
    "Pharma & Regulatory": {
        "fda approval",
        "regulatory",
        "madrigal pharmaceuticals",
        "mash",
    },
    "AI/Computing & Platforms": {
        "ai computing",
        "data center",
        "ai platform",
    },
    "AI in Healthcare": {
        "ai in healthcare",
        "clinical decision support",
        "health ai",
    },
}

WEAK_DOMAIN_KEYWORDS = {
    "Healthcare Delivery & Equity": {
        "access",
        "equity",
        "disparit",
        "rural",
        "delivery",
        "care pathway",
        "quality improvement",
        "population health",
    },
    "Clinical Oncology": {
        "oncolog",
        "cancer",
        "tumor",
        "chemo",
        "immuno",
        "radiation",
        "surgery",
        "metast",
    },
    "Biomedical Research & Vaccines": {
        "vaccine",
        "immun",
        "antibody",
        "preclinical",
        "assay",
        "clinical trial",
    },
    "Biomedical Systems & Physiology": {
        "device",
        "wearable",
        "physiology",
        "microbiome",
        "microbial",
        "metabolism",
        "gut",
        "flatulence",
    },
    "Aging & Brain Health": {
        "alzheimer",
        "dementia",
        "neuro",
        "brain",
        "aging",
    },
    "Pharma & Regulatory": {
        "fda",
        "regulator",
        "label",
        "approval",
        "pharma",
        "drug",
        "safety",
    },
    "AI/Computing & Platforms": {
        "ai",
        "machine learning",
        "ml",
        "model",
        "inference",
        "compute",
        "cloud",
        "platform",
        "data center",
    },
    "AI in Healthcare": {
        "ai",
        "clinical",
        "diagnostic",
        "health",
        "patient",
    },
}


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
        return "Unknown"

    def count_keyword_hits(keywords: Iterable[str]) -> int:
        hits = set()
        for tag in tags:
            for kw in keywords:
                if kw in tag:
                    hits.add(kw)
        return len(hits)

    best_domain = "Unknown"
    best_score = 0
    best_strong = 0
    for domain in DOMAIN_ORDER:
        indicators = DOMAIN_INDICATORS.get(domain, set())
        score = len(tags.intersection(indicators))
        strong_indicators = STRONG_DOMAIN_INDICATORS.get(domain, set())
        strong_score = len(tags.intersection(strong_indicators))
        weak_keywords = WEAK_DOMAIN_KEYWORDS.get(domain, set())
        weak_score = count_keyword_hits(weak_keywords)

        total_score = (score * 2) + weak_score

        # Prefer AI in Healthcare when both AI + health context appear together.
        if domain == "AI in Healthcare":
            has_ai = any("ai" in tag for tag in tags)
            has_health = any(
                any(term in tag for term in ("health", "clinical", "diagnostic", "patient"))
                for tag in tags
            )
            if has_ai and has_health:
                total_score += 2

        if total_score > best_score:
            best_score = total_score
            best_strong = strong_score
            best_domain = domain
        elif total_score == best_score and total_score > 0:
            if strong_score > best_strong:
                best_strong = strong_score
                best_domain = domain
            elif strong_score == best_strong:
                best_domain = "Unknown"

    return best_domain if best_score > 0 else "Unknown"


def persist_tag_summary(
    session: Session,
    document_id: int,
    job_id: int | None,
    raw_tags: Iterable[str],
) -> Dict[str, object]:
    raw_tags = list(raw_tags or [])
    if not raw_tags:
        raw_tags = ["general"]

    alias_map = load_alias_map(session)
    canonical_tags = canonicalize_tags(raw_tags, alias_map)
    domain = infer_domain(canonical_tags)

    delete_stmt = delete(DocumentTagSummary).where(
        DocumentTagSummary.document_id == document_id
    )
    if job_id is not None:
        delete_stmt = delete_stmt.where(DocumentTagSummary.job_id == job_id)
    session.exec(delete_stmt)

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
