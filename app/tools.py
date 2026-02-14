from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional


_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?\n]?")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
MAX_QUOTE_CHARS = 320


@dataclass
class CitationResult:
    claim_text: str
    quote_text: str
    source_start: Optional[int]
    source_end: Optional[int]
    confidence: float


@dataclass
class RiskFlag:
    severity: str
    category: str
    text_span: str
    suggested_fix: str


def _sentences_with_offsets(text: str) -> List[tuple[str, int, int]]:
    sentences: List[tuple[str, int, int]] = []
    index = 0
    for match in _SENTENCE_RE.finditer(text):
        sentence = match.group(0)
        start = match.start()
        end = match.end()
        if sentence.strip():
            sentences.append((sentence.strip(), start, end))
        index = end
    if index < len(text):
        tail = text[index:].strip()
        if tail:
            sentences.append((tail, index, len(text)))
    return sentences


def _tokenize(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(value.lower()))


def _find_focus(sentence: str, claim_tokens: set[str]) -> int:
    lower = sentence.lower()
    indices = []
    for token in claim_tokens:
        if not token:
            continue
        idx = lower.find(token)
        if idx != -1:
            indices.append(idx)
    if indices:
        return min(indices)
    return max(0, len(sentence) // 2)


def _trim_span(text: str, focus_index: int, max_len: int) -> tuple[str, int, int]:
    if len(text) <= max_len:
        return text, 0, len(text)
    half = max_len // 2
    start = max(0, focus_index - half)
    end = min(len(text), start + max_len)
    start = max(0, end - max_len)
    return text[start:end], start, end


def citation_finder(source_text: str, claims: Iterable[str]) -> List[CitationResult]:
    source_text = source_text or ""
    sentences = _sentences_with_offsets(source_text)
    results: List[CitationResult] = []
    claims_list = [c for c in (claims or []) if str(c).strip()]
    if not claims_list:
        return results

    for claim in claims_list[:8]:
        claim_text = str(claim).strip()
        if not claim_text:
            continue
        claim_tokens = _tokenize(claim_text)
        best = None
        best_score = 0.0

        # Exact substring match first.
        idx = source_text.lower().find(claim_text.lower())
        if idx != -1:
            quote = source_text[idx : idx + len(claim_text)]
            trimmed, t_start, t_end = _trim_span(
                quote, max(0, len(quote) // 2), MAX_QUOTE_CHARS
            )
            results.append(
                CitationResult(
                    claim_text=claim_text,
                    quote_text=trimmed.strip(),
                    source_start=idx + t_start,
                    source_end=idx + t_end,
                    confidence=0.9,
                )
            )
            continue

        for sentence, start, end in sentences:
            sentence_tokens = _tokenize(sentence)
            if not sentence_tokens:
                continue
            overlap = claim_tokens.intersection(sentence_tokens)
            score = len(overlap) / max(1, len(claim_tokens))
            if score > best_score:
                best_score = score
                best = (sentence, start, end)

        if best and best_score >= 0.25:
            quote, start, end = best
            focus = _find_focus(quote, claim_tokens)
            trimmed, t_start, t_end = _trim_span(quote, focus, MAX_QUOTE_CHARS)
            results.append(
                CitationResult(
                    claim_text=claim_text,
                    quote_text=trimmed.strip(),
                    source_start=start + t_start,
                    source_end=start + t_end,
                    confidence=min(0.85, 0.4 + best_score),
                )
            )
        else:
            results.append(
                CitationResult(
                    claim_text=claim_text,
                    quote_text="",
                    source_start=None,
                    source_end=None,
                    confidence=0.2,
                )
            )

    return results[:8]


def count_supported_citations(citations: Iterable[CitationResult]) -> int:
    count = 0
    for item in citations:
        if item.quote_text and item.confidence >= 0.4:
            count += 1
    return count


def risk_checker(text: str) -> List[RiskFlag]:  # noqa: C901
    output = text or ""
    flags: List[RiskFlag] = []
    seen: set[str] = set()

    def _add(severity: str, category: str, span: str, fix: str) -> None:
        key = f"{category}:{span[:50].lower()}"
        if key not in seen:
            seen.add(key)
            flags.append(RiskFlag(severity=severity, category=category, text_span=span, suggested_fix=fix))

    # ── HIGH: absolute efficacy / outcome claims ──────────────────────────────
    _abs_outcome = re.compile(
        r"\b("
        r"will\s+(cure|eliminate|eradicate|prevent\s+all|guarantee)"
        r"|100\s*%\s*(response|efficacy|success|cure|effective)"
        r"|clinically\s+proven\s+to\s+(cure|eliminate|prevent)"
        r"|guarantees?\s+(efficacy|safety|response|cure)"
        r"|always\s+works?|never\s+fails?"
        r")\b",
        re.I,
    )
    for m in _abs_outcome.finditer(output):
        _add("high", "absolute efficacy claim", m.group(0),
             "Remove absolute language; cite supporting evidence and add uncertainty qualifiers.")

    # ── HIGH: off-label promotion indicators ─────────────────────────────────
    _off_label = re.compile(
        r"\b("
        r"off[\-\s]label"
        r"|unapproved\s+indication"
        r"|not\s+(yet\s+)?approved\s+for\s+\w+"
        r"|expand(ed|ing)?\s+(use|indication)\s+(beyond|outside)"
        r"|outside\s+(the\s+)?approved\s+(label|indication)"
        r"|use\s+in\s+(children|pediatric|paediatric)\s+\w+\s+not\s+approved"
        r")\b",
        re.I,
    )
    for m in _off_label.finditer(output):
        _add("high", "off-label promotion", m.group(0),
             "Remove off-label promotion; ensure all claims are within the approved label indication.")

    # ── HIGH: black-box / serious-safety minimisation ────────────────────────
    _downplay = re.compile(
        r"\b(minor|minimal|negligible|rare|well[\-\s]tolerated)\s+(side[\s\-]effects?|adverse\s+event|risk|concern)\b",
        re.I,
    )
    for m in _downplay.finditer(output):
        _add("high", "safety minimisation", m.group(0),
             "Do not minimise adverse events; include the complete safety profile and any boxed warnings.")

    _blackbox = re.compile(
        r"\b(black[\s\-]?box\s+warning|boxed\s+warning)\b", re.I
    )
    for m in _blackbox.finditer(output):
        _add("high", "boxed warning present", m.group(0),
             "A boxed warning is referenced — ensure it is prominently disclosed and not downplayed.")

    # ── HIGH: residual over-certain language ─────────────────────────────────
    _absolute = re.compile(r"\b(guarantees?|always|never|proves?\s+that)\b", re.I)
    for m in _absolute.finditer(output):
        _add("high", "over-certain language", m.group(0),
             "Add uncertainty qualifiers (e.g., may/might) or back the claim with a citation.")

    # ── MEDIUM: unsupported comparative claims ────────────────────────────────
    _comparative = re.compile(
        r"\b("
        r"best[\-\s]in[\-\s]class|best\s+(available|option|choice)"
        r"|superior(\s+to)?|outperforms?"
        r"|better\s+than|more\s+effective\s+than"
        r"|gold\s+standard"
        r"|first[\-\s]in[\-\s]class"
        r"|only\s+(drug|therapy|treatment|agent|option)"
        r")\b",
        re.I,
    )
    for m in _comparative.finditer(output):
        _add("medium", "unsupported comparative claim", m.group(0),
             "Specify the head-to-head data source. Cross-trial indirect comparisons must be labelled as such.")

    # ── MEDIUM: cross-trial comparison without head-to-head context ───────────
    _cross_trial = re.compile(
        r"\b(compared\s+(?:favorably\s+)?(?:to|with)\s+\w+|vs\.?\s+\w+|versus\s+\w+)\b.{0,100}"
        r"\b(trial|study|arm|cohort|data)\b",
        re.I,
    )
    for m in _cross_trial.finditer(output):
        snippet = m.group(0)[:90]
        _add("medium", "cross-trial comparison", snippet,
             "Label this as an indirect/cross-trial comparison. It is not equivalent to head-to-head evidence.")

    # ── MEDIUM: patient testimonials used as clinical evidence ────────────────
    _testimonial = re.compile(
        r"\b("
        r"patient(s)?\s+(report(ed|s)?|say(s)?|claim(s)?|described|noted|felt)"
        r"|anecdotal\s+evidence"
        r"|personal\s+experience\s+shows?"
        r"|patient\s+(testimonial|account|story)"
        r"|patient\s+feedback\s+suggests?"
        r")\b",
        re.I,
    )
    for m in _testimonial.finditer(output):
        _add("medium", "patient testimonial as evidence", m.group(0),
             "Patient anecdotes are not clinical evidence; cite controlled trial data instead.")

    # ── MEDIUM: absolute percentage claims without study context ─────────────
    _pct = re.compile(
        r"\b(100\s*%\s*(response|efficacy|success|survival|reduction)"
        r"|zero\s+(adverse|side)\s*events?"
        r"|0\s*%\s*(adverse|side)\s*events?)\b",
        re.I,
    )
    for m in _pct.finditer(output):
        _add("medium", "absolute percentage claim", m.group(0),
             "Absolute percentages require a citation with trial name, population size, and confidence interval.")

    # ── MEDIUM: unapproved paediatric / special-population use ───────────────
    _paeds = re.compile(
        r"\b(pediatric|paediatric|children|child|adolescent|neonatal).{0,60}"
        r"\b(use|treat(ment|ed)?|safe|efficacy|dose|dosing)\b",
        re.I,
    )
    for m in _paeds.finditer(output):
        snippet = m.group(0)[:90]
        _add("medium", "special population / unapproved use", snippet,
             "Confirm this population is within the approved label; otherwise this may constitute off-label promotion.")

    # ── LOW: missing limitation / uncertainty language ────────────────────────
    _caution = re.compile(
        r"\b(may|might|could|suggests?|limited|preliminary|potential|approximately|estimated|appears?\s+to)\b",
        re.I,
    )
    if output and not _caution.search(output):
        _add("low", "missing limitation language", "No limitation language detected.",
             "Add a limitation or uncertainty qualifier (e.g., 'data suggest', 'preliminary findings').")

    # ── LOW: efficacy claimed without any safety / adverse-event language ─────
    _efficacy = re.compile(r"\b(efficacy|effective|response\s+rate|clinical\s+benefit|positive\s+outcome)\b", re.I)
    _safety = re.compile(r"\b(adverse|side[\s\-]?effect|safety|tolerab|risk|warning|contraindic|toxicity)\b", re.I)
    if _efficacy.search(output) and not _safety.search(output):
        _add("low", "missing safety language",
             "Efficacy claimed without accompanying safety context.",
             "Include adverse event profile, safety data, or disclaimers alongside efficacy claims.")

    # ── LOW: undisclosed conflicts of interest / KOL payments ────────────────
    _disclosure = re.compile(
        r"\b("
        r"key\s+opinion\s+leader|kol"
        r"|speaker\s+(bureau|fee|honorari)"
        r"|paid\s+(consultant|advisor|speaker)"
        r"|financial\s+(relationship|arrangement|support|interest)"
        r"|advisory\s+board"
        r")\b",
        re.I,
    )
    for m in _disclosure.finditer(output):
        _add("low", "conflict of interest / disclosure", m.group(0),
             "Ensure all financial relationships and conflicts of interest are fully disclosed per applicable guidelines.")

    return flags[:12]
