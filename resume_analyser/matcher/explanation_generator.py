from __future__ import annotations

import re
from typing import Dict, List, Optional

def _candidate_name(parsed_resume: Dict) -> str:
    """Best-effort name detection from the resume's first non-empty line."""
    raw = (parsed_resume.get("_meta", {}).get("raw_text") or "").strip()
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Heuristic: name is short, capitalised, no punctuation other than dot/space
        if 2 <= len(line.split()) <= 5 and len(line) <= 60:
            words = line.split()
            if all(w[0].isupper() or w[0].isdigit() for w in words if w):
                return line
        break
    return "The candidate"

def _strongest_direct(links: List[Dict]) -> Optional[Dict]:
    direct_links = [l for l in links if l["link_type"] == "direct" and l["supporting_bullets"]]
    if not direct_links:
        return None
    direct_links.sort(key=lambda l: -l["strength"])
    return direct_links[0]

def _first_missing_required(links: List[Dict]) -> Optional[Dict]:
    for l in links:
        if l["link_type"] == "missing" and l["is_required"]:
            return l
    return None

def _first_inferred(links: List[Dict]) -> Optional[Dict]:
    for l in links:
        if l["link_type"] == "inferred":
            return l
    return None

def _quote(text: str, max_words: int = 22) -> str:
    """Truncate a bullet to a short citation, preserving word boundaries."""
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).rstrip(",;:.") + "…"

def _assert_grounded(citations: List[str], supporting_texts: List[str]) -> None:
    """Sanity check: every citation must be a substring (modulo trailing …)
    of at least one supporting bullet. Hard-asserts the no-hallucination
    invariant."""
    for c in citations:
        clean = c.rstrip("… ").strip()
        if not clean:
            continue
        if not any(_substring_lenient(clean, st) for st in supporting_texts):
            raise AssertionError(
                f"Citation not grounded in any supporting bullet: {c!r}"
            )

def _substring_lenient(needle: str, haystack: str) -> bool:
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
    return norm(needle) in norm(haystack)

def explain(
    parsed_resume: Dict,
    parsed_job: Dict,
    links: List[Dict],
    match_result: Dict,
) -> str:
    """Return a 3–4 sentence grounded explanation of the match."""
    candidate = _candidate_name(parsed_resume)
    job_title = parsed_job.get("title", "the role")
    label = match_result["label"]
    overall = match_result["overall_score"]

    sentences = [
        f"{candidate} is a {label} match for the {job_title} role "
        f"(overall score: {overall}/100)."
    ]

    citations: List[str] = []
    supporting_texts: List[str] = []

    best = _strongest_direct(links)
    if best:
        sb = best["supporting_bullets"][0]
        quote = _quote(sb["bullet_text"])
        citations.append(quote)
        supporting_texts.append(sb["bullet_text"])
        ctx = []
        if sb.get("role_title"):
            ctx.append(sb["role_title"])
        if sb.get("company"):
            ctx.append(sb["company"])
        ctx_str = ", ".join(ctx) if ctx else "the resume"
        sentences.append(
            f"Strongest evidence: {best['requirement_name']} demonstrated by "
            f"\"{quote}\" at {ctx_str}."
        )
    else:
        sentences.append("No directly evidenced requirements were found.")

    missing = _first_missing_required(links)
    if missing:
        sentences.append(
            f"Key gap: no evidence of {missing['requirement_name']} found in the resume."
        )
    else:
        inferred = _first_inferred(links)
        if inferred:
            sentences.append(
                f"Note: {inferred['requirement_name']} was inferred rather than "
                f"directly demonstrated and should be verified."
            )
        else:
            sentences.append("All required skills are directly evidenced.")

    if label == "high":
        sentences.append("Recommend progressing to interview.")
    elif label == "medium":
        sentences.append("Recommend a screening call to clarify gaps.")
    else:
        sentences.append("Likely not a fit for this role.")

    # Grounding assertion: only quoted spans need to be substring-grounded.
    _assert_grounded(citations, supporting_texts)

    return " ".join(sentences)
