"""
Matcher package public API.

Re-exports each pipeline function under the name expected by app.py, and
provides the `match_keywords_per_section` utility that checks requirement
keywords against resume sections (synonym-aware, lexical only).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from matcher.job_parser import parse as parse_job
from matcher.requirement_extractor import extract as extract_requirements
from matcher.matcher import score as match_score
from matcher.evidence_linker import link as link_evidence
from matcher.match_scorer import aggregate as aggregate_match
from matcher.explanation_generator import explain

__all__ = [
    "parse_job",
    "extract_requirements",
    "match_score",
    "link_evidence",
    "aggregate_match",
    "explain",
    "match_keywords_per_section",
]

# Sections considered "direct" evidence (skill listed or role demonstrated).
_DIRECT_SECTIONS = {"experience", "skills"}

def _word_in(text_low: str, needle_low: str) -> bool:
    """Word-bounded substring check; allows + and # inside tokens (C++, C#)."""
    if not text_low or not needle_low:
        return False
    pattern = (
        r"(?<![A-Za-z0-9+#])"
        + re.escape(needle_low)
        + r"(?![A-Za-z0-9+#])"
    )
    return bool(re.search(pattern, text_low))

def match_keywords_per_section(
    sections: Dict,
    requirements: List[Dict],
    use_semantic: bool = False,
) -> Dict:
    """Check whether each requirement keyword appears in the resume sections.

    For each requirement we probe the resume text with the keyword and all its
    known synonyms (loaded from ``skill_synonyms.json``).  A hit in an
    *experience* or *skills* section is classified as **direct**; a hit
    anywhere else is **inferred**; no hit is **missing**.

    Returns::

        {
            "summary": {"n_keywords": int, "n_with_hit": int},
            "rows": [
                {
                    "keyword":          str,
                    "is_required":      bool,
                    "best_match_type":  "direct" | "inferred" | "missing",
                    "matched_section":  str | None,
                },
                ...
            ],
        }
    """
    try:
        from matcher.section_keyword_matcher import _equivalents
    except Exception:
        def _equivalents(base: str):  # type: ignore[misc]
            return {base.strip().lower()} if base.strip() else set()

    # Pre-build concatenated text per section (lowercase).
    section_texts: Dict[str, str] = {}
    for sec_name, sec_data in sections.items():
        if sec_name == "_meta":
            continue
        lines = sec_data.get("content", []) if isinstance(sec_data, dict) else []
        section_texts[sec_name] = " ".join(str(ln) for ln in lines).lower()

    # Ordered scan: check direct sections first so we prefer "direct" over
    # "inferred" when the keyword appears in both.
    ordered_sections = sorted(
        section_texts.keys(),
        key=lambda s: (0 if s in _DIRECT_SECTIONS else 1),
    )

    rows: List[Dict] = []
    for req in requirements:
        keyword = req.get("name", "")
        is_required = bool(req.get("is_required", True))
        if not keyword:
            continue

        # Build synonym-expanded search terms, longest first.
        terms = list(_equivalents(keyword.lower()))
        if not terms:
            terms = [keyword.lower()]
        terms.sort(key=len, reverse=True)

        best_type = "missing"
        matched_section: Optional[str] = None
        per_section: Dict[str, float] = {}

        for sec_name in ordered_sections:
            text_low = section_texts[sec_name]
            n_hits = sum(1 for term in terms if _word_in(text_low, term))
            score = 0.0
            if n_hits > 0:
                # Direct sections weighted strongest; cap at 1.0.
                base = 1.0 if sec_name in _DIRECT_SECTIONS else 0.55
                bonus = min(0.35, 0.1 * (n_hits - 1))
                score = min(1.0, base + bonus)
            per_section[sec_name] = round(score, 3)
            if n_hits == 0:
                continue
            if sec_name in _DIRECT_SECTIONS:
                if best_type != "direct":
                    best_type = "direct"
                    matched_section = sec_name
            elif best_type == "missing":
                best_type = "inferred"
                matched_section = sec_name

        rows.append({
            "keyword":         keyword,
            "is_required":     is_required,
            "best_match_type": best_type,
            "matched_section": matched_section,
            "per_section":     per_section,
        })

    n_keywords = len(rows)
    n_with_hit = sum(1 for r in rows if r["best_match_type"] != "missing")
    section_keys = sorted(
        section_texts.keys(),
        key=lambda s: (0 if s in _DIRECT_SECTIONS else 1, s),
    )

    return {
        "summary": {
            "n_keywords": n_keywords,
            "n_with_hit": n_with_hit,
        },
        "section_keys": section_keys,
        "rows": rows,
    }
