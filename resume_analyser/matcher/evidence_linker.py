from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

# Initial calibration thresholds. Documented in the dissertation's eval section.
DIRECT_THRESHOLD = 0.80
INFERRED_THRESHOLD = 0.55

# Sections worth treating as evidence sources, in priority order.
_EVIDENCE_SECTIONS = (
    "skills", "experience", "projects", "open_source", "summary",
    "certifications", "education", "training", "publications",
)

# Strength of evidence by section. Hits in EXPERIENCE or SKILLS are *direct*
# (the candidate has actually performed or explicitly listed the skill);
# hits in PROJECTS, CERTIFICATIONS, OPEN_SOURCE, etc. are *inferred* (a
# weaker signal — they suggest competence but don't prove sustained work
# experience). Sections not listed default to "inferred".
_DIRECT_SECTIONS = {"experience", "skills"}
_INFERRED_SECTIONS = {
    "projects", "certifications", "open_source", "summary", "education",
    "training", "publications", "awards", "volunteer", "languages",
}

def _classify_by_section(section_name: Optional[str]) -> str:
    """Return 'direct' if the section is Experience/Skills, else 'inferred'."""
    if section_name and section_name.lower() in _DIRECT_SECTIONS:
        return "direct"
    return "inferred"

def _name_in_text(req_name: str, text: str) -> bool:
    if not req_name or not text:
        return False
    pattern = r"(?<![A-Za-z0-9])" + re.escape(req_name.lower()) + r"(?![A-Za-z0-9])"
    return re.search(pattern, text.lower()) is not None

def _word_in(text_low: str, needle_low: str) -> bool:
    """Word-bounded substring check (with `+` and `#` allowed inside tokens
    so 'C++' / 'C#' don't get split)."""
    if not text_low or not needle_low:
        return False
    pattern = (
        r"(?<![A-Za-z0-9+#])"
        + re.escape(needle_low)
        + r"(?![A-Za-z0-9+#])"
    )
    return re.search(pattern, text_low) is not None

_TOKEN_STOP = {
    "and", "or", "the", "a", "an", "of", "in", "on", "at", "for", "with",
    "to", "from", "is", "are", "be", "have", "has", "had", "do", "does",
    "must", "should", "will", "would", "you", "your", "we", "our",
    "experience", "knowledge", "background", "ability", "able", "good",
    "strong", "solid", "excellent", "deep", "expertise", "proven",
    "track", "record", "hands", "on", "hands-on", "familiar", "familiarity",
    "understanding", "preferred", "required", "nice", "have",
    "year", "years", "yrs",
}

def _variants(token_low: str) -> List[str]:
    """Add naive singular/plural variants to widen exact-match coverage."""
    out = {token_low}
    if token_low.endswith("s") and len(token_low) > 3:
        out.add(token_low[:-1])
    elif len(token_low) >= 3 and not token_low.endswith("s"):
        out.add(token_low + "s")
    return list(out)

def _resolve_synonyms(req_name: str) -> List[str]:
    """Return the synonym set for a requirement name, lower-cased.
    Falls back to just the requirement itself if the lexicon is unavailable."""
    base = (req_name or "").strip().lower()
    if not base:
        return []
    try:
        from .section_keyword_matcher import _equivalents
        return list(_equivalents(base))
    except Exception:
        return [base]

def _build_search_terms(req: Dict) -> List[str]:
    """Comprehensive list of terms to search the resume for.

    Combines:
      1. The requirement name + its synonym set.
      2. For 'domain' / 'soft' requirements, the specific keywords detected
         inside req.raw_text (e.g. 'fintech', 'payments', 'communication').
      3. Lower-case content tokens of the requirement name, with stopwords
         removed and singular/plural variants added.
    Order is preserved (longest first) so multi-word terms match before
    their single-word components.
    """
    terms: List[str] = []
    seen = set()

    def _add(t: str):
        t = t.strip().lower()
        if t and len(t) >= 2 and t not in seen:
            seen.add(t)
            terms.append(t)

    name = (req.get("name") or "").strip()
    raw = (req.get("raw_text") or "").lower()

    for s in _resolve_synonyms(name):
        _add(s)

    cat = req.get("category", "skill")
    try:
        from .requirement_extractor import DOMAIN_KEYWORDS, SOFT_KEYWORDS
    except Exception:
        DOMAIN_KEYWORDS, SOFT_KEYWORDS = set(), set()
    if cat == "domain":
        for kw in DOMAIN_KEYWORDS:
            if kw in raw:
                for v in _variants(kw):
                    _add(v)
    elif cat == "soft":
        for kw in SOFT_KEYWORDS:
            if kw in raw:
                for v in _variants(kw):
                    _add(v)

    name_low = name.lower()
    if len(name_low.split()) > 1:
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9+#\.\-]{1,}", name_low):
            if tok in _TOKEN_STOP or len(tok) < 3:
                continue
            for v in _variants(tok):
                _add(v)

    # Sort longer terms first so 'machine learning' beats 'learning' when
    # iterating for a hit.
    terms.sort(key=len, reverse=True)
    return terms

def _scan_bullets_for_terms(
    terms: List[str], bullets: List[Dict]
) -> Optional[Tuple[int, str]]:
    """Scan raw bullets for any of the search terms. Returns
    (bullet_index, matched_alias) or None."""
    if not terms:
        return None
    for idx, b in enumerate(bullets):
        text_low = (b.get("text") or "").lower()
        for term in terms:
            if _word_in(text_low, term):
                return idx, term
    return None

def _scan_sections_for_terms(
    terms: List[str], sections: Optional[Dict[str, Dict]]
) -> Optional[Tuple[str, str, str]]:
    """Scan resume sections for any of the search terms. Returns
    (section_name, matching_line, matched_alias) or None."""
    if not sections or not terms:
        return None
    for sec_name in _EVIDENCE_SECTIONS:
        sec = sections.get(sec_name)
        if not sec:
            continue
        content = sec["content"] if isinstance(sec, dict) else sec
        for line in content:
            line_low = line.lower()
            for term in terms:
                if _word_in(line_low, term):
                    return sec_name, line, term
    return None

def _bullet_to_dict(b: Dict, score: float) -> Dict:
    return {
        "bullet_id": b.get("id"),
        "bullet_text": b["text"],
        "role_title": b.get("role_title"),
        "company": b.get("company"),
        "date_range": b.get("date_range"),
        "score": round(float(score), 4),
    }

def link(
    requirements: List[Dict],
    bullets: List[Dict],
    match_result: Dict,
    sections: Optional[Dict[str, Dict]] = None,
    direct_threshold: float = DIRECT_THRESHOLD,
    inferred_threshold: float = INFERRED_THRESHOLD,
) -> List[Dict]:
    """Classify each requirement and return EvidenceLink objects.

    Each EvidenceLink:
      {
        id, requirement_id, requirement_name,
        link_type: "direct" | "inferred" | "missing",
        strength: float,
        supporting_bullets: [{...}, ...] (top-3, even when missing),
        reasoning: Optional[str]
      }
    """
    matrix: np.ndarray = match_result.get("score_matrix")
    per_req = match_result.get("per_requirement", [])
    links: List[Dict] = []

    for idx, req in enumerate(requirements):
        if matrix is None or matrix.size == 0 or idx >= matrix.shape[0]:
            best_score = 0.0
            top_idxs: List[int] = []
        else:
            row = matrix[idx]
            best_score = float(row.max()) if row.size else 0.0
            top_idxs = [int(i) for i in np.argsort(-row)[:3]]

        top_bullets = [
            _bullet_to_dict(bullets[i], matrix[idx, i] if matrix is not None else 0.0)
            for i in top_idxs if i < len(bullets)
        ]

        # Classify
        if top_bullets:
            best_bullet_text = top_bullets[0]["bullet_text"]
        else:
            best_bullet_text = ""

        explicit_mention = _name_in_text(req["name"], best_bullet_text)

        if best_score >= direct_threshold and explicit_mention:
            link_type = "direct"
            strength = best_score
            reasoning = None
        elif best_score >= inferred_threshold:
            link_type = "inferred"
            strength = best_score
            reasoning = (
                f"No direct mention of '{req['name']}'; inferred from semantic "
                f"similarity (score {round(best_score, 2)})."
            )
        else:
            link_type = "missing"
            strength = 0.0
            reasoning = "No supporting evidence found in resume."

        # ---- Exact-word presence override (section-aware) ----------------
        # Build a comprehensive list of search terms (name + synonyms +
        # extracted domain/soft keywords + tokenised name with singular/plural
        # variants), then look for any of them in bullets first, then sections.
        # Classification depends on WHICH section the hit comes from:
        #   - hits in Experience / Skills  → "direct"   (sustained work / explicit listing)
        #   - hits in Projects / Certifications / etc.  → "inferred" (suggests competence)
        if link_type != "direct":
            search_terms = _build_search_terms(req)
            hit = _scan_bullets_for_terms(search_terms, bullets)
            if hit is not None:
                bi, alias = hit
                bullet_section = bullets[bi].get("section") or "experience"
                new_link_type = _classify_by_section(bullet_section)
                # Don't downgrade an already-stronger classification.
                if not (link_type == "direct" and new_link_type == "inferred"):
                    link_type = new_link_type
                synthetic_score = (
                    max(strength, 0.9) if new_link_type == "direct"
                    else max(strength, 0.7)
                )
                bullet_dict = _bullet_to_dict(bullets[bi], synthetic_score)
                top_bullets = [bullet_dict] + [
                    b for b in top_bullets if b.get("bullet_id") != bullet_dict["bullet_id"]
                ]
                top_bullets = top_bullets[:3]
                strength = synthetic_score
                pretty_sec = bullet_section.replace('_', ' ').title()
                if new_link_type == "direct":
                    reasoning = (
                        f"Found in a {pretty_sec} bullet via '{alias}'."
                        if alias.lower() != req["name"].lower()
                        else f"Found in a {pretty_sec} bullet."
                    )
                else:
                    reasoning = (
                        f"Inferred from {pretty_sec} section via '{alias}' "
                        f"(weaker than sustained work experience)."
                        if alias.lower() != req["name"].lower()
                        else f"Inferred from {pretty_sec} section "
                             f"(weaker than sustained work experience)."
                    )
            else:
                section_hit = _scan_sections_for_terms(search_terms, sections)
                if section_hit is not None:
                    sec_name, line, alias = section_hit
                    pretty_sec = sec_name.replace('_', ' ').title()
                    new_link_type = _classify_by_section(sec_name)
                    synthetic_score = (
                        max(strength, 0.85) if new_link_type == "direct"
                        else max(strength, 0.65)
                    )
                    synthetic = {
                        "bullet_id": f"section:{sec_name}",
                        "bullet_text": line,
                        "role_title": f"({pretty_sec} section)",
                        "company": None,
                        "date_range": None,
                        "score": synthetic_score,
                    }
                    top_bullets = [synthetic] + top_bullets
                    top_bullets = top_bullets[:3]
                    link_type = new_link_type
                    strength = synthetic_score
                    if new_link_type == "direct":
                        reasoning = (
                            f"Listed in resume's {pretty_sec} section via '{alias}'."
                            if alias.lower() != req["name"].lower()
                            else f"Listed in resume's {pretty_sec} section."
                        )
                    else:
                        reasoning = (
                            f"Inferred from {pretty_sec} section via '{alias}' "
                            f"(weaker than Experience/Skills evidence)."
                            if alias.lower() != req["name"].lower()
                            else f"Inferred from {pretty_sec} section "
                                 f"(weaker than Experience/Skills evidence)."
                        )

        links.append({
            "id": f"link_{idx + 1}",
            "requirement_id": req["id"],
            "requirement_name": req["name"],
            "requirement_category": req.get("category", "skill"),
            "is_required": req.get("is_required", True),
            "min_proficiency": req.get("min_proficiency", "mid"),
            "min_years": req.get("min_years"),
            "link_type": link_type,
            "strength": round(strength, 4),
            "supporting_bullets": top_bullets,
            "reasoning": reasoning,
        })
    return links
