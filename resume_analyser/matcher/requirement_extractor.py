from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Dict, List, Optional

# Reuse the analyser's skill knowledge.
from analyser.skill_extractor import _all_known_skills, _canonical_case

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

REQUIRED_CUES = ["must have", "required", "essential", "you must", "minimum"]
NICE_CUES = ["nice to have", "preferred", "bonus", "plus", "desirable", "ideally", "optional"]

YEARS_RE = re.compile(r"(\d+)\s*\+?\s*(?:years?|yrs?)", flags=re.IGNORECASE)
PROFICIENCY_HINTS = {
    "senior": ["senior", "lead", "principal", "expert", "advanced", "deep expertise"],
    "junior": ["junior", "entry", "graduate", "intern", "trainee", "fresher"],
    "mid":    ["mid", "intermediate", "associate", "professional"],
}

DOMAIN_KEYWORDS = {
    "fintech", "finance", "financial", "banking", "bank",
    "payment", "payments", "transaction", "transactions",
    "insurance", "lending", "credit", "trading",
    "healthcare", "medical", "biotech", "pharma", "clinical",
    "education", "edtech", "learning",
    "ecommerce", "e-commerce", "retail", "marketplace", "shopping",
    "logistics", "supply chain", "shipping", "delivery",
    "automotive", "telecom", "gaming", "media", "streaming",
    "advertising", "adtech",
    "saas", "b2b", "b2c", "enterprise", "platform",
    "government", "defence", "defense", "public sector",
    "security", "cybersecurity",
    "iot", "embedded",
    "cloud", "devops",
}

SOFT_KEYWORDS = {
    "communication", "teamwork", "leadership", "collaboration",
    "problem solving", "stakeholder", "presentation", "mentoring",
    "ownership", "self-motivated",
}

@lru_cache(maxsize=1)
def _proficiency_signals() -> Dict[str, List[str]]:
    with open(os.path.join(DATA_DIR, "proficiency_signals.json"), encoding="utf-8") as fh:
        return json.load(fh)

def _split_sentences(text: str) -> List[str]:
    """Split on bullets, newlines, and sentence-final punctuation."""
    if not text:
        return []
    # Normalise common bullets to newlines.
    normalised = re.sub(r"[•·▪◦●]+", "\n", text)
    # Split on newlines first.
    chunks = [c.strip(" -*\t") for c in normalised.split("\n") if c.strip()]
    out: List[str] = []
    for chunk in chunks:
        # Then split long chunks on sentence-final punctuation.
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", chunk)
        out.extend(p.strip() for p in parts if p.strip())
    return [s for s in out if len(s) >= 20]   # filter under-20-char fragments

def _classify_required(sentence: str) -> bool:
    low = sentence.lower()
    if any(c in low for c in NICE_CUES):
        return False
    return True   # required by default unless flagged otherwise

def _infer_proficiency(sentence: str) -> str:
    low = sentence.lower()
    signals = _proficiency_signals()
    for level in ("senior", "junior", "mid"):
        for signal in signals.get(level, []) + PROFICIENCY_HINTS.get(level, []):
            if signal in low:
                return level
    return "mid"

def _extract_years(sentence: str) -> Optional[int]:
    m = YEARS_RE.search(sentence)
    return int(m.group(1)) if m else None

def _categorise(skill: Optional[str], sentence: str, years: Optional[int]) -> str:
    low = sentence.lower()
    if any(d in low for d in DOMAIN_KEYWORDS):
        return "domain"
    if any(s in low for s in SOFT_KEYWORDS):
        return "soft"
    if years is not None and not skill:
        return "experience"
    if skill:
        return "skill"
    return "experience"

def _find_skills_in_sentence(sentence: str) -> List[str]:
    """Return all canonical skill names found in a requirement sentence."""
    low = sentence.lower()
    flat = _all_known_skills()
    found, seen = [], set()
    # Sort longest first so "Machine Learning" beats "Learning".
    for skill_lc in sorted(flat.keys(), key=len, reverse=True):
        if _word_in(low, skill_lc):
            canonical = _canonical_case(skill_lc)
            key = canonical.lower()
            if key not in seen:
                seen.add(key)
                found.append(canonical)
    return found

def _word_in(haystack: str, needle: str) -> bool:
    if needle not in haystack:
        return False
    pattern = r"(?<![A-Za-z0-9])" + re.escape(needle) + r"(?![A-Za-z0-9])"
    return re.search(pattern, haystack) is not None

def extract(parsed_job: Dict) -> List[Dict]:
    """Return a deduplicated list of Requirement dicts."""
    block = parsed_job.get("raw_requirements_text", "") or parsed_job.get("raw_text", "")
    sentences = _split_sentences(block)

    raw_reqs: List[Dict] = []
    for i, sent in enumerate(sentences, start=1):
        skills = _find_skills_in_sentence(sent)
        years = _extract_years(sent)
        proficiency = _infer_proficiency(sent)
        is_required = _classify_required(sent)

        # Construct one requirement per explicitly named skill. This preserves
        # multi-skill bullets such as "AWS, Docker and Kubernetes".
        if skills:
            for skill in skills:
                raw_reqs.append({
                    "id": f"req_{i}",
                    "name": skill,
                    "raw_text": sent,
                    "category": _categorise(skill, sent, years),
                    "min_proficiency": proficiency,
                    "is_required": is_required,
                    "min_years": years,
                })
        else:
            raw_reqs.append({
                "id": f"req_{i}",
                "name": _summarise(sent),
                "raw_text": sent,
                "category": _categorise(None, sent, years),
                "min_proficiency": proficiency,
                "is_required": is_required,
                "min_years": years,
            })

    return _dedupe(raw_reqs)

def _summarise(sentence: str) -> str:
    """Compact a sentence into a short noun-phrase-like requirement label."""
    cleaned = re.sub(r"^[\W\d]+", "", sentence).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if len(cleaned) > 60:
        cleaned = cleaned[:57].rstrip() + "…"
    return cleaned or sentence[:60]

_PROF_RANK = {"junior": 0, "mid": 1, "senior": 2}

def _dedupe(reqs: List[Dict]) -> List[Dict]:
    """Merge same-name requirements; keep the strictest proficiency / max years
    / strictest is_required (any True wins)."""
    by_name: Dict[str, Dict] = {}
    for r in reqs:
        key = r["name"].lower()
        if key not in by_name:
            by_name[key] = dict(r)
            continue
        existing = by_name[key]
        # Strictest proficiency
        if _PROF_RANK[r["min_proficiency"]] > _PROF_RANK[existing["min_proficiency"]]:
            existing["min_proficiency"] = r["min_proficiency"]
        # Max years
        if r.get("min_years") and (
            existing.get("min_years") is None or r["min_years"] > existing["min_years"]
        ):
            existing["min_years"] = r["min_years"]
        # Required wins over nice-to-have
        existing["is_required"] = existing["is_required"] or r["is_required"]
    # Reassign sequential ids so callers can rely on them.
    out = list(by_name.values())
    for i, r in enumerate(out, start=1):
        r["id"] = f"req_{i}"
    return out
