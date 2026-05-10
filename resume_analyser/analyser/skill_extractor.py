from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

_SECTION_CONFIDENCE: Dict[str, float] = {
    "skills":        1.0,
    "experience":    0.7,
    "projects":      0.7,
    "open_source":   0.7,
    "certifications": 0.6,
    "summary":       0.5,
    "education":     0.5,
    "training":      0.5,
    "awards":        0.4,
    "volunteer":     0.4,
}
_DEFAULT_CONFIDENCE = 0.4

def _load_json(filename: str) -> dict:
    path = os.path.join(DATA_DIR, filename)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)

@lru_cache(maxsize=1)
def _build_registry() -> Tuple[Dict[str, Tuple[str, float]], Dict[str, str]]:
    """Build and cache the skill registry.

    Returns:
        skill_map  — {lowercase_name: (category, weight)}
        case_map   — {lowercase_name: canonical_cased_name}
    """
    skill_map: Dict[str, Tuple[str, float]] = {}
    case_map: Dict[str, str] = {}

    def _register(name: str, category: str, weight: float) -> None:
        lc = name.strip().lower()
        if lc and lc not in skill_map:
            skill_map[lc] = (category, weight)
            case_map[lc] = name.strip()

    try:
        core = _load_json("core_skills.json")
        core_weight = float(core.get("weight", 3.0))
        for skill in core.get("skills", []):
            _register(skill, "core", core_weight)
    except Exception:
        pass

    try:
        soft = _load_json("soft_skills.json")
        soft_weight = float(soft.get("weight", 1.0))
        for skill in soft.get("skills", []):
            _register(skill, "soft", soft_weight)
    except Exception:
        pass

    try:
        graph = _load_json("skill_graph.json")
        for edge in graph.get("edges", []):
            if isinstance(edge, (list, tuple)):
                for node in edge:
                    _register(str(node), "supporting", 1.5)
    except Exception:
        pass

    try:
        synonyms = _load_json("skill_synonyms.json")
        for synset in synonyms.get("synsets", []):
            if not synset:
                continue
            # Determine canonical: first member that is already registered.
            canonical_lc: Optional[str] = None
            for term in synset:
                lc = term.strip().lower()
                if lc in skill_map:
                    canonical_lc = lc
                    break
            if canonical_lc is None:
                # No member is registered yet; use first entry as canonical.
                first: str = str(synset[0]).strip()
                canonical_lc = first.lower()
                if canonical_lc not in skill_map:
                    skill_map[canonical_lc] = ("supporting", 1.5)
                    case_map[canonical_lc] = first

            assert canonical_lc is not None
            cat, wt = skill_map[canonical_lc]
            canonical_name = case_map[canonical_lc]
            for term in synset:
                lc = term.strip().lower()
                if lc not in skill_map:
                    skill_map[lc] = (cat, wt)
                    case_map[lc] = canonical_name
    except Exception:
        pass

    return skill_map, case_map

# Public API

def _all_known_skills() -> Dict[str, Tuple[str, float]]:
    """Return mapping of {lowercase_skill_name: (category, weight)}."""
    skill_map, _ = _build_registry()
    return skill_map

def _canonical_case(skill_lc: str) -> str:
    """Return the canonical cased name for a lowercase skill token."""
    _, case_map = _build_registry()
    return case_map.get(skill_lc.strip().lower(), skill_lc.strip())

def _word_in(text_low: str, needle_low: str) -> bool:
    """Word-bounded substring check; permits + and # inside tokens (C++, C#)."""
    if not text_low or not needle_low:
        return False
    pattern = (
        r"(?<![A-Za-z0-9+#])"
        + re.escape(needle_low)
        + r"(?![A-Za-z0-9+#])"
    )
    return bool(re.search(pattern, text_low))

def extract_skills(sections: Dict[str, Dict]) -> List[Dict]:
    """Scan all resume sections for known skills.

    Returns a list of dicts, one per unique (skill, section) occurrence::

        {
            "skill":      str,    # canonical cased name
            "category":   str,    # "core" | "soft" | "supporting"
            "weight":     float,
            "confidence": float,  # section-based confidence 0.0–1.0
            "section":    str,    # section where the skill was found
        }

    The same skill can appear multiple times with different sections — this is
    intentional and consumed by consistency_checker to distinguish skills that
    are *listed* vs *demonstrated*.
    """
    flat = _all_known_skills()
    results: List[Dict] = []
    seen: Set[Tuple[str, str]] = set()

    # Sort longest names first so "Machine Learning" is matched before "Learning".
    sorted_skills = sorted(flat.keys(), key=len, reverse=True)

    for sec_name, sec_data in sections.items():
        if sec_name == "_meta":
            continue
        if not isinstance(sec_data, dict):
            continue
        content_lines = sec_data.get("content", [])
        if not content_lines:
            continue
        full_text = " ".join(str(line) for line in content_lines).lower()
        confidence = _SECTION_CONFIDENCE.get(sec_name, _DEFAULT_CONFIDENCE)

        for skill_lc in sorted_skills:
            key = (skill_lc, sec_name)
            if key in seen:
                continue
            if _word_in(full_text, skill_lc):
                seen.add(key)
                cat, weight = flat[skill_lc]
                results.append({
                    "skill":      _canonical_case(skill_lc),
                    "category":   cat,
                    "weight":     weight,
                    "confidence": confidence,
                    "section":    sec_name,
                })

    return results
