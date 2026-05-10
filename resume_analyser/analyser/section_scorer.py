from __future__ import annotations

from typing import Dict

EXPECTED = {
    "summary":        {"required": False, "weight": 1.0},
    "skills":         {"required": True,  "weight": 2.0},
    "experience":     {"required": True,  "weight": 3.0},
    "education":      {"required": True,  "weight": 1.5},
    "projects":       {"required": False, "weight": 1.5},
    "certifications": {"required": False, "weight": 0.8},
    "training":       {"required": False, "weight": 0.0},
    "awards":         {"required": False, "weight": 0.5},
    "publications":   {"required": False, "weight": 0.5},
}

def score_sections(sections: Dict[str, Dict]) -> Dict:
    """Score the resume's structural completeness."""
    achieved = 0.0
    total = 0.0
    breakdown = {}
    missing_required = []

    for name, spec in EXPECTED.items():
        weight = spec["weight"]
        total += weight
        present = name in sections and bool(sections[name].get("content"))
        if present:
            confidence = sections[name].get("confidence", 0.5)
            achieved += weight * confidence
        elif spec["required"]:
            missing_required.append(name)
        breakdown[name] = {
            "present": present,
            "required": spec["required"],
            "confidence": sections.get(name, {}).get("confidence", 0.0),
        }

    score = round((achieved / total) * 100) if total else 0
    return {
        "score": score,
        "breakdown": breakdown,
        "missing_required": missing_required,
        "n_present": sum(1 for v in breakdown.values() if v["present"]),
        "n_expected": len(EXPECTED),
    }
