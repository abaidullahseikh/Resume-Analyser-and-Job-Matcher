from __future__ import annotations

from typing import Dict, List

def _words(text: str) -> set:
    return {tok.strip(".,;:()[]").lower() for tok in text.split() if len(tok) > 2}

def check_consistency(sections: Dict[str, Dict], skills: List[Dict]) -> Dict:
    """Compare skills_section terms against experience+projects body terms."""
    skills_section_terms = set()
    for line in sections.get("skills", {}).get("content", []):
        skills_section_terms |= _words(line)

    body_terms = set()
    for sec in ("experience", "projects", "summary", "volunteer", "open_source"):
        for line in sections.get(sec, {}).get("content", []):
            body_terms |= _words(line)

    listed_skills = {s["skill"].lower() for s in skills if s["section"] == "skills"}
    used_skills = {s["skill"].lower() for s in skills if s["section"] != "skills"}

    listed_but_unused = sorted(listed_skills - used_skills - body_terms)
    used_but_unlisted = sorted(used_skills - listed_skills)

    if not listed_skills:
        score = 0.6 if used_skills else 0.0
    else:
        coverage = 1.0 - (len(listed_but_unused) / max(len(listed_skills), 1))
        score = max(0.0, coverage)

    return {
        "score": round(score * 100),
        "listed_but_unused": listed_but_unused[:15],
        "used_but_unlisted": used_but_unlisted[:15],
        "n_listed": len(listed_skills),
        "n_used_in_body": len(used_skills),
    }
