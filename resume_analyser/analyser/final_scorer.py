from __future__ import annotations

import re
from typing import Dict, List, Optional

from .preprocessor import preprocess, extract_bullets, extract_role_positions
from .skill_extractor import extract_skills
from .experience_scorer import score_experience, summarize_roles
from .section_scorer import score_sections
from .content_scorer import score_content
from .consistency_checker import check_consistency
from .keyword_ranker import rank_keywords
from .confidence_scorer import score_confidence
from .project_extractor import DOMAIN_LEXICON, extract_projects

DIMENSION_WEIGHTS = {
    "skills":       0.30,
    "experience":   0.25,
    "sections":     0.20,
    "content":      0.15,
    "consistency":  0.10,
}

def _word_in(text: str, keyword: str) -> bool:
    pattern = r"(?<![A-Za-z0-9])" + re.escape(keyword.lower()) + r"(?![A-Za-z0-9])"
    return re.search(pattern, text.lower()) is not None

def _collect_observed_domains(projects: List[Dict], bullets: List[Dict]) -> List[str]:
    """Return ordered unique domains seen in projects and experience bullets."""
    texts: List[str] = []
    for project in projects:
        raw_text = (project.get("raw_text") or "").strip()
        if raw_text:
            texts.append(raw_text)

    for bullet in bullets:
        section = (bullet.get("section") or "").lower()
        if section in {"projects", "open_source"}:
            continue
        combined = " ".join(
            part.strip()
            for part in (
                bullet.get("role_title") or "",
                bullet.get("company") or "",
                bullet.get("text") or "",
            )
            if part and part.strip()
        )
        if combined:
            texts.append(combined)

    observed: List[str] = []
    seen = set()
    for domain, keywords in DOMAIN_LEXICON.items():
        if domain in seen:
            continue
        for text in texts:
            if any(_word_in(text, keyword) for keyword in keywords):
                seen.add(domain)
                observed.append(domain)
                break
    return observed

def _skill_score(skills: List[Dict]) -> int:
    """Weighted skill richness score, normalised to 0–100."""
    if not skills:
        return 0
    total = sum(s["weight"] * s["confidence"] for s in skills)
    return min(round(total * 4), 100)            # ~25 weighted points = 100

def _generate_suggestions(
    skills: List[Dict],
    experience: Dict,
    sections_score: Dict,
    consistency: Dict,
    content: Dict,
) -> List[str]:
    sugg: List[str] = []
    if experience.get("weak_pct", 0) > 30:
        sugg.append(
            "Replace passive verbs (worked/helped/responsible) with action verbs"
            " like Designed, Led or Implemented."
        )
    if experience.get("strong_pct", 0) < 30:
        sugg.append(
            "Quantify impact: add numbers (% improvement, $ saved, users served)"
            " to your top experience bullets."
        )
    for missing in sections_score.get("missing_required", []):
        sugg.append(f"Add a '{missing.title()}' section — it's expected by ATS systems.")
    if consistency.get("listed_but_unused"):
        terms = ", ".join(consistency["listed_but_unused"][:3])
        sugg.append(f"Demonstrate the following listed skills in your bullets: {terms}.")
    if content.get("flesch", 50) < 25:
        sugg.append("Shorten sentences — current phrasing is hard to scan quickly.")
    if not any(s["category"] == "core" and s["confidence"] >= 0.9 for s in skills):
        sugg.append("Make sure at least one core technical skill is named explicitly.")
    return sugg[:5]

def _pick_best_project(projects: List[Dict]) -> Optional[Dict]:
    """Return the most impressive project based on skills breadth, detail and signals."""
    if not projects:
        return None

    def _score(p: Dict) -> float:
        return (
            len(p.get("skills") or []) * 3          # skill variety is the primary signal
            + (p.get("n_bullets") or 0)              # more bullets = more substantial work
            + (2 if p.get("url") else 0)             # live demo / repo link = polished
            + (1 if p.get("domain") else 0)          # applied context
            + (p.get("months_total") or 0) / 12      # longer duration = bigger project
        )

    return max(projects, key=_score)

def _generate_insights(
    skills: List[Dict],
    experience: Dict,
    sections_score: Dict,
) -> List[str]:
    insights: List[str] = []
    n_core = sum(1 for s in skills if s["category"] == "core")
    if n_core >= 5:
        insights.append(f"Strong core-skill coverage ({n_core} core skills detected).")
    if experience.get("strong_pct", 0) >= 50:
        insights.append("Majority of bullets use action verbs with quantified impact.")
    if sections_score.get("n_present", 0) >= 5:
        insights.append("Resume covers a comprehensive set of expected sections.")
    if not insights:
        insights.append("Solid foundation — see suggestions for the highest-leverage fixes.")
    return insights[:5]

def build_final_analysis(raw_text: str) -> Dict:
    """Run the full analyser pipeline and return a single result blob."""
    sections = preprocess(raw_text)
    bullets = [b.to_dict() for b in extract_bullets(sections)]
    positions = extract_role_positions(sections)
    skills = extract_skills(sections)
    experience = score_experience(sections)
    sections_score = score_sections(sections)
    content = score_content(sections)
    consistency = check_consistency(sections, skills)
    keywords = rank_keywords(sections)
    confidence = score_confidence(sections, skills, experience, sections_score)
    projects = extract_projects(sections)
    role_experience = summarize_roles(bullets, positions=positions)
    observed_domains = _collect_observed_domains(projects, bullets)
    best_project = _pick_best_project(projects)

    sk_score = _skill_score(skills)
    final = round(
        DIMENSION_WEIGHTS["skills"]      * sk_score
        + DIMENSION_WEIGHTS["experience"] * experience["score"]
        + DIMENSION_WEIGHTS["sections"]   * sections_score["score"]
        + DIMENSION_WEIGHTS["content"]    * content["score"]
        + DIMENSION_WEIGHTS["consistency"] * consistency["score"]
    )
    final = max(0, min(final, 100))
    if final >= 75:
        band = "green"
    elif final >= 55:
        band = "yellow"
    else:
        band = "red"

    return {
        "final_score": final,
        "band": band,
        "dimensions": {
            "skills":      sk_score,
            "experience":  experience["score"],
            "sections":    sections_score["score"],
            "content":     content["score"],
            "consistency": consistency["score"],
        },
        "weights": DIMENSION_WEIGHTS,
        "skills": skills,
        "experience": experience,
        "section_breakdown": sections_score,
        "content": content,
        "consistency": consistency,
        "keywords": keywords,
        "confidence": confidence,
        "projects": projects,
        "best_project": best_project,
        "observed_domains": observed_domains,
        "role_experience": role_experience,
        "bullets": bullets,
        "sections": sections,
        "positions": positions,
        "insights": _generate_insights(skills, experience, sections_score),
        "suggestions": _generate_suggestions(
            skills, experience, sections_score, consistency, content
        ),
    }
