from __future__ import annotations

from typing import Dict, List

def score_confidence(
    sections: Dict[str, Dict],
    skills: List[Dict],
    experience: Dict,
    sections_score: Dict,
) -> Dict:
    """Aggregate per-stage reliability signals into a single 0–100 confidence."""
    reasons: List[str] = []

    n_present = sections_score.get("n_present", 0)
    n_expected = sections_score.get("n_expected", 1)
    coverage = n_present / n_expected
    if sections_score.get("missing_required"):
        reasons.append(
            f"Missing required sections: {', '.join(sections_score['missing_required'])}"
        )

    sec_confidences = [
        sec.get("confidence", 0.0)
        for name, sec in sections.items()
        if name != "_meta"
    ]
    sec_conf_mean = sum(sec_confidences) / len(sec_confidences) if sec_confidences else 0.0
    if sec_conf_mean < 0.5:
        reasons.append("Low average section content confidence")

    high_conf_skills = [s for s in skills if s["confidence"] >= 0.7]
    skill_signal = min(len(high_conf_skills) / 8.0, 1.0)  # saturates at 8 skills
    if skill_signal < 0.5:
        reasons.append("Few high-confidence skills extracted")

    n_bullets = experience.get("n_bullets", 0)
    bullet_signal = min(n_bullets / 6.0, 1.0)             # saturates at 6 bullets
    if n_bullets < 3:
        reasons.append("Very few experience bullets to evaluate")

    composite = (
        0.30 * coverage
        + 0.25 * sec_conf_mean
        + 0.25 * skill_signal
        + 0.20 * bullet_signal
    )
    label = "high" if composite >= 0.75 else "medium" if composite >= 0.5 else "low"
    return {
        "score": round(composite * 100),
        "label": label,
        "reasons": reasons,
        "details": {
            "section_coverage": round(coverage, 2),
            "mean_section_confidence": round(sec_conf_mean, 2),
            "skill_signal": round(skill_signal, 2),
            "bullet_signal": round(bullet_signal, 2),
        },
    }
