from __future__ import annotations

import re
from typing import Dict, List, Optional

from analyser.experience_scorer import summarize_roles, parse_duration_years

_FIRST_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")

def _resume_experience_summary(bullets: List[Dict]) -> Dict:
    """Return resume experience totals for scoring and display.

    Match scoring should use the overlap-adjusted calendar total so concurrent
    roles do not inflate the candidate's time-in-seat.
    """
    return summarize_roles(bullets)

def _required_years(links: List[Dict]) -> Optional[int]:
    """Return the largest min_years among experience-category requirements."""
    years = [l["min_years"] for l in links if l.get("min_years")]
    return max(years) if years else None

def _trajectory_score(bullets: List[Dict]) -> int:
    """Score whether titles seem to ascend (junior → mid → senior).

    Starts at 70; +15 if titles ascend in chronological order, -15 if descend.
    """
    titles_with_dates: List[tuple] = []
    seen = set()
    for b in bullets:
        title = (b.get("role_title") or "").strip().lower()
        date_range = (b.get("date_range") or "").strip()
        if not title or not date_range:
            continue
        m = _FIRST_YEAR_RE.search(date_range)
        if not m:
            continue
        start_year = int(m.group(1))
        if (title, start_year) in seen:
            continue
        seen.add((title, start_year))
        titles_with_dates.append((start_year, title))

    if len(titles_with_dates) < 2:
        return 70

    titles_with_dates.sort()  # chronological ascending
    levels = [_title_level(t) for _, t in titles_with_dates]
    if all(b >= a for a, b in zip(levels, levels[1:])) and levels[-1] > levels[0]:
        return 85
    if all(b <= a for a, b in zip(levels, levels[1:])) and levels[-1] < levels[0]:
        return 55
    return 70

def _title_level(title: str) -> int:
    """Return 0 (junior) / 1 (mid) / 2 (senior) / 3 (lead+)."""
    t = title.lower()
    if any(k in t for k in ("intern", "junior", "graduate", "trainee", "associate i ")):
        return 0
    if any(k in t for k in ("principal", "staff", "head", "director", "vp", "chief")):
        return 3
    if any(k in t for k in ("senior", "sr.", "lead")):
        return 2
    return 1

def aggregate(
    links: List[Dict],
    bullets: List[Dict],
    requirements: List[Dict],
    experience_summary: Optional[Dict] = None,
) -> Dict:
    """Roll evidence links into the headline match score and risk flags."""

    experience_summary = experience_summary or _resume_experience_summary(bullets)

    skill_links = [l for l in links if l["requirement_category"] in {"skill", "soft"}]
    if skill_links:
        weights = [1.5 if l["is_required"] else 1.0 for l in skill_links]
        weighted_sum = sum(w * l["strength"] for w, l in zip(weights, skill_links))
        skill_match = round((weighted_sum / sum(weights)) * 100)
    else:
        skill_match = 100   # no skill requirements → no penalty

    needed = _required_years(links)
    main_total_months = experience_summary.get(
        "main_experience_months",
        experience_summary.get("actual_total_months", 0),
    )
    have_years = round(main_total_months / 12.0, 1)
    if needed:
        experience_match = max(0, min(round((have_years / needed) * 100), 100))
    else:
        experience_match = 100

    domain_links = [l for l in links if l["requirement_category"] == "domain"]
    if domain_links:
        domain_match = round(max(l["strength"] for l in domain_links) * 100)
    else:
        domain_match = 100

    trajectory_match = _trajectory_score(bullets)

    overall = round(
        0.50 * skill_match
        + 0.20 * experience_match
        + 0.15 * domain_match
        + 0.15 * trajectory_match
    )
    overall = max(0, min(overall, 100))

    if overall >= 80:
        label = "high"
    elif overall >= 60:
        label = "medium"
    else:
        label = "low"

    inferred_count = sum(1 for l in links if l["link_type"] == "inferred")
    missing_required = [l for l in links if l["link_type"] == "missing" and l["is_required"]]
    has_all_axes = bool(skill_links) and (needed is not None) and bool(domain_links)
    if not missing_required and inferred_count == 0 and has_all_axes:
        confidence = "high"
    elif len(missing_required) >= 2:
        confidence = "low"
    elif inferred_count >= 1 or len(missing_required) == 1:
        confidence = "medium"
    else:
        confidence = "medium"

    risks: List[str] = []
    for l in missing_required:
        risks.append(f"Missing required: {l['requirement_name']}")
    if inferred_count >= 3:
        risks.append(f"{inferred_count} requirements inferred — verify in interview")

    # Experience comparison: under / match / over.
    if needed:
        actual_label = experience_summary.get(
            "main_experience_label",
            experience_summary.get("actual_total_label", "0 months"),
        )
        if have_years < needed - 0.25:
            risks.append(
                f"Under-qualified: JD needs {needed}+ yrs, "
                f"resume shows {actual_label} main experience"
            )
        elif have_years > needed * 1.5 + 1:
            risks.append(
                f"Over-qualified: JD needs {needed}+ yrs, "
                f"resume shows {actual_label} main experience"
            )
        # else within band — surfaced as a positive note, not a risk.

    return {
        "overall_score": overall,
        "label": label,
        "confidence": confidence,
        "breakdown": {
            "skill_match":      skill_match,
            "experience_match": experience_match,
            "domain_match":     domain_match,
            "trajectory_match": trajectory_match,
        },
        "weights": {
            "skill_match":      0.50,
            "experience_match": 0.20,
            "domain_match":     0.15,
            "trajectory_match": 0.15,
        },
        "estimated_resume_years": have_years,
        "estimated_resume_label": experience_summary.get(
            "main_experience_label",
            experience_summary.get("actual_total_label",
                                   experience_summary["merged_total_label"]),
        ),
        "work_training_resume_years": experience_summary.get("actual_total_years"),
        "work_training_resume_label": experience_summary.get("actual_total_label"),
        "collected_resume_years": experience_summary["total_years"],
        "collected_resume_label": experience_summary["total_label"],
        "resume_positions": experience_summary["total_positions"],
        "required_years": needed,
        "experience_verdict": (
            "match" if not needed
            else "under" if have_years < needed - 0.25
            else "over"  if have_years > needed * 1.5 + 1
            else "match"
        ),
        "experience_kind_buckets": experience_summary.get("kind_buckets", {}),
        "resume_positions_list": [
            {
                "kind": r["kind"],
                "role_title": r["role_title"],
                "company": r["company"],
                "date_range": r["date_range"],
                "duration_label": r["duration_label"],
                "years_int": r["years_int"],
                "months_int": r["months_int"],
                "section": r.get("section"),
                "is_current": r.get("is_current", False),
                "is_recent_year": r.get("is_recent_year", False),
                "recent_label": r.get("recent_label", "0 months"),
            }
            for r in experience_summary.get("roles", [])
            if r.get("role_title") or r.get("company")
        ],
        "risks": risks,
        "n_direct":   sum(1 for l in links if l["link_type"] == "direct"),
        "n_inferred": inferred_count,
        "n_missing":  sum(1 for l in links if l["link_type"] == "missing"),
        "n_missing_required": len(missing_required),
    }
