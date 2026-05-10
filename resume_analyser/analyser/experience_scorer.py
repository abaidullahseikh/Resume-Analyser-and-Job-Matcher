from __future__ import annotations

import json
import os
import re
from datetime import date
from functools import lru_cache
from typing import Dict, List, Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def _today_ym() -> tuple[int, int]:
    """Return (year, month) for today — evaluated fresh on every call so that
    long-running server processes never use a stale 'current' month."""
    t = date.today()
    return t.year, t.month


_NUMERIC_MONTH_DATE_RE = re.compile(r"\b(0?[1-9]|1[0-2])/((?:19|20)\d{2})\b")
_MONTH_NAMES_SHORT = ("", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

def _normalize_date_str(s: str) -> str:
    """Convert numeric month formats like '09/2020' → 'Sep 2020' so the
    standard month-year regex can handle them."""
    def _repl(m: re.Match) -> str:
        return f"{_MONTH_NAMES_SHORT[int(m.group(1))]} {m.group(2)}"
    return _NUMERIC_MONTH_DATE_RE.sub(_repl, s)

_MONTH_IDX: Dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_MONTH_YEAR_RE = re.compile(
    r"(?:(?P<month>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+)?"
    r"(?P<year>(?:19|20)\d{2})",
    flags=re.IGNORECASE,
)
_PRESENT_RE = re.compile(r"\b(?:present|current|now)\b", flags=re.IGNORECASE)

def parse_duration_years(date_range: str) -> float:
    """Return the duration in (fractional) years for a date_range string.

    Handles year-only ranges ("2018 - 2020"), month/year ranges
    ("Aug 2018 - Dec 2020"), and open-ended ranges ("Jan 2021 - present").
    Falls back to 0.0 for unparseable input.
    """
    if not date_range:
        return 0.0
    text = _normalize_date_str(date_range.strip())
    matches = list(_MONTH_YEAR_RE.finditer(text))
    if not matches:
        return 0.0

    has_present = bool(_PRESENT_RE.search(text))

    start = matches[0]
    start_year = int(start.group("year"))
    start_month_str = (start.group("month") or "").lower()[:3]
    start_month = _MONTH_IDX.get(start_month_str)

    if has_present:
        end_year, end_month = _today_ym()
    elif len(matches) >= 2:
        end = matches[-1]
        end_year = int(end.group("year"))
        end_month_str = (end.group("month") or "").lower()[:3]
        end_month = _MONTH_IDX.get(end_month_str)
    else:
        # Only a single year was given — can't infer a duration.
        return 0.0

    if start_month is None and end_month is None:
        # Year-only on both sides — straight year diff.
        return max(0.0, float(end_year - start_year))

    start_month = start_month or 1
    end_month = end_month or 12
    months = (end_year * 12 + end_month) - (start_year * 12 + start_month)
    return max(0.0, months / 12.0)

def parse_duration_months(date_range: str) -> int:
    """Whole-month duration for a date_range string (rounded to nearest month).
    Returns 0 if unparseable. Used by summarize_roles for month-precision
    totals."""
    iv = _parse_range_months(date_range)
    if iv is None:
        return 0
    start, end = iv
    return max(0, end - start)

_INTERNSHIP_HINTS = (
    "intern", "internship", "trainee", "graduate program", "co-op",
    "coop", "apprentice", "training", "bootcamp", "fellowship",
)

def classify_position_kind(
    role_title: Optional[str],
    company: Optional[str] = None,
    section: Optional[str] = None,
) -> str:
    """Classify a position as one of:
      - 'job'        full-time work experience (default)
      - 'internship' time-limited training role
      - 'project'    a project/learning entry (no employer relationship)
    Used by the analyst summary so the user can see which positions are
    real work history vs internships vs projects (which we surface as the
    'learning way' bucket, weaker than work experience).
    """
    title = (role_title or "").lower()
    comp = (company or "").lower()
    sec = (section or "").lower()

    if sec in {"projects", "open_source"}:
        return "project"
    if sec == "training":
        return "internship"

    if not company and not role_title:
        return "project"
    if not company:
        # Project entries detected by the project_extractor have no company.
        # If the title looks like a project name (no role keywords), call it
        # a project.
        if any(k in title for k in ("project", "open-source", "open source",
                                    "contributor", "contribution", "personal")):
            return "project"

    for hint in _INTERNSHIP_HINTS:
        if hint in title or hint in comp:
            return "internship"
    return "job"

def format_years_months(months: int) -> str:
    """Render a month count as 'X years Y months' (omitting zero parts)."""
    if months <= 0:
        return "0 months"
    years, mons = divmod(months, 12)
    parts = []
    if years:
        parts.append(f"{years} year{'s' if years != 1 else ''}")
    if mons:
        parts.append(f"{mons} month{'s' if mons != 1 else ''}")
    return " ".join(parts) if parts else "0 months"

def _month_parts(months: int) -> tuple[int, int]:
    """Split a whole-month duration into (years, remaining_months)."""
    return divmod(max(0, months), 12)

def _parse_range_months(date_range: str) -> Optional[tuple]:
    """Return (start_month_idx, end_month_idx) where month_idx = year*12+month.
    Returns None if the range can't be parsed."""
    if not date_range:
        return None
    text = _normalize_date_str(date_range.strip())
    matches = list(_MONTH_YEAR_RE.finditer(text))
    if not matches:
        return None
    has_present = bool(_PRESENT_RE.search(text))

    s = matches[0]
    s_year = int(s.group("year"))
    s_month = _MONTH_IDX.get((s.group("month") or "").lower()[:3]) or 1

    if has_present:
        e_year, e_month = _today_ym()
    elif len(matches) >= 2:
        e = matches[-1]
        e_year = int(e.group("year"))
        e_month = _MONTH_IDX.get((e.group("month") or "").lower()[:3]) or 12
    else:
        return None
    return (s_year * 12 + s_month, e_year * 12 + e_month)

def _overlap_months(left: Optional[tuple], right: tuple) -> int:
    """Return whole months of overlap between two month-index ranges."""
    if left is None:
        return 0
    start = max(left[0], right[0])
    end = min(left[1], right[1])
    return max(0, end - start)

def _role_sort_key(role: Dict) -> tuple:
    """Sort by most recent end date, then latest start date."""
    iv = role.get("_interval")
    if iv is None:
        return (-1, -1)
    return (iv[1], iv[0])

def _public_role(role: Optional[Dict]) -> Optional[Dict]:
    """Drop private helper fields before returning role data to templates."""
    if role is None:
        return None
    return {k: v for k, v in role.items() if not k.startswith("_")}

def _merge_intervals(intervals: List[tuple]) -> List[tuple]:
    """Merge overlapping (start, end) month intervals."""
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_iv[0]]
    for start, end in sorted_iv[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged

def summarize_roles(bullets: List, positions: Optional[List[Dict]] = None) -> Dict:
    """Group bullets by (role, company, date_range) and compute years per role.

        Per-role years are the duration of that single role. The primary total is
        the collected sum across all job positions. An overlap-adjusted calendar
        total is also returned so downstream match scoring can avoid double-counting
        concurrent roles.

    Returns:
        {
          "roles": [
            {"role_title", "company", "date_range", "years", "n_bullets"},
            ...
          ],
                    "total_years": float,           # collected sum across roles
                    "merged_total_years": float,    # overlap-adjusted calendar total
          "n_roles": int,
        }
    """
    groups: Dict = {}
    order: List = []

    def _ensure_group(
        role: Optional[str],
        company: Optional[str],
        dates: Optional[str],
        section: Optional[str],
    ) -> Optional[tuple]:
        # Normalise so casing/spacing variants ("Google", "google", "Google ")
        # and seniority variants ("Senior X" vs "X") collapse to one group.
        role_norm = (
            re.sub(r"\s+", " ", (role or "").strip().lower())
        ) or None
        company_norm = (
            re.sub(r"\s+", " ",
                   re.sub(r"\b(inc|llc|ltd|limited|corp|corporation|company|co)\b\.?",
                          "", (company or "").strip().lower()))
        ).strip() or None
        dates_norm = dates or None
        section_norm = section or None
        if not (role_norm or company_norm):
            return None
        key = (role_norm, company_norm, section_norm)

        # Wildcard merge — if the company is missing, merge into any existing
        # group with the same (role, section, date) and a real company. Same
        # in reverse: an incoming entry with a company should absorb a
        # previously-stored "no company" entry that shares role+section+date.
        # This collapses ghost "Web Developer (no company)" entries with the
        # real "Web Developer @ Webslice" entry that has the same date.
        if key not in groups:
            for k_existing in list(groups):
                er, ec, es = k_existing[0], k_existing[1], k_existing[2]
                if er != role_norm or es != section_norm:
                    continue
                # one side missing company AND dates align (or either missing)
                if (ec is None) != (company_norm is None):
                    ed = groups[k_existing].get("date_range")
                    if (not ed) or (not dates_norm) or (ed == dates_norm):
                        key = k_existing
                        break

        if key not in groups:
            groups[key] = {
                "role_title": role,           # keep original casing
                "company": company,
                "date_range": dates_norm,
                "section": section_norm,
                "n_bullets": 0,
            }
            order.append(key)
        else:
            # Merge — keep the more complete date / longer title / real company.
            existing = groups[key]
            if dates_norm and not existing.get("date_range"):
                existing["date_range"] = dates_norm
            if role and len(role) > len(existing.get("role_title") or ""):
                existing["role_title"] = role
            if company and not existing.get("company"):
                existing["company"] = company
        return key

    for p in positions or []:
        getter = p.get if isinstance(p, dict) else (lambda k, _p=p: getattr(_p, k, None))
        _ensure_group(
            getter("role_title"),
            getter("company"),
            getter("date_range"),
            getter("section"),
        )

    for b in bullets:
        getter = b.get if isinstance(b, dict) else (lambda k, _b=b: getattr(_b, k, None))
        role = getter("role_title")
        company = getter("company")
        dates = getter("date_range")
        section = getter("section")
        # A "role" requires an employer or job title — bullets carrying only
        # a date range are project entries, surfaced separately in the
        # Projects card, and must not be counted as work experience.
        key = _ensure_group(role, company, dates, section)
        if key is None:
            continue
        groups[key]["n_bullets"] += 1

    roles = []
    intervals: List[tuple] = []
    raw_sum_months = 0
    for key in order:
        g = groups[key]
        dr = g["date_range"] or ""
        months = parse_duration_months(dr)
        years_int, months_int = _month_parts(months)
        iv = _parse_range_months(dr)
        _ty, _tm = _today_ym()
        last_year_window = (_ty * 12 + _tm - 12, _ty * 12 + _tm)
        recent_months = _overlap_months(iv, last_year_window)
        g["kind"] = classify_position_kind(
            g["role_title"], g["company"], g.get("section")
        )
        g["years"] = months / 12.0                       # precise decimal years
        g["months_total"] = months                       # whole months
        g["years_int"] = years_int                       # for "X years"
        g["months_int"] = months_int                     # for "Y months"
        g["duration_label"] = format_years_months(months)
        g["is_current"] = bool(_PRESENT_RE.search(dr))
        g["is_recent_year"] = recent_months > 0
        g["recent_months"] = recent_months
        g["recent_label"] = format_years_months(recent_months)
        g["_interval"] = iv
        raw_sum_months += months
        if iv is not None:
            intervals.append(iv)
        roles.append(g)

    merged = _merge_intervals(intervals)
    merged_total_months = sum(end - start for start, end in merged)
    merged_total_years_int, merged_total_months_int = _month_parts(merged_total_months)
    total_years_int, total_months_int = _month_parts(raw_sum_months)

    # Per-kind buckets so the dashboard can show jobs, internships/training,
    # and projects separately.
    kind_buckets: Dict[str, Dict] = {
        "job":        {"months": 0, "count": 0, "intervals": []},
        "internship": {"months": 0, "count": 0, "intervals": []},
        "project":    {"months": 0, "count": 0, "intervals": []},
    }
    interval_lookup = []
    for r in roles:
        m = r["months_total"]
        iv = r.get("_interval")
        kind_buckets[r["kind"]]["months"] += m
        kind_buckets[r["kind"]]["count"] += 1
        if iv is not None:
            kind_buckets[r["kind"]]["intervals"].append(iv)
        interval_lookup.append(iv)

    for kind, bucket in kind_buckets.items():
        merged_iv = _merge_intervals(bucket["intervals"])
        merged_months = sum(e - s for s, e in merged_iv)
        bucket["merged_months"] = merged_months
        bucket["label"] = format_years_months(bucket["months"])
        bucket["merged_label"] = format_years_months(merged_months)
        del bucket["intervals"]   # not template-friendly

    # Main experience is the EXPERIENCE/job bucket. Actual work exposure keeps
    # internships/training as a separate but still work-like signal.
    main_intervals = []
    actual_intervals = []
    for r, iv in zip(roles, interval_lookup):
        if r["kind"] == "job" and iv is not None:
            main_intervals.append(iv)
        if r["kind"] in ("job", "internship") and iv is not None:
            actual_intervals.append(iv)
    main_merged = _merge_intervals(main_intervals)
    main_total_months = sum(e - s for s, e in main_merged)
    actual_merged = _merge_intervals(actual_intervals)
    actual_total_months = sum(e - s for s, e in actual_merged)

    work_roles = [r for r in roles if r["kind"] in ("job", "internship")]
    recent_year_roles = sorted(
        [r for r in work_roles if r.get("is_recent_year")],
        key=_role_sort_key,
        reverse=True,
    )
    current_roles = sorted(
        [r for r in work_roles if r.get("is_current")],
        key=_role_sort_key,
        reverse=True,
    )
    latest_role = max(work_roles, key=_role_sort_key) if work_roles else None
    roles_sorted = sorted(roles, key=_role_sort_key, reverse=True)
    roles_public = [_public_role(r) for r in roles_sorted]

    return {
        "roles": roles_public,
        "kind_buckets": kind_buckets,
        "main_experience_months": main_total_months,
        "main_experience_years": round(main_total_months / 12.0, 1),
        "main_experience_label": format_years_months(main_total_months),
        "actual_total_months": actual_total_months,
        "actual_total_years": round(actual_total_months / 12.0, 1),
        "actual_total_label": format_years_months(actual_total_months),
        "latest_role": _public_role(latest_role),
        "current_roles": [_public_role(r) for r in current_roles],
        "recent_year_roles": [_public_role(r) for r in recent_year_roles],
        "total_positions": len(roles),
        "total_months": raw_sum_months,
        "total_years": round(raw_sum_months / 12.0, 1),
        "total_years_int": total_years_int,
        "total_months_int": total_months_int,
        "total_label": format_years_months(raw_sum_months),
        "merged_total_months": merged_total_months,
        "merged_total_years": round(merged_total_months / 12.0, 1),
        "merged_total_years_int": merged_total_years_int,
        "merged_total_months_int": merged_total_months_int,
        "merged_total_label": format_years_months(merged_total_months),
        "raw_sum_months": raw_sum_months,
        "raw_sum_label": format_years_months(raw_sum_months),
        "raw_sum_years": round(raw_sum_months / 12.0, 1),
        "n_roles": len(roles),
    }
METRIC_RE = re.compile(
    r"(\d+(?:\.\d+)?\s*%|\$\s*\d+[\d,\.]*\s*[KMB]?|"
    r"\b\d+(?:,\d{3})+\b|\b\d{2,}\b)",
    flags=re.IGNORECASE,
)

@lru_cache(maxsize=1)
def _strong() -> set:
    with open(os.path.join(DATA_DIR, "strong_verbs.json"), encoding="utf-8") as fh:
        return {v.lower() for v in json.load(fh)["verbs"]}

@lru_cache(maxsize=1)
def _weak() -> set:
    with open(os.path.join(DATA_DIR, "weak_verbs.json"), encoding="utf-8") as fh:
        return {v.lower() for v in json.load(fh)["verbs"]}

def _classify_bullet(text: str) -> str:
    """Return one of 'strong', 'moderate', 'weak'."""
    if len(text.split()) < 3:
        return "weak"
    first_word = text.strip().split()[0].lower().strip(".,;:-")
    has_metric = bool(METRIC_RE.search(text))
    is_strong_verb = first_word in _strong()
    is_weak_verb = first_word in _weak() or any(w in text.lower() for w in _weak())

    if is_strong_verb and has_metric:
        return "strong"
    if is_strong_verb or has_metric:
        return "moderate"
    if is_weak_verb:
        return "weak"
    return "moderate"

def score_experience(sections: Dict[str, Dict]) -> Dict:
    """Classify every bullet in experience+projects and aggregate.

    Returns:
        {
          "score": 0-100,
          "strong_pct": float, "moderate_pct": float, "weak_pct": float,
          "n_bullets": int,
          "bullets": [{text, label}, ...]
        }
    """
    counts = {"strong": 0, "moderate": 0, "weak": 0}
    classified: List[Dict] = []

    for sec_name in ("experience", "projects"):
        sec = sections.get(sec_name)
        if not sec:
            continue
        for line in sec["content"]:
            line = line.strip()
            if len(line) < 4:
                continue
            label = _classify_bullet(line)
            counts[label] += 1
            classified.append({"text": line, "label": label, "section": sec_name})

    n = sum(counts.values())
    if n == 0:
        return {
            "score": 0,
            "strong_pct": 0.0, "moderate_pct": 0.0, "weak_pct": 0.0,
            "n_bullets": 0,
            "bullets": [],
        }

    strong_pct = counts["strong"] / n
    moderate_pct = counts["moderate"] / n
    weak_pct = counts["weak"] / n
    # Weighted score: strong=1.0, moderate=0.6, weak=0.2
    raw = 1.0 * strong_pct + 0.6 * moderate_pct + 0.2 * weak_pct
    return {
        "score": round(raw * 100),
        "strong_pct": round(strong_pct * 100, 1),
        "moderate_pct": round(moderate_pct * 100, 1),
        "weak_pct": round(weak_pct * 100, 1),
        "n_bullets": n,
        "bullets": classified,
    }
