from __future__ import annotations

import re
from typing import Dict, List, Optional

from .skill_extractor import _all_known_skills, _canonical_case
from .experience_scorer import parse_duration_months, format_years_months, _month_parts

DOMAIN_LEXICON: Dict[str, List[str]] = {
    "fintech":      ["fintech", "finance", "banking", "payment", "trading",
                     "lending", "insurance", "credit", "stock"],
    "healthcare":   ["healthcare", "medical", "hospital", "patient",
                     "clinical", "biotech", "pharma", "diagnosis"],
    "ecommerce":    ["ecommerce", "e-commerce", "retail", "shopping",
                     "marketplace", "checkout", "cart"],
    "education":    ["education", "edtech", "learning", "course",
                     "student", "tutoring", "lms"],
    "logistics":    ["logistics", "shipping", "supply chain", "delivery",
                     "fleet", "warehouse"],
    "media":        ["media", "video", "streaming", "content",
                     "publishing", "newsroom"],
    "gaming":       ["game", "gaming", "esports", "leaderboard"],
    "saas":         ["saas", "b2b", "enterprise platform"],
    "ml/ai":        ["machine learning", "ml model", "deep learning",
                     "ai assistant", "neural network", "nlp", "computer vision"],
    "data":         ["data pipeline", "etl", "data warehouse", "analytics",
                     "data lake", "dashboard"],
    "security":     ["security", "cybersecurity", "encryption",
                     "vulnerability", "authentication"],
    "iot":          ["iot", "embedded", "sensor", "raspberry pi", "arduino"],
    "cloud":        ["cloud", "aws", "azure", "gcp", "serverless"],
    "open source":  ["open-source", "open source", "github contribution",
                     "pull request", "accepted pr", "merged pr"],
}

# A "domain name" can also be a literal URL — a recruiter wants to follow it.
URL_RE = re.compile(
    r"(https?://[^\s)>\]]+|github\.com/[^\s)>\]]+|"
    r"[\w\-]+\.(?:io|com|org|net|dev|app)(?:/[^\s)>\]]*)?)",
    flags=re.IGNORECASE,
)

DATE_HINT_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|"
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b|"
    r"\bpresent\b|\bcurrent\b",
    flags=re.IGNORECASE,
)

BULLET_PREFIX_RE = re.compile(r"^[\s\-•·*▪◦●]+")

# Header detection
def _looks_like_header(line: str) -> bool:
    """A header line introduces a new project. Heuristics:
    - not a bullet line (no leading '-', '•', '*')
    - reasonably short
    - either contains a date hint, OR begins with a capitalised noun phrase
      and does NOT begin with a typical action verb.
    """
    stripped = line.strip()
    if not stripped:
        return False
    if BULLET_PREFIX_RE.match(line):
        return False
    if len(stripped.split()) > 14:
        return False
    if DATE_HINT_RE.search(stripped):
        return True
    first = stripped.split()[0]
    if not first[0].isupper():
        return False
    # Action-verb tail is a strong signal of a body bullet, not a header.
    low = first.lower()
    if low.endswith(("ed", "ing")):
        return False
    if low in {"developed", "built", "designed", "implemented", "created",
              "led", "led", "deployed", "launched", "improved", "reduced"}:
        return False
    return True

def _detect_domain(text: str) -> Optional[str]:
    low = text.lower()
    for domain, keywords in DOMAIN_LEXICON.items():
        for kw in keywords:
            pattern = r"(?<![A-Za-z0-9])" + re.escape(kw) + r"(?![A-Za-z0-9])"
            if re.search(pattern, low):
                return domain
    return None

def _detect_url(text: str) -> Optional[str]:
    m = URL_RE.search(text)
    return m.group(0).rstrip(".,;:") if m else None

def _problem_statement(body: List[str], header: str) -> str:
    """The "problem" the project addresses.

    Heuristic: take the first body bullet's first sentence — STAR-style bullets
    usually open with the situation/task. If no body, fall back to the header.
    """
    if not body:
        return header.strip()
    first = body[0]
    sentences = re.split(r"(?<=[.!?])\s+", first, maxsplit=1)
    return sentences[0].strip()

def _skills_in(text: str) -> List[Dict]:
    flat = _all_known_skills()
    low = text.lower()
    found, seen = [], set()
    # Sort longest-first so "Machine Learning" beats "Learning".
    for skill_lc in sorted(flat.keys(), key=len, reverse=True):
        if skill_lc in seen:
            continue
        pattern = r"(?<![A-Za-z0-9])" + re.escape(skill_lc) + r"(?![A-Za-z0-9])"
        if re.search(pattern, low):
            cat, weight = flat[skill_lc]
            seen.add(skill_lc)
            found.append({
                "skill": _canonical_case(skill_lc),
                "category": cat,
                "weight": weight,
            })
    return found

def _clean_name(header: str) -> str:
    """Strip date ranges and bullet markers from a header to leave the project name."""
    name = BULLET_PREFIX_RE.sub("", header).strip()
    # Remove a trailing date range / dash-date tail.
    name = re.sub(
        r"\s*[\-–—|]\s*(?:[A-Za-z]+\s+)?\d{4}.*$",
        "",
        name,
    ).strip()
    name = re.sub(r"\s*\(\s*\d{4}.*\)\s*$", "", name).strip()
    return name or header.strip()

_INLINE_PROJECT_PREFIX_RE = re.compile(
    r"^\s*(?:project|personal project|side project|key project|academic project)\s*[:\-–]\s*",
    flags=re.IGNORECASE,
)

def _collect_content(sections: Dict[str, Dict], names: List[str]) -> List[str]:
    """Concatenate the content lines of any of the given canonical sections."""
    out: List[str] = []
    for name in names:
        section = sections.get(name)
        if not section:
            continue
        content = section["content"] if isinstance(section, dict) else section
        out.extend(content)
        out.append("")  # boundary so headers from another section start fresh
    return out

def _scan_inline_projects(sections: Dict[str, Dict]) -> List[Dict]:
    """Detect lines like 'Project: Built X' or 'Personal Project - Did Y'
    anywhere outside the project/open-source sections, and return them as
    one-line project groups."""
    found: List[Dict] = []
    for sec_name, section in sections.items():
        if sec_name in ("_meta", "projects", "open_source"):
            continue
        if not isinstance(section, dict):
            continue
        for line in section.get("content", []):
            stripped = line.strip()
            m = _INLINE_PROJECT_PREFIX_RE.match(stripped)
            if not m:
                continue
            remainder = stripped[m.end():].strip()
            if not remainder:
                continue
            found.append({"header": remainder, "body": []})
    return found

def extract_projects(sections: Dict[str, Dict]) -> List[Dict]:
    """Return one structured project dict per detected project entry.

    Sources scanned, in order:
      1. The 'projects' section (any of its many alternate headers).
      2. The 'open_source' section — each contribution counts as a project.
      3. Inline 'Project:' / 'Personal Project:' prefixed lines anywhere else
         (a common pattern in single-column engineering resumes that don't
         carve out a dedicated section).

    Each project:
        {
          "name", "problem", "skills", "domain", "url",
          "date_range", "months_total", "years_int", "months_int",
          "duration_label", "n_bullets", "raw_text"
        }
    """
    content: List[str] = _collect_content(sections, ["projects", "open_source"])

    # Group lines into header → body groups.
    groups: List[Dict] = []
    current: Optional[Dict] = None
    for raw in content:
        line = raw.rstrip()
        if not line.strip():
            if current is not None:
                groups.append(current)
                current = None
            continue
        if _looks_like_header(line):
            if current:
                groups.append(current)
            current = {"header": line.strip(), "body": []}
        else:
            if current is None:
                current = {"header": "Untitled Project", "body": []}
            current["body"].append(BULLET_PREFIX_RE.sub("", line).strip())
    if current:
        groups.append(current)

    # Add inline-prefixed project mentions found in other sections.
    groups.extend(_scan_inline_projects(sections))

    projects: List[Dict] = []
    for g in groups:
        full_text = (g["header"] + " " + " ".join(g["body"])).strip()
        date_range = _extract_date_range(g["header"])
        months = parse_duration_months(date_range or "")
        years_int, months_int = _month_parts(months)
        projects.append({
            "name":           _clean_name(g["header"]),
            "problem":        _problem_statement(g["body"], g["header"]),
            "skills":         _skills_in(full_text),
            "domain":         _detect_domain(full_text),
            "url":            _detect_url(full_text),
            "date_range":     date_range,
            "months_total":   months,
            "years_int":      years_int,
            "months_int":     months_int,
            "duration_label": format_years_months(months) if months else None,
            "n_bullets":      len(g["body"]),
            "raw_text":       full_text,
        })
    return projects

_DATE_RANGE_IN_HEADER_RE = re.compile(
    r"((?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+)?\d{4}"
    r"\s*[-–to]+\s*"
    r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+)?(?:\d{4}|present|current|now))",
    flags=re.IGNORECASE,
)

def _extract_date_range(header: str) -> Optional[str]:
    """Pull the date range out of a project header line if one is present."""
    if not header:
        return None
    m = _DATE_RANGE_IN_HEADER_RE.search(header)
    return m.group(1).strip() if m else None
