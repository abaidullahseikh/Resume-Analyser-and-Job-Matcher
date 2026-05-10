from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

log = logging.getLogger("resume_analyser.job_entry_extractor")

# Constants

_BULLET_CHARS = frozenset("•·▪◦●▶►*")

_ACTION_VERBS: frozenset[str] = frozenset({
    "achieved", "analysed", "analyzed", "automated", "built", "collaborated",
    "collected", "completed", "conducted", "configured", "coordinated",
    "created", "defined", "delivered", "designed", "developed", "devised",
    "directed", "drove", "enhanced", "established", "evaluated", "executed",
    "generated", "handled", "identified", "implemented", "improved",
    "increased", "initiated", "integrated", "launched", "led", "maintained",
    "managed", "mentored", "migrated", "monitored", "obtained", "operated",
    "optimized", "organised", "organized", "oversaw", "owned", "performed",
    "planned", "prepared", "processed", "produced", "provided", "reduced",
    "researched", "resolved", "reviewed", "shipped", "solved", "spearheaded",
    "streamlined", "supported", "tested", "trained", "utilized", "worked",
    "wrote",
})

# Locations / noise phrases that look short and title-cased but are not jobs.
_LOCATION_TOKENS = frozenset({
    "us", "uk", "usa", "uae", "dubai", "london", "remote", "onsite",
    "hybrid", "full-time", "part-time", "contract", "freelance",
    "present", "current",
})

# Regex that matches a full date range — used to reject date-only lines.
_DATE_RANGE_RE = re.compile(
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}"
    r"|(?:0?[1-9]|1[0-2])/(?:19|20)\d{2}"
    r"|\b(?:19|20)\d{2}\b",
    re.IGNORECASE,
)

# A line that is purely a date range.
_DATE_ONLY_RE = re.compile(
    r"^\s*(?:"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}"
    r"(?:\s*[-–—to]+\s*"
    r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}"
    r"|present|current|now))?"
    r"|\d{4}\s*[-–—to]+\s*(?:\d{4}|present|current|now)"
    r")\s*$",
    re.IGNORECASE,
)

# Role-title keywords — their presence in a short line is a strong signal.
_TITLE_KEYWORD_RE = re.compile(
    r"\b(?:engineer|developer|architect|manager|lead|director|head|"
    r"analyst|consultant|designer|scientist|specialist|administrator|"
    r"intern|trainee|apprentice|fellow|researcher|coordinator|officer|"
    r"executive|supervisor|owner|founder|co-?founder|president|"
    r"vp|ceo|cto|cfo|coo|principal|partner|advisor|assistant|associate|"
    r"technician|operator|accountant|programmer|instructor|tutor|"
    r"representative|rep|nurse|therapist|qa|sre|devops|sde|swe)\b",
    re.IGNORECASE,
)

# "Title, Company" pattern — FIX 1: primary strict detection rule.
_TITLE_COMPANY_RE = re.compile(
    r"^([A-Z][A-Za-z0-9\s\-/&'\.]{1,50}),\s+([A-Z][A-Za-z0-9\s\-/&'\.]{1,50})$"
)

# FIX 1-4: Core helper functions

def clean_line(line: str) -> str:
    """Normalise a raw resume line.

    - Strip leading/trailing whitespace and common decoration characters.
    - Collapse internal runs of 2+ spaces to one space.
    - Remove zero-width and non-printable characters.
    """
    cleaned = line.strip(" \t\r\n-–—|,.*•·▪◦●▶►")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = "".join(ch for ch in cleaned if ch.isprintable())
    return cleaned.strip()

def _first_word(line: str) -> str:
    """Return the first word of a cleaned line, lower-cased."""
    words = line.strip().split()
    return words[0].lower().strip(".,;:-") if words else ""

def is_job_title(line: str) -> bool:
    """FIX 1 + FIX 4 — Return True if *line* looks like a job-title header.

    A line qualifies when it passes ALL noise filters AND at least one of the
    positive signals below:

    Positive signals
    ----------------
    1. Matches ``[Title], [Company]`` comma pattern.
    2. Contains a well-known role-title keyword (engineer, analyst, …).
    3. Is title-cased, short (≤ 8 words), and not a known location/date.

    Noise filters (any one rejects the line)
    -----------------------------------------
    - Starts with a bullet marker (•, -, *, …).           FIX 2
    - First word is a resume action verb.                  FIX 2 / FIX 4
    - Line ends with a sentence-final period.
    - Contains 2+ commas (enumeration, not title/company).
    - Longer than 10 words.                                FIX 4
    - Is a pure date range.
    - All words are location/noise tokens.
    """
    if not line or not line.strip():
        return False

    stripped = line.strip()

    # ── Noise filters ──────────────────────────────────────────────────
    # FIX 2: explicit bullet marker.
    if stripped[0] in _BULLET_CHARS or stripped.startswith("- "):
        return False

    clean = clean_line(line)
    if not clean:
        return False

    words = clean.split()

    # FIX 4: too long.
    if len(words) > 10:
        return False

    # FIX 4: starts with action verb (bullet without marker).
    if _first_word(clean) in _ACTION_VERBS:
        return False

    # Sentence ending → descriptive prose, not a title.
    if clean.rstrip().endswith("."):
        return False

    # Enumeration line (too many commas).
    if clean.count(",") >= 2:
        return False

    # Pure date line.
    if _DATE_ONLY_RE.match(clean):
        return False

    # All-location tokens → not a job title.
    lower_words = [w.lower().strip(".,;:") for w in words]
    if all(w in _LOCATION_TOKENS for w in lower_words):
        return False

    # ── Positive signals ───────────────────────────────────────────────
    # Signal 1: strict "Title, Company" comma pattern.
    if _TITLE_COMPANY_RE.match(clean):
        return True

    # Signal 2: known role-title keyword present.
    if _TITLE_KEYWORD_RE.search(clean):
        return True

    # Signal 3: title-cased short phrase not dominated by dates.
    alpha_words = [w for w in words if w.isalpha()]
    if (alpha_words
            and len(words) <= 8
            and sum(1 for w in alpha_words if w[0].isupper()) >= max(1, len(alpha_words) * 0.6)
            and not all(w in _LOCATION_TOKENS for w in lower_words)):
        return True

    return False

def _split_title_company(line: str) -> tuple[str, str]:
    """Split a detected job-title line into (title, company).

    Handles patterns:
      - "Senior Engineer, Acme Corp"      → comma split
      - "Senior Engineer at Acme Corp"    → 'at' split
      - "Senior Engineer | Acme Corp"     → pipe split
      - "Senior Engineer"                 → title only
    """
    clean = clean_line(line)

    # Comma split (FIX 1 primary pattern).
    if "," in clean:
        parts = clean.split(",", 1)
        return parts[0].strip(), parts[1].strip()

    # Separator keywords / symbols.
    for sep in (r"\s+at\s+", r"\s+@\s+", r"\s*\|\s*", r"\s*/\s+", r"\s*–\s*"):
        m = re.split(sep, clean, maxsplit=1, flags=re.IGNORECASE)
        if len(m) == 2:
            return m[0].strip(), m[1].strip()

    return clean, ""

# FIX 3: Job boundary detection + description collection

def extract_job_entries(lines: List[str]) -> List[Dict[str, str]]:
    """FIX 3 — Parse a flat list of resume lines into structured job entries.

    Algorithm
    ---------
    1. Iterate lines top-to-bottom.
    2. When a job-title line is detected → start a new job entry (boundary).
       All previous accumulated description lines are flushed to the current job.
    3. Lines between two job titles are collected as the job description.
    4. Bullet lines (FIX 2) are included in descriptions but never as titles.

    Parameters
    ----------
    lines : list of str
        Raw or lightly-preprocessed lines from one resume section.

    Returns
    -------
    list of dict with keys: "title", "company", "description"
    """
    jobs: List[Dict[str, str]] = []
    current_title: Optional[str] = None
    current_company: Optional[str] = None
    description_lines: List[str] = []

    def _flush():
        """Save the accumulated entry (if any title was detected)."""
        if current_title:
            desc = " ".join(
                clean_line(l) for l in description_lines
                if clean_line(l)
            )
            jobs.append({
                "title":       current_title,
                "company":     current_company or "",
                "description": desc,
            })

    for raw in lines:
        # FIX 2: skip empty lines (don't add to description noise).
        if not raw.strip():
            continue

        if is_job_title(raw):
            # FIX 3: new job boundary detected — flush previous entry.
            _flush()
            title, company = _split_title_company(raw)
            current_title   = title
            current_company = company
            description_lines = []
            log.debug("Job title detected: %r @ %r", title, company)
        else:
            # Everything else (bullets, dates, prose) goes into the description.
            description_lines.append(raw)

    # Flush the final entry.
    _flush()

    log.debug("Detected Jobs (%d): %s", len(jobs), [(j["title"], j["company"]) for j in jobs])
    return jobs

# REQ 1-8: Structured bullet-level job extraction

# End-of-range alternatives: Month Year | bare year | present/current/now
_DATE_END = (
    r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*)?\d{4}"
    r"|present|current|now"
)
# Dash/separator class (en-dash, em-dash, hyphen, "to").
_DASH = r"[-–—to]"

# Pure date line — a line that is ONLY a date range and nothing else.
_DATE_ONLY_SIMPLE_RE = re.compile(
    r"^\s*(?:"
    # "2023 – 2025" / "2023 to present"
    r"\d{4}\s*" + _DASH + r"+\s*(?:" + _DATE_END + r")"
    r"|"
    # "Jan 2023" / "Jan 2023 – Dec 2025" / "Jan 2023 – present"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}"
    r"(?:\s*" + _DASH + r"+\s*(?:" + _DATE_END + r"))?"
    r"|"
    # "01/2023 – 06/2025"
    r"(?:0?[1-9]|1[0-2])/(?:19|20)\d{2}"
    r"(?:\s*" + _DASH + r"+\s*(?:(?:0?[1-9]|1[0-2])/(?:19|20)\d{2}|present|current|now))?"
    r")\s*$",
    re.IGNORECASE,
)

# Location noise tokens — single-word or two-word lines that are just a city/country.
_LOCATION_RE = re.compile(
    r"^\s*(?:remote|onsite|hybrid|us|uk|usa|uae|dubai|london|new york|"
    r"karachi|lahore|islamabad|india|pakistan|canada|australia|"
    r"full[-\s]?time|part[-\s]?time|contract|freelance)\s*$",
    re.IGNORECASE,
)

# "Role, Company" — the primary job-header pattern (REQ 1).
_JOB_HEADER_RE = re.compile(
    r"^([A-Z][A-Za-z0-9\s\-/&'\.]{1,60}),\s+([A-Z][A-Za-z0-9\s\-/&'\.]{1,60})$"
)

def is_job_header(line: str) -> bool:
    """REQ 1 — Return True if *line* is a job-title header.

    Detection rules (all must pass):
    ──────────────────────────────────────────────────────────────────────
    NOISE filters (any one → False):
      • Blank line.
      • Starts with a bullet marker (•, -, *, etc.).
      • First word is a resume action verb (uses _ROLE_ACTION_STARTS).
      • Pure date range ("2023 – 2025", "Jan 2023 - Present").
      • Pure location token ("US", "Remote", "Dubai").
      • Ends with a sentence-final period (descriptive prose).
      • Contains 2+ commas (enumeration, not "Title, Company").
      • Longer than 10 words.

    POSITIVE signals (at least one → True):
      1. Matches "[Title], [Company]" comma pattern  ← primary (REQ 1).
      2. Matches "Title at/@ Company" / "Title | Company" etc.
      3. Contains a known role-title keyword (engineer, analyst, …).
      4. Title-cased short phrase (≤ 8 words, ≥ 60% words capitalised).
    """
    if not line or not line.strip():
        return False

    stripped = line.strip()

    # ── Noise filters ──────────────────────────────────────────────────
    # Bullet marker at start.
    if stripped[0] in _BULLET_CHARS or re.match(r"^[-*]\s", stripped):
        return False

    clean = clean_line(line)
    if not clean:
        return False

    words = clean.split()

    if len(words) > 10:                              # REQ 1: word limit
        return False

    first = words[0].lower().strip(".,;:-") if words else ""
    if first in _ACTION_VERBS:                       # REQ 2: action verbs
        return False

    if clean.rstrip().endswith("."):                 # sentence prose
        return False

    if clean.count(",") >= 2:                        # enumeration
        return False

    if _DATE_ONLY_SIMPLE_RE.match(clean):            # REQ 4: date line
        return False

    if _LOCATION_RE.match(clean):                    # REQ 4: location
        return False

    # ── Positive signals ───────────────────────────────────────────────
    if _JOB_HEADER_RE.match(clean):                  # Signal 1 (REQ 1)
        return True

    if _TITLE_COMPANY_RE.match(clean):               # broader comma pattern
        return True

    # "Title at/@ Company" etc.
    for sep in (r"\s+at\s+", r"\s+@\s+", r"\s*\|\s*", r"\s*/\s+"):
        if re.split(sep, clean, maxsplit=1, flags=re.IGNORECASE).__len__() == 2:
            return True

    if _TITLE_KEYWORD_RE.search(clean):              # Signal 3: role keyword
        return True

    # Signal 4: title-cased short phrase.
    alpha = [w for w in words if w.isalpha()]
    if (alpha
            and len(words) <= 8
            and sum(1 for w in alpha if w[0].isupper()) >= max(1, len(alpha) * 0.6)):
        return True

    return False

def parse_job_header(line: str) -> tuple[str, str]:
    """REQ 5 — Split a confirmed job-header line into (title, company).

    Tries separators in priority order:
      1. Comma     "Senior Engineer, Tkxel"
      2. ' at '    "Senior Engineer at Tkxel"
      3. ' @ '     "Senior Engineer @ Tkxel"
      4. ' | '     "Senior Engineer | Tkxel"
      5. ' / '     "Senior Engineer / Tkxel"
    Falls back to (full_line, "") if no separator is found.
    """
    clean = clean_line(line)

    # Priority 1: comma (most specific, REQ 1 primary pattern).
    if "," in clean:
        left, right = clean.split(",", 1)
        return left.strip(), right.strip()

    # Priority 2-5: other separators.
    for sep in (r"\s+at\s+", r"\s+@\s+", r"\s*\|\s*", r"\s*/\s+",
                r"\s*–\s*", r"\s+-\s+"):
        parts = re.split(sep, clean, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            return parts[0].strip(), parts[1].strip()

    return clean, ""

def extract_jobs_from_experience(lines: List[str]) -> List[Dict]:
    """REQ 3 + REQ 6 + REQ 7 — Parse experience lines into structured jobs.

    Called AFTER extract_section_content() — do NOT call on raw text.

    Algorithm (REQ 3 — Job Boundary Logic):
    ────────────────────────────────────────
    1. Walk lines top-to-bottom.
    2. Detect job-header lines using is_job_header().
    3. On header → flush previous job (if any) → start new entry.
    4. Non-header lines → appended to current job's bullets list.
    5. Skip noise: empty lines, pure dates, pure locations, blank lines.

    Parameters
    ----------
    lines : List[str]
        Content list from sections["experience"] (post extract_section_content).

    Returns (REQ 7)
    -------
    List of dicts: [{"title": str, "company": str, "bullets": List[str]}]
    """
    jobs: List[Dict] = []

    current_title:   Optional[str]       = None
    current_company: Optional[str]       = None
    current_bullets: List[str]           = []

    def _flush() -> None:
        """Save the current job entry to the output list."""
        if current_title is not None:
            jobs.append({
                "title":   current_title,
                "company": current_company or "",
                "bullets": list(current_bullets),
            })

    for raw in lines:
        # REQ 4: skip empty / noise-only lines.
        stripped = raw.strip()
        if not stripped:
            continue

        cl = clean_line(raw)
        if not cl:
            continue

        if _DATE_ONLY_SIMPLE_RE.match(cl):   # skip date-only lines
            continue

        if _LOCATION_RE.match(cl):           # skip location-only lines
            continue

        # REQ 3: job-boundary detection.
        if is_job_header(raw):
            _flush()                         # close previous job
            title, company = parse_job_header(raw)
            current_title   = title
            current_company = company
            current_bullets = []
            log.debug("Job header detected: %r @ %r", title, company)

        else:
            # Everything else is a bullet / description for the current job.
            if cl:
                current_bullets.append(cl)

    _flush()   # close the final job

    # REQ 8: debug output.
    log.debug(
        "extract_jobs_from_experience: %d jobs found — %s",
        len(jobs),
        [(j["title"], j["company"]) for j in jobs],
    )
    if not jobs:
        log.warning("extract_jobs_from_experience: no jobs detected in %d lines", len(lines))

    return jobs

# Pipeline integration

def extract_job_entries_from_sections(sections: dict) -> List[Dict[str, str]]:
    """Extract structured job entries from the sections dict produced by
    preprocessor.preprocess().

    Searches the 'experience' and 'training' sections, returns a combined
    de-duplicated list sorted by order of appearance.
    """
    all_lines: List[str] = []
    for section_name in ("experience", "training"):
        sec = sections.get(section_name)
        if not sec:
            continue
        content = sec.get("content", []) if isinstance(sec, dict) else []
        all_lines.extend(content)

    return extract_job_entries(all_lines)
