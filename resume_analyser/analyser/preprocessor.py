from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

SECTION_PATTERNS: Dict[str, List[str]] = {
    "summary": [
        r"^summary$",
        r"^professional summary$",
        r"^executive summary$",
        r"^career summary$",
        r"^profile$",
        r"^professional profile$",
        r"^career profile$",
        r"^about(?: me)?$",
        r"^personal statement$",
        r"^career objective$",
        r"^objective$",
        r"^overview$",
    ],
    "skills": [
        r"^skills$",
        r"^technical skills$",
        r"^key skills$",
        r"^core skills$",
        r"^core competenc(?:y|ies)$",
        r"^competenc(?:y|ies)$",
        r"^areas of expertise$",
        r"^expertise$",
        r"^proficien(?:cy|cies)$",
        r"^skill ?set$",
        r"^professional skills$",
        r"^tools?\s*(?:&|and)\s*technolog(?:y|ies)$",
        r"^tech(?:nology)? stack$",
        r"^technolog(?:y|ies)$",
        r"^technical proficien(?:cy|cies)$",
        r"^skills?\s*(?:&|and|/)\s*(?:tools?|technolog(?:y|ies)|expertise|languages?)$",
        r"^tools?\s*(?:&|and|/)\s*skills?$",
        r"^technical\s*(?:&|and)\s*(?:professional)?\s*skills$",
        r"^it skills$",
        r"^hard skills$",
        r"^primary skills$",
        r"^additional skills$",
        r"^skills summary$",
        r"^summary of skills$",
    ],
    "experience": [
        r"^experience$",
        r"^professional experience$",
        r"^work experience$",
        r"^employment(?: history)?$",
        r"^employment record$",
        r"^career history$",
        r"^work history$",
        r"^professional background$",
        r"^career experience$",
        r"^relevant experience$",
        r"^industry experience$",
        r"^related experience$",
        r"^job history$",
        r"^positions held$",
        r"^professional record$",
        r"^career progression$",
        r"^appointments$",
        r"^work\s*(?:&|and)\s*professional experience$",
        r"^prior experience$",
        r"^selected experience$",
        r"^work$",
        r"^my experience$",
        r"^experience and projects$",
    ],
    "education": [
        r"^education$",
        r"^educational background$",
        r"^academic background$",
        r"^academic qualifications$",
        r"^educational qualifications$",
        r"^education\s*(?:&|and)\s*training$",
        r"^academic history$",
        r"^academic credentials$",
        r"^qualifications$",
        r"^degrees?$",
        r"^educational details$",
    ],
    "projects": [
        r"^projects$",
        r"^selected projects$",
        r"^key projects$",
        r"^personal projects$",
        r"^academic projects$",
        r"^relevant projects$",
        r"^side projects$",
        r"^project experience$",
        r"^project work$",
    ],
    "certifications": [
        r"^certifications?$",
        r"^certificates?$",
        r"^professional certifications?$",
        r"^licenses?\s*(?:&|and)\s*certifications?$",
        r"^certifications?\s*(?:&|and)\s*training$",
        r"^credentials$",
        r"^professional credentials$",
        r"^accreditations?$",
        r"^licenses?$",
        r"^training\s*(?:&|and)\s*certifications?$",
    ],
    "awards": [
        r"^awards?$",
        r"^achievements?$",
        r"^accomplishments?$",
        r"^key achievements?$",
        r"^awards?\s*(?:&|and)\s*honou?rs$",
        r"^honou?rs$",
        r"^honou?rs?\s*(?:&|and)\s*awards?$",
        r"^recognitions?$",
        r"^awards?\s*(?:&|and)\s*achievements?$",
    ],
    "publications": [
        r"^publications?$",
        r"^research$",
        r"^research\s*(?:&|and)\s*publications?$",
        r"^papers$",
        r"^published works?$",
        r"^scholarly works?$",
    ],
    "languages": [
        r"^languages$",
        r"^language skills$",
        r"^language proficiency$",
        r"^languages known$",
        r"^linguistic skills$",
    ],
    "volunteer": [
        r"^volunteering$",
        r"^volunteer experience$",
        r"^volunteer work$",
        r"^community service$",
        r"^community involvement$",
        r"^civic engagement$",
        r"^social impact$",
    ],
    "interests": [
        r"^interests$",
        r"^hobbies$",
        r"^hobbies\s*(?:&|and)\s*interests$",
        r"^personal interests$",
        r"^extracurricular activities$",
    ],
    "references": [
        r"^references$",
        r"^professional references$",
        r"^references available upon request$",
        r"^referees$",
        r"^character references$",
    ],
    "contact": [
        r"^contact$",
        r"^contact information$",
        r"^contact details$",
        r"^personal information$",
        r"^personal details$",
        r"^get in touch$",
    ],
    "training": [
        r"^training$",
        r"^internships?$",
        r"^internship experience$",
        r"^internships?\s*(?:&|and|/)\s*training$",
        r"^training\s*(?:&|and|/)\s*internships?$",
        r"^apprenticeships?$",
        r"^traineeships?$",
        r"^bootcamps?$",
        r"^professional development$",
        r"^workshops?\s*(?:&|and)\s*training$",
        r"^courses$",
        r"^continuous learning$",
        r"^professional training$",
    ],
    "open_source": [
        r"^open[\s\-]source$",
        r"^open[\s\-]source contributions?$",
        r"^community contributions?$",
        r"^oss contributions?$",
        r"^github contributions?$",
    ],
}

NOISE_WORDS = {
    "worked", "task", "responsible", "various", "miscellaneous", "etc",
    "and", "the", "of", "in", "for", "to", "a", "an", "with",
}

# Data classes
@dataclass
class Section:
    name: str
    content: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self):
        return asdict(self)

@dataclass
class Bullet:
    """Atomic experience unit consumed by the matcher."""
    id: str
    text: str
    role_title: Optional[str] = None
    company: Optional[str] = None
    date_range: Optional[str] = None
    section: Optional[str] = None        # source section name (experience / projects / …)

    def to_dict(self):
        return asdict(self)

# Text normalisation
def clean_text(raw: str) -> str:
    """Normalise unicode, strip control chars, collapse whitespace."""
    if not raw:
        return ""
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\x0b\x0c]+", " ", text)
    # remove zero-width / non-printable characters
    text = "".join(ch for ch in text if ch.isprintable() or ch == "\n")
    # squeeze multiple spaces (preserve newlines)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

_LEADING_DECOR_RE = re.compile(r"^[^A-Za-z0-9]+")        # emojis, dashes, bullets…
_TRAILING_DECOR_RE = re.compile(
    r"[\s\-=_~.─-▟*•·▪◦●►◇◆■□▶◀\|]+$"
)
_TRAILING_DATE_RE = re.compile(
    r"\s+(?:\d{4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4})"
    r"\s*[-–—to/]+\s*"
    r"(?:\d{4}|present|current|now|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4})\s*$",
    flags=re.IGNORECASE,
)
_TRAILING_COLON_RE = re.compile(r"[:\|;,]+\s*.*$")        # "Skills: Python..." → "Skills"

def _is_header_line(line: str) -> Optional[str]:
    """Return canonical section name if line matches a header pattern.

    Two passes:
      1. Aggressive clean-up (leading/trailing decorations, trailing dates,
         trailing "colon: rest of line"), then exact pattern match.
      2. Strict prefix match — pattern matches a leading substring AND the
         remaining text is empty (or only punctuation/decoration). This
         catches "EXPERIENCE  ____________" / "Skills | Tools" without
         over-classifying real content lines like
         "Open-source contributor to FastAPI" as headers.
    """
    raw = line.strip()
    if not raw:
        return None

    candidate = _LEADING_DECOR_RE.sub("", raw)         # drop emojis / decorations
    candidate = _TRAILING_DECOR_RE.sub("", candidate)  # drop trailing decorations
    candidate = _TRAILING_DATE_RE.sub("", candidate)   # drop trailing date ranges
    candidate = _TRAILING_COLON_RE.sub("", candidate)  # "Skills: Python..." → "Skills"
    candidate = candidate.rstrip(":|;,").strip().lower()
    if not candidate or len(candidate) > 80:
        return None

    # Pass 1 — exact match.
    for canonical, patterns in SECTION_PATTERNS.items():
        for pat in patterns:
            if re.match(pat, candidate, flags=re.IGNORECASE):
                return canonical

    # Pass 2 — strict prefix match. The pattern must consume the start of
    # the candidate AND nothing meaningful may remain after it.
    if len(candidate) > 50:
        return None
    for canonical, patterns in SECTION_PATTERNS.items():
        for pat in patterns:
            inner = pat.lstrip("^").rstrip("$")
            m = re.match(r"(?:" + inner + r")", candidate, flags=re.IGNORECASE)
            if m:
                remainder = candidate[m.end():].strip(" \t-_.|:,;()/&")
                if not remainder:
                    return canonical
    return None

def detect_sections(text: str) -> List[Dict]:
    """Return ordered list of {name, start_line, end_line} markers."""
    lines = text.split("\n")
    headers = []
    for idx, line in enumerate(lines):
        name = _is_header_line(line)
        if name:
            headers.append({"name": name, "start": idx})
    # close intervals
    for i, h in enumerate(headers):
        h["end"] = headers[i + 1]["start"] if i + 1 < len(headers) else len(lines)
    return headers

_BULLET_START_RE = re.compile(r"^\s*[\-•·*▪◦●]\s+")

def _is_continuation(prev: Optional[str], line: str) -> bool:
    """Heuristic: line is a continuation of the previous bullet if it starts
    with whitespace OR with a lowercase letter, AND we already have a bullet
    in progress that didn't end with a sentence-final mark followed by a
    capital later."""
    if prev is None or not line:
        return False
    if line[0].isspace():
        return True
    if line[0].islower():
        return True
    return False

def extract_section_content(text: str, headers: List[Dict]) -> Dict[str, List[str]]:
    """Return {section_name: [logical bullet lines]} — continuation lines
    (wrapped second / third lines of a multi-line bullet) are merged into the
    previous bullet so the matcher and experience scorer see one bullet per
    achievement, not one per visual line."""
    lines = text.split("\n")
    out: Dict[str, List[str]] = {}
    for h in headers:
        body = lines[h["start"] + 1: h["end"]]
        grouped: List[str] = []
        current: Optional[str] = None
        for raw in body:
            if not raw.strip():
                if current is not None:
                    grouped.append(current)
                    current = None
                continue

            is_new = bool(_BULLET_START_RE.match(raw))
            cleaned = raw.strip(" -•·*▪◦●\t")

            if is_new:
                if current is not None:
                    grouped.append(current)
                current = cleaned
            elif _is_continuation(current, raw):
                current = (current + " " + cleaned).strip() if current else cleaned
            else:
                if current is not None:
                    grouped.append(current)
                current = cleaned

        if current is not None:
            grouped.append(current)

        if h["name"] in out:
            out[h["name"]].extend(grouped)
        else:
            out[h["name"]] = grouped
    return out

def validate_section(name: str, content: List[str]) -> bool:
    """Cheap sanity check: section is valid if it has at least one usable line."""
    if not content:
        return False
    joined = " ".join(content).strip()
    return len(joined) >= 5

def assign_confidence(section: Section) -> float:
    """Confidence ∈ [0, 1] based on length, density and noise ratio."""
    if not section.content:
        return 0.0
    joined = " ".join(section.content).lower()
    tokens = re.findall(r"[a-zA-Z]+", joined)
    if not tokens:
        return 0.0
    noise = sum(1 for t in tokens if t in NOISE_WORDS)
    noise_ratio = noise / len(tokens)
    length_score = min(len(tokens) / 60.0, 1.0)        # saturates at ~60 tokens
    density_score = 1.0 - noise_ratio
    return round(0.6 * length_score + 0.4 * density_score, 3)

# Bullet extraction (used by matcher)
_MONTH_NAME_ALT = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)"
)
_DATE_RANGE_RE = re.compile(
    # "Jan 2020 - Dec 2022", "Jan. 2020 – Present", "Jan 2020 — Present"
    # Handles hyphen (-), en-dash (–, U+2013), em-dash (—, U+2014), and "to".
    r"(" + _MONTH_NAME_ALT + r"\.?\s*\d{4}\s*[-\u2013\u2014to]+\s*"
    r"(?:" + _MONTH_NAME_ALT + r"\.?\s*\d{4}|present|current|now))"
    # "2020 - 2022", "2020 to present", "2020 — present"
    r"|(\d{4}\s*[-\u2013\u2014to]+\s*(?:\d{4}|present|current|now))"
    # "01/2020 - 12/2022", "01/2020 - present"
    r"|((?:0?[1-9]|1[0-2])/(?:19|20)\d{2}\s*[-\u2013\u2014to]+\s*"
    r"(?:(?:0?[1-9]|1[0-2])/(?:19|20)\d{2}|present|current|now))",
    flags=re.IGNORECASE,
)

# Lines that are pure date fragments (used to reject them being parsed
# as role/company). e.g. "2024", "June 2025", "Jan 2023", "01/2024".
_DATE_FRAGMENT_RE = re.compile(
    r"^\s*(?:" + _MONTH_NAME_ALT + r"\.?\s*)?\d{4}\s*$"
    r"|^\s*(?:0?[1-9]|1[0-2])/(?:19|20)\d{2}\s*$",
    flags=re.IGNORECASE,
)

# Title <separator> Company. Comma is intentionally NOT a separator — it
# matches too many ordinary content sentences ("Django, Flask, and React,
# delivering scalable systems"). Dashes must be surrounded by whitespace.
# Slash, colon, and tilde are also accepted because resumes often use them as
# heading dividers ("Title / Company", "Title: Company", "Title ~ Company").
_AT_COMPANY_RE = re.compile(
    "^(.{2,80}?)(?:\\s+at\\s+|\\s+@\\s+|\\s*\\|\\s*|"
    "\\s+[-\\u2013\\u2014]\\s+|\\s+\\u2022\\s+|"
    "\\s*/\\s*|\\s+:\\s+|\\s+~\\s+)"
    "(.{2,80})$",
    flags=re.IGNORECASE,
)

_ROLE_TITLE_HINT_RE = re.compile(
    r"\b(?:engineer|developer|architect|manager|lead|director|head|"
    r"analyst|consultant|designer|scientist|specialist|administrator|"
    r"intern|trainee|apprentice|fellow|researcher|coordinator|officer|"
    r"executive|supervisor|owner|founder|co\-?founder|president|"
    r"vp|ceo|cto|cfo|coo|cmo|cio|chro|chief|principal|partner|"
    r"agent|advisor|assistant|associate|ambassador|"
    r"technician|operator|accountant|auditor|planner|strategist|"
    r"marketer|writer|editor|producer|recruiter|copywriter|"
    r"programmer|technologist|instructor|tutor|professor|teacher|"
    r"representative|rep|salesperson|nurse|therapist|"
    r"qa|sre|devops|sde|swe)\b",
    flags=re.IGNORECASE,
)

# Location-only tokens — must NEVER be classified as a company.
_LOCATION_BLACKLIST = frozenset({
    "us", "usa", "uk", "ksa", "uae", "eu", "eea",
    "remote", "onsite", "hybrid", "global", "international",
    "dubai", "london", "singapore", "tokyo", "sydney", "berlin",
    "paris", "amsterdam", "toronto", "seattle", "boston", "austin",
    "karachi", "lahore", "islamabad", "delhi", "mumbai", "bangalore",
    "new york", "san francisco", "los angeles", "hong kong",
    "saudi arabia", "united kingdom", "united states", "united arab emirates",
    "india", "pakistan", "germany", "france", "spain", "italy", "japan",
    "china", "australia", "canada", "mexico", "brazil",
    "full-time", "part-time", "full time", "part time", "contract", "freelance",
    "europe", "asia", "africa", "north america", "south america",
})

# Seniority keywords stripped from titles for fuzzy dedup (e.g. so that
# "Senior Software Engineer" and "Software Engineer" merge).
_SENIORITY_PREFIX_RE = re.compile(
    r"\b(?:junior|jr\.?|senior|sr\.?|lead|staff|principal|head|chief|"
    r"associate|assistant|entry[-\s]level|intermediate|mid[-\s]level)\b\.?",
    flags=re.IGNORECASE,
)

def _normalize_role_for_match(text: Optional[str]) -> Optional[str]:
    """Strip seniority keywords + lower-case so role variants merge.

    "Senior Software Engineer" → "software engineer"
    "Sr. Software Engineer"   → "software engineer"
    "Software Engineer"        → "software engineer"
    """
    if not text:
        return None
    t = _SENIORITY_PREFIX_RE.sub("", text.lower())
    return re.sub(r"\s+", " ", t).strip()

def detect_seniority(role_title: Optional[str]) -> str:
    """Classify a role title into a seniority bucket.

    Returns one of: 'junior', 'mid', 'senior', 'lead', 'manager', 'executive'.
    """
    if not role_title:
        return "mid"
    t = role_title.lower()
    if re.search(r"\b(?:director|vp|svp|evp|ceo|cto|cfo|coo|cio|cmo|chro|"
                 r"president|chair|chief)\b", t):
        return "executive"
    if re.search(r"\b(?:manager|head|owner|founder)\b", t):
        return "manager"
    if re.search(r"\b(?:lead|staff|principal)\b", t):
        return "lead"
    if re.search(r"\b(?:senior|sr\.?)\b", t):
        return "senior"
    if re.search(r"\b(?:junior|jr\.?|intern|trainee|graduate|apprentice|"
                 r"entry[-\s]level)\b", t):
        return "junior"
    return "mid"

_COMPANY_SUFFIX_RE = re.compile(
    r"\b(?:inc|llc|ltd|limited|corp|corporation|company|co|"
    r"gmbh|plc|sa|s\.a|ag|bv|kk|pty|pvt|oy|oyj|srl|s\.r\.o|"
    r"labs|systems|solutions|technologies|tech|studio|studios|"
    r"group|holdings|partners|ventures|capital|industries|"
    r"agency|consulting|networks|media|global|international|"
    r"enterprise|enterprises)\b\.?$",
    flags=re.IGNORECASE,
)

_ROLE_ACTION_STARTS = {
    # Common resume bullet action verbs — lines starting with these are
    # treated as bullet content, never as role/company headers.
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
}

def _looks_title_like(line: str) -> bool:
    """Return True when a short clean line looks like a role title."""
    clean = line.strip(" -–|,.*•·")
    if not clean or len(clean.split()) > 8:
        return False
    if clean.rstrip().endswith("."):
        return False
    first = clean.split()[0].lower().strip(".,;:-")
    if first in _ROLE_ACTION_STARTS:
        return False
    # Known title keyword — definitive signal.
    if _ROLE_TITLE_HINT_RE.search(clean):
        return True
    # Fallback: title-cased short phrase covers non-standard roles like
    # "Scrum Master", "Growth Lead", "Operations", "Head of Product".
    alpha_words = [w for w in clean.split() if w.isalpha()]
    if alpha_words and sum(1 for w in alpha_words if w[0].isupper()) >= max(1, len(alpha_words) * 0.6):
        return True
    return False

def _is_location(line: str) -> bool:
    """Return True when *line* is a city/country/work-mode token, never a company."""
    if not line:
        return False
    norm = re.sub(r"\s+", " ", line.lower().strip(" .,-–"))
    return norm in _LOCATION_BLACKLIST

def _looks_company_like(line: str) -> bool:
    """Return True when a short clean line looks more like an employer than a title."""
    clean = line.strip(" -–|,.*•·")
    if not clean or len(clean.split()) > 6:
        return False
    if clean.rstrip().endswith("."):
        return False
    if _ROLE_TITLE_HINT_RE.search(clean):
        return False
    # IMPROVEMENT — never classify a location as a company.
    if _is_location(clean):
        return False
    if _COMPANY_SUFFIX_RE.search(clean):
        return True
    words = clean.split()
    # Reject single-word ALL-UPPERCASE short lines (US, UK, KSA …) even if
    # they slipped past the blacklist.
    if len(words) == 1 and len(words[0]) <= 4 and words[0].isupper():
        return False
    # Extended from 4 to 5 words to cover multi-word names like
    # "National Bank of Australia" or "Red Hat Europe".
    return len(words) <= 5 and all(w[:1].isupper() for w in words if w)

def _parse_role_header(line: str, had_date: bool) -> Optional[tuple[str, str]]:
    """Return (role, company) if line looks like a role header.

    Rejects content sentences: lines ending with a period (sentences),
    lines with 2+ internal commas (enumerations), and lines whose company
    half is itself a long sentence. Real role headers are short and
    structurally clean: 'Title at/-/| Company [- date]'.

    When had_date is True (the date was on the same line), also accepts
    title-only lines (no company) and comma-separated "Title, Company" pairs,
    since the date context greatly reduces false-positive risk.
    """
    if not line or len(line.split()) > 14:
        return None
    if line.rstrip().endswith("."):
        return None
    # When a date anchors the line we tolerate richer comma punctuation
    # (e.g. trailing "Inc., San Francisco, CA"). Without that anchor we
    # keep the strict 2-comma guard to reject prose enumerations.
    comma_limit = 4 if had_date else 2
    if line.count(",") >= comma_limit:
        return None

    match = _AT_COMPANY_RE.match(line)
    if match:
        role = match.group(1).strip(" -|,")
        company = match.group(2).strip(" -|,")
        if not role or not company:
            return None
        if len(company.split()) > 6:
            return None
        if "," in company:
            return None
        # Reject date-fragment halves: "2024 - June 2025" must not be parsed
        # as role="2024", company="June 2025".
        if _DATE_FRAGMENT_RE.match(role) or _DATE_FRAGMENT_RE.match(company):
            return None
        first = role.split()[0].lower().strip(".,;:-")
        if not had_date and first in _ROLE_ACTION_STARTS:
            return None
        # The structural separator matched by _AT_COMPANY_RE (|, -, /, :, ~,
        # at, @) is sufficient signal on its own. Requiring a title keyword
        # or company-suffix hint causes non-standard but valid titles such as
        # "Operations", "Marketing", "Scrum Master" to be silently dropped.
        return role, company

    # No separator found.  When we know there's a date context, try
    # comma-separated "Title, Company" and plain title-only forms.
    if not had_date:
        return None

    # "Title, Company" — comma allowed here because the date anchor lowers
    # false-positive risk enough to outweigh the ambiguity.
    if line.count(",") == 1:
        left, right = line.split(",", 1)
        left = left.strip()
        right = right.strip()
        if (left and right
                and len(left.split()) <= 10
                and len(right.split()) <= 6
                and not left.rstrip().endswith(".")
                and "," not in right):
            first = left.split()[0].lower().strip(".,;:-")
            if first not in _ROLE_ACTION_STARTS:
                return left, right

    # Title-only — a short clean phrase is the entire role context.
    words = line.split()
    # Require ≥ 2 words AND a role keyword AND not a location, so that
    # bare skill/location words ("JavaScript", "Dubai") are never promoted
    # to standalone roles.
    if (len(words) >= 2 and len(words) <= 10
            and not _is_location(line)
            and _ROLE_TITLE_HINT_RE.search(line)):
        first = words[0].lower().strip(".,;:-")
        if first not in _ROLE_ACTION_STARTS:
            return line.strip(), ""

    return None

def _parse_comma_header(line: str) -> Optional[tuple[str, str]]:
    """FIX: Detect 'Role, Company' pattern safely.

    Catches the very common "Senior Software Engineer, Google" format that
    _parse_role_header rejects (because it has no `at`/`|`/`-` separator and
    no inline date). Strict word-count + verb checks prevent false positives
    on prose enumerations.
    """
    if not line or len(line.split()) > 10:
        return None

    if line.count(",") != 1:
        return None

    left, right = [x.strip() for x in line.split(",", 1)]

    if not left or not right:
        return None

    # Reject bullet-like lines.
    first = left.split()[0].lower().strip(".,;:-")
    if first in _ROLE_ACTION_STARTS:
        return None

    # Reject long phrases on either side.
    if len(left.split()) > 6 or len(right.split()) > 6:
        return None

    # Must look like a title AND contain a role keyword — prevents bare tech
    # terms like "JavaScript, ensuring timely delivery." from being parsed
    # as a role (left side) and company (right side).
    if not _looks_title_like(left):
        return None
    if not _ROLE_TITLE_HINT_RE.search(left):
        return None

    return left, right

def _scan_forward_for_date(content: List[str], from_idx: int,
                           max_lookahead: int = 8) -> Optional[str]:
    """Look at the next few non-empty lines for a date range.

    Default window widened to 8 lines for LinkedIn-style resumes that put:
        Role
        Company
        Location
        Country
        ... bullets ...
        Date
    Stops scanning if it hits another role header so role A doesn't steal
    role B's date.
    """
    looked = 0
    for j in range(from_idx + 1, len(content)):
        line = content[j].strip()
        if not line:
            continue
        date_match = _DATE_RANGE_RE.search(line)
        line_no_date = (
            _DATE_RANGE_RE.sub("", line).strip(" -\u2013\u2014|,") if date_match else line
        )
        if _parse_role_header(line_no_date, bool(date_match)):
            return None
        looked += 1
        if looked > max_lookahead:
            break
        if date_match:
            return (date_match.group(1) or date_match.group(2)
                    or date_match.group(3)).strip()
    return None

def _date_completeness(date_str: Optional[str]) -> int:
    """Score how complete a date_range string is (higher = better).

    Used to pick the best version when merging duplicate role entries:
      - "Jan 2023 - Dec 2025"  → 4   (start month + start year + end month + end year)
      - "Jan 2023 - Present"   → 3
      - "2023 - 2025"          → 2
      - "2023"                 → 1
      - None / ""              → 0
    """
    if not date_str:
        return 0
    score = 0
    # Years.
    score += len(re.findall(r"\b(?:19|20)\d{2}\b", date_str))
    # Month names.
    if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", date_str, re.IGNORECASE):
        score += 1
    if len(re.findall(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", date_str, re.IGNORECASE)) >= 2:
        score += 1
    # Open-ended ranges count as one extra signal.
    if re.search(r"\b(?:present|current|now)\b", date_str, re.IGNORECASE):
        score += 1
    return score

def _better_date(a: Optional[str], b: Optional[str]) -> Optional[str]:
    """Pick the more complete of two date_range strings."""
    if not a:
        return b
    if not b:
        return a
    return a if _date_completeness(a) >= _date_completeness(b) else b

_ROLE_SECTIONS = ("experience", "training", "projects", "volunteer", "open_source")

_BULLET_MARKERS = frozenset("•·▪◦●▶►-*")

def _is_bullet_line(raw: str) -> bool:
    """FIX 2 — Return True when a line is clearly a bullet point.

    Catches both marked bullets (•, -, *, etc.) and unmarked prose that
    starts with a known action verb (common after bullet-marker stripping).
    """
    stripped = raw.strip()
    if not stripped:
        return False
    # Explicit bullet marker at start.
    if stripped[0] in _BULLET_MARKERS:
        return True
    if _BULLET_START_RE.match(raw):
        return True
    # Unmarked bullet — first word is a common action verb.
    first = stripped.split()[0].lower().strip(".,;:-") if stripped else ""
    return first in _ROLE_ACTION_STARTS

def _confidence_filter(line: str) -> bool:
    """High-confidence check that a line is a genuine job title.

    A line qualifies as a role only when it passes ALL of:
      - 2 to 8 words (rejects single-word skills like "JavaScript")
      - not a location ("Dubai", "KSA", "Remote")
      - not an action-verb start ("Built ...", "Designed ...")
      - contains a role keyword (engineer, analyst, …) OR is in
        "Role, Company" form with a role keyword on the left side
    """
    clean = line.strip(" -–|,.*•·")
    if not clean:
        return False
    words = clean.split()
    if len(words) < 2 or len(words) > 8:        # ≥ 2 words required
        return False
    if _is_location(clean):                     # locations are never roles
        return False
    first = words[0].lower().strip(".,;:-") if words else ""
    if first in _ROLE_ACTION_STARTS:
        return False
    # "Role, Company" — left side must contain a role keyword.
    if "," in clean:
        left = clean.split(",", 1)[0].strip()
        if left and len(left.split()) <= 6 and _ROLE_TITLE_HINT_RE.search(left):
            return True
    # Plain role line — must contain a role keyword.
    return bool(_ROLE_TITLE_HINT_RE.search(clean))

def _detect_role_headers(content: List[str]) -> tuple[List[tuple], set]:
    """Return detected role headers and the line indices they consume.

    Applies FIX 1 (strict title detection), FIX 2 (bullet exclusion),
    FIX 3 (job boundaries), FIX 4 (confidence filter).

    The result is a pair of:
      - role_starts: [(line_index, role, company, date_range), ...]
      - claimed_indices: set of line indices consumed as header/company lines
    """
    role_starts: List[tuple] = []
    claimed_indices: set = set()
    company_anchor: Optional[str] = None

    for i, raw in enumerate(content):
        if i in claimed_indices:
            continue
        line = raw.strip()
        if len(line) < 4:
            continue
        # FIX 2 — skip bullet lines immediately.
        if _is_bullet_line(raw):
            continue
        date_match = _DATE_RANGE_RE.search(line)
        had_date = bool(date_match)
        inline_date = (
            (date_match.group(1) or date_match.group(2) or date_match.group(3)).strip()
            if date_match else None
        )
        line_no_date = (
            _DATE_RANGE_RE.sub("", line).strip(" -\u2013\u2014|,") if date_match else line
        )

        # Company-first format:
        #   Acme Corp
        #   Senior Backend Engineer
        #   Jan 2023 - present
        #   ...
        if (not had_date and _looks_company_like(line_no_date)):
            next_idx = None
            next_line = None
            for j in range(i + 1, len(content)):
                nxt = content[j].strip()
                if nxt:
                    next_idx = j
                    next_line = nxt
                    break
            if next_idx is not None and next_line is not None:
                next_date_match = _DATE_RANGE_RE.search(next_line)
                next_no_date = (
                    _DATE_RANGE_RE.sub("", next_line).strip(" -\u2013\u2014|,")
                    if next_date_match else next_line
                )
                title_has_date = bool(next_date_match)
                title_has_followup_date = bool(_scan_forward_for_date(content, next_idx, max_lookahead=2))
                if _looks_title_like(next_no_date) and (title_has_date or title_has_followup_date):
                    company_anchor = line_no_date.strip() or None
                    claimed_indices.add(i)
                    continue

        header = _parse_role_header(line_no_date, had_date)

        if not header:
            header = _parse_comma_header(line_no_date)

        if header:
            role, company = header
            company_idx_used: Optional[int] = None
            # Title-only header (no company on its own line): look forward
            # for a short company-shaped line before the bullets begin so the
            # role doesn't render as "Title @ —" on the dashboard.
            if not company:
                for j in range(i + 1, min(i + 4, len(content))):
                    nxt = content[j].strip()
                    if not nxt:
                        continue
                    if _BULLET_START_RE.match(content[j]):
                        break
                    if _DATE_RANGE_RE.search(nxt):
                        # Pure date line — keep scanning past it.
                        if _DATE_RANGE_RE.sub("", nxt).strip(" -–|,.") == "":
                            continue
                        break
                    nc = nxt.strip(" -–|,.*•·")
                    if (nc
                            and len(nc.split()) <= 6
                            and not nc.rstrip().endswith(".")
                            and not _looks_title_like(nc)
                            and not _ROLE_ACTION_STARTS.intersection(
                                {nc.split()[0].lower().strip(".,;:-")})):
                        company = nc
                        company_idx_used = j
                    break
            company = company or company_anchor
            role_date = inline_date or _scan_forward_for_date(content, i)
            # FIX 4 — stronger same-job dedup (handles missing-company case).
            already_exists = any(
                _same_job(role, company, r, c)
                for _, r, c, _ in role_starts
            )
            if not already_exists:
                role_starts.append((i, role, company, role_date))
            if company:
                company_anchor = company
            claimed_indices.add(i)
            if company_idx_used is not None:
                claimed_indices.add(company_idx_used)
            continue

        if had_date:
            continue

        # Fallback for multi-line format: "Title\n[Company\n]Date\n- bullets"
        # Also handles reversed layout: "Title\nDate\nCompany\n- bullets"
        # FIX 1 + 4 — only proceed when the line passes the confidence filter.
        clean = line.strip(" -–|,.*•·")
        if (clean
                and not clean.rstrip().endswith(".")
                and _confidence_filter(clean)):
            # max_lookahead=8 to handle LinkedIn-style:
            #   Role / Company / Location / Country / bullets / Date
            fwd_date = _scan_forward_for_date(content, i, max_lookahead=8)
            if fwd_date:
                company_candidate: Optional[str] = None
                company_idx: Optional[int] = None
                date_passed = False
                # Scan forward both before AND after the date line for a company.
                for j in range(i + 1, min(i + 6, len(content))):
                    nxt = content[j].strip()
                    if not nxt:
                        continue
                    if _DATE_RANGE_RE.search(nxt):
                        date_passed = True
                        continue   # keep scanning past the date line
                    if _BULLET_START_RE.match(nxt):
                        break      # still-marked bullet — stop
                    nc = nxt.strip(" -–|,.*•·")
                    nc_first = nc.split()[0].lower().strip(".,;:-") if nc else ""
                    # Reject lines that are clearly resume bullets (action verbs,
                    # long sentences, ends with period) even after marker removal.
                    if not nc:
                        continue
                    if nc_first in _ROLE_ACTION_STARTS:
                        break      # bullet content, not a company
                    if nc.rstrip().endswith(".") or nc.count(",") >= 2:
                        break
                    if len(nc.split()) > 6:
                        break
                    if (not _looks_title_like(nc)
                            and not _COMPANY_SUFFIX_RE.search(nc)
                            and not all(w[:1].isupper() for w in nc.split() if w.isalpha())):
                        if not date_passed:
                            break  # doesn't look like a company, stop before date
                        continue   # after date, keep looking
                    company_candidate = nc
                    company_idx = j
                    break
                    if not date_passed:
                        break
                company_value = company_candidate or company_anchor

                # If the title line itself is "Role, Company", split it so the
                # company gets attributed correctly instead of being baked into
                # the title text.
                role_value = clean
                if not company_value and clean.count(",") == 1:
                    left, right = (s.strip() for s in clean.split(",", 1))
                    if (left and right
                            and len(left.split()) <= 8
                            and len(right.split()) <= 6):
                        role_value, company_value = left, right

                # FIX 4 — stronger same-job dedup.
                already_exists = any(
                    _same_job(role_value, company_value, r, c)
                    for _, r, c, _ in role_starts
                )
                if not already_exists:
                    role_starts.append((i, role_value, company_value, fwd_date))
                claimed_indices.add(i)
                if company_idx is not None:
                    claimed_indices.add(company_idx)
                if company_value:
                    company_anchor = company_value

    import logging as _logging
    _log = _logging.getLogger("resume_analyser.preprocessor")
    if role_starts:
        _log.debug(
            "Detected Jobs (%d): %s",
            len(role_starts),
            [(r, c, d) for _, r, c, d in role_starts],
        )
    return role_starts, claimed_indices

def _normalize(text: Optional[str]) -> Optional[str]:
    """Lower-case, strip common company suffixes, collapse whitespace.

    Used for fuzzy de-duplication so that "Google Inc", "Google LLC", and
    plain "Google" all collapse to the same key.
    """
    if not text:
        return None

    text = text.lower().strip()
    text = re.sub(
        r"\b(inc|llc|ltd|limited|corp|corporation|company|co)\b\.?",
        "",
        text,
    )
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def _same_job(r1, c1, r2, c2) -> bool:
    """Two role headers describe the same job when:
      - role titles match after seniority + casing normalisation, AND
      - either company is missing, or normalised companies match.

    Lets ("Senior Software Engineer", None) merge with
    ("Software Engineer", "Google") — both collapse to ("software engineer", *).
    """
    r1n = _normalize_role_for_match(r1)
    r2n = _normalize_role_for_match(r2)
    if not r1n or not r2n or r1n != r2n:
        return False
    c1n = _normalize(c1)
    c2n = _normalize(c2)
    if not c1n or not c2n:
        return True
    return c1n == c2n

# Block-based role extraction (production-grade rewrite)
# A resume "block" is a contiguous run of non-empty lines separated by one or
# more blank lines. Each block is expected to contain at most ONE job entry.
# This block-based approach is dramatically more robust than line-by-line
# scanning for LinkedIn-style resumes where role / company / location / date
# are stacked on consecutive lines.

# A line qualifies as a role title only when it contains a strong role
# keyword OR a seniority keyword AND is short enough.
_VALID_ROLE_RE = re.compile(
    r"\b(?:engineer|developer|architect|manager|lead|director|head|"
    r"analyst|consultant|designer|scientist|specialist|administrator|"
    r"intern|trainee|fellow|researcher|coordinator|officer|executive|"
    r"supervisor|founder|president|vp|ceo|cto|cfo|coo|principal|partner|"
    r"advisor|associate|technician|programmer|technologist|instructor|"
    r"teacher|representative|qa|sre|devops|sde|swe)\b",
    flags=re.IGNORECASE,
)

def _is_valid_role(line: str) -> bool:
    """Strict role-title validator: ≥2 words AND contains a role keyword."""
    if not line:
        return False
    clean = line.strip(" -–|,.*•·")
    if not clean:
        return False
    words = clean.split()
    if len(words) < 2 or len(words) > 10:
        return False
    if clean.rstrip().endswith("."):
        return False
    if _is_location(clean):
        return False
    if _is_bullet_line(clean):
        return False
    # Action-verb start → bullet content, not a role.
    if words[0].lower().strip(".,;:-") in _ROLE_ACTION_STARTS:
        return False
    if "," in clean:
        # "Role, Company" — left side must be the role.
        left = clean.split(",", 1)[0].strip()
        return bool(_VALID_ROLE_RE.search(left))
    return bool(_VALID_ROLE_RE.search(clean))

def _is_valid_company(line: str) -> bool:
    """Strict company validator: short, not a location, not a date, not a role."""
    if not line:
        return False
    clean = line.strip(" -–|,.*•·")
    if not clean:
        return False
    words = clean.split()
    if len(words) > 6 or not words:
        return False
    if clean.rstrip().endswith("."):
        return False
    if _is_location(clean):
        return False
    if _is_bullet_line(clean):
        return False
    if _DATE_RANGE_RE.search(clean) or _DATE_FRAGMENT_RE.match(clean):
        return False
    if words[0].lower().strip(".,;:-") in _ROLE_ACTION_STARTS:
        return False
    # If it has a role keyword, it's the role line, not the company.
    if _VALID_ROLE_RE.search(clean):
        return False
    if _COMPANY_SUFFIX_RE.search(clean):
        return True
    # Plain title-cased short phrase ("Careem", "Acme Corp", "Red Hat Europe").
    return all(w[:1].isupper() for w in words if w.isalpha())

_SINGLE_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

def _extract_date_from_block(lines: List[str]) -> Optional[str]:
    """Find the first date range in the block, or fall back to a single year.

    A single-year match (e.g. "2025") is normalised to "{year} - present" so
    downstream duration calc treats it as ongoing from that year through the
    current month — counted as ≥ 1 year of experience.
    """
    for line in lines:
        m = _DATE_RANGE_RE.search(line)
        if m:
            return (m.group(1) or m.group(2) or m.group(3)).strip()
    for line in lines:
        stripped = line.strip()
        if not stripped or len(stripped) > 12:
            continue
        ym = _SINGLE_YEAR_RE.fullmatch(stripped)
        if ym:
            return f"{ym.group(0)} - Present"
        # Allow lines like "Month YYYY" alone — treat as start.
        mm = re.fullmatch(
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}",
            stripped, flags=re.IGNORECASE,
        )
        if mm:
            return f"{mm.group(0)} - Present"
    return None

def group_job_blocks(content: List[str]) -> List[List[str]]:
    """Group section content into job blocks.

    Boundaries: blank lines AND the appearance of a new role-title line
    (since extract_section_content() strips blank lines, role-title boundary
    detection is what actually separates LinkedIn-style blocks).
    """
    blocks: List[List[str]] = []
    current: List[str] = []
    for raw in content:
        if not raw or not raw.strip():
            if current:
                blocks.append(current)
                current = []
            continue
        # New role title → start a new block (and close the previous one).
        if _is_valid_role(raw) and current:
            blocks.append(current)
            current = []
        current.append(raw)
    if current:
        blocks.append(current)
    return blocks

def _extract_job_from_block(lines: List[str]) -> Optional[Dict[str, Optional[str]]]:
    """Return a structured job dict from a single block, or None."""
    role: Optional[str] = None
    company: Optional[str] = None

    for line in lines:
        clean = line.strip(" -–|,.*•·").strip()
        if not clean:
            continue
        # First valid title wins.
        if role is None and _is_valid_role(clean):
            if "," in clean:
                left, right = (s.strip() for s in clean.split(",", 1))
                role = left
                if not company and _is_valid_company(right):
                    company = right
            else:
                # "Role at Company" / "Role @ Company" / "Role | Company"
                m = _AT_COMPANY_RE.match(clean)
                if m:
                    left = m.group(1).strip(" -|,")
                    right = m.group(2).strip(" -|,")
                    if _VALID_ROLE_RE.search(left):
                        role = left
                        if not company and _is_valid_company(right):
                            company = right
                    else:
                        role = clean
                else:
                    role = clean
            continue
        # First valid company after role wins.
        if role and not company and _is_valid_company(clean):
            company = clean

    if not role:
        return None

    date_range = _extract_date_from_block(lines)
    return {
        "role_title": role,
        "company": company,
        "date_range": date_range,
        "seniority": detect_seniority(role),
    }

def extract_role_positions(sections: Dict[str, Dict]) -> List[Dict[str, Optional[str]]]:
    """Return clean structured role list from resume sections.

    Block-based parser (rewrite):
      1. Split each section's content into blocks separated by blank lines.
      2. For every block, find the first valid role title and the first
         valid company; extract any date range that appears anywhere in the
         block.
      3. Reject single-word skills, sentences, bullets, locations, and dates
         from being misclassified as role/company.
      4. Deduplicate on (normalized role, normalized company), merging
         duplicates by picking the best date and longest title.
      5. Use a section-level last-seen-date fallback for blocks whose date
         is genuinely on a separate, later block.
    """
    positions: List[Dict[str, Optional[str]]] = []
    seen = set()

    for section_name in _ROLE_SECTIONS:
        section = sections.get(section_name)
        if not section:
            continue
        content = section["content"] if isinstance(section, dict) else section
        last_seen_date: Optional[str] = None

        for block in group_job_blocks(content):
            job = _extract_job_from_block(block)
            if not job:
                continue

            role         = job["role_title"]
            company      = job["company"]
            date_range   = job["date_range"]

            # Strip locations that slipped through.
            if company and _is_location(company):
                company = None

            # Training section: a "role + date" without a real company is
            # almost always a course/certification name (e.g. "Web Developer
            # Jan 2024 - Mar 2024"). Drop these — they are not jobs/internships.
            if section_name == "training" and not company:
                continue
            # Same defensive rule for projects/open_source — a project
            # without an organisation isn't a paid position.
            if section_name in ("projects", "open_source") and not company:
                continue
            # Section-level date memory fallback.
            if not date_range and last_seen_date:
                date_range = last_seen_date
            elif date_range:
                last_seen_date = date_range

            existing = next(
                (p for p in positions if _same_job(
                    role, company, p["role_title"], p["company"]
                )),
                None,
            )
            if existing:
                existing["date_range"] = _better_date(
                    existing.get("date_range"), date_range
                )
                if not existing["company"] and company:
                    existing["company"] = company
                _existing_role = existing.get("role_title") or ""
                if role and len(role) > len(_existing_role):
                    existing["role_title"] = role
                    existing["seniority"] = detect_seniority(role)
                continue

            key = (_normalize_role_for_match(role), _normalize(company))
            if key in seen:
                continue
            seen.add(key)
            positions.append({
                "role_title": role,
                "company":    company or None,
                "date_range": date_range or None,
                "seniority":  detect_seniority(role),
                "section":    section_name,
            })

    return positions

def extract_bullets(sections: Dict[str, Dict]) -> List[Bullet]:
    """Walk narrative sections and emit one Bullet per line.

    Two-pass within each section:
      1. Walk lines once to find role-header positions and harvest a single
         date_range per role (looking forward up to 3 lines if the date
         isn't on the header line itself).
      2. Walk again, attaching every bullet to the most recent role header
         seen — which now carries a stable date_range for the whole block.
         This prevents the same role from being split into two positions
         (one with date_range=None, one with the date) when only some of
         its bullets happen to mention the date inline.
    """
    bullets: List[Bullet] = []
    counter = 0
    for section_name in _ROLE_SECTIONS:
        section = sections.get(section_name)
        if not section:
            continue
        content = section["content"] if isinstance(section, dict) else section

        # ---- Pass 1: locate role headers and their dates ------------------
        role_starts, claimed_indices = _detect_role_headers(content)

        # ---- Pass 2: emit bullets attached to the right role -------------
        def _role_for(idx: int):
            current = (None, None, None)
            for start_idx, role, company, date in role_starts:
                if start_idx <= idx:
                    current = (role, company, date)
                else:
                    break
            return current

        header_indices = claimed_indices
        for i, raw in enumerate(content):
            line = raw.strip()
            if len(line) < 4:
                continue
            if i in header_indices:
                continue                              # don't re-emit the header line itself
            date_match = _DATE_RANGE_RE.search(line)
            line_no_date = (
                _DATE_RANGE_RE.sub("", line).strip(" -\u2013\u2014|,") if date_match else line
            )
            # Skip lines that are pure date strings.
            if not line_no_date or line_no_date.strip(" -\u2013\u2014|,.") == "":
                continue

            role, company, date_range = _role_for(i)

            # Training-section special case removed — promoting a bare dated
            # line to a synthetic "role" is what produced ghost positions like
            # "Web Developer Jan 2024 - Mar 2024" on the dashboard. Course
            # names without a real company should stay as bullets, not become
            # their own job.

            if len(line_no_date.split()) >= 3:
                counter += 1
                bullets.append(Bullet(
                    id=f"bullet_{counter}",
                    text=line_no_date,
                    role_title=role,
                    company=company,
                    date_range=date_range,
                    section=section_name,
                ))
    return bullets

# Top-level entry
def preprocess(raw_text: str) -> Dict[str, Dict]:
    """Run full preprocessing pipeline.

    Returns a dict like:
        {
          "skills":     {"content": [...], "confidence": 0.95},
          "experience": {"content": [...], "confidence": 0.80},
          ...
        }
    A "_meta" key carries auxiliary data such as raw cleaned text.
    """
    cleaned = clean_text(raw_text)
    headers = detect_sections(cleaned)
    raw_sections = extract_section_content(cleaned, headers)

    output: Dict[str, Dict] = {}
    for name, content in raw_sections.items():
        if not validate_section(name, content):
            continue
        sec = Section(name=name, content=content)
        sec.confidence = assign_confidence(sec)
        output[name] = sec.to_dict()

    output["_meta"] = {
        "raw_text": cleaned,
        "n_sections": len(output),
        "n_chars": len(cleaned),
    }
    return output
