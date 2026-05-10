from __future__ import annotations

import re
from typing import Dict, List, Optional

# Section headers commonly used in JDs to introduce requirements.
REQ_HEADER_PATTERNS = [
    r"^requirements\b",
    r"^qualifications\b",
    r"^required qualifications\b",
    r"^minimum qualifications\b",
    r"^what you'?ll need\b",
    r"^what we'?re looking for\b",
    r"^must have\b",
    r"^you have\b",
    r"^your skills\b",
    r"^skills(?: required)?\b",
    r"^responsibilities\b",
    r"^the role\b",
    r"^about you\b",
]

REQ_END_PATTERNS = [
    r"^benefits\b", r"^perks\b", r"^what we offer\b", r"^we offer\b",
    r"^about us\b", r"^the company\b", r"^why join\b",
    r"^how to apply\b", r"^apply now\b", r"^equal opportunity\b",
    r"^diversity\b", r"^our culture\b",
]

_AT_COMPANY_RE = re.compile(
    r"\b(?:at|@|with)\s+([A-Z][A-Za-z0-9&.,\-' ]{2,40})\b"
)
_HIRING_RE = re.compile(
    r"^([A-Z][A-Za-z0-9&.,\-' ]{2,40})\s+is\s+(?:hiring|looking|seeking)",
    flags=re.MULTILINE,
)
_LOCATION_RE = re.compile(
    r"\b(?:Location|Based in|Office)\s*[:\-]\s*([A-Za-z, ]{3,60})",
    flags=re.IGNORECASE,
)

def _matches_any(line: str, patterns: List[str]) -> bool:
    norm = line.strip().lower().rstrip(":")
    return any(re.match(p, norm) for p in patterns)

def parse(raw_text: str) -> Dict:
    """Return a ParsedJob dict.

    Output:
        {
          "title": str,
          "company": Optional[str],
          "location": Optional[str],
          "summary": str,
          "raw_requirements_text": str,
          "raw_text": str
        }
    """
    text = (raw_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [ln.strip() for ln in text.split("\n")]
    non_empty = [ln for ln in lines if ln]

    title = non_empty[0] if non_empty else "Unknown role"
    title = re.sub(r"\s+at\s+.+$", "", title, flags=re.IGNORECASE).strip(" -|,")

    company: Optional[str] = None
    m = _HIRING_RE.search(text)
    if m:
        company = m.group(1).strip()
    else:
        m = _AT_COMPANY_RE.search(non_empty[0] if non_empty else "")
        if m:
            company = m.group(1).strip()

    location: Optional[str] = None
    m = _LOCATION_RE.search(text)
    if m:
        location = m.group(1).strip().rstrip(".")

    summary_lines: List[str] = []
    for ln in non_empty[1:]:
        if _matches_any(ln, REQ_HEADER_PATTERNS) or _matches_any(ln, REQ_END_PATTERNS):
            break
        summary_lines.append(ln)
        if len(" ".join(summary_lines)) > 400:
            break
    summary = " ".join(summary_lines).strip()

    raw_req = _extract_requirements_block(lines)

    return {
        "title": title or "Unknown role",
        "company": company,
        "location": location,
        "summary": summary,
        "raw_requirements_text": raw_req,
        "raw_text": text,
    }

def _extract_requirements_block(lines: List[str]) -> str:
    """Pull out everything between a requirements-style header and the next
    boilerplate header (benefits / about us / etc). If no header exists,
    return the whole document."""
    start: Optional[int] = None
    end: Optional[int] = None
    for i, ln in enumerate(lines):
        if start is None and _matches_any(ln, REQ_HEADER_PATTERNS):
            start = i + 1
            continue
        if start is not None and _matches_any(ln, REQ_END_PATTERNS):
            end = i
            break
    if start is None:
        return "\n".join(lines).strip()
    if end is None:
        end = len(lines)
    block = "\n".join(lines[start:end]).strip()
    return block or "\n".join(lines).strip()
