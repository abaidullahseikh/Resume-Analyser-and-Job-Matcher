"""
Flask entry point — Resume Analyser & Job Matcher.

Routes:
    GET  /         single-page upload (resume + JD side-by-side)
    POST /analyse  run both analyser and matcher pipelines, render dashboard
"""

from __future__ import annotations

import io
import logging
import time
from typing import Dict, List, Optional

from flask import Flask, render_template, request

from analyser import build_final_analysis
from matcher import (
    parse_job,
    extract_requirements,
    match_score,
    link_evidence,
    aggregate_match,
    explain,
    match_keywords_per_section,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("resume_analyser")

_LIGATURES = {
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
    "ﬅ": "ft", "ﬆ": "st",
    "‐": "-", "–": "-", "—": "-", "−": "-",
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "•": "-", "●": "-", "▪": "-", "◦": "-",
    " ": " ",   # non-breaking space
    "​": "",    # zero-width space
}

import re as _re

_SECTION_HDR_RE = _re.compile(
    r"((?:PROFESSIONAL\s+)?(?:EXPERIENCE|EMPLOYMENT|WORK\s+HISTORY)"
    r"|EDUCATION|SKILLS?|PROJECTS?|CERTIFICATIONS?"
    r"|AWARDS?|PUBLICATIONS?|VOLUNTEER|TRAINING"
    r"|SUMMARY|OBJECTIVE|PROFILE|ACTIVITIES|HONORS?)"
    r"(?=[\s\n:]|$)",
    _re.IGNORECASE,
)

def _normalise_pdf_text(text: str) -> str:
    """Clean common PDF extraction artefacts before the analyser sees them."""
    if not text:
        return ""

    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = _re.sub(r"-\n([a-z])", r"\1", text)

    def _inject_newlines(m: _re.Match) -> str:
        s = m.string
        pos = m.start()
        line_start = s.rfind("\n", 0, pos) + 1
        prefix = s[line_start:pos]
        if not prefix.strip():
            return "\n\n" + m.group(1)
        if prefix.endswith("  ") or prefix.endswith("\t"):
            return "\n\n" + m.group(1)
        return m.group(0)
    text = _SECTION_HDR_RE.sub(_inject_newlines, text)

    text = _re.sub(r"\n{3,}", "\n\n", text)

    text = "\n".join(line.rstrip() for line in text.splitlines())

    def _collapse_letter_spaced(line: str) -> str:
        def _repl(m: _re.Match) -> str:
            chars = m.group(0).split()
            return "".join(chars)
        # 3+ single-letter tokens separated by single spaces.
        return _re.sub(r"\b(?:[A-Z]\s){2,}[A-Z]\b", _repl, line)
    text = "\n".join(_collapse_letter_spaced(ln) for ln in text.splitlines())

    text = _SECTION_HDR_RE.sub(_inject_newlines, text)

    text = _split_fused_dates(text)

    text = "\n".join(_re.sub(r"[ \t]{2,}", " ", line) for line in text.splitlines())

    text = _re.sub(r"\n{3,}", "\n\n", text)

    return text

_FUSED_DATE_RE = _re.compile(
    r"(?<=\S)"                          # must follow non-whitespace
    r"[ \t]{2,}"                        # 2+ horizontal spaces (not a newline)
    r"("
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
    r"\s+\d{4}"                         # Month YYYY
    r"|(?:0?[1-9]|1[0-2])/\d{4}"       # MM/YYYY
    r"|\d{4}"                           # bare YYYY
    r")"
    r"(?:\s*[-\u2013\u2014to]+\s*"
    r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}"
    r"|(?:0?[1-9]|1[0-2])/\d{4}"
    r"|\d{4}|[Pp]resent|[Cc]urrent|[Nn]ow))?",
    _re.IGNORECASE,
)

def _split_fused_dates(text: str) -> str:
    """Insert a newline before a date range that was fused onto a role line.

    PDFs often produce lines like:
      "Senior Software Engineer at Google Jan 2023 - Present"
    when the original resume had the date on the right margin.
    We break them into two lines so the preprocessor can pair them correctly.
    """
    lines = text.splitlines()
    out = []
    for line in lines:
        m = _FUSED_DATE_RE.search(line)
        if m and m.start() > 0:
            before = line[: m.start()].rstrip()
            date_part = line[m.start() :].strip()
            if before:
                out.append(before)
                out.append(date_part)
                continue
        out.append(line)
    return "\n".join(out)

def _pdfplumber_extract_page(page) -> str:
    """Reconstruct page text from word-level bounding boxes.

    Standard `extract_text()` uses fixed tolerances that can merge words
    from different visual lines (especially when dates are right-aligned).
    We group words by their vertical centre (top coordinate) instead,
    which preserves one-word-per-visual-line fidelity. We also break a row
    where horizontal gaps between adjacent words exceed a column-gap
    threshold so two-column PDFs don't fuse left and right columns into
    one line.
    """
    try:
        words = page.extract_words(
            x_tolerance=2,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=True,
        )
    except Exception:
        return page.extract_text(x_tolerance=2, y_tolerance=3) or ""

    if not words:
        return ""

    from collections import defaultdict
    rows: dict = defaultdict(list)
    for w in words:
        bucket = round(w["top"] / 4) * 4
        rows[bucket].append(w)

    widths = [w["x1"] - w["x0"] for w in words if w.get("x1") and w.get("x0")]
    median_w = sorted(widths)[len(widths) // 2] if widths else 6.0
    col_gap = max(40.0, median_w * 4.0)

    lines: List[str] = []
    for bucket in sorted(rows):
        row_words = sorted(rows[bucket], key=lambda w: w["x0"])
        if not row_words:
            continue
        segments: List[List[str]] = [[row_words[0]["text"]]]
        prev_x1 = row_words[0]["x1"]
        for w in row_words[1:]:
            if w["x0"] - prev_x1 > col_gap:
                segments.append([w["text"]])         # column break
            else:
                segments[-1].append(w["text"])
            prev_x1 = w["x1"]
        for seg in segments:
            lines.append(" ".join(seg))

    return "\n".join(lines)

def _extract_pdf_text(file_storage) -> str:
    """Extract text from a Werkzeug FileStorage holding a PDF.

    Tries three extractors and scores each candidate by structural quality
    (word count and average words/line) rather than raw character length —
    pdfplumber's per-word y-bucketing can otherwise win on length while
    producing one-word-per-line output that the section/role detector
    can't parse.
      1. pdfminer.six — best reading-order reconstruction for prose
      2. pdfplumber  — best layout for two-column PDFs (right-aligned dates)
      3. PyPDF2      — fast fallback
    Output is run through _normalise_pdf_text() to fix ligatures and the
    common -\\n hyphen-wrap pattern that breaks word-boundary regexes.
    """
    file_storage.stream.seek(0)
    raw = file_storage.stream.read()
    candidates: List[tuple[str, str]] = []   # (source_name, text)

    try:
        from pdfminer.high_level import extract_text as _pm_extract
        txt = _pm_extract(io.BytesIO(raw)) or ""
        if txt.strip():
            candidates.append(("pdfminer", txt))
    except Exception as e:
        log.info("pdfminer unavailable: %s", e)

    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            chunks = []
            for page in pdf.pages:
                txt = _pdfplumber_extract_page(page)
                if txt.strip():
                    chunks.append(txt)
            if chunks:
                candidates.append(("pdfplumber", "\n\n".join(chunks)))
    except Exception as e:
        log.warning("pdfplumber failed: %s", e)

    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        chunks = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                chunks.append(txt)
        if chunks:
            candidates.append(("PyPDF2", "\n\n".join(chunks)))
    except Exception as e:
        log.warning("PyPDF2 failed: %s", e)

    if not candidates:
        return ""

    def _quality(text: str) -> float:
        # Reward word count and discourage extreme line fragmentation.
        words = text.split()
        n_words = len(words)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines or not n_words:
            return 0.0
        avg_words_per_line = n_words / len(lines)
        # Penalise outputs where lines average <2 words (one-word-per-line PDFs)
        # but cap the bonus so a reasonable layout (~6 words/line) doesn't
        # dominate over genuinely longer prose extraction.
        line_bonus = min(avg_words_per_line, 8.0) / 8.0
        return n_words * (0.4 + 0.6 * line_bonus)

    best_name, best = max(candidates, key=lambda c: _quality(c[1]))
    log.info("PDF extractor selected: %s (%d candidates)", best_name, len(candidates))
    return _normalise_pdf_text(best)

def _resolve_resume_text(form, files) -> str:
    """Pick whichever input the user provided for the resume."""
    pasted = (form.get("resume_text") or "").strip()
    if pasted:
        return pasted
    file = files.get("resume_file")
    if file and file.filename:
        if file.filename.lower().endswith(".pdf"):
            return _extract_pdf_text(file)
        try:
            file.stream.seek(0)
            return file.stream.read().decode("utf-8", errors="ignore")
        except Exception:
            return ""
    return ""

def _resolve_job_text(form, files) -> str:
    pasted = (form.get("job_text") or "").strip()
    if pasted:
        return pasted
    file = files.get("job_file")
    if file and file.filename:
        if file.filename.lower().endswith(".pdf"):
            return _extract_pdf_text(file)
        try:
            file.stream.seek(0)
            return file.stream.read().decode("utf-8", errors="ignore")
        except Exception:
            return ""
    return ""

def _first_supporting_text(link: Dict) -> str:
    bullets = link.get("supporting_bullets") or []
    if not bullets:
        return ""
    return (bullets[0].get("bullet_text") or "").strip()

_FALLBACK_DATE_PAIR_RE = _re.compile(
    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}"
    r"|(?:0?[1-9]|1[0-2])/(?:19|20)\d{2}"
    r"|\b(?:19|20)\d{2}\b)"
    r"\s*[-–—to]+\s*"
    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}"
    r"|(?:0?[1-9]|1[0-2])/(?:19|20)\d{2}"
    r"|\b(?:19|20)\d{2}\b"
    r"|[Pp]resent|[Cc]urrent|[Nn]ow)",
    flags=_re.IGNORECASE,
)

def _inject_experience_fallback(
    analysis: Dict, match_result: Dict, sections: Dict
) -> None:
    """If the strict pipeline returned 0 months, recompute totals from raw
    section text using a permissive date scanner. Mutates match_result."""
    from analyser.experience_scorer import (
        _parse_range_months,
        _merge_intervals,
        format_years_months,
    )

    if match_result.get("estimated_resume_years") and match_result["estimated_resume_years"] > 0:
        return

    candidate_sections = ("experience", "training", "projects")
    intervals_by_section: Dict[str, List[tuple]] = {s: [] for s in candidate_sections}
    for sec_name in candidate_sections:
        sec = sections.get(sec_name)
        if not sec:
            continue
        content = sec["content"] if isinstance(sec, dict) else sec
        text = "\n".join(content) if isinstance(content, list) else str(content)
        for m in _FALLBACK_DATE_PAIR_RE.finditer(text):
            iv = _parse_range_months(f"{m.group(1)} - {m.group(2)}")
            if iv is not None and iv[1] > iv[0]:
                intervals_by_section[sec_name].append(iv)

    job_iv = intervals_by_section["experience"]
    intern_iv = intervals_by_section["training"]
    proj_iv = intervals_by_section["projects"]

    def _months(ivs):
        return sum(e - s for s, e in _merge_intervals(ivs))

    job_m = _months(job_iv)
    intern_m = _months(intern_iv)
    proj_m = _months(proj_iv)
    work_training = _months(job_iv + intern_iv)
    collected = _months(job_iv + intern_iv + proj_iv)

    if collected == 0:
        return

    if job_m > 0 and not match_result.get("estimated_resume_years"):
        match_result["estimated_resume_years"] = round(job_m / 12.0, 1)
        match_result["estimated_resume_label"] = format_years_months(job_m)
    if work_training > 0:
        match_result["work_training_resume_years"] = round(work_training / 12.0, 1)
        match_result["work_training_resume_label"] = format_years_months(work_training)
    match_result["collected_resume_years"] = round(collected / 12.0, 1)
    match_result["collected_resume_label"] = format_years_months(collected)

    n_pos = match_result.get("resume_positions") or (
        len(job_iv) + len(intern_iv) + len(proj_iv)
    )
    match_result["resume_positions"] = n_pos

    buckets = match_result.get("experience_kind_buckets") or {}
    for key, m in (("job", job_m), ("internship", intern_m), ("project", proj_m)):
        b = buckets.get(key) or {"months": 0, "count": 0, "merged_months": 0,
                                  "label": "0 months", "merged_label": "0 months"}
        b["merged_months"] = max(int(b.get("merged_months") or 0), m)
        b["months"] = max(int(b.get("months") or 0), m)
        b["merged_label"] = format_years_months(b["merged_months"])
        b["label"] = format_years_months(b["months"])
        buckets[key] = b
    match_result["experience_kind_buckets"] = buckets

def _build_roles_breakdown(analysis: Dict) -> List[Dict]:
    """Return every detected role with its bullets, sorted jobs → interns → projects.

    A role is ALWAYS included even when no bullets match the exact
    (role_title, company, date_range) triple — the dashboard must show the
    position itself even if bullet attribution failed.
    """
    roles = (analysis.get("role_experience") or {}).get("roles", []) or []

    # If the role_experience pipeline produced nothing, fall back to the raw
    # position list from extract_role_positions so the dashboard always shows
    # at least the detected position names.
    if not roles:
        from analyser.experience_scorer import (
            classify_position_kind,
            format_years_months,
            parse_duration_months,
        )
        for p in (analysis.get("positions") or []):
            getter = p.get if isinstance(p, dict) else (lambda k, _p=p: getattr(_p, k, None))
            rt = getter("role_title") or None
            co = getter("company") or None
            dr = getter("date_range") or ""
            if not (rt or co):
                continue
            months = parse_duration_months(dr)
            roles.append({
                "role_title": rt,
                "company": co,
                "date_range": dr,
                "kind": classify_position_kind(rt, co, getter("section")),
                "duration_label": format_years_months(months),
                "years": months / 12.0,
                "is_current": bool(_re.search(r"present|current|now", dr, _re.I)),
                "is_recent_year": False,
                "n_bullets": 0,
            })

    bullets = analysis.get("bullets") or []
    quality_map: Dict[str, str] = {
        b["text"]: b["label"]
        for b in (analysis.get("experience") or {}).get("bullets", [])
    }
    kind_order = {"job": 0, "internship": 1, "project": 2}
    result = []
    for role in sorted(roles, key=lambda r: kind_order.get(r.get("kind", "job"), 9)):
        rt = role.get("role_title")
        co = role.get("company")
        dr = role.get("date_range")
        role_bullets = [
            {"text": b["text"], "label": quality_map.get(b["text"], "moderate")}
            for b in bullets
            if b.get("role_title") == rt
            and b.get("company") == co
            and b.get("date_range") == dr
        ]
        # Fallback: if exact triple-match yielded nothing, fall back to
        # (role_title, company) so a role still gets its bullets even if the
        # date_range was harvested differently between passes.
        if not role_bullets and (rt or co):
            role_bullets = [
                {"text": b["text"], "label": quality_map.get(b["text"], "moderate")}
                for b in bullets
                if b.get("role_title") == rt
                and b.get("company") == co
            ]
        result.append(dict(role, role_bullets=role_bullets))
    return result

_DEGREE_LEVELS = {
    "phd": 5, "doctorate": 5, "doctoral": 5,
    "master": 4, "msc": 4, "mba": 4, "meng": 4, "mres": 4, "postgraduate": 4,
    "bachelor": 3, "bsc": 3, "ba": 3, "beng": 3, "undergraduate": 3, "degree": 3,
    "diploma": 2, "hnd": 2, "foundation": 2,
    "a-level": 1, "a level": 1, "high school": 1, "secondary": 1, "gcse": 1,
}

_FIELD_KEYWORDS: Dict[str, List[str]] = {
    "computer science": ["computer science", "computing", "software engineering", "cs"],
    "data science": ["data science", "data analytics", "data analysis"],
    "mathematics": ["mathematics", "maths", "math", "statistics", "statistical"],
    "engineering": ["engineering", "electrical", "mechanical", "civil", "chemical"],
    "business": ["business", "management", "commerce", "administration", "mba"],
    "information technology": ["information technology", "it ", "information systems"],
    "artificial intelligence": ["artificial intelligence", "ai", "machine learning", "deep learning"],
    "finance": ["finance", "accounting", "economics", "financial"],
    "science": ["science", "physics", "chemistry", "biology", "biochemistry"],
}

_DEGREE_RE = _re.compile(
    r"\b("
    r"ph\.?d|doctorate|doctoral|"
    r"m\.?sc|m\.?res|m\.?eng|m\.?b\.?a|master(?:s|'s)?|postgraduate|"
    r"b\.?sc|b\.?a|b\.?eng|b\.?tech|bachelor(?:s|'s)?|undergraduate|"
    r"diploma|h\.?n\.?d|foundation degree|"
    r"a[\s-]levels?|a level|high school|secondary school|gcse"
    r")\b",
    _re.IGNORECASE,
)

_FIELD_RE = _re.compile(
    r"\b(computer science|data science|software engineering|information technology|"
    r"artificial intelligence|machine learning|mathematics|statistics|physics|"
    r"chemistry|biology|biochemistry|engineering|electrical engineering|"
    r"mechanical engineering|civil engineering|business|management|"
    r"economics|finance|accounting|psychology|nursing|medicine|law|"
    r"design|architecture|media|communications|marketing)\b",
    _re.IGNORECASE,
)

def _fix_joined_words(text: str) -> str:
    """Insert spaces before uppercase letters that follow lowercase letters
    where a space is clearly missing (e.g. 'ComputerScience' → 'Computer Science')."""
    # Insert space before a capital that follows a lowercase (camelCase joins)
    text = _re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # Insert space before digits that follow letters with no space
    text = _re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)
    text = _re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)
    # Collapse any double spaces
    text = _re.sub(r" {2,}", " ", text)
    return text.strip()

def _parse_education_entries(sections: Dict) -> List[Dict]:
    """Extract structured degree entries from the education section."""
    edu_sec = sections.get("education")
    if not edu_sec:
        return []
    lines = edu_sec.get("content", []) if isinstance(edu_sec, dict) else []

    entries = []
    for line in lines:
        line = _fix_joined_words(line.strip())
        if not line or len(line) < 4:
            continue
        degree_m = _DEGREE_RE.search(line)
        if not degree_m:
            continue
        field_m = _FIELD_RE.search(line)
        degree_label = degree_m.group(0).upper().replace(".", "")
        field_label  = field_m.group(0).title() if field_m else ""
        name = f"{degree_label}" + (f" — {field_label}" if field_label else "")
        entries.append({"name": name, "full": line})

    return entries

def _match_education(sections: Dict, parsed_job: Dict) -> Dict:
    """Compare resume education section against JD education requirements."""
    jd_text = (parsed_job.get("raw_text") or "").lower()

    # ── Detect what the JD requires ──────────────────────────────────────
    required_level = 0
    required_level_label = ""
    for label, level in sorted(_DEGREE_LEVELS.items(), key=lambda x: -x[1]):
        if label in jd_text:
            required_level = level
            required_level_label = label.title()
            break

    required_fields: List[str] = []
    for field, keywords in _FIELD_KEYWORDS.items():
        if any(kw in jd_text for kw in keywords):
            required_fields.append(field.title())

    if not required_level and not required_fields:
        return {"required": False, "status": "none", "required_fields": []}

    # ── Scan resume education section ────────────────────────────────────
    edu_sec = sections.get("education")
    edu_lines = (edu_sec.get("content") or []) if isinstance(edu_sec, dict) else []
    edu_text = " ".join(edu_lines).lower()

    found_level = 0
    found_level_label = "—"
    for label, level in sorted(_DEGREE_LEVELS.items(), key=lambda x: -x[1]):
        if label in edu_text:
            found_level = level
            found_level_label = label.title()
            break

    found_fields = [f for f, kws in _FIELD_KEYWORDS.items() if any(kw in edu_text for kw in kws)]

    # ── Determine status ──────────────────────────────────────────────────
    level_ok = (found_level >= required_level) if required_level else True
    field_ok = (not required_fields) or bool(
        set(f.lower() for f in found_fields) &
        set(f.lower() for f in required_fields)
    )

    if level_ok and field_ok:
        status = "match"
    elif level_ok or field_ok:
        status = "partial"
    else:
        status = "missing"

    return {
        "required": True,
        "status": status,
        "required_level": required_level,
        "required_label": required_level_label or "Degree",
        "found_level": found_level,
        "found_label": found_level_label,
        "required_fields": required_fields,
        "found_fields": [f.title() for f in found_fields],
    }

def _section_gap_rows(section_match: Optional[Dict]) -> List[Dict]:
    if not section_match:
        return []
    rows = section_match.get("rows") or []
    missing = [r for r in rows if r.get("best_match_type") == "missing"]
    missing.sort(key=lambda r: (not r.get("is_required", True), r.get("keyword", "")))
    return missing[:10]

def _direct_strengths(links: List[Dict]) -> List[Dict]:
    direct_required = [
        l for l in links
        if l.get("link_type") == "direct" and l.get("is_required", True)
    ]
    direct_required.sort(key=lambda l: -float(l.get("strength") or 0))
    out = []
    for link in direct_required[:5]:
        out.append({
            "title": link["requirement_name"],
            "score": round(float(link.get("strength") or 0) * 100),
            "evidence": _first_supporting_text(link),
        })
    return out

def _push_action(
    actions: List[Dict],
    priority: str,
    title: str,
    detail: str,
    evidence: str = "",
) -> None:
    actions.append({
        "priority": priority,
        "title": title,
        "detail": detail,
        "evidence": evidence,
    })

def _build_action_plan(
    analysis: Dict,
    links: List[Dict],
    match_result: Dict,
    section_match: Optional[Dict],
) -> List[Dict]:
    actions: List[Dict] = []

    missing_required = [
        l for l in links
        if l.get("link_type") == "missing" and l.get("is_required", True)
    ]
    for link in missing_required[:4]:
        _push_action(
            actions,
            "high",
            f"Add evidence for {link['requirement_name']}",
            "A required job requirement has no grounded resume evidence.",
            link.get("reasoning") or "",
        )

    inferred_required = [
        l for l in links
        if l.get("link_type") == "inferred" and l.get("is_required", True)
    ]
    inferred_required.sort(key=lambda l: float(l.get("strength") or 0))
    for link in inferred_required[:3]:
        _push_action(
            actions,
            "medium",
            f"Make {link['requirement_name']} explicit",
            "The matcher can infer this skill, but the resume should name it in a work bullet.",
            _first_supporting_text(link),
        )

    if match_result.get("experience_verdict") == "under":
        _push_action(
            actions,
            "high",
            "Clarify professional experience duration",
            "The JD year requirement is above the resume's professional experience estimate.",
            match_result.get("estimated_resume_label", ""),
        )

    weak_bullets = [
        b for b in analysis.get("experience", {}).get("bullets", [])
        if b.get("label") == "weak"
    ]
    if weak_bullets:
        _push_action(
            actions,
            "medium",
            "Rewrite weak experience bullets",
            "Replace passive wording with action verbs and measurable outcomes.",
            weak_bullets[0].get("text", ""),
        )

    for row in _section_gap_rows(section_match)[:3]:
        _push_action(
            actions,
            "low" if not row.get("is_required") else "medium",
            f"Cover keyword: {row.get('keyword')}",
            "This JD keyword is absent from the detected resume sections.",
            "",
        )

    for suggestion in analysis.get("suggestions", [])[:3]:
        _push_action(actions, "low", "Resume polish", suggestion)

    seen = set()
    deduped = []
    rank = {"high": 0, "medium": 1, "low": 2}
    for item in actions:
        key = (item["priority"], item["title"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    deduped.sort(key=lambda item: rank.get(item["priority"], 9))
    return deduped[:8]

def _build_evidence_audit(
    links: List[Dict],
    requirements: List[Dict],
    section_match: Optional[Dict],
) -> Dict:
    required = [r for r in requirements if r.get("is_required", True)]
    direct_required = [
        l for l in links
        if l.get("is_required", True) and l.get("link_type") == "direct"
    ]
    inferred_required = [
        l for l in links
        if l.get("is_required", True) and l.get("link_type") == "inferred"
    ]
    missing_required = [
        l for l in links
        if l.get("is_required", True) and l.get("link_type") == "missing"
    ]
    total_required = max(1, len(required))
    keyword_summary = (section_match or {}).get("summary", {})
    n_keywords = keyword_summary.get("n_keywords") or 0
    n_with_hit = keyword_summary.get("n_with_hit") or 0
    return {
        "required_total": len(required),
        "required_direct": len(direct_required),
        "required_inferred": len(inferred_required),
        "required_missing": len(missing_required),
        "required_coverage_pct": round(
            ((len(direct_required) + len(inferred_required)) / total_required) * 100
        ),
        "keyword_coverage_pct": round((n_with_hit / n_keywords) * 100) if n_keywords else 0,
        "keyword_missing": _section_gap_rows(section_match),
    }

def _build_impression_pack(
    analysis: Dict,
    requirements: List[Dict],
    links: List[Dict],
    match_result: Dict,
    section_match: Optional[Dict],
) -> Dict:
    dimensions = match_result.get("breakdown", {})
    weakest_axis = None
    if dimensions:
        weakest_key = min(dimensions, key=lambda k: dimensions[k])
        weakest_axis = {
            "name": weakest_key.replace("_", " ").title(),
            "score": dimensions[weakest_key],
        }

    return {
        "audit": _build_evidence_audit(links, requirements, section_match),
        "strengths": _direct_strengths(links),
        "action_plan": _build_action_plan(analysis, links, match_result, section_match),
        "weakest_axis": weakest_axis,
        "resume_suggestions": analysis.get("suggestions", [])[:5],
        "insights": analysis.get("insights", [])[:5],
    }

# Routes
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyse", methods=["POST"])
def analyse():
    t0 = time.time()
    timings: Dict[str, float] = {}

    resume_text = _resolve_resume_text(request.form, request.files)
    job_text = _resolve_job_text(request.form, request.files)

    if not resume_text or not job_text:
        return render_template(
            "index.html",
            error="Please provide both a resume and a job description.",
        )

    # ---- Resume analyser pipeline ----------------------------------------
    t = time.time()
    analysis = build_final_analysis(resume_text)
    timings["analyser"] = time.time() - t

    # Bullets and sections are already attached to the analyser output —
    # reuse them so the matcher pipeline doesn't re-extract.
    sections = analysis["sections"]
    bullets = analysis["bullets"]

    # ---- Matcher pipeline -------------------------------------------------
    t = time.time()
    parsed_job = parse_job(job_text)
    timings["job_parse"] = time.time() - t

    t = time.time()
    requirements = extract_requirements(parsed_job)
    timings["req_extract"] = time.time() - t

    t = time.time()
    match_out = match_score(requirements, bullets)
    timings["matcher"] = time.time() - t

    t = time.time()
    links = link_evidence(requirements, bullets, match_out, sections=sections)
    match_result = aggregate_match(
        links,
        bullets,
        requirements,
        analysis.get("role_experience"),
    )
    timings["link_aggregate"] = time.time() - t

    t = time.time()
    explanation = explain(sections, parsed_job, links, match_result)
    timings["explain"] = time.time() - t

    # Per-section keyword × heading match (synonym-aware).
    t = time.time()
    section_match = match_keywords_per_section(
        sections, requirements, use_semantic=False
    )
    timings["section_match"] = time.time() - t

    impression = _build_impression_pack(
        analysis, requirements, links, match_result, section_match
    )

    timings["total"] = time.time() - t0

    # Group links for the dashboard accordion (direct, inferred, missing).
    grouped: Dict[str, List[Dict]] = {"direct": [], "inferred": [], "missing": []}
    for l in links:
        grouped[l["link_type"]].append(l)
    grouped["direct"].sort(key=lambda l: -l["strength"])
    grouped["inferred"].sort(key=lambda l: -l["strength"])
    grouped["missing"].sort(key=lambda l: (not l["is_required"], l["requirement_name"]))

    roles_breakdown = _build_roles_breakdown(analysis)
    education_match = _match_education(sections, parsed_job)
    education_entries = _parse_education_entries(sections)

    # Fallback experience detection: when the strict role/date pipeline above
    # produced 0 months (date format the strict matcher couldn't link), scan
    # the raw experience/training/projects section text for any year/month
    # ranges and inject computed totals so the dashboard always shows a value.
    _inject_experience_fallback(analysis, match_result, sections)

    return render_template(
        "dashboard.html",
        analysis=analysis,
        parsed_job=parsed_job,
        requirements=requirements,
        match=match_result,
        links_grouped=grouped,
        explanation=explanation,
        section_match=section_match,
        impression=impression,
        roles_breakdown=roles_breakdown,
        education_match=education_match,
        education_entries=education_entries,
        timings={k: round(v, 3) for k, v in timings.items()},
    )

# Debug helpers
@app.route("/debug-extract", methods=["GET", "POST"])
def debug_extract():
    """Upload a PDF and see the raw text the parser receives."""
    from flask import Response
    if request.method == "POST":
        file = request.files.get("resume_file")
        if not file or not file.filename:
            return Response("No file uploaded.", mimetype="text/plain")
        text = _extract_pdf_text(file)
        header = f"=== Extracted text ({len(text)} chars) ===\n\n"
        return Response(header + text, mimetype="text/plain; charset=utf-8")
    return Response(
        """<!doctype html><html><body>
        <h2>PDF extraction debugger</h2>
        <form method=post enctype=multipart/form-data>
          <input type=file name=resume_file accept=".pdf"><br><br>
          <button type=submit>Extract text</button>
        </form></body></html>""",
        mimetype="text/html",
    )

@app.route("/debug-positions", methods=["GET", "POST"])
def debug_positions():
    """Upload a PDF or paste text — see every detected job position."""
    from flask import Response
    if request.method == "POST":
        from analyser.preprocessor import preprocess, extract_role_positions
        pasted = (request.form.get("resume_text") or "").strip()
        file = request.files.get("resume_file")
        if pasted:
            text = pasted
        elif file and file.filename:
            text = _extract_pdf_text(file) if file.filename.lower().endswith(".pdf") \
                else file.stream.read().decode("utf-8", errors="ignore")
        else:
            return Response("No input provided.", mimetype="text/plain")

        sections = preprocess(text)
        positions = extract_role_positions(sections)

        rows = []
        for i, p in enumerate(positions, 1):
            getter = p.get if isinstance(p, dict) else (lambda k, _p=p: getattr(_p, k, None))
            rows.append(
                f"<tr><td>{i}</td>"
                f"<td>{getter('role_title') or '—'}</td>"
                f"<td>{getter('company') or '—'}</td>"
                f"<td>{getter('date_range') or '—'}</td>"
                f"<td>{getter('section') or '—'}</td></tr>"
            )

        table = (
            "<table border=1 cellpadding=6 style='border-collapse:collapse'>"
            "<tr><th>#</th><th>Role Title</th><th>Company</th>"
            "<th>Date Range</th><th>Section</th></tr>"
            + ("".join(rows) if rows else "<tr><td colspan=5>No positions detected</td></tr>")
            + "</table>"
        )

        extracted_block = (
            f"<details><summary>Extracted text ({len(text)} chars)</summary>"
            f"<pre style='white-space:pre-wrap;font-size:12px'>{text[:8000]}</pre></details>"
        )

        return Response(
            f"""<!doctype html><html><body>
            <h2>Detected positions ({len(positions)})</h2>
            {table}<br>{extracted_block}
            <br><a href='/debug-positions'>&#8592; Try another</a>
            </body></html>""",
            mimetype="text/html",
        )

    return Response(
        """<!doctype html><html><body>
        <h2>Position detection debugger</h2>
        <form method=post enctype=multipart/form-data>
          <p><b>Upload PDF:</b><br>
          <input type=file name=resume_file accept=".pdf,.txt"></p>
          <p><b>Or paste text:</b><br>
          <textarea name=resume_text rows=12 cols=70></textarea></p>
          <button type=submit>Detect positions</button>
        </form></body></html>""",
        mimetype="text/html",
    )

# Main
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
