from __future__ import annotations

import re
from typing import Dict

_flesch_reading_ease = None
try:
    from textstat import flesch_reading_ease as _flesch_reading_ease  # type: ignore[attr-defined]
    _TEXTSTAT_OK = True
except Exception:                                 # pragma: no cover
    _TEXTSTAT_OK = False

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "as", "is", "was", "were", "be",
    "been", "being", "are", "am", "i", "you", "he", "she", "it", "we",
    "they", "this", "that", "these", "those", "my", "your", "their",
    "our", "his", "her", "its", "have", "has", "had", "will", "would",
    "could", "should", "do", "does", "did", "not", "no", "so", "than",
    "then", "there", "here", "where", "when", "while", "also", "very",
}

def _flesch(text: str) -> float:
    if _TEXTSTAT_OK and _flesch_reading_ease is not None and text:
        try:
            return float(_flesch_reading_ease(text))
        except Exception:
            pass
    sentences = max(len(re.findall(r"[.!?]+", text)), 1)
    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        return 0.0
    total_syllables = sum(_estimate_syllables(w) for w in words)
    return 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (total_syllables / len(words))

def _estimate_syllables(word: str) -> int:
    word = word.lower()
    vowels = "aeiouy"
    count, prev_vowel = 0, False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)

def _lexical_density(text: str) -> float:
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    if not tokens:
        return 0.0
    content_tokens = [t for t in tokens if t not in STOPWORDS]
    return len(content_tokens) / len(tokens)

def score_content(sections: Dict[str, Dict]) -> Dict:
    """Score readability and lexical density."""
    full_text = " ".join(
        line
        for sec_name, sec in sections.items()
        if sec_name != "_meta"
        for line in sec.get("content", [])
    )

    if not full_text.strip():
        return {
            "score": 0,
            "flesch": 0.0,
            "lexical_density": 0.0,
            "n_words": 0,
        }

    flesch = _flesch(full_text)
    density = _lexical_density(full_text)
    words = len(re.findall(r"[A-Za-z]+", full_text))

    if 30 <= flesch <= 60:
        flesch_score = 1.0
    elif 60 < flesch <= 80 or 20 <= flesch < 30:
        flesch_score = 0.75
    else:
        flesch_score = 0.5

    if 0.55 <= density <= 0.75:
        density_score = 1.0
    elif 0.45 <= density < 0.55 or 0.75 < density <= 0.85:
        density_score = 0.75
    else:
        density_score = 0.5

    composite = 0.5 * flesch_score + 0.5 * density_score
    return {
        "score": round(composite * 100),
        "flesch": round(flesch, 2),
        "lexical_density": round(density, 3),
        "n_words": words,
    }
