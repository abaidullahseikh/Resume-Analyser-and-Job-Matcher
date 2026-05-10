"""
Section Keyword Matcher — synonym index
Provides `_equivalents(base)` used by evidence_linker to expand a requirement
keyword into its full synonym set before scanning resume sections.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Set

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

@lru_cache(maxsize=1)
def _build_index() -> dict:
    """Return {lowercase_term: frozenset_of_lowercase_synonyms}."""
    path = os.path.join(DATA_DIR, "skill_synonyms.json")
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}

    index: dict = {}
    for synset in data.get("synsets", []):
        lc_set = frozenset(t.strip().lower() for t in synset if t.strip())
        for term in lc_set:
            index[term] = lc_set
    return index

def _equivalents(base: str) -> Set[str]:
    """Return the set of lowercase synonyms for *base*, including *base* itself."""
    lc = base.strip().lower()
    if not lc:
        return set()
    index = _build_index()
    return set(index.get(lc, {lc}))
