from __future__ import annotations

from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer

NOISE = {
    "worked", "task", "responsible", "various", "etc", "miscellaneous",
    "general", "able", "experience", "experienced", "duties", "role",
}

def rank_keywords(sections: Dict[str, Dict], top_n: int = 25) -> Dict:
    """Compute TF-IDF over per-section pseudo-documents.

    Returns:
        {
          "top_keywords": [{term, weight}, ...],
          "n_terms": int
        }
    """
    docs: List[str] = []
    for name, sec in sections.items():
        if name == "_meta":
            continue
        text = " ".join(sec.get("content", []))
        if text.strip():
            docs.append(text)

    if not docs:
        return {"top_keywords": [], "n_terms": 0}

    try:
        vec = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=400,
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z+\-#\.0-9]{1,}\b",
        )
        matrix = vec.fit_transform(docs)
    except ValueError:
        return {"top_keywords": [], "n_terms": 0}

    weights = matrix.sum(axis=0).A1
    terms = vec.get_feature_names_out()

    pairs = [
        {"term": t, "weight": round(float(w), 4)}
        for t, w in zip(terms, weights)
        if t not in NOISE and len(t) > 1
    ]
    pairs.sort(key=lambda x: -x["weight"])
    return {"top_keywords": pairs[:top_n], "n_terms": len(pairs)}
