from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

_cross_encoder = None
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-2-v2"

def _get_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)
    return _cross_encoder

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _bullet_to_text(b: Dict) -> str:
    """Render a bullet for the cross-encoder, using role/company context."""
    role = b.get("role_title") or ""
    company = b.get("company") or ""
    prefix = f"{role} at {company}: " if (role or company) else ""
    return f"{prefix}{b['text']}".strip()

def _requirement_to_text(r: Dict) -> str:
    raw = (r.get("raw_text") or "").strip()
    name = (r.get("name") or "").strip()
    if name and name.lower() not in raw.lower():
        return f"{name} | {raw}"
    return raw or name

def score(
    requirements: List[Dict],
    bullets: List[Dict],
    direct_threshold: float = 0.6,
    batch_size: int = 32,
) -> Dict:
    """Compute a (#reqs x #bullets) score matrix and per-requirement summaries.

    Parameters
    ----------
    requirements : list of Requirement dicts (from requirement_extractor)
    bullets      : list of Bullet dicts (from preprocessor.extract_bullets)
    direct_threshold : per-pair score above which we count "coverage"
    batch_size   : cross-encoder batch size

    Returns
    -------
    dict with:
      score_matrix   : np.ndarray of shape (R, B), float in [0,1]
      per_requirement: list, length R, each entry has best_score, top_3, coverage
    """
    if not requirements or not bullets:
        return {
            "score_matrix": np.zeros((max(len(requirements), 0), max(len(bullets), 0))),
            "per_requirement": [
                {"req_id": r["id"], "best_score": 0.0, "top_3": [], "coverage": 0}
                for r in requirements
            ],
        }

    eligible_bullet_idx = [i for i, b in enumerate(bullets) if len(b["text"].split()) >= 4]
    if not eligible_bullet_idx:
        eligible_bullet_idx = list(range(len(bullets)))

    pairs: List[List[str]] = []
    pair_index: List = []   # (req_idx, bullet_idx)
    for ri, req in enumerate(requirements):
        rt = _requirement_to_text(req)
        for bi in eligible_bullet_idx:
            pairs.append([rt, _bullet_to_text(bullets[bi])])
            pair_index.append((ri, bi))

    raw_logits = _safe_predict(pairs, batch_size=batch_size)
    probs = _sigmoid(np.asarray(raw_logits, dtype=float))

    matrix = np.zeros((len(requirements), len(bullets)), dtype=float)
    for (ri, bi), p in zip(pair_index, probs):
        matrix[ri, bi] = float(p)

    per_req: List[Dict] = []
    for ri, req in enumerate(requirements):
        row = matrix[ri]
        if row.size == 0:
            per_req.append({"req_id": req["id"], "best_score": 0.0, "top_3": [], "coverage": 0})
            continue
        best_score = float(row.max())
        top_3 = list(np.argsort(-row)[:3])
        coverage = int((row > direct_threshold).sum())
        per_req.append({
            "req_id": req["id"],
            "best_score": round(best_score, 4),
            "top_3": [int(i) for i in top_3],
            "coverage": coverage,
        })

    return {"score_matrix": matrix, "per_requirement": per_req}

def _safe_predict(pairs: List[List[str]], batch_size: int) -> List[float]:
    """Run the cross-encoder; if model can't be loaded, fall back to a
    deterministic lexical-overlap baseline so the system still functions."""
    try:
        encoder = _get_encoder()
        return list(encoder.predict(pairs, batch_size=batch_size, show_progress_bar=False))
    except Exception:
        return [_lexical_baseline(a, b) for a, b in pairs]

def _lexical_baseline(req_text: str, bullet_text: str) -> float:
    """Token-overlap fallback (~Jaccard) when the cross-encoder is unavailable.

    Returned as a logit so the same sigmoid is applied downstream.
    """
    a = {t.lower().strip(".,;:") for t in req_text.split() if len(t) > 2}
    b = {t.lower().strip(".,;:") for t in bullet_text.split() if len(t) > 2}
    if not a or not b:
        return -4.0
    jaccard = len(a & b) / len(a | b)
    # Map [0,1] Jaccard into a useful logit range.
    return -4.0 + 8.0 * jaccard
