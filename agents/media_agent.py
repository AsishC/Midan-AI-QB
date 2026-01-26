"""Media Agent

Purpose
-------
Generate candidate media URLs for questions and store them in the
`MediaCandidate` table.

Stability rules (production):
- Do not crash a request because one question fails.
- Do not swallow exceptions silently (return errors and print tracebacks).

This module uses Wikimedia Commons search (no API key) as a baseline.
"""

from __future__ import annotations

import os
import requests
import re
import traceback
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from database import MediaCandidate, Question
from utils.wikipedia_tools import wikimedia_search_images

# Optional sources (require API keys). These are merged with Wikimedia/Openverse results.
def _unsplash_search_images(query: str, limit: int = 10):
    key = os.getenv("UNSPLASH_ACCESS_KEY") or os.getenv("UNSPLASH_KEY")
    if not key:
        return []
    try:
        url = "https://api.unsplash.com/search/photos"
        params = {"query": query, "per_page": min(max(limit, 1), 30)}
        headers = {"Authorization": f"Client-ID {key}"}
        if os.getenv("DEBUG_MEDIA", "0") == "1":
            print(f"[DEBUG_MEDIA] Unsplash API request: q={query!r} limit={params['per_page']}")
        if os.getenv("DEBUG_MEDIA","0")=="1":
            print(f"[DEBUG_MEDIA] Unsplash API request: q={query!r} limit={limit}")
        if os.getenv("DEBUG_MEDIA","0")=="1":
           print(f"[DEBUG_MEDIA] Pexels API request: q={query!r} limit={limit}")
        resp = requests.get(url, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json() or {}
        out = []
        for item in (data.get("results") or []):
            urls = item.get("urls") or {}
            full = urls.get("regular") or urls.get("full") or urls.get("raw")
            thumb = urls.get("thumb") or urls.get("small")
            if full:
                out.append({
                    "url": full,
                    "thumb_url": thumb or "",
                    "title": item.get("description") or item.get("alt_description") or "",
                    "license": "Unsplash",
                    "source": "Unsplash",
                })
        return out[:limit]
    except Exception as e:
        if os.getenv("DEBUG_MEDIA", "0") == "1":
            print(f"[DEBUG_MEDIA] Unsplash API error: {e}")
        return []

def _pexels_search_images(query: str, limit: int = 10):
    key = os.getenv("PEXELS_API_KEY") or os.getenv("PEXELS_KEY")
    if not key:
        return []
    try:
        url = "https://api.pexels.com/v1/search"
        params = {"query": query, "per_page": min(max(limit, 1), 80)}
        headers = {"Authorization": key}
        if os.getenv("DEBUG_MEDIA", "0") == "1":
            print(f"[DEBUG_MEDIA] Pexels API request: q={query!r} limit={params['per_page']}")
        if os.getenv("DEBUG_MEDIA","0")=="1":
            print(f"[DEBUG_MEDIA] Unsplash API request: q={query!r} limit={limit}")
        resp = requests.get(url, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json() or {}
        out = []
        for item in (data.get("photos") or []):
            src = item.get("src") or {}
            full = src.get("large") or src.get("original") or src.get("medium")
            thumb = src.get("tiny") or src.get("small")
            if full:
                out.append({
                    "url": full,
                    "thumb_url": thumb or "",
                    "title": item.get("alt") or "",
                    "license": "Pexels",
                    "source": "Pexels",
                })
        return out[:limit]
    except Exception as e:
        if os.getenv("DEBUG_MEDIA", "0") == "1":
            print(f"[DEBUG_MEDIA] Pexels API error: {e}")
        return []

def _merge_candidates(*lists):
    seen = set()
    merged = []
    for lst in lists:
        for c in (lst or []):
            u = (c.get("url") or "").strip()
            if not u or u in seen:
                continue
            seen.add(u)
            merged.append(c)
    return merged


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _build_media_query(q: Question) -> str:
    """Build a pragmatic search query for Wikimedia.

    We bias toward the *answer* (entity name) plus a small amount of context
    (category/subtopic) to reduce ambiguity.
    """

    parts: List[str] = []
    if getattr(q, "answer_en", None):
        parts.append(str(q.answer_en))
    if getattr(q, "subtopic", None):
        parts.append(str(q.subtopic))
    # If answer is missing (should not happen), fall back to question text.
    if not parts and getattr(q, "question_en", None):
        parts.append(str(q.question_en))
    return _normalize_space(" ".join(parts))


def collect_media_candidates(q_or_query, limit: int = 8) -> List[Dict[str, Any]]:
    """Return a list of candidate dicts for a question.

    Important behavior (v30 stable):
    - Always attempt ALL enabled sources (Commons/Openverse, Unsplash, Pexels).
    - A failure in one source must not prevent calling the others.
    - Results are merged + deduplicated by URL.
    """

    # Accept either a Question instance or a raw query string
    if isinstance(q_or_query, str):
        query = q_or_query.strip()
    else:
        query = _build_media_query(q_or_query)
    if not query:
        return []

    if os.getenv("DEBUG_MEDIA", "0") == "1":
        print(f"[DEBUG_MEDIA] Media query built: {query!r} (limit={limit})")

    wikimedia = []
    unsplash = []
    pexels = []

    # 1) Wikimedia/Openverse (no key)
    try:
        wikimedia = wikimedia_search_images(query, limit=limit) or []
    except Exception as e:
        if os.getenv("DEBUG_MEDIA", "0") == "1":
            print(f"[DEBUG_MEDIA] Wikimedia/Openverse call error: {e}")
        wikimedia = []

    # 2) Unsplash (requires key)
    try:
        unsplash = _unsplash_search_images(query, limit=limit) or []
    except Exception as e:
        if os.getenv("DEBUG_MEDIA", "0") == "1":
            print(f"[DEBUG_MEDIA] Unsplash call error: {e}")
        unsplash = []

    # 3) Pexels (requires key)
    try:
        pexels = _pexels_search_images(query, limit=limit) or []
    except Exception as e:
        if os.getenv("DEBUG_MEDIA", "0") == "1":
            print(f"[DEBUG_MEDIA] Pexels call error: {e}")
        pexels = []

    results = _merge_candidates(wikimedia, unsplash, pexels)

    if os.getenv("DEBUG_MEDIA", "0") == "1":
        # Also show whether keys were detected (without printing the key).
        has_unsplash = bool(os.getenv("UNSPLASH_ACCESS_KEY") or os.getenv("UNSPLASH_KEY"))
        has_pexels = bool(os.getenv("PEXELS_API_KEY") or os.getenv("PEXELS_KEY"))
        print(
            f"[DEBUG_MEDIA] results: wikimedia={len(wikimedia)} unsplash={len(unsplash)} "
            f"pexels={len(pexels)} merged={len(results)} keys:unsplash={has_unsplash} pexels={has_pexels}"
        )

    candidates: List[Dict[str, Any]] = []
    for r in results or []:
        url = (r.get("url") or "").strip()
        if not url:
            continue
        candidates.append({
            "url": url,
            "source": (r.get("source") or "").strip(),
            "title": (r.get("title") or "").strip(),
            "license": (r.get("license") or "").strip(),
            "thumb_url": (r.get("thumb_url") or "").strip(),
        })

    return candidates


def run_media_agent_for_question_ids(db: Session, ids: List[int]) -> Dict[str, Any]:
    """Generate media candidates for the provided question IDs.

    Returns a structured status dictionary, for logging/audit.
    """

    stats: Dict[str, Any] = {
        "requested": len(ids),
        "processed": 0,
        "questions_with_candidates": 0,
        "candidates_inserted": 0,
        "errors": [],
    }

    if not ids:
        return stats

    for qid in ids:
        try:
            q = db.query(Question).filter(Question.id == qid).first()
            if not q:
                continue

            stats["processed"] += 1

            # Append mode: keep existing candidates and only insert new URLs.
            existing_urls = {row.url for row in db.query(MediaCandidate.url).filter(MediaCandidate.question_id == qid).all()}
            max_rank_row = db.query(func.max(MediaCandidate.rank)).filter(MediaCandidate.question_id == qid).scalar()
            next_rank = int(max_rank_row) + 1 if max_rank_row is not None else 0

            candidates = collect_media_candidates(q)
            if os.getenv("DEBUG_MEDIA", "0") == "1":
                print(f"[media_agent] Q#{qid} answer={getattr(q,'answer_en',None)!r} candidates={len(candidates)}")
            if candidates:
                stats["questions_with_candidates"] += 1

            for idx, c in enumerate(candidates):
                u = (c.get('url') or '').strip()
                if not u or u in existing_urls:
                    continue
                existing_urls.add(u)
                mc = MediaCandidate(
                    question_id=qid,
                    url=u,
                    source=c.get("source", ""),
                    rank=next_rank + idx,
                    meta={
                        "title": c.get("title", ""),
                        "license": c.get("license", ""),
                        "thumb_url": c.get("thumb_url", ""),
                    },
                )
                db.add(mc)
                stats["candidates_inserted"] += 1

            # Mark media as needing review if candidates exist and no final URL.
            if candidates and not (getattr(q, "final_media_url", None) or "").strip():
                if getattr(q, "media_status", None) != "APPROVED":
                    q.media_status = "REVIEW_REQUIRED"

            db.commit()

        except Exception as e:
            db.rollback()
            traceback.print_exc()
            stats["errors"].append({"question_id": qid, "error": str(e)})

    return stats


def build_media_hint(*args: str) -> str:
    """Build a short, human-readable hint for media generation.

    The UI has used multiple call signatures over time. We support:
      - build_media_hint(question_text, answer_text)
      - build_media_hint(cat_en, cat_ar, subtopic, q_en, q_ar, a_en, region)

    Extra/empty values are ignored.
    """

    vals = [(_normalize_space(a) if isinstance(a, str) else "") for a in args]

    # v1/v2 signature: (question, answer)
    if len(vals) <= 2:
        question_text = vals[0] if len(vals) > 0 else ""
        answer_text = vals[1] if len(vals) > 1 else ""
        if question_text and answer_text:
            return f"{answer_text} ({question_text})"
        return answer_text or question_text

    # v3 signature: (cat_en, cat_ar, subtopic, q_en, q_ar, a_en, region)
    cat_en = vals[0] if len(vals) > 0 else ""
    subtopic = vals[2] if len(vals) > 2 else ""
    q_en = vals[3] if len(vals) > 3 else ""
    a_en = vals[5] if len(vals) > 5 else ""
    region = vals[6] if len(vals) > 6 else ""

    parts: List[str] = []
    if cat_en:
        parts.append(cat_en)
    if subtopic:
        parts.append(subtopic)
    core = " | ".join(parts)

    hint_bits: List[str] = []
    if a_en:
        hint_bits.append(a_en)
    if q_en:
        hint_bits.append(q_en)
    hint = " — ".join([b for b in hint_bits if b])

    if core and hint:
        out = f"{core}: {hint}"
    else:
        out = core or hint
    if region:
        out = f"{out} [{region}]" if out else f"[{region}]"
    return out