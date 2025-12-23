import os
from typing import List, Tuple, Dict

from sqlalchemy.orm import Session

from database import Question, MediaCandidate, Category
from utils.unsplash_pexels_api import search_unsplash, search_pexels
from utils.wikipedia_tools import wikimedia_search_images


def build_media_hint(
    category_name: str,
    subtopic: str = "",
    hint: str = "",
    stem_en: str = "",
    answer_en: str = "",
    question_type: str = "",
    region: str = "",
) -> str:
    """
    Build a short English query string for media search.

    The query is used for Unsplash / Pexels / Wikimedia etc. We bias strongly
    toward the final *answer* text so that selected media is tightly related
    to the correct answer, not just the broad category.
    """
    pieces: List[str] = []

    # Put the exact answer first – this is the most important signal.
    if answer_en:
        pieces.append(answer_en)

    # Then supporting context.
    if subtopic:
        pieces.append(subtopic)
    if hint:
        pieces.append(hint)
    if stem_en:
        pieces.append(stem_en)
    if category_name:
        pieces.append(category_name)

    # Light regional flavour without over-constraining geography.
    if region and region.lower() not in ("", "global", "world"):
        pieces.append(region)

    qtype = (question_type or "").lower()
    if qtype in ("picture", "image", "photo", "logo", "icon"):
        pieces.append("photo")
    elif qtype in ("video", "clip"):
        pieces.append("video still")
    elif qtype in ("audio", "sound", "music"):
        pieces.append("audio")

    query = " ".join(pieces).strip()
    return query or category_name


def _collect_urls(
    query: str,
) -> List[Tuple[str, str, float]]:
    """
    Return a list of (url, source, base_score).
    The base_score is later adjusted by the caller if needed.
    """
    if not query:
        return []

    out: List[Tuple[str, str, float]] = []

    try:
        unsplash_results = search_unsplash(query)
        for r in unsplash_results:
            url = r if isinstance(r, str) else (r.get("url") if isinstance(r, dict) else None)
            if url:
                out.append((url, "unsplash", 0.80))
    except Exception as e:
        print("[MEDIA] Unsplash error:", e)

    try:
        pexels_results = search_pexels(query)
        for r in pexels_results:
            url = r if isinstance(r, str) else (r.get("url") if isinstance(r, dict) else None)
            if url:
                out.append((url, "pexels", 0.75))
    except Exception as e:
        print("[MEDIA] Pexels error:", e)

    try:
        wiki_results = wikimedia_search_images(query)
        for r in wiki_results:
            url = r if isinstance(r, str) else (r.get("url") if isinstance(r, dict) else None)
            if url:
                out.append((url, "wikimedia", 0.70))
    except Exception as e:
        print("[MEDIA] Wikimedia error:", e)

    return out


def _pick_best(
    urls: List[Tuple[str, str, float]],
) -> Tuple[str, str, float]:
    """
    Very simple: pick the highest-scoring candidate.
    """
    if not urls:
        return "", "", 0.0
    urls_sorted = sorted(urls, key=lambda x: x[2], reverse=True)
    return urls_sorted[0]


def run_media_agent_for_question_ids(db: Session, question_ids: List[int]) -> None:
    """
    For each question id, build a media query and select one candidate media URL.
    """
    if not question_ids:
        return

    categories: Dict[int, Category] = {
        c.id: c for c in db.query(Category).filter(Category.id.in_(
            list({qid for qid in question_ids if qid})) ) # dummy to avoid empty IN
    }

    # Fallback: just query all categories once if the above was too narrow
    if not categories:
        categories = {c.id: c for c in db.query(Category).all()}

    for qid in question_ids:
        q = db.query(Question).get(qid)
        if not q:
            continue

        cat = categories.get(q.category_id) if q.category_id else None
        cat_name = cat.name_en if cat else ""

        query = build_media_hint(
            category_name=cat_name,
            subtopic=q.subtopic or "",
            hint=q.hint or "",
            stem_en=q.stem_en or "",
            answer_en=q.answer_en or "",
            question_type=q.question_type or "",
            region=getattr(cat, "scope", "") or "",
        )

        candidate_urls = _collect_urls(query)
        if not candidate_urls:
            q.media_status = "failed"
            db.commit()
            continue

        best_url, best_source, best_score = _pick_best(candidate_urls)

        # Save candidate rows
        db.query(MediaCandidate).filter(
            MediaCandidate.question_id == q.id
        ).delete()

        for url, source, score in candidate_urls:
            mc = MediaCandidate(
                question_id=q.id,
                url=url,
                source=source,
                score=score,
            )
            db.add(mc)

        # Attach the best one on the question
        q.media_query = query
        q.media_url = best_url
        q.media_type = "image"
        q.media_status = "PENDING"
        q.media_selected_source = best_source
        q.media_selected_score = str(best_score)

        db.commit()