"""
MEDIA AGENT
-----------
Robust, defensive media collector for Midan AI Question Bank.

Goals (per v31 fixes):
- Live media candidates stored in MediaCandidate table
- Fallback chain: Wikimedia Commons → Openverse → Unsplash/Pexels → AI image gen
- Backward compatible with older call sites:
    - collect_media_candidates("query string") -> list[dict]
    - collect_media_candidates([question_ids], db=Session) -> list[MediaCandidate]
    - run_media_agent_for_question_ids(db, [ids]) OR run_media_agent_for_question_ids([ids])
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
import os
import logging
import re
import time
import json
import traceback
import base64
import hashlib
logger = logging.getLogger(__name__)


import requests

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# DB imports (safe)
try:
    from database import SessionLocal, MediaCandidate, Question, Category
except Exception:
    SessionLocal = None  # type: ignore
    MediaCandidate = None  # type: ignore
    Question = None  # type: ignore
    Category = None  # type: ignore


def _ua_headers() -> Dict[str, str]:
    ua = os.getenv("HTTP_USER_AGENT") or os.getenv("USER_AGENT")
    if not ua:
        ua = "MidanAIQB/1.0 (media-agent; contact: support@midan.ai)"
    return {"User-Agent": ua}

# ---------------------------
# RECIPE / INGREDIENTS HEURISTICS
# ---------------------------

_RECIPE_KEYWORDS_EN = {"recipe","recipes","cooking","cook","cuisine","food","dish","kitchen","meal","ingredients"}
_RECIPE_KEYWORDS_AR = {"وصفة","وصفات","طبخ","مطبخ","طعام","أكل","مكونات","مكوّنات","مأكولات","وصفه"}

def _is_recipe_context(cat_obj: Any, q_obj: Any) -> bool:
    try:
        ce = (getattr(cat_obj, "name_en", "") or "").lower()
        ca = (getattr(cat_obj, "name_ar", "") or "")
        if any(k in ce for k in _RECIPE_KEYWORDS_EN):
            return True
        if any(k in ca for k in _RECIPE_KEYWORDS_AR):
            return True
    except Exception:
        pass
    # Also detect from question text
    try:
        se = (getattr(q_obj, "stem_en", "") or "").lower()
        sa = (getattr(q_obj, "stem_ar", "") or "")
        if any(k in se for k in _RECIPE_KEYWORDS_EN):
            return True
        if any(k in sa for k in _RECIPE_KEYWORDS_AR):
            return True
    except Exception:
        pass
    return False

def _recipe_query_variants(dish: str) -> List[str]:
    dish = (dish or "").strip()
    if dish:
        return [
            f"ingredients for {dish} flat lay",
            f"{dish} ingredients top view",
            f"raw ingredients for {dish}",
            f"{dish} recipe ingredients",
            f"{dish} spices and ingredients",
        ]
    # generic recipe ingredients imagery
    return [
        "raw cooking ingredients flat lay",
        "assorted raw ingredients top view",
        "kitchen ingredients flat lay",
        "spices vegetables ingredients flat lay",
    ]




# ---------------------------
# Candidate scoring (quality + suitability)
# ---------------------------

_IMG_EXT = ('.jpg','.jpeg','.png','.webp')
_VID_EXT = ('.mp4','.webm','.mov')
_AUD_EXT = ('.mp3','.wav','.ogg','.m4a')

def _quality_score(url: str, source: str, media_type: str, query: str = '') -> float:
    """Heuristic quality score in [0,1]. Used for auto-pick best candidate.

    We deliberately keep it metadata-light (works even when sources only provide URLs).
    """
    u = (url or '').lower()
    s = (source or '').lower()
    mt = (media_type or 'picture').lower()

    # Base weights by source
    w = 0.45
    if 'unsplash' in s: w = 0.78
    elif 'pexels_video' in s: w = 0.80
    elif 'pexels' in s: w = 0.74
    elif 'youtube' in s: w = 0.72
    elif 'openverse_audio' in s: w = 0.76
    elif 'openverse' in s: w = 0.70
    elif 'commons' in s: w = 0.68
    elif 'ai' in s: w = 0.62

    # Penalize thumbnails / tiny images
    thumb_pen = 0.0
    if any(t in u for t in ['thumb', 'thumbnail', 'small', '150px', '200px', 'lowres']):
        thumb_pen -= 0.12

    # Prefer direct file URLs
    ext_bonus = 0.0
    if mt in ('picture','image'):
        if u.endswith(_IMG_EXT) or any(x in u for x in ['images.unsplash.com','images.pexels.com','upload.wikimedia.org']):
            ext_bonus += 0.08
        if 'upload.wikimedia.org' in u and '/thumb/' not in u:
            ext_bonus += 0.05
    elif mt in ('video',):
        if 'youtube.com' in u or 'youtu.be' in u:
            ext_bonus += 0.06
        if u.endswith(_VID_EXT):
            ext_bonus += 0.08
    elif mt in ('audio',):
        if u.endswith(_AUD_EXT):
            ext_bonus += 0.08

    # Query fit (very light)
    q = (query or '').lower()
    fit = 0.0
    if q and any(k in q for k in ['ingredients','raw','flat lay','top view']):
        # modest boost when we're searching for raw ingredients (recipe polish)
        if any(k in u for k in ['ingredient','ingredients','raw','spice','vegetable','kitchen']):
            fit += 0.06

    score = w + thumb_pen + ext_bonus + fit
    if score < 0.0: score = 0.0
    if score > 1.0: score = 1.0
    return float(score)

# ---------------------------
# MEDIA HINT BUILDER
# ---------------------------

def build_media_hint(*args, **kwargs) -> str:
    """Backward-compatible media hint generator."""
    if len(args) == 1 and isinstance(args[0], str):
        return args[0].strip()

    cat_en = args[0] if len(args) > 0 else ""
    cat_ar = args[1] if len(args) > 1 else ""
    q_en   = args[2] if len(args) > 2 else ""
    q_ar   = args[3] if len(args) > 3 else ""
    difficulty = args[4] if len(args) > 4 else ""
    region     = args[5] if len(args) > 5 else ""
    topic      = args[6] if len(args) > 6 else ""

    cat_en = kwargs.get("category_en", cat_en) or ""
    cat_ar = kwargs.get("category_ar", cat_ar) or ""
    q_en   = kwargs.get("question_en", q_en) or ""
    q_ar   = kwargs.get("question_ar", q_ar) or ""
    difficulty = kwargs.get("difficulty", difficulty) or ""
    region     = kwargs.get("region", region) or ""
    topic      = kwargs.get("topic", topic) or ""

    parts = []
    if region: parts.append(str(region))
    if cat_en: parts.append(str(cat_en))
    if topic: parts.append(str(topic))
    if q_en: parts.append(str(q_en))
    if difficulty: parts.append(str(difficulty))
    if cat_ar: parts.append(str(cat_ar))
    if q_ar: parts.append(str(q_ar))

    return " | ".join(p.strip() for p in parts if p).strip() or "general"


# ---------------------------
# Source fetchers
# ---------------------------

def _commons_search(query: str, limit: int = 10) -> List[str]:
    try:
        from utils.wikipedia_tools import search_commons_images
        return (search_commons_images(query=query, limit=limit) or [])[:limit]
    except Exception:
        return []

def _openverse_search(query: str, limit: int = 10) -> List[str]:
    try:
        url = "https://api.openverse.engineering/v1/images"
        resp = requests.get(url, params={"q": query, "page_size": limit}, headers=_ua_headers(), timeout=20)
        if resp.status_code >= 400:
            return []
        data = resp.json() if resp.content else {}
        out: List[str] = []
        for r in (data.get("results") or []):
            u = r.get("url") or r.get("thumbnail")
            if u:
                out.append(u)
        return out[:limit]
    except Exception:
        return []


def _openverse_search_audio(query: str, limit: int = 10) -> List[str]:
    """Openverse audio search (requires no API key)."""
    try:
        url = "https://api.openverse.engineering/v1/audio"
        resp = requests.get(url, params={"q": query, "page_size": limit}, headers=_ua_headers(), timeout=20)
        if resp.status_code >= 400:
            return []
        data = resp.json() if resp.content else {}
        out: List[str] = []
        for r in (data.get("results") or []):
            # Openverse audio items often have "url" (landing) and "audio_url" or "filetype" variants.
            u = r.get("audio_url") or r.get("url")
            if u:
                out.append(u)
        return out[:limit]
    except Exception:
        return []

def _pexels_search_video(query: str, limit: int = 10) -> List[str]:
    key = os.getenv("PEXELS_API_KEY") or os.getenv("PEXELS_KEY") or ""
    if not key:
        return []
    try:
        url = "https://api.pexels.com/videos/search"
        resp = requests.get(url, params={"query": query, "per_page": limit}, headers={"Authorization": key, **_ua_headers()}, timeout=20)
        if resp.status_code >= 400:
            return []
        data = resp.json() if resp.content else {}
        out: List[str] = []
        for v in (data.get("videos") or []):
            files = v.get("video_files") or []
            # choose the first downloadable link
            for f in files:
                link = f.get("link")
                if link:
                    out.append(link)
                    break
        return out[:limit]
    except Exception:
        return []

def _youtube_search(query: str, limit: int = 6) -> List[str]:
    """YouTube search via Data API v3 (requires YOUTUBE_API_KEY). Returns watch URLs."""
    key = os.getenv("YOUTUBE_API_KEY") or os.getenv("YOUTUBE_KEY") or ""
    if not key:
        return []
    try:
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": limit,
            "key": key,
            "safeSearch": "strict",
        }
        resp = requests.get(url, params=params, headers=_ua_headers(), timeout=20)
        if resp.status_code >= 400:
            return []
        data = resp.json() if resp.content else {}
        out: List[str] = []
        for item in (data.get("items") or []):
            vid = ((item.get("id") or {}).get("videoId") or "").strip()
            if vid:
                out.append(f"https://www.youtube.com/watch?v={vid}")
        return out[:limit]
    except Exception:
        return []
def _unsplash_search(query: str, limit: int = 10) -> List[str]:
    key = os.getenv("UNSPLASH_ACCESS_KEY") or os.getenv("UNSPLASH_KEY") or ""
    if not key:
        return []
    try:
        url = "https://api.unsplash.com/search/photos"
        resp = requests.get(url, params={"query": query, "per_page": limit}, headers={"Authorization": f"Client-ID {key}", **_ua_headers()}, timeout=20)
        if resp.status_code >= 400:
            return []
        data = resp.json() if resp.content else {}
        out: List[str] = []
        for r in (data.get("results") or []):
            u = (((r.get("urls") or {}).get("regular")) or ((r.get("urls") or {}).get("full")))
            if u:
                out.append(u)
        return out[:limit]
    except Exception:
        return []

def _pexels_search(query: str, limit: int = 10) -> List[str]:
    key = os.getenv("PEXELS_API_KEY") or os.getenv("PEXELS_KEY") or ""
    if not key:
        return []
    try:
        url = "https://api.pexels.com/v1/search"
        resp = requests.get(url, params={"query": query, "per_page": limit}, headers={"Authorization": key, **_ua_headers()}, timeout=20)
        if resp.status_code >= 400:
            return []
        data = resp.json() if resp.content else {}
        out: List[str] = []
        for r in (data.get("photos") or []):
            src = r.get("src") or {}
            u = src.get("large") or src.get("original") or src.get("medium")
            if u:
                out.append(u)
        return out[:limit]
    except Exception:
        return []


def _ai_image_fallback(query: str, n: int = 2) -> List[str]:
    """Generate a few image candidates using OpenAI Images.

    Notes:
    - Newer OpenAI Images responses often return base64 (`b64_json`) instead of public URLs.
    - Our UI expects a URL, so we persist generated bytes under /static/media/generated and
      return a local /static/... URL.
    """
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY") or ""
    if not api_key or OpenAI is None:
        return []
    try:
        client = OpenAI(api_key=api_key)
        if not hasattr(client, "images"):
            return []

        # Prefer base64 to avoid relying on externally-hosted URLs.
        kwargs = dict(
            model=os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1"),
            prompt=query,
            size=os.getenv("OPENAI_IMAGE_SIZE", "1024x1024"),
            n=max(1, min(int(n), 4)),
        )
        # Some SDK versions support response_format; keep it best-effort.
        try:
            kwargs["response_format"] = "b64_json"
        except Exception:
            pass

        resp = None
        model_primary = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
        model_fallbacks = [m.strip() for m in os.getenv("OPENAI_IMAGE_MODEL_FALLBACKS", "gpt-image-1,dall-e-3").split(",") if m.strip()]
        # Ensure primary first
        models = [model_primary] + [m for m in model_fallbacks if m != model_primary]
        last_err = None
        for mname in models:
            for attempt in range(3):
                try:
                    kwargs["model"] = mname
                    # dall-e-3 supports n=1 in many configurations; keep safe
                    if mname.lower().startswith("dall-e") and kwargs.get("n", 1) > 1:
                        kwargs["n"] = 1
                    resp = client.images.generate(**kwargs)
                    break
                except Exception as e:
                    last_err = e
                    msg = str(e)
                    # Transient server errors: retry with backoff
                    if ("server_error" in msg) or ("Error code: 500" in msg):
                        try:
                            import time
                            time.sleep(1 * (2 ** attempt))
                        except Exception:
                            pass
                        continue
                    # Quota / 429 / auth errors: don't spin on retries for this model
                    break
            if resp is not None:
                break

        if resp is None:
            logger.warning("OpenAI image generation failed: %s", str(last_err))
            return []

        # Persist generated images on disk.
        # - Local dev default: ./static/media/generated
        # - Render persistent disk: set PERSIST_MEDIA_DIR=/var/data/media
        here = os.path.dirname(__file__)
        root = os.path.abspath(os.path.join(here, ".."))
        persist_dir = (os.getenv("PERSIST_MEDIA_DIR") or "").strip()
        if persist_dir:
            out_dir = os.path.abspath(persist_dir)
        else:
            out_dir = os.path.join(root, "static", "media", "generated")
        os.makedirs(out_dir, exist_ok=True)

        out: List[str] = []
        for d in getattr(resp, "data", []) or []:
            # 1) If a hosted URL exists, use it.
            u = getattr(d, "url", None)
            if u:
                out.append(u)
                continue

            # 2) Otherwise, handle base64 payloads.
            b64 = getattr(d, "b64_json", None) or getattr(d, "base64", None)
            if not b64:
                continue
            try:
                img_bytes = base64.b64decode(b64)
            except Exception:
                continue

            # Deterministic filename to avoid duplicates
            h = hashlib.sha1((query + str(len(out))).encode("utf-8")).hexdigest()[:16]
            fname = f"ai_{h}.png"
            fpath = os.path.join(out_dir, fname)
            try:
                if not os.path.exists(fpath):
                    with open(fpath, "wb") as f:
                        f.write(img_bytes)
                # If we're writing to a persistent directory, serve via /persist-media.
                if persist_dir:
                    out.append(f"/persist-media/{fname}")
                else:
                    out.append(f"/static/media/generated/{fname}")
            except Exception:
                # If filesystem write fails, fallback to data URL so UI still renders.
                out.append("data:image/png;base64," + b64)

        return out
    except Exception:
        return []


def _fallback_chain(query: str, media_type: str = "picture", limit: int = 12) -> List[Dict[str, Any]]:
    """Return ordered candidates with source metadata."""
    query = (query or "").strip()
    if not query:
        return []

    # Commons can be unreliable (403). We still try, but we don't rely on it.
    urls: List[Dict[str, Any]] = []
    mt = (media_type or "picture").strip().lower()

    # ---------------------------
    # IMAGE / LOGO / PICTURE
    # ---------------------------
    if mt in ("picture", "image", "logo"):
        # Source-quota strategy: by default we pull a small fixed number from each source
        # (commons/openverse/unsplash/pexels/ai) to ensure diversity and predictable volume.
        #
        # Note: Unsplash/Pexels/YouTube require API keys; if keys are missing, those sources return [].
        per_source = int(os.getenv("PER_SOURCE_LIMIT", "2") or 2)
        per_source = max(1, min(per_source, 5))

        # Allow recipe-style variants separated by '|'
        variants = [query]
        if "|" in query:
            variants = [v.strip() for v in query.split("|") if v.strip()]

        def _take_from_variants(fetch_fn, source_name: str, base_score: float) -> None:
            nonlocal urls
            remaining = per_source - sum(1 for c in urls if c.get("source") == source_name)
            if remaining <= 0:
                return
            for vq in variants:
                got = fetch_fn(vq, limit=remaining) or []
                for i, u in enumerate(got):
                    if not u:
                        continue
                    urls.append({"source": source_name, "url": u, "score": max(base_score - i * 0.03, 0.1)})
                    remaining -= 1
                    if remaining <= 0:
                        return

        # Always attempt commons + openverse (no keys). Commons may 403/empty but we still try.
        _take_from_variants(_commons_search, "commons", 0.90)
        _take_from_variants(_openverse_search, "openverse", 0.80)

        # For logos, stock photography sources often return irrelevant jerseys/people.
        # Default behavior: skip Unsplash/Pexels for mt=logo unless explicitly enabled.
        allow_stock_for_logo = (os.getenv("ALLOW_STOCK_FOR_LOGO", "0").strip() == "1")
        if mt != "logo" or allow_stock_for_logo:
            # Unsplash / Pexels (require keys)
            _take_from_variants(_unsplash_search, "unsplash", 0.78)
            _take_from_variants(_pexels_search, "pexels", 0.76)

        # AI images: always try to include per_source AI candidates when enabled,
        # but only works when OPENAI_API_KEY is set.
        enable_ai = (os.getenv("ENABLE_AI_MEDIA", "1").strip() != "0")
        if enable_ai:
            ai_existing = sum(1 for c in urls if c.get("source") == "ai")
            need = max(0, per_source - ai_existing)
            if need > 0:
                ai = _ai_image_fallback(variants[0] if variants else query, n=need) or []
                for i, u in enumerate(ai):
                    if not u:
                        continue
                    urls.append({"source": "ai", "url": u, "score": max(0.60 - i * 0.03, 0.1)})

        # Cap to a predictable size: at most per_source from each source
        # (dedupe later will reduce further).
        # We keep the original 'limit' as a hard ceiling.
        if limit is not None:
            urls = urls[: int(limit)]
    # ---------------------------
    # VIDEO
    # ---------------------------
    elif mt in ("video", "youtube"):
        # Prefer free stock video first (Pexels), then YouTube if key is configured.
        pxv = _pexels_search_video(query, limit=limit)
        for i, u in enumerate(pxv):
            urls.append({"source": "pexels_video", "url": u, "score": max(0.78 - i * 0.03, 0.25)})

        if len(urls) < limit:
            yt = _youtube_search(query, limit=min(6, limit))
            for i, u in enumerate(yt):
                urls.append({"source": "youtube", "url": u, "score": max(0.72 - i * 0.03, 0.2)})
                if len(urls) >= limit:
                    break

        # (Optional future) Wikimedia Commons video search can be added here if needed.

    # ---------------------------
    # AUDIO
    # ---------------------------
    elif mt in ("audio", "sound", "voice"):
        # Openverse has strong audio coverage.
        ova = _openverse_search_audio(query, limit=limit)
        for i, u in enumerate(ova):
            urls.append({"source": "openverse_audio", "url": u, "score": max(0.78 - i * 0.03, 0.25)})
            if len(urls) >= limit:
                break

        # (Optional future) Commons audio search / TTS fallback can be added here.

    else:
        # unknown type → treat as image
        return _fallback_chain(query, media_type="picture", limit=limit)
        for i, u in enumerate(ai):
            urls.append({"source": "ai", "url": u, "score": max(0.55 - i * 0.02, 0.1)})
    # Apply heuristic quality adjustment (better auto-pick)
    try:
        for c in urls:
            u = (c.get('url') or '').strip()
            if not u:
                continue
            base = float(c.get('score') or 0.0)
            q = float(_quality_score(u, str(c.get('source') or ''), mt, query))
            # Blend base rank score with quality score
            c['score'] = max(0.0, min(1.0, base * 0.75 + q * 0.25))
    except Exception:
        pass


    # de-dupe by URL
    seen = set()
    out = []
    for c in urls:
        u = c.get("url")
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(c)
    return out[:limit]


# ---------------------------
# DB integration
# ---------------------------

def _derive_search_query(q_obj: Any, cat_obj: Any) -> str:
    # Prefer explicit query fields
    for attr in ("media_query",):
        if hasattr(q_obj, attr):
            v = getattr(q_obj, attr)
            if isinstance(v, str) and v.strip():
                return v.strip()

    # media_intent_json may contain media_search_query
    try:
        mij = getattr(q_obj, "media_intent_json", None)
        if mij:
            data = json.loads(mij)
            if isinstance(data, dict):
                msq = (data.get("media_search_query") or "").strip()
                if msq:
                    return msq
    except Exception:
        pass

    # fallback: category + subtopic + hint
    parts = []
    if cat_obj is not None:
        parts.append((getattr(cat_obj, "name_en", "") or getattr(cat_obj, "name_ar", "") or "").strip())
    for attr in ("subtopic", "hint", "answer_en", "answer_ar"):
        if hasattr(q_obj, attr):
            v = getattr(q_obj, attr)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
    if hasattr(q_obj, "stem_en") and isinstance(q_obj.stem_en, str) and q_obj.stem_en.strip():
        parts.append(q_obj.stem_en.strip())
    query = " ".join([p for p in parts if p])[:220]
    query = re.sub(r"\s+", " ", query).strip()

    # Recipe optimization: for picture questions in recipe categories,
    # bias searches toward *raw ingredients* images (flat lay / top view).
    try:
        qtype = (getattr(q_obj, "question_type", "") or "").lower().strip()
        if qtype in ("picture", "image", "photo") and _is_recipe_context(cat_obj, q_obj):
            dish = (getattr(q_obj, "answer_en", "") or getattr(q_obj, "answer_ar", "") or "").strip()
            # If the stored answer is long, fall back to the generic query.
            if len(dish) > 80:
                dish = ""
            variants = _recipe_query_variants(dish)
            # Collapse variants into a strong single query string for downstream providers.
            query = " | ".join(variants[:3])
    except Exception:
        pass

    return query

def run_media_agent_for_question_ids(*args, **kwargs) -> None:
    """Populate MediaCandidate rows for provided question IDs.

    Supported call signatures:
      - run_media_agent_for_question_ids([1,2,3])
      - run_media_agent_for_question_ids(db, [1,2,3])
    """
    db = None
    question_ids: Iterable[int] = []
    if args:
        if len(args) == 1:
            question_ids = args[0]  # type: ignore
        elif len(args) >= 2:
            db = args[0]
            question_ids = args[1]  # type: ignore
    if "db" in kwargs and db is None:
        db = kwargs["db"]
    if "question_ids" in kwargs:
        question_ids = kwargs["question_ids"]

    if not question_ids:
        return

    ids: List[int] = []
    for qid in question_ids:
        try:
            ids.append(int(qid))
        except Exception:
            continue
    if not ids:
        return

    if SessionLocal is None or MediaCandidate is None or Question is None:
        return

    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True

    try:
        for qid in ids:
            q_obj = db.query(Question).get(int(qid))
            if not q_obj:
                continue

            # Collect existing candidates (we append by default)
            existing_rows = db.query(MediaCandidate).filter(MediaCandidate.question_id == qid).all()
            existing_urls = set([(r.url or "").strip() for r in existing_rows if (r.url or "").strip()])

            # If force refresh is enabled, clear existing candidates first
            force_refresh = str(os.getenv("MEDIA_AGENT_FORCE_REFRESH", "0")).strip() in ("1","true","yes","on")
            if force_refresh and existing_rows:
                db.query(MediaCandidate).filter(MediaCandidate.question_id == qid).delete()
                existing_urls = set()

            for c in candidates:
                u = (c.get("url") or "").strip()
                if not u or u in existing_urls:
                    continue
                existing_urls.add(u)
                db.add(
                    MediaCandidate(
                        question_id=qid,
                        source=c.get("source") or "search",
                        url=u,
                        title=c.get("title"),
                        score=float(c.get("score") or 0.0),
                        meta_json=json.dumps(c, ensure_ascii=False),
                        selected=(u == getattr(q_obj, "url", None)),
                    )
                )
        db.commit()
    except Exception:
        db.rollback()
    finally:
        if close_db:
            try:
                db.close()
            except Exception:
                pass


def collect_media_candidates(question_ids: Any, db=None, limit: int = 12, media_type: str = "picture") -> List[Any]:
    """Backward compatible:

    - If question_ids is a str: treat as query and return list[dict] candidates.
    - Else: treat as IDs, run agent if needed, return MediaCandidate ORM rows.
    """
    if not question_ids:
        return []

    # Query-string call
    if isinstance(question_ids, str):
        return _fallback_chain(question_ids, media_type=media_type, limit=limit)

    # Normalize input to list of ints
    if not isinstance(question_ids, (list, tuple, set)):
        question_ids = [question_ids]

    ids: List[int] = []
    for x in question_ids:
        try:
            ids.append(int(x))
        except Exception:
            continue
    if not ids:
        return []

    # Populate candidates if missing
    try:
        run_media_agent_for_question_ids(db, ids)
    except Exception:
        pass

    if SessionLocal is None or MediaCandidate is None:
        return []

    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True

    try:
        q = db.query(MediaCandidate).filter(MediaCandidate.question_id.in_(ids)).order_by(MediaCandidate.score.desc())
        return q.limit(max(1, int(limit)) * len(ids)).all()
    except Exception:
        traceback.print_exc()
        return []
    finally:
        if close_db:
            try:
                db.close()
            except Exception:
                pass




# ---------------------------
# Source Health Checks (Agents panel)
# ---------------------------

def check_media_source_health() -> Dict[str, Dict[str, str]]:
    """Return per-source health dict: {source: {status: green|yellow|red, detail: str}}"""
    out: Dict[str, Dict[str, str]] = {}

    def ok(name: str, detail: str = "OK"):
        out[name] = {"status": "green", "detail": detail}

    def warn(name: str, detail: str):
        out[name] = {"status": "yellow", "detail": detail}

    def bad(name: str, detail: str):
        out[name] = {"status": "red", "detail": detail}

    # Openverse (no key)
    try:
        r = requests.get(
            "https://api.openverse.engineering/v1/images",
            params={"q": "test", "page_size": 1},
            headers=_ua_headers(),
            timeout=8,
        )
        if r.status_code < 400:
            ok("openverse", "reachable")
        else:
            bad("openverse", f"http {r.status_code}")
    except Exception as e:
        bad("openverse", f"error: {type(e).__name__}")

    # Wikimedia Commons (often flaky / 403)
    try:
        r = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": "test", "format": "json", "srlimit": 1},
            headers=_ua_headers(),
            timeout=8,
        )
        if r.status_code == 403:
            warn("commons", "403 (degraded)")
        elif r.status_code < 400:
            ok("commons", "reachable")
        else:
            warn("commons", f"http {r.status_code}")
    except Exception as e:
        warn("commons", f"error: {type(e).__name__}")

    # Unsplash
    ukey = os.getenv("UNSPLASH_ACCESS_KEY") or os.getenv("UNSPLASH_KEY") or ""
    if not ukey:
        warn("unsplash", "missing key")
    else:
        try:
            r = requests.get(
                "https://api.unsplash.com/search/photos",
                params={"query": "test", "per_page": 1, "client_id": ukey},
                headers=_ua_headers(),
                timeout=8,
            )
            if r.status_code < 400:
                ok("unsplash", "reachable")
            else:
                bad("unsplash", f"http {r.status_code}")
        except Exception as e:
            bad("unsplash", f"error: {type(e).__name__}")

    # Pexels (images/video)
    pkey = os.getenv("PEXELS_API_KEY") or os.getenv("PEXELS_KEY") or ""
    if not pkey:
        warn("pexels", "missing key")
    else:
        try:
            r = requests.get(
                "https://api.pexels.com/v1/search",
                params={"query": "test", "per_page": 1},
                headers={"Authorization": pkey, **_ua_headers()},
                timeout=8,
            )
            if r.status_code < 400:
                ok("pexels", "reachable")
            else:
                bad("pexels", f"http {r.status_code}")
        except Exception as e:
            bad("pexels", f"error: {type(e).__name__}")

    # YouTube
    ykey = os.getenv("YOUTUBE_API_KEY") or os.getenv("YOUTUBE_KEY") or ""
    if not ykey:
        warn("youtube", "missing key")
    else:
        try:
            r = requests.get(
                "https://www.googleapis.com/youtube/v3/search",
                params={"part": "snippet", "q": "test", "type": "video", "maxResults": 1, "key": ykey},
                headers=_ua_headers(),
                timeout=8,
            )
            if r.status_code < 400:
                ok("youtube", "reachable")
            else:
                bad("youtube", f"http {r.status_code}")
        except Exception as e:
            bad("youtube", f"error: {type(e).__name__}")

    # AI media
    enable_ai = str(os.getenv("ENABLE_AI_MEDIA", "1")).strip() not in ("0", "false", "no", "off")
    okey = os.getenv("OPENAI_API_KEY") or ""
    if not enable_ai:
        warn("ai", "disabled")
    elif not okey:
        warn("ai", "missing OPENAI_API_KEY")
    else:
        ok("ai", "enabled")

    return out

__all__ = ["build_media_hint", "collect_media_candidates", "run_media_agent_for_question_ids"]