
import os
import requests


def _ua_headers():
    # Wikimedia APIs require a descriptive User-Agent. Use env if provided.
    ua = os.getenv("HTTP_USER_AGENT") or os.getenv("USER_AGENT")
    if not ua:
        ua = "MidanAIQB/1.0 (contact: support@midan.ai)"
    return {"User-Agent": ua}


def openverse_search_images(query: str, limit: int = 10):
    """Openverse image search (no key required). Returns direct image/thumbnail URLs."""
    try:
        url = "https://api.openverse.engineering/v1/images"
        params = {"q": query, "page_size": limit}
        resp = requests.get(url, params=params, headers=_ua_headers(), timeout=20)
        if resp.status_code >= 400:
            return []
        data = resp.json() if resp.content else {}
        out = []
        for r in (data.get("results") or []):
            u = r.get("url") or r.get("thumbnail")
            if u:
                out.append(u)
        return out[:limit]
    except Exception as e:
        if os.getenv("DEBUG_MEDIA") == "1":
            print(f"[DEBUG_MEDIA] Openverse error: {e}")
        return []


def search_commons_images(query: str, limit: int = 10):
    """Search Wikimedia Commons for images.

    Returns list of direct image URLs. If Commons blocks (403) or fails, falls back to Openverse.
    """
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url",
        "format": "json",
    }
    try:
        if os.getenv("DEBUG_MEDIA") == "1":
            print(f"[DEBUG_MEDIA] Commons API request: q={query!r} limit={limit}")
        resp = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params=params,
            headers=_ua_headers(),
            timeout=15,
        )
        if resp.status_code == 403:
            if os.getenv("DEBUG_MEDIA") == "1":
                print("[DEBUG_MEDIA] Commons returned 403; fallback to Openverse")
            return openverse_search_images(query, limit)

        resp.raise_for_status()
        data = resp.json() if resp.content else {}
        pages = data.get("query", {}).get("pages", {}) or {}
        urls = []
        for page in pages.values():
            for ii in page.get("imageinfo", []) or []:
                url = ii.get("url")
                if url:
                    urls.append(url)
        # If Commons returns empty, try Openverse
        if not urls:
            return openverse_search_images(query, limit)
        return urls[:limit]
    except Exception as e:
        if os.getenv("DEBUG_MEDIA") == "1":
            print(f"[DEBUG_MEDIA] Commons API error: {e}")
        return openverse_search_images(query, limit)


def wikimedia_search_images(query: str, max_results: int = 8):
    """Compatibility wrapper.

    Returns a list of dicts: {"url": <direct-image-url>}.
    """
    urls = search_commons_images(query=query, limit=max_results)
    return [{"url": u} for u in urls]
