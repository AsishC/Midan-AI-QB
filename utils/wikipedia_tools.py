
import os
import requests


def _ua_headers():
    # Wikimedia APIs require a descriptive User-Agent. Use env if provided.
    ua = os.getenv('HTTP_USER_AGENT') or os.getenv('USER_AGENT')
    if not ua:
        ua = 'MidanAIQB/1.0 (contact: support@midan.ai)'
    return {'User-Agent': ua}



def _openverse_fallback(query: str, limit: int = 10):
    """Fallback image search using Openverse (no API key required)."""
    try:
        url = "https://api.openverse.engineering/v1/images"
        params = {"q": query, "page_size": limit}
        if os.getenv("DEBUG_MEDIA","0")=="1":
            print(f"[DEBUG_MEDIA] Openverse API request: q={query!r} limit={limit}")
        resp = requests.get(url, params=params, headers=_ua_headers(), timeout=20)
        if resp.status_code == 403:
            if os.getenv('DEBUG_MEDIA') == '1':
                print('[DEBUG_MEDIA] Commons returned 403, using Openverse fallback')
            return _openverse_fallback(query, limit)
        if resp.status_code == 403:
            if os.getenv('DEBUG_MEDIA') == '1':
                print('[DEBUG_MEDIA] Commons returned 403, using Openverse fallback')
            return _openverse_fallback(query, limit)
        resp.raise_for_status()
        data = resp.json()
        out = []
        for r in (data.get("results") or []):
            u = r.get("url") or r.get("thumbnail")
            if u:
                out.append(u)
        return out[:limit]
    except Exception as e:
        if os.getenv("DEBUG_MEDIA") == "1":
            print(f"[DEBUG_MEDIA] Openverse fallback error: {e}")
        return []


def search_commons_images(query: str, limit: int = 10):
    """Search Wikimedia Commons for images.

    Returns list of direct image URLs. Uses only public API (free).
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
        resp = requests.get("https://commons.wikimedia.org/w/api.php", params=params, timeout=10)
        if resp.status_code == 403:
            if os.getenv('DEBUG_MEDIA') == '1':
                print('[DEBUG_MEDIA] Commons returned 403, using Openverse fallback')
            return _openverse_fallback(query, limit)
        resp.raise_for_status()
        data = resp.json()
        if os.getenv("DEBUG_MEDIA") == "1":
            # Keep log size bounded
            preview = str(data)
            print(f"[DEBUG_MEDIA] Commons API response: status={resp.status_code} bytes={len(resp.content)} preview={preview[:500]}")
        pages = data.get("query", {}).get("pages", {})
        urls = []
        for page in pages.values():
            for ii in page.get("imageinfo", []):
                url = ii.get("url")
                if url:
                    urls.append(url)
        return urls
    except Exception as e:
        if os.getenv("DEBUG_MEDIA") == "1":
            print(f"[DEBUG_MEDIA] Commons API error: {e}")
        return []


# Compatibility wrapper so older code that imports `wikimedia_search_images`
# continues to work. It simply delegates to `search_commons_images`.
def wikimedia_search_images(query: str, max_results: int = 8):
    """Compatibility wrapper.

    Returns a list of dicts: {"url": <direct-image-url>}.
    """
    urls = search_commons_images(query=query, limit=max_results)
    return [{"url": u} for u in urls]

