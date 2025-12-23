
import requests

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
        resp = requests.get("https://commons.wikimedia.org/w/api.php", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        urls = []
        for page in pages.values():
            for ii in page.get("imageinfo", []):
                url = ii.get("url")
                if url:
                    urls.append(url)
        return urls
    except Exception:
        return []
    
    # Compatibility wrapper so older code that imports `wikimedia_search_images`
# continues to work. It simply delegates to `search_commons_images`.
def wikimedia_search_images(query: str, max_results: int = 8):
    """Compatibility wrapper.

    Returns a list of dicts: {"url": <direct-image-url>}.
    """
    urls = search_commons_images(query=query, limit=max_results)
    return [{"url": u} for u in urls]

