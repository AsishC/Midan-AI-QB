
import os
import requests

UNSPLASH_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
PEXELS_KEY = os.getenv("PEXELS_API_KEY")

def search_unsplash(query: str, limit: int = 5):
    if not UNSPLASH_KEY:
        return []
    url = "https://api.unsplash.com/search/photos"
    params = {"query": query, "per_page": limit}
    headers = {"Authorization": f"Client-ID {UNSPLASH_KEY}"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        return [{"url": p["urls"]["regular"], "title": p.get("description") or p.get("alt_description") or ""} for p in data.get("results", [])]
    except Exception:
        return []

def search_pexels(query: str, limit: int = 5):
    if not PEXELS_KEY:
        return []
    url = "https://api.pexels.com/v1/search"
    params = {"query": query, "per_page": limit}
    headers = {"Authorization": PEXELS_KEY}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        return [{"url": p["src"]["large"], "title": p.get("alt") or ""} for p in data.get("photos", [])]
    except Exception:
        return []
