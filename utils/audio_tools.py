
def pick_audio_thumbnail(url: str) -> str:
    """In free version we just return the URL itself (audio clip).

    Real implementation could cut to 5–10 seconds etc.
    """
    return url
