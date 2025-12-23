
def detect_text_in_image(url: str) -> bool:
    """Placeholder OCR check.

    Free/local: we do not actually run OCR. The moderator remains final gate.
    We simply treat filename hints like 'logo' as text-free.
    """
    if not url:
        return False
    lowered = url.lower()
    for bad in ["scoreboard", "subtitle", "caption", "tweet"]:
        if bad in lowered:
            return True
    return False
