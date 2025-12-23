
def score_relevance(question_text: str, answer_text: str, image_url: str) -> float:
    """Placeholder CLIP-like scorer.

    In this free/local version we do not run a heavy model.
    We simply return a dummy score so that the UI can sort candidates.
    """
    if not image_url:
        return 0.0
    # very rough heuristic: longer question => slightly higher base
    base = min(len(question_text) / 200.0, 1.0)
    return round(0.4 + base * 0.4, 2)
