
from .ocr_detector import detect_text_in_image
from .clip_model import score_relevance

def filter_images(question_text: str, answer_text: str, urls, max_n: int = 10):
    cleaned = []
    for u in urls:
        if not u:
            continue
        if detect_text_in_image(u):
            continue
        score = score_relevance(question_text, answer_text, u)
        cleaned.append((u, score))
    cleaned.sort(key=lambda t: t[1], reverse=True)
    return cleaned[:max_n]
