
def prefers_face_crop(question_type: str) -> bool:
    return question_type in ("picture", "logo")
