def auto_select_media(candidates):
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x.get('score', 0), reverse=True)[0]
