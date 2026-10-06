def _to_top_left(bbox):
    """Apple Vision의 좌표(좌하단 기준)를 좌상단 기준 [x, y, w, h]로 바꾼다."""
    x, y, w, h = bbox
    top = 1 - y - h

    x0, y0 = max(0.0, x), max(0.0, top)
    x1, y1 = min(1.0, x + w), min(1.0, top + h)

    return [round(x0, 4), round(y0, 4), round(x1 - x0, 4), round(y1 - y0, 4)]


def read_text(image_path, languages=("ko-KR", "en-US")):
    """간판 사진에서 글자를 읽는다. (macOS 전용: Apple Vision)

    OCR을 쓸 수 없는 환경이면 None을 반환한다.
    """
    try:
        from ocrmac import ocrmac
    except ImportError:
        return None

    results = ocrmac.OCR(
        str(image_path),
        language_preference=list(languages),
    ).recognize()

    lines = [
        {
            "text": text,
            "confidence": round(float(confidence), 3),
            "bbox": _to_top_left(bbox),
        }
        for text, confidence, bbox in results
    ]

    if not lines:
        return {"text": "", "confidence": None, "lines": []}

    mean_conf = sum(line["confidence"] for line in lines) / len(lines)

    return {
        "text": " ".join(line["text"] for line in lines),
        "confidence": round(mean_conf, 3),
        "lines": lines,
    }
