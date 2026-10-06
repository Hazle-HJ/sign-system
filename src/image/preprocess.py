from pathlib import Path

import cv2
import numpy as np


def load_image(path):
    """한글 경로에서도 동작하도록 imread 대신 imdecode를 사용한다."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {path}")

    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"이미지로 읽을 수 없는 파일입니다: {path}")

    return image


def resize_image(image, max_size=1600):
    """긴 변이 max_size를 넘을 때만 비율을 유지해 축소한다."""
    h, w = image.shape[:2]
    scale = min(max_size / w, max_size / h, 1)

    if scale == 1:
        return image

    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
