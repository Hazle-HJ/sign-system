import cv2
import numpy as np


def crop_region(image, bbox, pad=0.15):
    """정규화된 bbox [x, y, w, h]를 여백(pad)과 함께 자른다.

    반환: (잘라낸 이미지, 원래 상자의 위치 (x0, y0, x1, y1) - 잘라낸 이미지 기준 픽셀)
    """
    H, W = image.shape[:2]
    x, y, bw, bh = bbox

    bx0, by0 = int(x * W), int(y * H)
    bx1, by1 = int((x + bw) * W), int((y + bh) * H)
    pw, ph = int((bx1 - bx0) * pad), int((by1 - by0) * pad)

    cx0, cy0 = max(0, bx0 - pw), max(0, by0 - ph)
    cx1, cy1 = min(W, bx1 + pw), min(H, by1 + ph)

    crop = image[cy0:cy1, cx0:cx1]
    inner = (bx0 - cx0, by0 - cy0, bx1 - cx0, by1 - cy0)
    return crop, inner


def text_mask(crop, inner):
    """색을 두 덩어리로 나눠 글자 부분을 255, 나머지를 0으로 하는 마스크를 만든다.

    반환: (마스크, 남은 덩어리 개수)
    """
    h, w = crop.shape[:2]
    x0, y0, x1, y1 = inner
    bw, bh = x1 - x0, y1 - y0

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)

    scale = min(1.0, 500 / max(h, w))
    small = cv2.resize(lab, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else lab
    data = small.reshape(-1, 3)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    cv2.setRNGSeed(0)
    _, _, centers = cv2.kmeans(data, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

    dist = np.linalg.norm(lab[:, :, None, :] - centers[None, None, :, :], axis=3)
    labels = dist.argmin(axis=2)

    # 상자 안에서 면적이 더 작은 색을 글자라고 본다
    inside = labels[y0:y1, x0:x1]
    ink = 0 if (inside == 0).sum() < (inside == 1).sum() else 1
    mask = (labels == ink).astype(np.uint8) * 255

    # 작은 잡음을 없애고, 갈라진 틈을 메운다
    mask = cv2.medianBlur(mask, 3)
    k = max(3, int(0.03 * min(bh, bw))) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 상자 안에 중심이 있고, 상자보다 지나치게 크지 않은 덩어리만 후보로 둔다
    n, comp, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cand = []
    for i in range(1, n):
        cx, cy = cent[i]
        in_box = x0 <= cx <= x1 and y0 <= cy <= y1
        too_big = (
            stats[i, cv2.CC_STAT_WIDTH] > 1.3 * bw
            or stats[i, cv2.CC_STAT_HEIGHT] > 1.3 * bh
        )
        if in_box and not too_big:
            cand.append(i)

    if not cand:
        return np.zeros_like(mask), 0

    largest = max(stats[i, cv2.CC_STAT_AREA] for i in cand)
    keep = [i for i in cand if stats[i, cv2.CC_STAT_AREA] >= 0.08 * largest]

    return np.isin(comp, keep).astype(np.uint8) * 255, len(keep)


def mask_quality(mask, inner, n_kept):
    """마스크를 믿을 수 있는지 간단히 판정한다. 기준은 8장 실험에서 정한 값이다."""
    x0, y0, x1, y1 = inner
    total = int((mask > 0).sum())
    if total == 0 or n_kept < 2:
        return "low"

    inside = int((mask[y0:y1, x0:x1] > 0).sum())
    fill = inside / max(1, (x1 - x0) * (y1 - y0))
    outside = (total - inside) / total

    if fill < 0.05 or outside > 0.2:
        return "low"
    return "ok"


def measure_text(mask, line, image_shape):
    """마스크에서 획 굵기, 글자 크기, 글자 폭 비율을 잰다. (정규화하지 않은 원래 값)"""
    H, W = image_shape[:2]
    _, _, bw, bh = line["bbox"]
    bw_px, bh_px = bw * W, bh * H

    if bw_px > bh_px * 1.2:
        orientation = "horizontal"
    elif bh_px > bw_px * 1.2:
        orientation = "vertical"
    else:
        orientation = "unknown"

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    foreground = dist[mask > 0]
    if foreground.size == 0:
        return None

    stroke_width = 4 * float(foreground.mean())
    char_size = min(bw_px, bh_px)

    n_chars = len(line["text"].replace(" ", ""))
    char_aspect = None
    if orientation == "horizontal" and n_chars > 0:
        char_aspect = bw_px / (n_chars * bh_px)

    return {
        "line_text": line["text"],
        "orientation": orientation,
        "stroke_width_px": round(stroke_width, 2),
        "char_size_px": round(char_size, 1),
        "stroke_to_height": round(stroke_width / char_size, 3),
        "char_aspect": None if char_aspect is None else round(char_aspect, 3),
    }


def draw_preview(image, bbox):
    """글자 영역을 빨간 상자로 표시한 확인용 이미지."""
    H, W = image.shape[:2]
    x, y, bw, bh = bbox
    preview = image.copy()
    cv2.rectangle(
        preview,
        (int(x * W), int(y * H)),
        (int((x + bw) * W), int((y + bh) * H)),
        (0, 0, 255),
        3,
    )
    return preview


def save_image(path, image):
    """경로에 한글이 있어도 저장되도록 imencode를 사용한다."""
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"이미지를 저장하지 못했습니다: {path}")
    buffer.tofile(str(path))
