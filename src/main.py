import json
import sys
from pathlib import Path

from config import PROJECT_ROOT
from image.preprocess import load_image, resize_image
from image.color import analyze_color
from image.geometry import analyze_edges
from image.text_region import (
    crop_region,
    draw_preview,
    mask_quality,
    measure_text,
    save_image,
    text_mask,
)
from ocr.reader import read_text
from ai.vision import observe_sign
from analysis.schema import SignAnalysis

OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# 사람이 직접 지정하는 글자 영역. 형식:
#   {"sign_002": [left, top, right, bottom]}   (확인용 preview 이미지 기준 픽셀)
MANUAL_BOXES_PATH = PROJECT_ROOT / "data" / "manual" / "boxes.json"


def ai_box_to_xywh(box_2d):
    """Gemini의 [ymin, xmin, ymax, xmax] (0~1000)를 [x, y, w, h] (0~1)로 바꾼다."""
    if not box_2d or len(box_2d) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = [float(v) / 1000 for v in box_2d]
    except (TypeError, ValueError):
        return None

    ymin, xmin = max(0.0, ymin), max(0.0, xmin)
    ymax, xmax = min(1.0, ymax), min(1.0, xmax)
    if xmax <= xmin or ymax <= ymin:
        return None

    return [
        round(xmin, 4),
        round(ymin, 4),
        round(xmax - xmin, 4),
        round(ymax - ymin, 4),
    ]


def load_manual_box(stem, image_shape):
    if not MANUAL_BOXES_PATH.exists():
        return None

    data = json.loads(MANUAL_BOXES_PATH.read_text(encoding="utf-8"))
    entry = data.get(stem)
    if not entry or len(entry) != 4:
        return None

    H, W = image_shape[:2]
    left, top, right, bottom = entry
    left, right = max(0, left), min(W, right)
    top, bottom = max(0, top), min(H, bottom)
    if right <= left or bottom <= top:
        return None

    return [
        round(left / W, 4),
        round(top / H, 4),
        round((right - left) / W, 4),
        round((bottom - top) / H, 4),
    ]


def choose_region(image, ocr, ai, stem):
    """글자 영역을 정한다. 우선순위: 사람 지정 > AI > OCR의 가장 큰 글자."""
    main = ((ai or {}).get("observation") or {}).get("main_text") or {}
    ai_text = main.get("text") or ""

    box = load_manual_box(stem, image.shape)
    if box:
        return "manual", box, ""

    box = ai_box_to_xywh(main.get("box_2d"))
    if box:
        return "ai", box, ai_text

    lines = [l for l in (ocr or {}).get("lines", []) if l.get("bbox")]
    if lines:
        line = max(lines, key=lambda l: l["bbox"][2] * l["bbox"][3])
        return "ocr_largest", line["bbox"], line["text"]

    return None


def measure_text_region(image, stem, region):
    """정해진 영역을 잘라 OpenCV로 측정하고, 확인용 이미지를 저장한다."""
    if region is None:
        return None

    source, box, text = region

    crop, inner = crop_region(image, box)
    if crop.size == 0:
        return None

    mask, n_kept = text_mask(crop, inner)
    quality = mask_quality(mask, inner, n_kept)

    for folder in ("crops", "masks", "previews"):
        (PROCESSED_DIR / folder).mkdir(parents=True, exist_ok=True)

    save_image(PROCESSED_DIR / "crops" / f"{stem}_text.png", crop)
    save_image(PROCESSED_DIR / "masks" / f"{stem}_mask.png", mask)
    save_image(
        PROCESSED_DIR / "previews" / f"{stem}_preview.png",
        draw_preview(image, box),
    )

    # 측정은 상자 안쪽 영역 기준이므로 상자 크기를 그대로 넘긴다
    result = measure_text(mask, {"text": text, "bbox": box}, image.shape)
    if result is None:
        return None

    result["region_source"] = source
    result["box"] = box
    result["mask_quality"] = quality
    return result


def run_vision_ai(image):
    """Vision AI가 실패해도 OpenCV/OCR 결과는 잃지 않는다."""
    try:
        ai = observe_sign(image)
    except Exception as e:
        print(f"[경고] Vision AI 실패: {e}")
        return None, "failed"

    if ai is None:
        return None, "unavailable"

    return ai, "complete"


def analyze_sign(image_path):
    image_path = Path(image_path)

    image = load_image(image_path)
    image = resize_image(image)

    # OCR은 해상도가 높은 원본 파일로 읽는다
    ocr = read_text(image_path)

    ai, ai_status = run_vision_ai(image)

    region = choose_region(image, ocr, ai, image_path.stem)
    opencv_text = measure_text_region(image, image_path.stem, region)

    result = {
        "id": image_path.stem,
        "source": {
            "filename": image_path.name,
            "width": image.shape[1],   # 리사이즈 후 크기
            "height": image.shape[0],
        },
        "ocr": ocr if ocr is not None else {},
        "color": analyze_color(image),
        "geometry": analyze_edges(image),
        # 0~1로 정규화된 typography는 간판이 모인 뒤 채운다
        "typography": {
            "weight": None,
            "width": None,
            "roundness": None,
            "condensation": None,
            "irregularity": None,
        },
        "opencv_text": opencv_text,
        "ai": ai,
        "status": {
            "opencv": "complete",
            "ocr": "complete" if ocr is not None else "unavailable",
            "vision_ai": ai_status,
        },
    }

    # 스키마 검증: 범위를 벗어난 값이 있으면 여기서 오류가 난다
    return SignAnalysis(**result)


def main():
    if len(sys.argv) < 2:
        print("사용법: python src/main.py data/raw/signs/sign_001.jpg")
        return

    image_path = Path(sys.argv[1])
    analysis = analyze_sign(image_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{image_path.stem}.json"

    output_path.write_text(
        analysis.model_dump_json(indent=2),
        encoding="utf-8",
    )

    print(f"분석 완료: {output_path}")


if __name__ == "__main__":
    main()
