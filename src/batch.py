import json
import sys
import time

from config import PROJECT_ROOT
from main import OUTPUT_DIR, analyze_sign

SIGNS_DIR = PROJECT_ROOT / "data" / "raw" / "signs"
EXTENSIONS = {".jpg", ".jpeg", ".png"}
PAUSE_SECONDS = 15  # 무료 한도를 아끼기 위해 사진 사이에 쉬는 시간


def is_complete(json_path):
    """OpenCV, OCR, Vision AI가 끝났고 주 글자 위치(main_text)까지 있는 결과인가."""
    if not json_path.exists():
        return False
    data = json.loads(json_path.read_text(encoding="utf-8"))
    status = data.get("status", {})
    if not all(status.get(k) == "complete" for k in ("opencv", "ocr", "vision_ai")):
        return False
    obs = ((data.get("ai") or {}).get("observation")) or {}
    return bool(obs.get("main_text"))


def main():
    force = "--force" in sys.argv

    all_files = sorted(p for p in SIGNS_DIR.iterdir() if not p.name.startswith("."))
    images = [p for p in all_files if p.suffix.lower() in EXTENSIONS]
    skipped_types = [p.name for p in all_files if p.suffix.lower() not in EXTENSIONS]

    if skipped_types:
        print(f"[알림] 읽을 수 없는 형식이라 건너뜁니다: {', '.join(skipped_types)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, image_path in enumerate(images, start=1):
        json_path = OUTPUT_DIR / f"{image_path.stem}.json"
        print(f"\n[{i}/{len(images)}] {image_path.name}")

        if not force and is_complete(json_path):
            print("  이미 분석이 끝나 건너뜁니다. (다시 하려면 --force)")
            continue

        try:
            analysis = analyze_sign(image_path)
            json_path.write_text(
                analysis.model_dump_json(indent=2),
                encoding="utf-8",
            )
            print(f"  저장: {json_path.name}")
        except Exception as e:
            print(f"  [오류] {image_path.name}: {e}")

        if i < len(images):
            time.sleep(PAUSE_SECONDS)

    # 결과 요약표
    print("\n" + "=" * 78)
    print(f"{'id':<10}{'AI가 고른 주 글자':<18}{'영역출처':<13}{'획/높이':<9}{'AI굵기':<8}{'AI상태'}")
    print("-" * 78)

    for image_path in images:
        json_path = OUTPUT_DIR / f"{image_path.stem}.json"
        if not json_path.exists():
            print(f"{image_path.stem:<10}(결과 없음)")
            continue

        d = json.loads(json_path.read_text(encoding="utf-8"))
        obs = ((d.get("ai") or {}).get("observation")) or {}
        main_text = (obs.get("main_text") or {}).get("text") or "-"
        cv_text = d.get("opencv_text") or {}

        source = cv_text.get("region_source", "-")
        ratio = cv_text.get("stroke_to_height")
        weight = (obs.get("typography") or {}).get("weight")
        vision = (d.get("status") or {}).get("vision_ai", "-")

        print(
            f"{d['id']:<10}{main_text[:8]:<18}{source:<13}"
            f"{'-' if ratio is None else ratio:<9}"
            f"{'-' if weight is None else weight:<8}"
            f"{vision}"
        )


if __name__ == "__main__":
    main()
