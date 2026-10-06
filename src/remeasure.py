import json

from config import PROJECT_ROOT
from image.preprocess import load_image, resize_image
from main import choose_region, measure_text_region

for jp in sorted((PROJECT_ROOT / "data" / "analysis").glob("sign_[0-9]*.json")):
    d = json.loads(jp.read_text(encoding="utf-8"))
    img_path = PROJECT_ROOT / "data" / "raw" / "signs" / d["source"]["filename"]

    image = resize_image(load_image(img_path))
    region = choose_region(image, d.get("ocr"), d.get("ai"), d["id"])
    d["opencv_text"] = measure_text_region(image, d["id"], region)

    jp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

    ot = d["opencv_text"] or {}
    print(d["id"], ot.get("region_source"), ot.get("stroke_to_height"), ot.get("mask_quality"))
