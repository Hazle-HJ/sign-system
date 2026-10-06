import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = ROOT / "data" / "analysis"
EXCLUDE_PATH = ROOT / "data" / "manual" / "exclude.json"

# 평균을 낼 AI 수치 항목: (그룹, 항목)
AI_FEATURES = [
    ("typography", k)
    for k in ("weight", "width", "roundness", "condensation",
              "irregularity", "stroke_variation", "decoration")
] + [
    ("material", k) for k in ("metal", "plastic", "acrylic", "paint", "fabric")
] + [
    ("frame", k) for k in ("presence", "thickness", "corner_roundness")
] + [
    ("lighting", k) for k in ("intensity", "glow")
] + [
    ("weathering", k)
    for k in ("fading", "rust", "stain", "scratch", "overall_age")
]


def summarize(values):
    """n, 평균, 표준편차(표본), 최솟값, 최댓값. 값이 1개면 표준편차는 None."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": n,
        "mean": round(statistics.mean(values), 3),
        "std": round(statistics.stdev(values), 3) if n > 1 else None,
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def main():
    records = []
    for path in sorted(ANALYSIS_DIR.glob("sign_[0-9]*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))

    manual_ex = {}
    if EXCLUDE_PATH.exists():
        manual_ex = json.loads(EXCLUDE_PATH.read_text(encoding="utf-8"))
    skipped = {d["id"]: manual_ex[d["id"]] for d in records if d["id"] in manual_ex}
    records = [d for d in records if d["id"] not in manual_ex]

    if not records:
        print("data/analysis/에 사용할 sign_[0-9]*.json 파일이 없습니다.")
        return

    excluded = {"manual": skipped, "opencv": {}, "ai": {}}
    opencv_values = {"stroke_to_height": [], "char_aspect": []}
    ai_values = {f"{g}.{k}": [] for g, k in AI_FEATURES}
    shapes, lights = Counter(), Counter()
    signs = []

    for d in records:
        sid = d["id"]
        ot = d.get("opencv_text") or {}
        obs = ((d.get("ai") or {}).get("observation")) or {}

        # --- OpenCV 글자 측정: 마스크를 믿을 수 있는 간판만 사용 ---
        if not ot:
            excluded["opencv"][sid] = "측정값 없음"
        elif ot.get("mask_quality") != "ok":
            excluded["opencv"][sid] = f"마스크 품질 {ot.get('mask_quality', 'unknown')}"
        else:
            opencv_values["stroke_to_height"].append(ot["stroke_to_height"])
            # 글자 폭 비율은 가로쓰기이고 글자 수를 아는 경우에만 쓴다
            if ot.get("char_aspect") is not None and ot.get("orientation") == "horizontal":
                opencv_values["char_aspect"].append(ot["char_aspect"])

        # --- AI 관찰: 분석이 끝난 간판만 사용 ---
        if (d.get("status") or {}).get("vision_ai") != "complete" or not obs:
            excluded["ai"][sid] = "Vision AI 결과 없음"
        else:
            for g, k in AI_FEATURES:
                v = (obs.get(g) or {}).get(k)
                if isinstance(v, (int, float)):
                    ai_values[f"{g}.{k}"].append(float(v))
            shape = (obs.get("frame") or {}).get("shape")
            light = (obs.get("lighting") or {}).get("type")
            if shape:
                shapes[shape] += 1
            if light:
                lights[light] += 1

        signs.append({
            "id": sid,
            "ocr_text": (d.get("ocr") or {}).get("text"),
            "ai_main_text": (obs.get("main_text") or {}).get("text"),
            "stroke_to_height": ot.get("stroke_to_height"),
            "mask_quality": ot.get("mask_quality"),
            "ai_weight": (obs.get("typography") or {}).get("weight"),
            "ai_overall_age": (obs.get("weathering") or {}).get("overall_age"),
        })

    summary = {
        "opencv": {k: summarize(v) for k, v in opencv_values.items()},
        "ai": {k: summarize(v) for k, v in ai_values.items()},
        "categorical": {
            "frame_shape": dict(shapes),
            "lighting_type": dict(lights),
        },
    }

    dataset = {
        "note": "표본 수(n)가 적은 잠정값입니다. 값을 0~1로 정규화하지 않았습니다.",
        "total_signs": len(records),
        "excluded": excluded,
        "summary": summary,
        "signs": signs,
    }

    out = ANALYSIS_DIR / "dataset.json"
    out.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- 화면 출력 ---
    print(f"간판 {len(records)}장 합치기 완료 -> {out}\n")
    print(f"{'항목':<26}{'n':>3}{'평균':>8}{'표준편차':>9}{'최소':>8}{'최대':>8}")
    print("-" * 62)

    def show(name, s):
        f = lambda x: "-" if x is None else f"{x:.3f}"
        print(f"{name:<26}{s['n']:>3}{f(s['mean']):>8}{f(s['std']):>9}{f(s['min']):>8}{f(s['max']):>8}")

    print("[OpenCV 글자 측정]")
    for k, s in summary["opencv"].items():
        show(k, s)
    print("[AI 관찰]")
    for k, s in summary["ai"].items():
        show(k, s)

    print("\n[범주형] 틀 모양:", dict(shapes), "/ 조명:", dict(lights))
    for kind, items in excluded.items():
        if items:
            print(f"[제외: {kind}]", items)


if __name__ == "__main__":
    main()
