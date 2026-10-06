import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = ROOT / "data" / "analysis"

# 자소 시안(슬라이더)의 기하 구조에서 온 값. 규칙 v0의 가정이다.
GLYPH_HEIGHT = 88      # 시안에서 글자 높이(단위)
STROKE_BASE = 3        # 시안의 획 굵기 = 3 + 15 * weight
STROKE_RANGE = 15
ASPECT_FACTOR = 0.98   # 시안에서 글자 가로세로 비 = 0.98 * (0.65 + 0.5 * width)
VARIATION_GAIN = 0.346 # 시안의 글자별 변동: 굵기의 상대 표준편차 = 0.346 * variation


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def pick(summary, group, key):
    s = (summary.get(group) or {}).get(key) or {}
    return s if s.get("n") else None


def feature(s, source):
    return {
        "base": s["mean"],
        "variation": s["std"],
        "n": s["n"],
        "source": source,
    }


def main():
    ds_path = ANALYSIS_DIR / "dataset.json"
    if not ds_path.exists():
        print("dataset.json이 없습니다. 먼저 python src/dataset.py를 실행하세요.")
        return

    ds = json.loads(ds_path.read_text(encoding="utf-8"))
    summary = ds["summary"]
    ids = [s["id"] for s in ds["signs"]]

    stroke = pick(summary, "opencv", "stroke_to_height")
    aspect = pick(summary, "opencv", "char_aspect")
    rnd = pick(summary, "ai", "typography.roundness")
    irr = pick(summary, "ai", "typography.irregularity")

    typeface = {}
    if stroke:
        typeface["stroke_to_height"] = feature(stroke, "opencv")
    if aspect:
        typeface["char_aspect"] = feature(aspect, "opencv")
    if rnd:
        typeface["roundness"] = feature(rnd, "ai")
    if irr:
        typeface["irregularity"] = feature(irr, "ai")

    # --- 자소 규칙 v0: DNA 값을 시안의 슬라이더 값으로 바꾼다 ---
    rule = {}
    if stroke:
        weight = clamp((stroke["mean"] * GLYPH_HEIGHT - STROKE_BASE) / STROKE_RANGE)
        rule["weight"] = round(weight, 2)
        if stroke["std"] is not None and stroke["mean"]:
            cv = stroke["std"] / stroke["mean"]
            rule["variation"] = round(clamp(cv / VARIATION_GAIN), 2)
    if aspect:
        width = clamp((aspect["mean"] / ASPECT_FACTOR - 0.65) / 0.5)
        rule["width"] = round(width, 2)
    if rnd:
        rule["roundness"] = round(clamp(rnd["mean"]), 2)
    if irr:
        rule["irregularity"] = round(clamp(irr["mean"]), 2)

    sign_system = {}
    for key, s in summary["ai"].items():
        group = key.split(".")[0]
        if group in ("material", "frame", "lighting", "weathering") and s["n"]:
            sign_system[key] = feature(s, "ai")
    sign_system["categorical"] = summary.get("categorical", {})

    dna = {
        "version": "v0-provisional",
        "based_on": ids,
        "excluded": ds.get("excluded", {}),
        "typeface": typeface,
        "jamo_rule": rule,
        "sign_system": sign_system,
        "assumptions": [
            "표본이 매우 적은 잠정값이며, 서로 다른 종류의 간판이 섞여 있다.",
            "굵기와 글자 폭은 OpenCV 측정을 쓰고, AI 굵기는 간판 간 구별이 약해 쓰지 않았다.",
            "둥글기와 불규칙은 AI 관찰값이라 참고값이다.",
            "OpenCV 값을 시안 슬라이더 값으로 바꾸는 식은 시안의 기하 구조를 기준으로 한 가정이다.",
        ],
    }

    out = ANALYSIS_DIR / "sign_dna.json"
    out.write_text(json.dumps(dna, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"SIGN DNA (잠정, 간판 {len(ids)}장: {', '.join(ids)}) -> {out}\n")
    print(f"{'타입 항목':<20}{'기준값':>8}{'변동(±)':>9}{'n':>4}  출처")
    print("-" * 50)
    for k, v in typeface.items():
        sd = "-" if v["variation"] is None else f"{v['variation']:.3f}"
        print(f"{k:<20}{v['base']:>8.3f}{sd:>9}{v['n']:>4}  {v['source']}")

    print("\n자소 규칙 v0 -> 시안 슬라이더 값")
    for k, v in rule.items():
        print(f"  {k}: {v}")

    for k, v in typeface.items():
        if v["n"] < 5:
            print(f"\n[주의] {k}는 n={v['n']}라서 평균이 불안정합니다.")


if __name__ == "__main__":
    main()
