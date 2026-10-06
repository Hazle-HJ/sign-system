import base64
import html
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "data" / "analysis"
PROCESSED = ROOT / "data" / "processed"
MANUAL = ROOT / "data" / "manual"
OUT = ROOT / "data" / "report.html"

LABELS = {
    "stroke_to_height": "획 굵기 / 글자 높이",
    "char_aspect": "글자 가로세로 비",
    "typography.weight": "굵기",
    "typography.width": "폭",
    "typography.roundness": "둥글기",
    "typography.condensation": "응축(촘촘함)",
    "typography.irregularity": "불규칙",
    "typography.stroke_variation": "굵고 가는 대비",
    "typography.decoration": "장식",
    "material.metal": "재료: 금속",
    "material.plastic": "재료: 플라스틱",
    "material.acrylic": "재료: 아크릴",
    "material.paint": "재료: 페인트",
    "material.fabric": "재료: 천",
    "frame.presence": "틀 있음",
    "frame.thickness": "틀 두께",
    "frame.corner_roundness": "틀 모서리 둥글기",
    "lighting.intensity": "조명 세기",
    "lighting.glow": "빛 번짐",
    "weathering.fading": "바램",
    "weathering.rust": "녹",
    "weathering.stain": "얼룩",
    "weathering.scratch": "긁힘·벗겨짐",
    "weathering.overall_age": "전체 노후도",
}
GROUPS = [
    ("typography", "글자 (AI 관찰)"),
    ("material", "재료 (AI 관찰)"),
    ("frame", "틀 (AI 관찰)"),
    ("lighting", "조명 (AI 관찰)"),
    ("weathering", "노후화 (AI 관찰)"),
]


def esc(x):
    return html.escape(str(x))


def fmt(v, digits=2):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def img_uri(path, max_w=720):
    """이미지를 줄여서 HTML 안에 직접 넣는다. (파일 하나로 열 수 있게)"""
    if not path.exists():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def load_json(path, default=None):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def bar(mean, std):
    """0~1 값의 평균(점)과 평균±표준편차(띠)를 보여 주는 작은 막대."""
    if mean is None:
        return ""
    s = std or 0
    lo, hi = max(0, mean - s), min(1, mean + s)
    return (
        '<div class="track">'
        f'<div class="band" style="left:{lo*100:.1f}%;width:{(hi-lo)*100:.1f}%"></div>'
        f'<div class="dot" style="left:{mean*100:.1f}%"></div></div>'
    )


def sign_card(d, excluded):
    sid = d.get("id", "?")
    ot = d.get("opencv_text") or {}
    obs = ((d.get("ai") or {}).get("observation")) or {}
    status = d.get("status") or {}
    ty = obs.get("typography") or {}
    wr = obs.get("weathering") or {}

    badges = []
    cls = ""
    if sid in excluded:
        badges.append(f'<span class="b gray">통계 제외 (직접 지정)</span>')
        cls = "off"
    if ot.get("mask_quality") == "low":
        badges.append('<span class="b red">마스크 low · 굵기 값 제외</span>')
    elif ot.get("mask_quality") == "ok":
        badges.append('<span class="b green">마스크 ok</span>')
    if status.get("vision_ai") != "complete":
        badges.append('<span class="b red">Vision AI 없음</span>')

    p = img_uri(PROCESSED / "previews" / f"{sid}_preview.png")
    m = img_uri(PROCESSED / "masks" / f"{sid}_mask.png", 420)
    figs = ""
    if p:
        figs += f'<figure><img src="{p}" alt="{esc(sid)} 글자 영역"><figcaption>글자 영역 (빨간 상자)</figcaption></figure>'
    if m:
        figs += f'<figure><img src="{m}" alt="{esc(sid)} 마스크"><figcaption>마스크 (흰색 = 글자로 인식)</figcaption></figure>'

    mat = obs.get("material") or {}
    top_mat = max(mat, key=lambda k: mat[k] or 0) if mat else None
    rows = [
        ("읽은 글자 (OCR)", (d.get("ocr") or {}).get("text") or "-"),
        ("AI가 고른 주 글자", (obs.get("main_text") or {}).get("text") or "-"),
        ("영역 출처", ot.get("region_source", "-")),
        ("획 굵기 / 글자 높이", fmt(ot.get("stroke_to_height"), 3)),
        ("글자 가로세로 비", fmt(ot.get("char_aspect"), 3)),
        ("AI 굵기", fmt(ty.get("weight"))),
        ("AI 둥글기 / 불규칙", f'{fmt(ty.get("roundness"))} / {fmt(ty.get("irregularity"))}'),
        ("AI 노후도", fmt(wr.get("overall_age"))),
        ("재료(가장 높은 것)", f"{top_mat} {fmt(mat.get(top_mat))}" if top_mat else "-"),
        ("틀 모양 / 조명", f'{(obs.get("frame") or {}).get("shape") or "-"} / {(obs.get("lighting") or {}).get("type") or "-"}'),
    ]
    dl = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in rows)
    notes = "".join(f"<li>{esc(n)}</li>" for n in (obs.get("notes") or []))
    notes = f'<ul class="notes">{notes}</ul>' if notes else ""

    return (
        f'<article class="card {cls}"><header><h3>{esc(sid)}</h3>'
        f'<div class="badges">{"".join(badges)}</div></header>'
        f'<div class="imgs">{figs}</div><dl>{dl}</dl>{notes}</article>'
    )


def summary_table(signs, excluded):
    head = "<tr><th>간판</th><th>AI가 고른 글자</th><th>영역 출처</th><th>획/높이</th><th>마스크</th><th>AI 굵기</th><th>AI 노후도</th><th>통계 사용</th></tr>"
    body = ""
    for d in signs:
        ot = d.get("opencv_text") or {}
        obs = ((d.get("ai") or {}).get("observation")) or {}
        used = "제외(직접)" if d["id"] in excluded else (
            "AI만" if ot.get("mask_quality") != "ok" else "AI + OpenCV")
        body += (
            f"<tr><td>{esc(d['id'])}</td>"
            f"<td>{esc((obs.get('main_text') or {}).get('text') or '-')}</td>"
            f"<td>{esc(ot.get('region_source', '-'))}</td>"
            f"<td>{fmt(ot.get('stroke_to_height'), 3)}</td>"
            f"<td>{esc(ot.get('mask_quality', '-'))}</td>"
            f"<td>{fmt((obs.get('typography') or {}).get('weight'))}</td>"
            f"<td>{fmt((obs.get('weathering') or {}).get('overall_age'))}</td>"
            f"<td>{used}</td></tr>"
        )
    return f"<table>{head}{body}</table>"


def stats_section(ds):
    if not ds:
        return "<p>dataset.json이 없습니다. python src/dataset.py를 먼저 실행하세요.</p>"
    summ = ds.get("summary", {})
    out = ""

    rows = ""
    for k, s in (summ.get("opencv") or {}).items():
        rows += (
            f"<tr><td>{esc(LABELS.get(k, k))}</td><td>{s['n']}</td><td>{fmt(s['mean'], 3)}</td>"
            f"<td>{fmt(s['std'], 3)}</td><td>{fmt(s['min'], 3)}</td><td>{fmt(s['max'], 3)}</td></tr>"
        )
    out += (
        "<h3>OpenCV 글자 측정 (마스크가 ok인 간판만)</h3><table>"
        "<tr><th>항목</th><th>n</th><th>평균</th><th>표준편차</th><th>최소</th><th>최대</th></tr>"
        f"{rows}</table>"
    )

    ai = summ.get("ai") or {}
    for g, title in GROUPS:
        rows = ""
        for k, s in ai.items():
            if not k.startswith(g + "."):
                continue
            rows += (
                f"<tr><td>{esc(LABELS.get(k, k))}</td><td>{s['n']}</td><td>{fmt(s['mean'])}</td>"
                f"<td>{fmt(s['std'])}</td><td class='barcell'>{bar(s['mean'], s['std'])}</td></tr>"
            )
        if rows:
            out += (
                f"<h3>{esc(title)}</h3><table><tr><th>항목</th><th>n</th><th>평균</th>"
                f"<th>표준편차</th><th>0 ──── 1 (점 = 평균, 띠 = ±표준편차)</th></tr>{rows}</table>"
            )

    cat = summ.get("categorical") or {}
    out += (
        f"<p class='small'>틀 모양: {esc(cat.get('frame_shape', {}))} · "
        f"조명: {esc(cat.get('lighting_type', {}))}</p>"
    )
    ex = ds.get("excluded") or {}
    for kind, label in (("manual", "직접 제외"), ("opencv", "OpenCV 통계에서 제외"), ("ai", "AI 통계에서 제외")):
        if ex.get(kind):
            items = ", ".join(f"{esc(k)} ({esc(v)})" for k, v in ex[kind].items())
            out += f"<p class='small'><b>{label}:</b> {items}</p>"
    return out


def dna_section(dna):
    if not dna:
        return "<p>sign_dna.json이 없습니다. python src/dna.py를 먼저 실행하세요.</p>"
    rows = ""
    for k, v in (dna.get("typeface") or {}).items():
        rows += (
            f"<tr><td>{esc(LABELS.get(k, k))}</td><td>{fmt(v['base'], 3)}</td>"
            f"<td>{fmt(v['variation'], 3)}</td><td>{v['n']}</td><td>{esc(v['source'])}</td></tr>"
        )
    rule = "".join(
        f"<tr><td>{esc(k)}</td><td>{fmt(v)}</td></tr>" for k, v in (dna.get("jamo_rule") or {}).items()
    )
    assumptions = "".join(f"<li>{esc(a)}</li>" for a in dna.get("assumptions", []))
    return (
        f"<p class='small'>버전: {esc(dna.get('version'))} · 사용한 간판: {esc(', '.join(dna.get('based_on', [])))}</p>"
        "<table><tr><th>글자 항목</th><th>기준값</th><th>변동(±)</th><th>n</th><th>출처</th></tr>"
        f"{rows}</table>"
        "<h3>자소 규칙 v0 → 시안 슬라이더 값</h3>"
        f"<table><tr><th>파라미터</th><th>값</th></tr>{rule}</table>"
        f"<h3>가정과 한계</h3><ul>{assumptions}</ul>"
    )


CSS = """
:root{--bg:#f3efe6;--card:#fbf9f3;--line:#d9d3c4;--text:#1c1b19;--sub:#5b574d;--accent:#b23a26;--green:#2f7d4f;--red:#b23a26}
@media (prefers-color-scheme:dark){:root{--bg:#1c1b19;--card:#2a2926;--line:#3d3b36;--text:#f3efe6;--sub:#b9b3a6;--accent:#e0795f;--green:#6fcf97;--red:#e0795f}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,"Apple SD Gothic Neo","Noto Sans KR",sans-serif;line-height:1.5}
main{max-width:1200px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:28px;margin:0 0 4px} h2{font-size:22px;margin:40px 0 12px;border-bottom:2px solid var(--line);padding-bottom:6px}
h3{font-size:16px;margin:22px 0 8px} .sub,.small{color:var(--sub);font-size:14px}
table{border-collapse:collapse;width:100%;font-size:14px;background:var(--card);border-radius:10px;overflow:hidden}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}
th{font-size:13px;color:var(--sub);font-weight:600}
.scroll{overflow-x:auto}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px}
.card.off{opacity:.55}
.card header{display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap}
.card h3{margin:0;font-size:18px}
.b{font-size:12px;padding:2px 8px;border-radius:99px;border:1px solid var(--line);margin-left:4px}
.b.green{color:var(--green);border-color:var(--green)} .b.red{color:var(--red);border-color:var(--red)} .b.gray{color:var(--sub)}
.imgs{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:10px 0}
figure{margin:0} figure img{width:100%;border-radius:8px;display:block;background:#000}
figcaption{font-size:12px;color:var(--sub);margin-top:2px}
dl{display:grid;grid-template-columns:140px 1fr;gap:3px 10px;margin:8px 0;font-size:14px}
dt{color:var(--sub)} dd{margin:0;font-weight:600}
.notes{font-size:13px;color:var(--sub);margin:6px 0 0;padding-left:18px}
.track{position:relative;height:10px;background:var(--line);border-radius:5px;min-width:160px}
.band{position:absolute;top:0;height:100%;background:var(--accent);opacity:.35;border-radius:5px}
.dot{position:absolute;top:-2px;width:14px;height:14px;margin-left:-7px;background:var(--accent);border-radius:50%}
.barcell{width:42%}
"""


def main():
    signs = []
    for path in sorted(ANALYSIS.glob("sign_[0-9]*.json")):
        signs.append(json.loads(path.read_text(encoding="utf-8")))
    if not signs:
        print("data/analysis/에 분석 결과가 없습니다.")
        return

    excluded = load_json(MANUAL / "exclude.json", {})
    ds = load_json(ANALYSIS / "dataset.json")
    dna = load_json(ANALYSIS / "sign_dna.json")

    cards = "".join(sign_card(d, excluded) for d in signs)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    page = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SIGN SYSTEM 분석 결과</title><style>{CSS}</style></head><body><main>
<h1>SIGN SYSTEM 분석 결과</h1>
<p class="sub">간판 {len(signs)}장 · 만든 시각 {now} · 이 화면은 data/report.html 파일입니다 (다시 만들려면 python src/report.py)</p>
<h2>1. 한눈에 보기</h2><div class="scroll">{summary_table(signs, excluded)}</div>
<h2>2. 간판별 결과</h2><p class="small">빨간 상자는 시스템이 '가게 이름 글자'라고 판단한 영역이고, 마스크는 그 안에서 글자로 인식한 부분입니다. 마스크가 글자를 잘 따라가는지 눈으로 확인하세요.</p>
<div class="grid">{cards}</div>
<h2>3. 통계 (평균과 변동)</h2><p class="small">n은 그 항목에 쓰인 간판 수입니다. 표본이 적으면 평균이 불안정합니다.</p>
<div class="scroll">{stats_section(ds)}</div>
<h2>4. SIGN DNA (잠정)</h2><div class="scroll">{dna_section(dna)}</div>
</main></body></html>"""

    OUT.write_text(page, encoding="utf-8")
    print(f"결과 화면 생성 완료 -> {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
