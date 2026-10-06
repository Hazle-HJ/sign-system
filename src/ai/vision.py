import json
import os
import time

import cv2
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from ai.prompt import SYSTEM_PROMPT, USER_PROMPT
from config import MODELS_TO_TRY


# --- Gemini에 요청할 응답 형식 (범위 검증은 schema.py에서 한다) ---
class _MainText(BaseModel):
    text: str | None = None
    box_2d: list[int] | None = None  # [ymin, xmin, ymax, xmax], 0~1000


class _Typography(BaseModel):
    weight: float | None = None
    width: float | None = None
    roundness: float | None = None
    condensation: float | None = None
    irregularity: float | None = None
    stroke_variation: float | None = None
    decoration: float | None = None


class _Material(BaseModel):
    metal: float | None = None
    plastic: float | None = None
    acrylic: float | None = None
    paint: float | None = None
    fabric: float | None = None


class _Frame(BaseModel):
    presence: float | None = None
    thickness: float | None = None
    shape: str | None = None
    corner_roundness: float | None = None


class _Lighting(BaseModel):
    type: str | None = None
    intensity: float | None = None
    glow: float | None = None


class _Weathering(BaseModel):
    fading: float | None = None
    rust: float | None = None
    stain: float | None = None
    scratch: float | None = None
    overall_age: float | None = None


class _Observation(BaseModel):
    main_text: _MainText | None = None
    typography: _Typography
    material: _Material
    frame: _Frame
    lighting: _Lighting
    weathering: _Weathering
    confidence: float | None = None
    notes: list[str] = []


def observe_sign(image):
    """간판 사진을 Gemini에 보내 구조화된 관찰 결과를 받는다.

    API 키가 없으면 None을 반환한다.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("이미지를 JPEG로 변환하지 못했습니다.")

    client = genai.Client(api_key=api_key)

    contents = [
        types.Part.from_bytes(data=buffer.tobytes(), mime_type="image/jpeg"),
        USER_PROMPT,
    ]

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=_Observation,
        temperature=0,
    )

    retry_codes = {429, 500, 503, 504}
    skip_codes = {400, 403, 404}  # 이 모델은 쓸 수 없다 -> 다음 모델로
    max_attempts = 2
    last_error = None

    for model in MODELS_TO_TRY:
        for attempt in range(1, max_attempts + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                return {
                    "model": model,
                    "observation": json.loads(response.text),
                }
            except errors.APIError as e:
                last_error = e
                message = str(getattr(e, "message", None) or e)
                short = message[:160].replace("\n", " ")

                if e.code in skip_codes:
                    print(f"{model}: 사용할 수 없음 ({e.code}) {short}")
                    break
                if e.code not in retry_codes:
                    raise
                if attempt < max_attempts:
                    wait = 40 if e.code == 429 else 10
                    print(f"{model}: ({e.code}) {short} / {wait}초 후 재시도")
                    time.sleep(wait)
                else:
                    print(f"{model}: 계속 실패({e.code}). 다음 모델로 넘어갑니다.")

    raise RuntimeError(f"모든 모델에서 Vision AI 요청이 실패했습니다. 마지막 오류: {last_error}")
