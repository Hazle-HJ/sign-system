from pydantic import BaseModel, Field


class Source(BaseModel):
    filename: str
    width: int
    height: int


class OcrLine(BaseModel):
    text: str
    confidence: float = Field(ge=0, le=1)
    bbox: list[float] | None = None  # [x, y, w, h], 0~1, 좌상단 기준


class Ocr(BaseModel):
    text: str = ""
    confidence: float | None = Field(default=None, ge=0, le=1)
    lines: list[OcrLine] = []


class Typography(BaseModel):
    # 아직 측정하지 않은 값은 None으로 둔다 (억지로 숫자를 넣지 않는다)
    weight: float | None = Field(default=None, ge=0, le=1)
    width: float | None = Field(default=None, ge=0, le=1)
    roundness: float | None = Field(default=None, ge=0, le=1)
    condensation: float | None = Field(default=None, ge=0, le=1)
    irregularity: float | None = Field(default=None, ge=0, le=1)


class OpenCvText(BaseModel):
    # 정규화하지 않은 원래 값. 0~1 변환은 간판이 모인 뒤(STEP 9~10)에 한다.
    line_text: str
    orientation: str
    stroke_width_px: float
    char_size_px: float
    stroke_to_height: float
    char_aspect: float | None = None
    # 어떤 방법으로 글자 영역을 정했는지: manual / ai / ocr_largest
    region_source: str = "unknown"
    box: list[float] | None = None  # [x, y, w, h], 0~1, 좌상단 기준
    mask_quality: str = "unknown"  # ok / low (low는 굵기 값을 믿지 않는다)


class Color(BaseModel):
    mean_saturation: float = Field(ge=0, le=1)
    mean_brightness: float = Field(ge=0, le=1)
    contrast: float = Field(ge=0, le=1)


class Geometry(BaseModel):
    edge_density: float = Field(ge=0, le=1)


# --- Vision AI가 관찰한 값 (OpenCV 값과 섞지 않고 따로 보관한다) ---
Score = float | None


class AiMainText(BaseModel):
    text: str | None = None
    box_2d: list[float] | None = None  # [ymin, xmin, ymax, xmax], 0~1000


class AiTypography(BaseModel):
    weight: Score = Field(default=None, ge=0, le=1)
    width: Score = Field(default=None, ge=0, le=1)
    roundness: Score = Field(default=None, ge=0, le=1)
    condensation: Score = Field(default=None, ge=0, le=1)
    irregularity: Score = Field(default=None, ge=0, le=1)
    stroke_variation: Score = Field(default=None, ge=0, le=1)
    decoration: Score = Field(default=None, ge=0, le=1)


class AiMaterial(BaseModel):
    metal: Score = Field(default=None, ge=0, le=1)
    plastic: Score = Field(default=None, ge=0, le=1)
    acrylic: Score = Field(default=None, ge=0, le=1)
    paint: Score = Field(default=None, ge=0, le=1)
    fabric: Score = Field(default=None, ge=0, le=1)


class AiFrame(BaseModel):
    presence: Score = Field(default=None, ge=0, le=1)
    thickness: Score = Field(default=None, ge=0, le=1)
    shape: str | None = None
    corner_roundness: Score = Field(default=None, ge=0, le=1)


class AiLighting(BaseModel):
    type: str | None = None
    intensity: Score = Field(default=None, ge=0, le=1)
    glow: Score = Field(default=None, ge=0, le=1)


class AiWeathering(BaseModel):
    fading: Score = Field(default=None, ge=0, le=1)
    rust: Score = Field(default=None, ge=0, le=1)
    stain: Score = Field(default=None, ge=0, le=1)
    scratch: Score = Field(default=None, ge=0, le=1)
    overall_age: Score = Field(default=None, ge=0, le=1)


class AiObservation(BaseModel):
    main_text: AiMainText | None = None
    typography: AiTypography
    material: AiMaterial
    frame: AiFrame
    lighting: AiLighting
    weathering: AiWeathering
    confidence: Score = Field(default=None, ge=0, le=1)
    notes: list[str] = []


class AiBlock(BaseModel):
    model: str
    observation: AiObservation


class Status(BaseModel):
    opencv: str = "pending"
    ocr: str = "pending"
    vision_ai: str = "pending"


class SignAnalysis(BaseModel):
    id: str
    source: Source
    ocr: Ocr
    color: Color
    geometry: Geometry
    typography: Typography
    opencv_text: OpenCvText | None = None
    ai: AiBlock | None = None
    status: Status
