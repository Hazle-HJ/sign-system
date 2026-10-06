from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# .env 파일의 GEMINI_API_KEY를 환경변수로 불러온다
load_dotenv(PROJECT_ROOT / ".env")

# 무료 티어에서 막히면 "gemini-2.5-flash"로 바꿔 본다
VISION_MODEL = "gemini-3.8-flash"
MODELS_TO_TRY = [VISION_MODEL, "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash"]
