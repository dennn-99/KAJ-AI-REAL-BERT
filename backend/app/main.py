from pathlib import Path
import json
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .models.schemas import AnalysisResponse, TrainingSample
from .services.media_pipeline import analyze_uploaded_media
from .services.text_pipeline import analyze_text_content, analyze_url_content
from .services.training_store import add_training_sample, list_training_samples, train_local_classifier

BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="KAJ AI",
    description="AI analisis artikel, media sosial, gambar, audio, dan video untuk deteksi AI/asli serta valid/hoaks.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "name": "KAJ AI"}


@app.post("/api/analyze/text", response_model=AnalysisResponse)
def analyze_text(text: str = Form(...), language: str = Form("auto")) -> AnalysisResponse:
    return analyze_text_content(text=text, requested_language=language)


@app.post("/api/analyze/url", response_model=AnalysisResponse)
def analyze_url(url: str = Form(...), language: str = Form("auto")) -> AnalysisResponse:
    return analyze_url_content(url=url, requested_language=language)


@app.post("/api/analyze/media", response_model=AnalysisResponse)
async def analyze_media(
    file: UploadFile = File(...),
    caption: Optional[str] = Form(None),
    language: str = Form("auto"),
) -> AnalysisResponse:
    return await analyze_uploaded_media(file=file, caption=caption or "", requested_language=language)


@app.get("/api/training/samples")
def samples() -> dict:
    return {"samples": list_training_samples()}


@app.get("/api/viral-tests")
def viral_tests() -> dict:
    path = BASE_DIR / "data" / "current_viral_tests.json"
    with path.open("r", encoding="utf-8") as handle:
        return {"tests": json.load(handle)}


@app.post("/api/training/samples")
def create_sample(sample: TrainingSample) -> dict:
    add_training_sample(sample)
    return {"status": "stored", "sample": sample}


@app.post("/api/training/train")
def train() -> dict:
    return train_local_classifier()
