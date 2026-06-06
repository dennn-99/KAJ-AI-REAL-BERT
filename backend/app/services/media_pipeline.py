import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import UploadFile

from ..models.schemas import AnalysisResponse, EvidenceItem
from .text_pipeline import analyze_text_content

IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
AUDIO_TYPES = {"audio/mpeg", "audio/wav", "audio/x-wav", "audio/mp4", "audio/ogg"}
VIDEO_TYPES = {"video/mp4", "video/webm", "video/quicktime", "video/x-msvideo"}


async def analyze_uploaded_media(file: UploadFile, caption: str, requested_language: str = "auto") -> AnalysisResponse:
    content = await file.read()
    digest = hashlib.sha256(content).hexdigest()[:16]
    media_type = _media_type(file.content_type or "", file.filename or "")
    synthetic_text = _media_summary(file.filename or "media", file.content_type or "unknown", len(content), digest, caption, media_type)
    result = analyze_text_content(synthetic_text, requested_language=requested_language, media_type=media_type)
    result.evidence.extend(_media_evidence(content, file.content_type or "", media_type))
    if media_type in {"image", "audio", "video"}:
        result.model_notes.append("Analisis media memakai metadata, caption, dan sinyal file. Tambahkan model vision/audio khusus untuk inspeksi piksel, frame, dan spektrogram produksi.")
    return result


def _media_type(content_type: str, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if content_type in IMAGE_TYPES or suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        return "image"
    if content_type in AUDIO_TYPES or suffix in {".mp3", ".wav", ".m4a", ".ogg"}:
        return "audio"
    if content_type in VIDEO_TYPES or suffix in {".mp4", ".webm", ".mov", ".avi"}:
        return "video"
    return "file"


def _media_summary(filename: str, content_type: str, size: int, digest: str, caption: str, media_type: str) -> str:
    return (
        f"Analisis {media_type} bernama {filename}. MIME {content_type}. "
        f"Ukuran {size} byte. Hash SHA256 pendek {digest}. Caption atau transkrip: {caption or 'tidak tersedia'}."
    )


def _media_evidence(content: bytes, content_type: str, media_type: str) -> list[EvidenceItem]:
    evidence = [
        EvidenceItem(title="Sidik jari file", detail=f"File {media_type} memiliki hash unik untuk pelacakan dan audit.", weight=0.15)
    ]
    if len(content) < 10_000:
        evidence.append(EvidenceItem(title="Ukuran file kecil", detail="Ukuran sangat kecil dapat berarti media terkompresi, cuplikan, atau metadata minim.", weight=-0.1))
    if content_type == "image/png" and b"AI" in content[:4096]:
        evidence.append(EvidenceItem(title="Metadata AI", detail="Ada token metadata yang mengarah pada proses AI/generatif.", weight=-0.35))
    return evidence
