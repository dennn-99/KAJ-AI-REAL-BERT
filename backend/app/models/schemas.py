from typing import Literal

from pydantic import BaseModel, Field

ValidityLabel = Literal["valid", "perlu_verifikasi", "hoaks"]
OriginLabel = Literal["asli", "ai_generated", "campuran", "tidak_pasti"]


class Entity(BaseModel):
    text: str
    type: str


class ScoreItem(BaseModel):
    label: str
    score: float = Field(ge=0, le=1)


class EvidenceItem(BaseModel):
    title: str
    detail: str
    weight: float = Field(ge=-1, le=1)


class AnalysisResponse(BaseModel):
    media_type: str
    language: str
    validity_label: ValidityLabel
    origin_label: OriginLabel
    confidence: float = Field(ge=0, le=1)
    hoax_probability: float = Field(ge=0, le=1)
    ai_probability: float = Field(ge=0, le=1)
    sentiment: ScoreItem
    entities: list[Entity]
    explanation: str
    evidence: list[EvidenceItem]
    recommendations: list[str]
    model_notes: list[str]


class TrainingSample(BaseModel):
    text: str = Field(min_length=12)
    validity_label: ValidityLabel
    origin_label: OriginLabel = "asli"
    language: str = "id"
    source: str = "manual"
    notes: str = ""
