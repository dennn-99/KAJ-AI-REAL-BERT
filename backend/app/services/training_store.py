import json
import os
from pathlib import Path

try:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
except Exception:
    joblib = None
    TfidfVectorizer = None
    LogisticRegression = None
    Pipeline = None

from ..models.schemas import TrainingSample
from .bert_classifier import LOCAL_CLASSIFIER_PATH, get_classifier

DATA_DIR = Path(os.getcwd()) / "data"
SAMPLES_PATH = DATA_DIR / "training_samples.jsonl"

SEED_SAMPLES = [
    TrainingSample(text="BMKG melalui kanal resmi menjelaskan bahwa klaim prediksi gempa megathrust pada tanggal tertentu adalah hoaks.", validity_label="hoaks", source="seed"),
    TrainingSample(text="Kementerian mengumumkan rilis resmi disertai nomor surat dan tautan domain pemerintah.", validity_label="valid", source="seed"),
    TrainingSample(text="Sebarkan sekarang, listrik dan ATM akan mati tujuh hari tanpa sumber resmi.", validity_label="hoaks", origin_label="campuran", source="seed"),
    TrainingSample(text="Laporan data statistik disertai metodologi, tanggal publikasi, dan narasumber yang dapat diverifikasi.", validity_label="valid", source="seed"),
]


def _ensure_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SAMPLES_PATH.exists():
        with SAMPLES_PATH.open("w", encoding="utf-8") as handle:
            for sample in SEED_SAMPLES:
                handle.write(sample.model_dump_json() + "\n")


def list_training_samples() -> list[dict]:
    _ensure_data()
    samples: list[dict] = []
    with SAMPLES_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def add_training_sample(sample: TrainingSample) -> None:
    _ensure_data()
    with SAMPLES_PATH.open("a", encoding="utf-8") as handle:
        handle.write(sample.model_dump_json() + "\n")


def train_local_classifier() -> dict:
    if any(item is None for item in [joblib, TfidfVectorizer, LogisticRegression, Pipeline]):
        return {
            "status": "failed",
            "message": "Dependensi scikit-learn/joblib belum terpasang. Jalankan pip install -r requirements.txt.",
            "sample_count": len(list_training_samples()),
        }

    samples = [TrainingSample(**item) for item in list_training_samples()]
    labels = [sample.validity_label for sample in samples]
    texts = [sample.text for sample in samples]
    if len(set(labels)) < 2:
        return {"status": "failed", "message": "Minimal perlu dua label berbeda untuk training.", "sample_count": len(samples)}

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=6000)),
            ("classifier", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )
    pipeline.fit(texts, labels)
    Path(LOCAL_CLASSIFIER_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, LOCAL_CLASSIFIER_PATH)
    get_classifier.cache_clear()
    return {
        "status": "trained",
        "sample_count": len(samples),
        "labels": sorted(set(labels)),
        "model_path": LOCAL_CLASSIFIER_PATH,
        "note": "Model lokal ini melengkapi pipeline BERT dan berguna untuk iterasi dataset cepat.",
    }
