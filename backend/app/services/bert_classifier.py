import os
import re
from functools import lru_cache

try:
    import joblib
except Exception:
    joblib = None

from ..models.schemas import ScoreItem

MODEL_DIR = os.path.join(os.getcwd(), "data", "models")
LOCAL_CLASSIFIER_PATH = os.path.join(MODEL_DIR, "kaj_local_classifier.joblib")

HOAX_TERMS = {
    "sebarkan",
    "viral",
    "darurat",
    "konspirasi",
    "tanpa sumber",
    "jangan percaya media",
    "pemerintah menyembunyikan",
    "klik link",
    "hadiah",
    "bansos cair",
    "pendaftaran",
    "100%",
    "terbukti",
}

VALID_TERMS = {
    "menurut",
    "rilis resmi",
    "data",
    "laporan",
    "sumber",
    "konfirmasi",
    "klarifikasi",
    "berdasarkan",
    "nomor surat",
}

AI_TERMS = {
    "gambar tampak terlalu halus",
    "suara sintetis",
    "deepfake",
    "ai generated",
    "generated speech",
    "manipulasi ai",
    "wajah tidak konsisten",
}


class BertClassifier:
    def __init__(self) -> None:
        self.model_name = os.getenv("KAJ_BERT_MODEL", "bert-base-multilingual-cased")
        self.transformer_ready = False
        self.local_classifier = None
        self.tokenizer = None
        self.model = None
        self._load_local_classifier()
        self._try_load_transformer()

    def _load_local_classifier(self) -> None:
        if joblib is not None and os.path.exists(LOCAL_CLASSIFIER_PATH):
            try:
                self.local_classifier = joblib.load(LOCAL_CLASSIFIER_PATH)
            except Exception:
                self.local_classifier = None

    def _try_load_transformer(self) -> None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            allow_download = os.getenv("KAJ_ALLOW_MODEL_DOWNLOAD", "0") == "1"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=not allow_download)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, local_files_only=not allow_download)
            self.transformer_ready = True
        except Exception:
            self.transformer_ready = False

    def classify(self, text: str) -> tuple[float, float, list[str]]:
        notes: list[str] = []
        if self.local_classifier is not None:
            try:
                probabilities = self.local_classifier.predict_proba([text])[0]
                classes = list(self.local_classifier.classes_)
                hoax_probability = float(probabilities[classes.index("hoaks")]) if "hoaks" in classes else 0.35
                valid_probability = float(probabilities[classes.index("valid")]) if "valid" in classes else 0.35
                ai_probability = self._ai_probability(text)
                notes.append("Klasifikasi memakai model lokal hasil pelatihan dataset KAJ AI.")
                return hoax_probability, ai_probability, notes
            except Exception:
                notes.append("Model lokal gagal dipakai, fallback ke skor semantik.")

        if self.transformer_ready:
            notes.append(f"Backbone BERT tersedia: {self.model_name}. Tambahkan fine-tuned head untuk akurasi produksi.")
        else:
            notes.append("Backbone BERT belum dimuat; sistem memakai fallback lexical-semantic agar prototipe tetap berjalan.")

        return self._semantic_fallback(text), self._ai_probability(text), notes

    def _semantic_fallback(self, text: str) -> float:
        lowered = text.lower()
        hoax_hits = sum(1 for term in HOAX_TERMS if term in lowered)
        valid_hits = sum(1 for term in VALID_TERMS if term in lowered)
        url_count = len(re.findall(r"https?://|www\.", lowered))
        exclamation_count = lowered.count("!")
        all_caps_tokens = len(re.findall(r"\b[A-Z]{4,}\b", text))
        suspicious = hoax_hits + min(url_count, 2) * 0.5 + min(exclamation_count, 5) * 0.08 + min(all_caps_tokens, 5) * 0.08
        grounding = valid_hits * 0.55
        score = 0.35 + suspicious * 0.11 - grounding * 0.09
        return round(max(0.05, min(0.95, score)), 2)

    def _ai_probability(self, text: str) -> float:
        lowered = text.lower()
        hits = sum(1 for term in AI_TERMS if term in lowered)
        repeated_phrases = len(re.findall(r"\b(\w+\s+\w+)\b(?=.*\b\1\b)", lowered))
        score = 0.18 + hits * 0.18 + min(repeated_phrases, 4) * 0.04
        return round(max(0.03, min(0.92, score)), 2)


@lru_cache(maxsize=1)
def get_classifier() -> BertClassifier:
    return BertClassifier()


def probability_to_label(hoax_probability: float) -> ScoreItem:
    if hoax_probability >= 0.68:
        return ScoreItem(label="hoaks", score=hoax_probability)
    if hoax_probability <= 0.32:
        return ScoreItem(label="valid", score=1 - hoax_probability)
    return ScoreItem(label="perlu_verifikasi", score=1 - abs(0.5 - hoax_probability))
