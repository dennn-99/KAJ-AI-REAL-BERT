import re

import requests

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

from ..models.schemas import AnalysisResponse, EvidenceItem
from .bert_classifier import get_classifier, probability_to_label
from .language import detect_language
from .nlp_features import extract_entities, sentiment_score

TRUSTED_DOMAINS = {
    "bmkg.go.id",
    "komdigi.go.id",
    "kemenag.go.id",
    "kemenkeu.go.id",
    "pln.co.id",
    "who.int",
    "turnbackhoax.id",
    "cekfakta.tempo.co",
    "cekfakta.com",
}

VIRAL_TEST_CASES = [
    {
        "title": "Klaim Hantavirus 2026 sudah diprediksi sejak 2022",
        "label": "hoaks",
        "source": "Global Fact-Check Database / TurnBackHoax, dipublikasikan 3 Juni 2026",
    },
    {
        "title": "Klaim listrik dan ATM mati selama 7 hari",
        "label": "hoaks",
        "source": "Komdigi, klarifikasi 22 Januari 2026",
    },
    {
        "title": "Klaim BMKG memprediksi gempa megathrust pada 2026",
        "label": "hoaks",
        "source": "Global Fact-Check Database / TurnBackHoax, dipublikasikan 1 Maret 2026",
    },
]


def analyze_text_content(text: str, requested_language: str = "auto", media_type: str = "text") -> AnalysisResponse:
    normalized = " ".join(text.split())
    classifier = get_classifier()
    hoax_probability, ai_probability, model_notes = classifier.classify(normalized)
    validity = probability_to_label(hoax_probability)
    language = detect_language(normalized, requested_language)
    sentiment = sentiment_score(normalized)
    entities = extract_entities(normalized)
    evidence = _build_evidence(normalized, hoax_probability, ai_probability)
    confidence = round((validity.score * 0.7) + ((1 - abs(ai_probability - 0.5)) * 0.3), 2)
    origin_label = _origin_label(ai_probability)
    explanation = _explain(validity.label, hoax_probability, ai_probability, evidence)
    recommendations = _recommendations(validity.label, origin_label)

    return AnalysisResponse(
        media_type=media_type,
        language=language,
        validity_label=validity.label,
        origin_label=origin_label,
        confidence=confidence,
        hoax_probability=hoax_probability,
        ai_probability=ai_probability,
        sentiment=sentiment,
        entities=entities,
        explanation=explanation,
        evidence=evidence,
        recommendations=recommendations,
        model_notes=model_notes,
    )


def analyze_url_content(url: str, requested_language: str = "auto") -> AnalysisResponse:
    response = requests.get(url, timeout=8, headers={"User-Agent": "KAJ-AI/1.0"})
    response.raise_for_status()
    if BeautifulSoup is not None:
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title and soup.title.string else url
        paragraphs = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p")[:18])
    else:
        title_match = re.search(r"<title[^>]*>(.*?)</title>", response.text, flags=re.IGNORECASE | re.DOTALL)
        title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else url
        paragraphs = " ".join(re.findall(r"<p[^>]*>(.*?)</p>", response.text, flags=re.IGNORECASE | re.DOTALL)[:18])
        paragraphs = re.sub(r"<[^>]+>", " ", paragraphs)
    text = f"{title}. {paragraphs}".strip()
    result = analyze_text_content(text=text[:8000], requested_language=requested_language, media_type="url")
    domain = re.sub(r"^https?://(www\.)?", "", url).split("/")[0].lower()
    if any(domain.endswith(trusted) for trusted in TRUSTED_DOMAINS):
        result.evidence.append(EvidenceItem(title="Sumber tepercaya", detail=f"Domain {domain} ada dalam daftar sumber rujukan.", weight=0.45))
        result.hoax_probability = max(0.04, round(result.hoax_probability - 0.18, 2))
        if result.hoax_probability <= 0.32:
            result.validity_label = "valid"
    return result


def _build_evidence(text: str, hoax_probability: float, ai_probability: float) -> list[EvidenceItem]:
    lowered = text.lower()
    evidence: list[EvidenceItem] = []
    if any(term in lowered for term in ["tanpa sumber", "sebarkan", "jangan percaya media", "klik link"]):
        evidence.append(EvidenceItem(title="Pola narasi mencurigakan", detail="Teks memakai ajakan sebar/klik atau melemahkan sumber pembanding.", weight=-0.45))
    if any(term in lowered for term in ["menurut", "data", "konfirmasi", "klarifikasi", "rilis resmi"]):
        evidence.append(EvidenceItem(title="Ada sinyal rujukan", detail="Teks memuat istilah yang biasanya menyertai rujukan atau klarifikasi.", weight=0.35))
    if ai_probability > 0.55:
        evidence.append(EvidenceItem(title="Indikasi konten AI", detail="Narasi memuat sinyal generatif seperti deepfake, suara sintetis, atau manipulasi AI.", weight=-0.35))
    if hoax_probability > 0.68:
        evidence.append(EvidenceItem(title="Risiko hoaks tinggi", detail="Kombinasi klaim sensasional dan minim konteks menaikkan risiko misinformasi.", weight=-0.55))
    if not evidence:
        evidence.append(EvidenceItem(title="Konteks terbatas", detail="Tidak ada sinyal kuat; sistem menyarankan verifikasi silang dengan sumber resmi.", weight=0))
    return evidence


def _origin_label(ai_probability: float) -> str:
    if ai_probability >= 0.72:
        return "ai_generated"
    if ai_probability >= 0.52:
        return "campuran"
    if ai_probability <= 0.28:
        return "asli"
    return "tidak_pasti"


def _explain(label: str, hoax_probability: float, ai_probability: float, evidence: list[EvidenceItem]) -> str:
    strongest = max(evidence, key=lambda item: abs(item.weight))
    return (
        f"KAJ AI mengklasifikasikan konten sebagai {label} dengan probabilitas hoaks "
        f"{hoax_probability:.0%} dan probabilitas AI {ai_probability:.0%}. "
        f"Faktor utama: {strongest.title.lower()}."
    )


def _recommendations(validity_label: str, origin_label: str) -> list[str]:
    recommendations = [
        "Bandingkan klaim dengan kanal resmi pemerintah, media kredibel, atau database cek fakta.",
        "Periksa tanggal, konteks lokasi, dan apakah ada kutipan sumber primer.",
    ]
    if validity_label != "valid":
        recommendations.insert(0, "Jangan sebarkan konten sebelum ada pembanding dari sumber resmi.")
    if origin_label in {"ai_generated", "campuran", "tidak_pasti"}:
        recommendations.append("Untuk media visual/audio, lakukan reverse image search dan analisis metadata bila tersedia.")
    return recommendations
