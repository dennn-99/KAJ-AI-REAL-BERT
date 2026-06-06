import re
from collections import Counter

from ..models.schemas import Entity, ScoreItem

POSITIVE_WORDS = {
    "valid",
    "resmi",
    "terverifikasi",
    "akurat",
    "benar",
    "aman",
    "transparan",
    "confirmed",
    "verified",
}

NEGATIVE_WORDS = {
    "hoaks",
    "palsu",
    "penipuan",
    "bahaya",
    "panik",
    "fitnah",
    "menyesatkan",
    "provokasi",
    "scam",
    "fake",
}

ORG_HINTS = {
    "BMKG",
    "Kemenag",
    "Kemenkeu",
    "Komdigi",
    "Kementerian",
    "Polri",
    "PLN",
    "WHO",
    "Meta",
    "Instagram",
}


def sentiment_score(text: str) -> ScoreItem:
    lowered = text.lower()
    phrase_penalty = 1 if "tanpa sumber resmi" in lowered or "tidak ada sumber resmi" in lowered else 0
    tokens = re.findall(r"\w+", lowered)
    counts = Counter(tokens)
    positive = sum(counts[word] for word in POSITIVE_WORDS)
    negative = sum(counts[word] for word in NEGATIVE_WORDS) + phrase_penalty
    total = positive + negative
    if total == 0:
        return ScoreItem(label="netral", score=0.5)
    score = positive / total
    if score > 0.62:
        return ScoreItem(label="positif", score=round(score, 2))
    if score < 0.38:
        return ScoreItem(label="negatif", score=round(1 - score, 2))
    return ScoreItem(label="campuran", score=round(0.5 + abs(score - 0.5), 2))


def extract_entities(text: str) -> list[Entity]:
    entities: list[Entity] = []
    for org in ORG_HINTS:
        if re.search(rf"\b{re.escape(org)}\b", text, flags=re.IGNORECASE):
            entities.append(Entity(text=org, type="ORG"))

    date_patterns = [
        r"\b\d{1,2}\s+(Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)\s+\d{4}\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b20\d{2}\b",
    ]
    for pattern in date_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = match.group(0)
            if not any(entity.text == value for entity in entities):
                entities.append(Entity(text=value, type="DATE"))

    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", text):
        value = match.group(0)
        if value not in ORG_HINTS and not any(entity.text == value for entity in entities):
            entities.append(Entity(text=value, type="PERSON_OR_PLACE"))

    return entities[:16]
