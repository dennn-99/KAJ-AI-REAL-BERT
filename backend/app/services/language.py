try:
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 7
except Exception:
    detect = None

LANGUAGE_NAMES = {
    "id": "Bahasa Indonesia",
    "en": "English",
    "ms": "Bahasa Melayu",
    "jv": "Basa Jawa",
    "su": "Basa Sunda",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "ar": "العربية",
    "zh-cn": "中文",
}


def detect_language(text: str, requested_language: str = "auto") -> str:
    if requested_language and requested_language != "auto":
        return LANGUAGE_NAMES.get(requested_language, requested_language)
    if detect is None:
        return "Bahasa Indonesia"
    try:
        code = detect(text[:1000])
    except Exception:
        code = "id"
    return LANGUAGE_NAMES.get(code, code)
