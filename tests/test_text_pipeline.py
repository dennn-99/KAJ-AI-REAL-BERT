from backend.app.services.text_pipeline import analyze_text_content


def test_hoax_like_claim_is_flagged_for_verification_or_hoax():
    result = analyze_text_content("Sebarkan sekarang! Listrik dan ATM akan mati selama tujuh hari tanpa sumber resmi.")
    assert result.validity_label in {"hoaks", "perlu_verifikasi"}
    assert result.hoax_probability >= 0.45


def test_official_sounding_report_scores_lower_risk():
    result = analyze_text_content("Menurut rilis resmi BMKG, data cuaca dipublikasikan dengan metodologi dan sumber yang dapat diverifikasi.")
    assert result.hoax_probability <= 0.5
    assert result.sentiment.label in {"netral", "positif", "campuran"}
