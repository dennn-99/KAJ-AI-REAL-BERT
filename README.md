# KAJ AI

KAJ AI adalah prototipe analisis media multimodal untuk menilai apakah konten berita, postingan media sosial, gambar, audio, atau video kemungkinan asli/AI-generated serta valid/hoaks. Backend memakai pendekatan BERT-first untuk klasifikasi teks, ditambah analisis sentimen, ekstraksi entitas, sinyal sumber, dan fitur multimodal ringan.

## Fitur

- Analisis teks, URL, gambar, audio, dan video.
- Klasifikasi validitas: `valid`, `perlu_verifikasi`, atau `hoaks`.
- Deteksi indikasi AI/generated media.
- Pipeline BERT multilingual dengan fallback lokal jika model belum tersedia.
- Analisis sentimen dan ekstraksi entitas untuk memperkaya konteks semantik.
- Menu pelatihan dataset untuk menambah contoh valid/hoaks/AI.
- Contoh pengujian isu viral dan terkini berbasis sumber cek fakta.
- REST API FastAPI dan dashboard web.

## Menjalankan

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```

Buka dashboard:

```text
https://kaj-ai-real.vercel.app/
```

## BERT Model

Secara default backend mencoba memuat model dari `KAJ_BERT_MODEL`, lalu fallback ke `bert-base-multilingual-cased`.

```powershell
$env:KAJ_BERT_MODEL="bert-base-multilingual-cased"
$env:KAJ_ALLOW_MODEL_DOWNLOAD="1"
```

Untuk produksi, fine-tune model sequence classification pada dataset internal KAJ AI berlabel `valid`, `hoaks`, `ai_generated`, dan `perlu_verifikasi`.

## Catatan

Analisis KAJ AI adalah sistem pendukung verifikasi, bukan pengganti cek fakta editorial. Hasil terbaik diperoleh dengan dataset lokal yang rutin dilatih dan pembanding dari sumber resmi.
