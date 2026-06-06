const state = {
  mode: "text",
};

const fallbackViralTests = [
  {
    claim: "Hantavirus 2026 sudah diprediksi oleh akun X sejak 2022.",
    expected_label: "hoaks",
    source: "Global Fact-Check Database / TurnBackHoax, 3 Juni 2026",
  },
  {
    claim: "Listrik dan ATM di Indonesia akan mati selama tujuh hari.",
    expected_label: "hoaks",
    source: "Komdigi, 22 Januari 2026",
  },
  {
    claim: "BMKG memprediksi gempa megathrust akan terjadi pada 2026.",
    expected_label: "hoaks",
    source: "Global Fact-Check Database / TurnBackHoax, 1 Maret 2026",
  },
];

document.querySelectorAll(".nav-button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".nav-button").forEach((item) => item.classList.remove("active"));
    document.querySelectorAll(".view").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    document.getElementById(button.dataset.view).classList.add("active");
    if (button.dataset.view === "training") loadSamples();
  });
});

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    state.mode = button.dataset.mode;
    document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    document.getElementById("text-input").classList.toggle("hidden", state.mode !== "text");
    document.getElementById("url-input").classList.toggle("hidden", state.mode !== "url");
    document.getElementById("media-inputs").classList.toggle("hidden", state.mode !== "media");
  });
});

document.getElementById("analysis-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const result = document.getElementById("result");
  result.className = "result";
  result.innerHTML = "<h2>Hasil Analisis</h2><p>Menganalisis konten...</p>";

  try {
    const language = document.getElementById("language").value;
    const form = new FormData();
    form.append("language", language);

    let endpoint = "/api/analyze/text";
    if (state.mode === "text") {
      form.append("text", document.getElementById("text-input").value);
    } else if (state.mode === "url") {
      endpoint = "/api/analyze/url";
      form.append("url", document.getElementById("url-input").value);
    } else {
      endpoint = "/api/analyze/media";
      const file = document.getElementById("file-input").files[0];
      if (!file) throw new Error("Pilih file media terlebih dahulu.");
      form.append("file", file);
      form.append("caption", document.getElementById("caption-input").value);
    }

    const response = await fetch(endpoint, { method: "POST", body: form });
    if (!response.ok) throw new Error(await response.text());
    renderResult(await response.json());
  } catch (error) {
    result.innerHTML = `<h2>Hasil Analisis</h2><p>${escapeHtml(error.message)}</p>`;
  }
});

document.getElementById("training-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = {
    text: document.getElementById("train-text").value,
    validity_label: document.getElementById("train-validity").value,
    origin_label: document.getElementById("train-origin").value,
    source: document.getElementById("train-source").value || "manual",
    language: document.getElementById("language").value === "auto" ? "id" : document.getElementById("language").value,
  };
  const response = await fetch("/api/training/samples", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (response.ok) {
    document.getElementById("training-form").reset();
    loadSamples();
  }
});

document.getElementById("train-model").addEventListener("click", async () => {
  const button = document.getElementById("train-model");
  button.textContent = "Melatih...";
  const response = await fetch("/api/training/train", { method: "POST" });
  const data = await response.json();
  button.textContent = data.status === "trained" ? `Terlatih (${data.sample_count} sampel)` : "Latih Model Lokal";
});

function renderResult(data) {
  const result = document.getElementById("result");
  result.className = "result";
  result.innerHTML = `
    <h2>Hasil Analisis</h2>
    <p>
      <span class="badge ${data.validity_label}">${labelText(data.validity_label)}</span>
      <span class="badge ${data.origin_label}">${labelText(data.origin_label)}</span>
    </p>
    <p>${escapeHtml(data.explanation)}</p>
    <div class="score-grid">
      ${metric("Hoaks", percent(data.hoax_probability))}
      ${metric("AI", percent(data.ai_probability))}
      ${metric("Confidence", percent(data.confidence))}
    </div>
    <div class="item">
      <h3>Konteks Semantik</h3>
      <p>Bahasa: ${escapeHtml(data.language)}. Sentimen: ${escapeHtml(data.sentiment.label)} (${percent(data.sentiment.score)}).</p>
      <div class="entities">${data.entities.map((entity) => `<span class="badge">${escapeHtml(entity.text)} · ${escapeHtml(entity.type)}</span>`).join("") || "<span class=\"muted\">Tidak ada entitas kuat.</span>"}</div>
    </div>
    <h3>Bukti</h3>
    <div class="evidence-list">${data.evidence.map((item) => `<div class="item"><h3>${escapeHtml(item.title)}</h3><p>${escapeHtml(item.detail)}</p></div>`).join("")}</div>
    <h3>Rekomendasi</h3>
    <div class="evidence-list">${data.recommendations.map((item) => `<div class="item">${escapeHtml(item)}</div>`).join("")}</div>
    <h3>Catatan Model</h3>
    <div class="evidence-list">${data.model_notes.map((item) => `<div class="item">${escapeHtml(item)}</div>`).join("")}</div>
  `;
}

function metric(label, value) {
  return `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;
}

async function loadSamples() {
  const container = document.getElementById("samples");
  container.innerHTML = "<p>Memuat...</p>";
  const response = await fetch("/api/training/samples");
  const data = await response.json();
  container.innerHTML = data.samples
    .slice()
    .reverse()
    .map((sample) => `<div class="item"><h3>${labelText(sample.validity_label)} · ${labelText(sample.origin_label)}</h3><p>${escapeHtml(sample.text)}</p><small>${escapeHtml(sample.source)}</small></div>`)
    .join("");
}

function renderViralTests() {
  const container = document.getElementById("viral-tests");
  fetch("/api/viral-tests")
    .then((response) => response.ok ? response.json() : { tests: fallbackViralTests })
    .then((data) => renderViralList(data.tests || fallbackViralTests))
    .catch(() => renderViralList(fallbackViralTests));
}

function renderViralList(tests) {
  const container = document.getElementById("viral-tests");
  container.innerHTML = tests
    .map((test) => `<div class="item"><h3>${escapeHtml(test.claim)}</h3><p><span class="badge hoaks">${labelText(test.expected_label)}</span></p><p class="muted">${escapeHtml(test.source)}${test.published ? ` · ${escapeHtml(test.published)}` : ""}</p><button data-claim="${escapeHtml(test.claim)}">Uji Klaim</button></div>`)
    .join("");
  container.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelector('[data-view="analysis"]').click();
      document.querySelector('[data-mode="text"]').click();
      document.getElementById("text-input").value = button.dataset.claim;
      document.getElementById("analysis-form").requestSubmit();
    });
  });
}

function labelText(value) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function percent(value) {
  return `${Math.round(value * 100)}%`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[char]));
}

renderViralTests();
