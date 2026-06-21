const paths = {
  trainingMetrics: "/outputs/experiments/aaa_training_metrics.json",
};

const state = {
  liveBusy: false,
  liveDebounce: null,
};

const fmt = (value, digits = 3) => Number(value).toFixed(digits);
const pct = (value) => `${(Number(value) * 100).toFixed(1)}%`;

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Failed to load ${path}`);
  return response.json();
}

async function readJsonResponse(response) {
  const text = await response.text();
  try {
    return JSON.parse(text);
  } catch {
    const compact = text.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
    throw new Error(compact || `Server returned non-JSON response (${response.status}).`);
  }
}

const metricLabels = {
  mAP: "mAP",
  AP50: "AP50",
  AP75: "AP75",
  APS: "AP_s",
  APM: "AP_m",
  APL: "AP_l",
};

function metricCard(label, value, accent, isBest = false) {
  return `
    <article class="metric-card ${isBest ? "best-card" : ""}" style="border-top: 4px solid ${accent}">
      <span>${label}</span>
      <strong>${value}</strong>
    </article>
  `;
}

function renderBars(target, rows, color) {
  const element = document.querySelector(target);
  element.innerHTML = rows
    .map(
      ([label, value]) => `
        <div class="bar-row">
          <strong>${label}</strong>
          <div class="bar-track"><div class="bar-fill" style="width: ${pct(value)}; background: ${color}"></div></div>
          <span>${fmt(value)}</span>
        </div>
      `,
    )
    .join("");
}

function getBestEpoch(rows) {
  return rows.reduce((best, row) => (row.mAP > best.mAP ? row : best), rows[0]);
}

function getMaxima(rows, keys) {
  return Object.fromEntries(keys.map((key) => [key, Math.max(...rows.map((row) => Number(row[key])))]));
}

function renderInsights(data, best, latest, maxima) {
  const delta = latest.mAP - data.metrics[0].mAP;
  const checkpoint = data.best_checkpoint || "checkpoint not recorded";
  document.querySelector("#run-insights").innerHTML = `
    <div class="insight-list">
      <div>
        <span>Best checkpoint</span>
        <strong>${checkpoint}</strong>
      </div>
      <div>
        <span>mAP gain</span>
        <strong>${delta >= 0 ? "+" : ""}${fmt(delta)} from epoch 1 to epoch ${latest.epoch}</strong>
      </div>
      <div>
        <span>Peak small-object AP</span>
        <strong>${fmt(maxima.APS)}</strong>
      </div>
      <div>
        <span>Peak large-object AP</span>
        <strong>${fmt(maxima.APL)}</strong>
      </div>
    </div>
  `;
}

function renderLineChart(target, rows, series) {
  const width = 760;
  const height = 320;
  const pad = { left: 52, right: 24, top: 26, bottom: 42 };
  const values = series.flatMap((item) => rows.map((row) => row[item.key]));
  const min = Math.floor((Math.min(...values) - 0.01) * 100) / 100;
  const max = Math.ceil((Math.max(...values) + 0.01) * 100) / 100;
  const x = (index) => pad.left + (index * (width - pad.left - pad.right)) / (rows.length - 1);
  const y = (value) => pad.top + ((max - value) * (height - pad.top - pad.bottom)) / (max - min);
  const yTicks = [min, (min + max) / 2, max];

  const lines = series
    .map((item) => {
      const points = rows.map((row, index) => `${x(index)},${y(row[item.key])}`).join(" ");
      const circles = rows
        .map(
          (row, index) =>
            `<circle cx="${x(index)}" cy="${y(row[item.key])}" r="${row[item.key] === item.max ? 5 : 3.5}" fill="${item.color}" class="${row[item.key] === item.max ? "best-point" : ""}" />`,
        )
        .join("");
      return `<polyline points="${points}" fill="none" stroke="${item.color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />${circles}`;
    })
    .join("");

  document.querySelector(target).innerHTML = `
    <svg class="svg-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="AP trend line chart">
      ${yTicks
        .map(
          (tick) => `
            <line x1="${pad.left}" y1="${y(tick)}" x2="${width - pad.right}" y2="${y(tick)}" class="grid-line" />
            <text x="${pad.left - 12}" y="${y(tick) + 4}" text-anchor="end" class="axis-label">${fmt(tick, 2)}</text>
          `,
        )
        .join("")}
      <line x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}" class="axis-line" />
      ${rows
        .map(
          (row, index) =>
            `<text x="${x(index)}" y="${height - 14}" text-anchor="middle" class="axis-label">E${row.epoch}</text>`,
        )
        .join("")}
      ${lines}
    </svg>
    <div class="chart-legend">
      ${series.map((item) => `<span><i style="background:${item.color}"></i>${item.label}</span>`).join("")}
    </div>
  `;
}

function renderGroupedBarChart(target, rows, series) {
  const width = 760;
  const height = 320;
  const pad = { left: 52, right: 24, top: 26, bottom: 42 };
  const max = Math.ceil((Math.max(...series.flatMap((item) => rows.map((row) => row[item.key]))) + 0.02) * 10) / 10;
  const groupWidth = (width - pad.left - pad.right) / rows.length;
  const barWidth = Math.min(18, (groupWidth - 18) / series.length);
  const y = (value) => pad.top + ((max - value) * (height - pad.top - pad.bottom)) / max;
  const yTicks = [0, max / 2, max];

  const bars = rows
    .map((row, rowIndex) => {
      const groupX = pad.left + rowIndex * groupWidth;
      const epochLabel = `<text x="${groupX + groupWidth / 2}" y="${height - 14}" text-anchor="middle" class="axis-label">E${row.epoch}</text>`;
      const rects = series
        .map((item, seriesIndex) => {
          const x = groupX + 9 + seriesIndex * (barWidth + 4);
          const barY = y(row[item.key]);
          const barH = height - pad.bottom - barY;
          const bestClass = row[item.key] === item.max ? "best-bar" : "";
          return `<rect x="${x}" y="${barY}" width="${barWidth}" height="${barH}" rx="4" fill="${item.color}" class="${bestClass}" />`;
        })
        .join("");
      return `${rects}${epochLabel}`;
    })
    .join("");

  document.querySelector(target).innerHTML = `
    <svg class="svg-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Object-size AP grouped bar chart">
      ${yTicks
        .map(
          (tick) => `
            <line x1="${pad.left}" y1="${y(tick)}" x2="${width - pad.right}" y2="${y(tick)}" class="grid-line" />
            <text x="${pad.left - 12}" y="${y(tick) + 4}" text-anchor="end" class="axis-label">${fmt(tick, 2)}</text>
          `,
        )
        .join("")}
      <line x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}" class="axis-line" />
      ${bars}
    </svg>
    <div class="chart-legend">
      ${series.map((item) => `<span><i style="background:${item.color}"></i>${item.label}</span>`).join("")}
    </div>
  `;
}

function renderTrainingResults(data) {
  const rows = data.metrics;
  const best = getBestEpoch(rows);
  const latest = rows[rows.length - 1];
  const metricKeys = ["mAP", "AP50", "AP75", "APS", "APM", "APL"];
  const maxima = getMaxima(rows, metricKeys);

  document.querySelector("#dataset-chip").textContent = `${data.source} | ${rows.length} epochs | current run`;
  document.querySelector("#metric-grid").innerHTML = [
    metricCard("Best mAP", fmt(best.mAP), "var(--green)", true),
    metricCard("Best AP50", fmt(maxima.AP50), "var(--teal)", true),
    metricCard("Best AP75", fmt(maxima.AP75), "var(--blue)", true),
    metricCard("Best epoch", best.epoch, "var(--gold)", true),
    metricCard("Latest mAP", fmt(latest.mAP), "var(--coral)"),
    metricCard("Source", data.source, "var(--ink)"),
  ].join("");

  renderInsights(data, best, latest, maxima);

  renderLineChart("#trend-chart", rows, [
    { key: "mAP", label: "mAP", color: "var(--green)", max: maxima.mAP },
    { key: "AP50", label: "AP50", color: "var(--teal)", max: maxima.AP50 },
    { key: "AP75", label: "AP75", color: "var(--blue)", max: maxima.AP75 },
  ]);

  renderGroupedBarChart("#size-chart", rows, [
    { key: "APS", label: "AP_s", color: "var(--coral)", max: maxima.APS },
    { key: "APM", label: "AP_m", color: "var(--gold)", max: maxima.APM },
    { key: "APL", label: "AP_l", color: "var(--blue)", max: maxima.APL },
  ]);

  renderBars(
    "#map-bars",
    rows.map((row) => [`epoch ${row.epoch}`, row.mAP]),
    "var(--green)",
  );

  document.querySelector("#best-epoch-label").textContent = `epoch ${best.epoch}`;
  renderBars(
    "#best-bars",
    [
      ["mAP", best.mAP],
      ["AP50", best.AP50],
      ["AP75", best.AP75],
      ["AP_s", best.APS],
      ["AP_m", best.APM],
      ["AP_l", best.APL],
    ],
    "var(--blue)",
  );

  const headers = ["epoch", "mAP", "AP50", "AP75", "APS", "APM", "APL"];
  document.querySelector("#training-head").innerHTML = `
    <tr>${headers.map((header) => `<th>${header}</th>`).join("")}</tr>
  `;
  document.querySelector("#training-body").innerHTML = rows
    .map(
      (row) => `
        <tr>
          ${headers
            .map((header) => {
              const value = row[header];
              const isBest = header !== "epoch" && Number(value) === maxima[header];
              return `<td class="${isBest ? "best-cell" : ""}">${header === "epoch" ? value : fmt(value)}</td>`;
            })
            .join("")}
        </tr>
      `,
    )
    .join("");
}

function renderDetectionList(target, detections, emptyText) {
  document.querySelector(target).innerHTML =
    detections.length === 0
      ? `<div class="detection-item"><strong>${emptyText}</strong><span>Adjust prompt or thresholds</span></div>`
      : detections
          .map(
            (item) => `
              <div class="detection-item">
                <div>
                  <strong>${item.phrase}</strong>
                  <span>box ${item.box.map((v) => Math.round(v)).join(", ")}</span>
                </div>
                <strong>${fmt(item.score, 2)}</strong>
              </div>
            `,
          )
          .join("");
}

function resultToDetections(result, threshold = 0) {
  return result.phrases
    .map((phrase, index) => ({
      phrase,
      score: result.scores[index],
      box: result.boxes[index],
    }))
    .filter((item) => item.score >= threshold)
    .sort((a, b) => b.score - a.score);
}

async function refreshStatus() {
  try {
    const status = await fetchJson("/api/status");
    const deviceText = status.cuda ? status.device : "CPU";
    document.querySelector("#runtime-status").textContent = status.cuda ? "CUDA ready" : "CPU mode";
    document.querySelector("#runtime-detail").textContent =
      `${deviceText}${status.model_loaded ? " | model loaded" : " | model lazy-load"} | ${status.checkpoint_name}`;
    document.querySelector("#brand-device").textContent = `${deviceText} inference`;
  } catch {
    document.querySelector("#runtime-status").textContent = "Static mode";
    document.querySelector("#runtime-detail").textContent = "Start frontend/server.py for live inference";
    document.querySelector("#brand-device").textContent = "Local inference";
  }
}

function setLiveMessage(message, busy = false) {
  state.liveBusy = busy;
  document.querySelector("#live-message").textContent = message;
  document.querySelector("#run-live").disabled = busy;
  document.querySelector("#run-live").textContent = busy ? "Running..." : "Run detection";
}

async function runLiveInference() {
  if (state.liveBusy) return;
  const imageInput = document.querySelector("#live-image");
  const text = document.querySelector("#live-text").value.trim();
  if (!imageInput.files[0]) {
    setLiveMessage("Choose an image before running detection.");
    return;
  }
  if (!text) {
    setLiveMessage("Enter a text prompt before running detection.");
    return;
  }

  const form = new FormData();
  form.append("image", imageInput.files[0]);
  form.append("text", text);
  form.append("box_threshold", document.querySelector("#live-box").value);
  form.append("text_threshold", document.querySelector("#live-text-threshold").value);

  setLiveMessage("Running Grounding DINO on your GPU...", true);
  const started = performance.now();
  try {
    const response = await fetch("/api/infer", { method: "POST", body: form });
    const data = await readJsonResponse(response);
    if (!response.ok) throw new Error(data.error || "Inference failed.");

    const elapsed = ((performance.now() - started) / 1000).toFixed(2);
    document.querySelector("#live-preview").src = `${data.annotated_url}?t=${Date.now()}`;
    document.querySelector("#empty-stage").classList.add("hidden");
    document.querySelector("#live-result-title").textContent = "Live predictions";
    document.querySelector("#live-result-summary").textContent =
      `${data.num_detections} detections | ${data.inference_time}s model | ${elapsed}s total`;
    renderDetectionList("#live-detection-list", resultToDetections(data, 0), "No detections returned");
    setLiveMessage(`Done: ${data.num_detections} detections for "${data.text_prompt}".`);
    refreshStatus();
  } catch (error) {
    setLiveMessage(error.message);
  }
}

function scheduleAutoRun() {
  if (!document.querySelector("#auto-run").checked) return;
  window.clearTimeout(state.liveDebounce);
  state.liveDebounce = window.setTimeout(runLiveInference, 900);
}

function bindEvents() {
  document.querySelector("#live-form").addEventListener("submit", (event) => {
    event.preventDefault();
    runLiveInference();
  });

  document.querySelector("#live-image").addEventListener("change", (event) => {
    const file = event.target.files[0];
    document.querySelector("#file-label").textContent = file ? file.name : "Choose an image";
    if (file) {
      document.querySelector("#live-preview").src = URL.createObjectURL(file);
      document.querySelector("#empty-stage").classList.add("hidden");
      document.querySelector("#live-result-summary").textContent = "Preview ready";
      scheduleAutoRun();
    }
  });

  ["#live-text", "#live-box", "#live-text-threshold"].forEach((selector) => {
    document.querySelector(selector).addEventListener("input", () => {
      document.querySelector("#live-box-value").textContent = fmt(document.querySelector("#live-box").value, 2);
      document.querySelector("#live-text-value").textContent = fmt(document.querySelector("#live-text-threshold").value, 2);
      scheduleAutoRun();
    });
  });

  document.querySelector("#auto-run").addEventListener("change", scheduleAutoRun);

  document.querySelectorAll("[data-copy]").forEach((button) => {
    button.addEventListener("click", async () => {
      await navigator.clipboard.writeText(button.dataset.copy);
      const original = button.textContent;
      button.textContent = "Copied";
      setTimeout(() => {
        button.textContent = original;
      }, 1100);
    });
  });
}

async function init() {
  bindEvents();
  refreshStatus();

  const trainingMetrics = await fetchJson(paths.trainingMetrics);
  renderTrainingResults(trainingMetrics);
}

init().catch((error) => {
  console.error(error);
  document.querySelector("#dataset-chip").textContent = "Failed to load current training results";
});
