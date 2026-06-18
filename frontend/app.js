const paths = {
  metrics: "/outputs/coco_eval/my_full_val2017/metrics.json",
  resultLog: "/result.txt",
  thresholdCsv: "/outputs/experiments/threshold_sensitivity/summary.csv",
  promptCsv: "/outputs/experiments/prompt_comparison/summary.csv",
};

const samples = ["000000000139", "000000000285", "000000000632", "000000000776", "000000000872"];

const figures = [
  ["01_metrics_overview.png", "COCO metric overview"],
  ["02_size_gap.png", "Object-size performance gap"],
  ["03_threshold_sensitivity.png", "Threshold sensitivity"],
  ["04_prompt_format_comparison.png", "Prompt format comparison"],
  ["05_success_cases_grid.jpg", "Successful qualitative cases"],
  ["06_failure_cases_grid.jpg", "Failure qualitative cases"],
];

const state = {
  selectedSample: samples[0],
  threshold: 0.35,
  liveFile: null,
  liveBusy: false,
  liveDebounce: null,
  predictions: {},
  experiments: {
    threshold: [],
    prompt: [],
  },
};

const fmt = (value, digits = 3) => Number(value).toFixed(digits);
const pct = (value) => `${(Number(value) * 100).toFixed(1)}%`;

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Failed to load ${path}`);
  return response.json();
}

async function fetchText(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Failed to load ${path}`);
  return response.text();
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines.shift().split(",");
  return lines.map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
  });
}

function parseRunLog(text) {
  const readNumber = (pattern) => {
    const match = text.match(pattern);
    return match ? Number(match[1]) : null;
  };
  return {
    inferenceSeconds: readNumber(/Inference complete: 5000 images in ([\d.]+)s/),
    secondsPerImage: readNumber(/\(([\d.]+) s\/image\)/),
    totalSeconds: readNumber(/Total time: ([\d.]+)s/),
    uniquePhrases: readNumber(/Mapped (\d+) unique phrases/),
  };
}

function metricCard(label, value, accent, suffix = "") {
  return `
    <article class="metric-card" style="border-top: 4px solid ${accent}">
      <span>${label}</span>
      <strong>${value}${suffix}</strong>
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

function renderMetrics(data, runLog) {
  const metrics = data.metrics;
  const totalMinutes = runLog.totalSeconds ? `${fmt(runLog.totalSeconds / 60, 1)} min` : "48.5 min";
  document.querySelector("#dataset-chip").textContent =
    `${data.num_requested_images} images | ${data.num_images_with_predictions} with predictions`;
  document.querySelector("#metric-grid").innerHTML = [
    metricCard("AP", fmt(metrics.AP), "var(--green)"),
    metricCard("AP50", fmt(metrics.AP50), "var(--teal)"),
    metricCard("AP75", fmt(metrics.AP75), "var(--blue)"),
    metricCard("Predictions", data.num_predictions.toLocaleString(), "var(--coral)"),
    metricCard("Images hit", `${data.num_images_with_predictions}/${data.num_requested_images}`, "var(--gold)"),
    metricCard("Total time", totalMinutes, "var(--ink)"),
  ].join("");

  renderBars(
    "#size-bars",
    [
      ["Small", metrics.APS],
      ["Medium", metrics.APM],
      ["Large", metrics.APL],
    ],
    "var(--green)",
  );

  renderBars(
    "#recall-bars",
    [
      ["AR1", metrics.AR1],
      ["AR10", metrics.AR10],
      ["AR100", metrics.AR100],
    ],
    "var(--blue)",
  );
}

function sampleImagePath(id) {
  return `/outputs/inference_demo/${id}_annotated.jpg`;
}

function sampleJsonPath(id) {
  return `/outputs/inference_demo/${id}_predictions.json`;
}

function renderSampleStrip() {
  document.querySelector("#sample-strip").innerHTML = samples
    .map(
      (id) => `
        <button class="sample-button ${id === state.selectedSample ? "active" : ""}" data-sample="${id}" aria-label="Open sample ${id}">
          <img src="${sampleImagePath(id)}" alt="Sample ${id}" />
        </button>
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

function renderSelectedSample() {
  const id = state.selectedSample;
  const prediction = state.predictions[id];
  document.querySelector("#selected-image").src = sampleImagePath(id);
  document.querySelector("#selected-name").textContent = `COCO ${id}`;

  if (!prediction) return;

  const detections = resultToDetections(prediction, state.threshold);
  document.querySelector("#selected-summary").textContent =
    `${detections.length} of ${prediction.num_detections} detections above threshold`;
  renderDetectionList("#detection-list", detections, "No detections above threshold");
}

function renderExperiment(type) {
  const rows = state.experiments[type];
  const labelKey = type === "threshold" ? "run" : "format";
  const displayHeaders =
    type === "threshold"
      ? ["run", "box_threshold", "text_threshold", "AP", "AP50", "APL", "avg_boxes_per_image"]
      : ["run", "format", "AP", "AP50", "APL", "avg_boxes_per_image"];

  document.querySelector("#experiment-chart").innerHTML = rows
    .map(
      (row) => `
        <article class="experiment-card">
          <span>${row[labelKey]}</span>
          <strong>${fmt(row.AP)}</strong>
          <div class="bar-track"><div class="bar-fill" style="width: ${pct(row.AP)}; background: var(--gold)"></div></div>
        </article>
      `,
    )
    .join("");

  document.querySelector("#experiment-head").innerHTML = `
    <tr>${displayHeaders.map((header) => `<th>${header}</th>`).join("")}</tr>
  `;
  document.querySelector("#experiment-body").innerHTML = rows
    .map(
      (row) => `
        <tr>
          ${displayHeaders
            .map((header) => {
              const value = row[header];
              const text = Number.isFinite(Number(value)) && header !== "run" ? fmt(value) : value;
              return `<td>${text}</td>`;
            })
            .join("")}
        </tr>
      `,
    )
    .join("");
}

function renderFigures() {
  document.querySelector("#figure-grid").innerHTML = figures
    .map(
      ([file, title]) => `
        <article class="figure-card">
          <img src="/outputs/visualizations/report_highlights/${file}" alt="${title}" />
          <strong>${title}</strong>
        </article>
      `,
    )
    .join("");
}

async function refreshStatus() {
  try {
    const status = await fetchJson("/api/status");
    document.querySelector("#runtime-status").textContent = status.cuda ? "CUDA ready" : "CPU mode";
    document.querySelector("#runtime-detail").textContent =
      `${status.device}${status.model_loaded ? " | model loaded" : " | model lazy-load"}`;
  } catch {
    document.querySelector("#runtime-status").textContent = "Static mode";
    document.querySelector("#runtime-detail").textContent = "Start frontend/server.py for live inference";
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
    const data = await response.json();
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
  document.querySelector("#score-threshold").addEventListener("input", (event) => {
    state.threshold = Number(event.target.value);
    document.querySelector("#threshold-value").textContent = fmt(state.threshold, 2);
    renderSelectedSample();
  });

  document.querySelector("#sample-strip").addEventListener("click", (event) => {
    const button = event.target.closest("[data-sample]");
    if (!button) return;
    state.selectedSample = button.dataset.sample;
    renderSampleStrip();
    renderSelectedSample();
  });

  document.querySelector(".segmented").addEventListener("click", (event) => {
    const button = event.target.closest("[data-experiment]");
    if (!button) return;
    document.querySelectorAll("[data-experiment]").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    renderExperiment(button.dataset.experiment);
  });

  document.querySelector("#live-form").addEventListener("submit", (event) => {
    event.preventDefault();
    runLiveInference();
  });

  document.querySelector("#live-image").addEventListener("change", (event) => {
    const file = event.target.files[0];
    state.liveFile = file;
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
  renderSampleStrip();
  renderFigures();
  refreshStatus();

  const [metrics, resultLog, thresholdCsv, promptCsv, ...predictionData] = await Promise.all([
    fetchJson(paths.metrics),
    fetchText(paths.resultLog).catch(() => ""),
    fetchText(paths.thresholdCsv),
    fetchText(paths.promptCsv),
    ...samples.map((id) => fetchJson(sampleJsonPath(id))),
  ]);

  state.experiments.threshold = parseCsv(thresholdCsv);
  state.experiments.prompt = parseCsv(promptCsv);
  samples.forEach((id, index) => {
    state.predictions[id] = predictionData[index];
  });

  renderMetrics(metrics, parseRunLog(resultLog));
  renderSelectedSample();
  renderExperiment("threshold");
}

init().catch((error) => {
  console.error(error);
  document.querySelector("#dataset-chip").textContent = "Failed to load local outputs";
});
