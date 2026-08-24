const state = { manifest: null, sampleId: null, region: "mid-gland", opacity: 0.72 };
const assetVersion = "6";

const elements = {
  sampleSelect: document.querySelector("#sample-select"),
  regionControl: document.querySelector("#region-control"),
  opacitySlider: document.querySelector("#opacity-slider"),
  opacityValue: document.querySelector("#opacity-value"),
  viewerTitle: document.querySelector("#viewer-title"),
  loadingState: document.querySelector("#loading-state"),
};

function activeSample() {
  return state.manifest.samples.find((sample) => sample.id === state.sampleId);
}

function assetRoot() {
  return `results-browser/assets/samples/${state.sampleId}/${state.region}`;
}

function formatMetric(value, unit = "%") {
  return `${Number(value).toFixed(2)}${unit}`;
}

function renderMetrics(region) {
  ["miraus", "baseline"].forEach((method) => {
    const values = region[method];
    document.querySelector(`#${method}-dice`).textContent = formatMetric(values.dice);
    document.querySelector(`#${method}-iou`).textContent = formatMetric(values.iou);
    document.querySelector(`#${method}-hd95`).textContent = formatMetric(values.hd95, " mm");
  });
}

function loadImage(image, source) {
  return new Promise((resolve, reject) => {
    image.onload = resolve;
    image.onerror = reject;
    image.src = source;
  });
}

async function render() {
  const sample = activeSample();
  const region = sample.regions[state.region];
  const regionLabel = state.region === "mid-gland"
    ? "Mid-gland"
    : state.region[0].toUpperCase() + state.region.slice(1);
  const root = assetRoot();

  elements.viewerTitle.textContent = `${sample.label} · ${regionLabel}`;
  elements.loadingState.textContent = "Loading results";
  elements.loadingState.classList.remove("hidden");

  const loads = [];
  document.querySelectorAll(".base-image").forEach((image) => {
    loads.push(loadImage(image, `${root}/original.webp?v=${assetVersion}`));
  });
  document.querySelectorAll(".overlay-image").forEach((image) => {
    image.style.opacity = state.opacity;
    loads.push(loadImage(image, `${root}/${image.dataset.overlay}?v=${assetVersion}`));
  });

  try {
    await Promise.all(loads);
    elements.loadingState.classList.add("hidden");
  } catch (error) {
    elements.loadingState.textContent = "Result unavailable";
    console.error(error);
  }
  renderMetrics(region);
}

function bindControls() {
  elements.sampleSelect.addEventListener("change", (event) => {
    state.sampleId = event.target.value;
    render();
  });

  elements.regionControl.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-region]");
    if (!button) return;
    state.region = button.dataset.region;
    elements.regionControl.querySelectorAll("button").forEach((candidate) => {
      candidate.classList.toggle("active", candidate === button);
    });
    render();
  });

  elements.opacitySlider.addEventListener("input", (event) => {
    state.opacity = Number(event.target.value) / 100;
    elements.opacityValue.textContent = `${event.target.value}%`;
    document.querySelectorAll(".overlay-image").forEach((image) => {
      image.style.opacity = state.opacity;
    });
  });
}

async function initialize() {
  try {
    const response = await fetch(`results-browser/samples.json?v=${assetVersion}`);
    if (!response.ok) throw new Error(`Manifest request failed: ${response.status}`);
    state.manifest = await response.json();
    state.sampleId = state.manifest.samples[0].id;
    state.manifest.samples.forEach((sample) => {
      const option = document.createElement("option");
      option.value = sample.id;
      option.textContent = sample.label;
      elements.sampleSelect.append(option);
    });
    elements.sampleSelect.value = state.sampleId;
    bindControls();
    render();
  } catch (error) {
    elements.loadingState.textContent = "Unable to load result manifest";
    console.error(error);
  }
}

initialize();
