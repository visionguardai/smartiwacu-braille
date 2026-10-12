// The model is deliberately stored as small files so it can be uploaded to
// hosts that reject individual files larger than 40 MB.  It is reassembled in
// memory before ONNX Runtime sees it, so inference remains exactly the same.
const MODEL_PARTS = [
  "model.onnx.part001",
  "model.onnx.part002",
  "model.onnx.part003",
  "model.onnx.part004",
  "model.onnx.part005",
  "model.onnx.part006",
  "model.onnx.part007",
  "model.onnx.part008"
];
const EXPECTED_INPUT = "input";
const EXPECTED_OUTPUTS = ["loc_pred", "cls_pred"];
// AngelinaReader's production inference pipeline is calibrated at width 1024.
// Source images may have any dimensions; they are resized proportionally and
// only the resulting tensor dimensions are rounded to the network's 32px stride.
const MODEL_INPUT_WIDTH = 1024;
const CLASS_THRESHOLD = 0.3;
const NMS_THRESHOLD = 0.02;
const ANCHOR_AREAS = [128.0, 288.0, 512.0];
const ASPECT_RATIOS = [0.5];
const SCALE_RATIOS = [1.0, Math.pow(2, 1 / 3), Math.pow(2, 2 / 3)];

const EN_MAP = {
  "1": "a", "12": "b", "14": "c", "145": "d", "15": "e",
  "124": "f", "1245": "g", "125": "h", "24": "i", "245": "j",
  "13": "k", "123": "l", "134": "m", "1345": "n", "135": "o",
  "1234": "p", "12345": "q", "1235": "r", "234": "s", "2345": "t",
  "136": "u", "1236": "v", "2456": "w", "1346": "x",
  "13456": "y", "1356": "z", "2": ",", "23": ";", "25": ":",
  "256": ".", "26": "?", "235": "!", "36": "-", "3": "'"
};

const elements = {
  status: document.getElementById("runtimeStatus"),
  file: document.getElementById("fileInput"),
  run: document.getElementById("runButton"),
  canvas: document.getElementById("paperCanvas"),
  translated: document.getElementById("translatedText"),
  braille: document.getElementById("brailleText"),
  info: document.getElementById("modelInfo"),
  language: document.getElementById("languageSelect"),
  enhance: document.getElementById("enhanceImage"),
  useRules: document.getElementById("useSavedRules"),
  showBoxes: document.getElementById("showBoxes"),
  boxColor: document.getElementById("boxColor"),
  showLabels: document.getElementById("showLabels"),
  labelPlacement: document.getElementById("labelPlacement"),
  labelColor: document.getElementById("labelColor"),
  labelSize: document.getElementById("labelSize"),
  folder: document.getElementById("folderInput"),
  runBatch: document.getElementById("runBatch"),
  openCamera: document.getElementById("openCamera"),
  penColor: document.getElementById("penColor"),
  penSize: document.getElementById("penSize"),
  undo: document.getElementById("undoMark"),
  redo: document.getElementById("redoMark"),
  clear: document.getElementById("clearMarks"),
  rotateLeft: document.getElementById("rotateLeft"),
  rotateRight: document.getElementById("rotateRight"),
  rotateAngle: document.getElementById("rotateAngle"),
  angle: document.getElementById("angleInput"),
  save: document.getElementById("saveImage")
};

const ctx = elements.canvas.getContext("2d", { willReadFrequently: true });
let sessionPromise = null;
let sourceImage = null;
let undoStack = [];
let redoStack = [];
let drawing = false;
let lastPoint = null;
let selectedFiles = [];
let currentFileName = "";
let cameraStream = null;
let canvasZoom = 1;
let resultCollection = [];

const extra = {
  translateEdited: document.getElementById("translateEdited"),
  copyText: document.getElementById("copyText"),
  saveText: document.getElementById("saveText"),
  saveBraille: document.getElementById("saveBraille"),
  zoomIn: document.getElementById("zoomIn"),
  zoomOut: document.getElementById("zoomOut"),
  noteText: document.getElementById("noteText"),
  noteBold: document.getElementById("noteBold"),
  noteItalic: document.getElementById("noteItalic"),
  addNote: document.getElementById("addNote"),
  capitalPrefixes: document.getElementById("capitalPrefixes"),
  numberPrefixes: document.getElementById("numberPrefixes"),
  singleRules: document.getElementById("singleRules"),
  sequenceRules: document.getElementById("sequenceRules"),
  saveRules: document.getElementById("saveRules"),
  resetRules: document.getElementById("resetRules"),
  exportRules: document.getElementById("exportRules"),
  importRules: document.getElementById("importRules"),
  historyList: document.getElementById("historyList"),
  clearHistory: document.getElementById("clearHistory"),
  batchStatus: document.getElementById("batchStatus"),
  savePreferences: document.getElementById("savePreferences"),
  resetPreferences: document.getElementById("resetPreferences"),
  cameraDialog: document.getElementById("cameraDialog"),
  cameraVideo: document.getElementById("cameraVideo"),
  captureCamera: document.getElementById("captureCamera"),
  captureAndRun: document.getElementById("captureAndRun"),
  closeCamera: document.getElementById("closeCamera")
};
extra.useEditedPaper = document.getElementById("useEditedPaper");
extra.downloadFormat = document.getElementById("downloadFormat");
extra.downloadAll = document.getElementById("downloadAll");
extra.shareResults = document.getElementById("shareResults");
extra.clearCollection = document.getElementById("clearCollection");
extra.paperGallery = document.getElementById("paperGallery");

function setStatus(text) {
  elements.status.textContent = text;
}

function dotsFromLabel(label) {
  let dots = "";
  for (let i = 0; i < 6; i += 1) {
    if (label & (1 << i)) dots += String(i + 1);
  }
  return dots;
}

function brailleFromLabel(label) {
  return String.fromCodePoint(0x2800 + label);
}

function textFromLabel(label) {
  const dots = dotsFromLabel(label);
  return EN_MAP[dots] || `~${dots}~`;
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function computeTargetSize(img) {
  if (!img || !Number.isFinite(img.width) || !Number.isFinite(img.height) || img.width < 1 || img.height < 1) {
    throw new Error("The selected image has invalid dimensions.");
  }
  const targetWidth = Math.ceil(MODEL_INPUT_WIDTH / 32) * 32;
  const scaled = img.height * targetWidth / img.width;
  const targetHeight = Math.max(32, Math.ceil(scaled / 32) * 32);
  return { width: targetWidth, height: targetHeight };
}

function pushUndo() {
  undoStack.push(ctx.getImageData(0, 0, elements.canvas.width, elements.canvas.height));
  if (undoStack.length > 30) undoStack.shift();
  redoStack = [];
}

function restoreImageData(data) {
  elements.canvas.width = data.width;
  elements.canvas.height = data.height;
  ctx.putImageData(data, 0, 0);
}

function drawSourceToCanvas(img) {
  const maxWidth = Math.max(640, Math.min(1400, img.width));
  const scale = maxWidth / img.width;
  elements.canvas.width = Math.round(img.width * scale);
  elements.canvas.height = Math.round(img.height * scale);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, elements.canvas.width, elements.canvas.height);
  ctx.drawImage(img, 0, 0, elements.canvas.width, elements.canvas.height);
}

async function readImageFromFile(file) {
  const bitmap = await createImageBitmap(file);
  const img = new Image();
  img.src = URL.createObjectURL(file);
  await img.decode();
  bitmap.close?.();
  return img;
}

async function getSession() {
  if (window.location.protocol === "file:") {
    throw new Error(
      "This ONNX app cannot run from a file:/// address. Close this page and open START_APP.cmd; it will launch the app at http://127.0.0.1:4174."
    );
  }
  if (!sessionPromise) {
    // ONNX Runtime 1.27 expects an object with `mjs` and `wasm` URLs, not a
    // filename-to-URL map. Both files used by the CPU/WASM backend are below
    // 20 MB; the optional large JSEP/asyncify runtimes remain split for upload.
    ort.env.wasm.wasmPaths = {
      mjs: new URL("ort-wasm-simd-threaded.mjs", document.baseURI).href,
      wasm: new URL("ort-wasm-simd-threaded.wasm", document.baseURI).href
    };
    // A single WASM thread works on ordinary static hosting without requiring
    // cross-origin isolation headers. There is intentionally no PyTorch or
    // checkpoint fallback in this browser application.
    ort.env.wasm.numThreads = 1;
    sessionPromise = loadModelBytes().then((modelBytes) => ort.InferenceSession.create(modelBytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all"
    })).then((session) => {
      if (!session.inputNames.includes(EXPECTED_INPUT)) {
        throw new Error(
          `Wrong ONNX model: expected input "${EXPECTED_INPUT}", found ${session.inputNames.join(", ")}`
        );
      }
      for (const outputName of EXPECTED_OUTPUTS) {
        if (!session.outputNames.includes(outputName)) {
          throw new Error(
            `Wrong ONNX model: expected output "${outputName}", found ${session.outputNames.join(", ")}`
          );
        }
      }
      return session;
    }).catch((error) => {
      sessionPromise = null;
      throw error;
    });
  }
  return sessionPromise;
}

async function loadModelBytes() {
  return loadBinaryParts(MODEL_PARTS, "model");
}

async function loadBinaryParts(urls, label) {
  const responses = await Promise.all(urls.map(async (url, index) => {
    setStatus(`Loading ${label} part ${index + 1} of ${urls.length}…`);
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Could not download ${url} (${response.status}).`);
    return new Uint8Array(await response.arrayBuffer());
  }));
  const totalLength = responses.reduce((sum, part) => sum + part.length, 0);
  const model = new Uint8Array(totalLength);
  let offset = 0;
  for (const part of responses) {
    model.set(part, offset);
    offset += part.length;
  }
  return model;
}

function preprocess(img) {
  const target = computeTargetSize(img);
  const off = document.createElement("canvas");
  off.width = target.width;
  off.height = target.height;
  const offCtx = off.getContext("2d", { willReadFrequently: true });
  if (elements.enhance.checked) offCtx.filter = "contrast(1.2) grayscale(1)";
  offCtx.fillStyle = "#ffffff";
  offCtx.fillRect(0, 0, off.width, off.height);
  offCtx.drawImage(img, 0, 0, off.width, off.height);
  const rgba = offCtx.getImageData(0, 0, off.width, off.height).data;
  const channels = [new Float32Array(off.width * off.height), new Float32Array(off.width * off.height), new Float32Array(off.width * off.height)];
  const mean = [0, 0, 0];
  for (let i = 0, p = 0; i < rgba.length; i += 4, p += 1) {
    channels[0][p] = rgba[i];
    channels[1][p] = rgba[i + 1];
    channels[2][p] = rgba[i + 2];
    mean[0] += rgba[i];
    mean[1] += rgba[i + 1];
    mean[2] += rgba[i + 2];
  }
  const n = off.width * off.height;
  for (let c = 0; c < 3; c += 1) mean[c] /= n;
  const std = [0, 0, 0];
  for (let c = 0; c < 3; c += 1) {
    for (let p = 0; p < n; p += 1) {
      const d = channels[c][p] - mean[c];
      std[c] += d * d;
    }
    std[c] = Math.max(Math.sqrt(std[c] / Math.max(1, n - 1)), 25.5);
  }
  const tensor = new Float32Array(3 * n);
  for (let p = 0; p < n; p += 1) {
    const r = (channels[0][p] - mean[0]) / (3 * std[0]);
    const g = (channels[1][p] - mean[1]) / (3 * std[1]);
    const b = (channels[2][p] - mean[2]) / (3 * std[2]);
    const gray = (r + g + b) / 3;
    tensor[p] = gray;
    tensor[n + p] = gray;
    tensor[2 * n + p] = gray;
  }
  return { tensor: new ort.Tensor("float32", tensor, [1, 3, off.height, off.width]), width: off.width, height: off.height };
}

function downloadBlob(name, contents, type = "text/plain;charset=utf-8") {
  const url = URL.createObjectURL(new Blob([contents], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) c = (c & 1) ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) crc = CRC_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

function zipHeader(size) {
  const bytes = new Uint8Array(size);
  return { bytes, view: new DataView(bytes.buffer) };
}

function concatBytes(parts) {
  const output = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
  let offset = 0;
  for (const part of parts) { output.set(part, offset); offset += part.length; }
  return output;
}

function createZip(entries) {
  const encoder = new TextEncoder();
  const locals = [];
  const centrals = [];
  let offset = 0;
  for (const entry of entries) {
    const name = encoder.encode(entry.name);
    const data = entry.data instanceof Uint8Array ? entry.data : encoder.encode(entry.data);
    const crc = crc32(data);
    const local = zipHeader(30);
    local.view.setUint32(0, 0x04034b50, true);
    local.view.setUint16(4, 20, true);
    local.view.setUint16(6, 0x0800, true);
    local.view.setUint16(8, 0, true);
    local.view.setUint32(14, crc, true);
    local.view.setUint32(18, data.length, true);
    local.view.setUint32(22, data.length, true);
    local.view.setUint16(26, name.length, true);
    locals.push(local.bytes, name, data);

    const central = zipHeader(46);
    central.view.setUint32(0, 0x02014b50, true);
    central.view.setUint16(4, 20, true);
    central.view.setUint16(6, 20, true);
    central.view.setUint16(8, 0x0800, true);
    central.view.setUint32(16, crc, true);
    central.view.setUint32(20, data.length, true);
    central.view.setUint32(24, data.length, true);
    central.view.setUint16(28, name.length, true);
    central.view.setUint32(42, offset, true);
    centrals.push(central.bytes, name);
    offset += local.bytes.length + name.length + data.length;
  }
  const centralBytes = concatBytes(centrals);
  const end = zipHeader(22);
  end.view.setUint32(0, 0x06054b50, true);
  end.view.setUint16(8, entries.length, true);
  end.view.setUint16(10, entries.length, true);
  end.view.setUint32(12, centralBytes.length, true);
  end.view.setUint32(16, offset, true);
  return new Blob([...locals, centralBytes, end.bytes], { type: "application/zip" });
}

function dataUrlBytes(dataUrl) {
  const binary = atob(dataUrl.split(",")[1]);
  return Uint8Array.from(binary, (char) => char.charCodeAt(0));
}

function collectionArchive() {
  const format = extra.downloadFormat.value;
  const entries = [];
  for (const item of resultCollection) {
    const stem = safeStem(item.name);
    if (format === "combined") {
      entries.push({ name: `${stem}.braille-and-translation.txt`, data: `BRAILLE\n${item.braille}\n\nTRANSLATION\n${item.translated}\n` });
    } else if (format === "text") {
      entries.push({ name: `${stem}.translated.txt`, data: `${item.translated}\n` });
    } else {
      entries.push(
        { name: `${stem}.marked.jpg`, data: dataUrlBytes(item.image) },
        { name: `${stem}.marked.brl`, data: `${item.braille}\n` },
        { name: `${stem}.marked.txt`, data: `${item.translated}\n` }
      );
    }
  }
  return createZip(entries);
}

function normalizeDots(value) {
  return [...new Set(String(value).match(/[1-6]/g) || [])].sort().join("");
}

function defaultRules() {
  return {
    capitalPrefixes: ["6", "46"],
    numberPrefixes: ["3456"],
    singleCells: { ...EN_MAP },
    sequences: []
  };
}

function rulesFromForm() {
  const parseMap = (text) => Object.fromEntries(
    text.split(/\r?\n/).map((line) => line.split("=")).filter((parts) => parts.length >= 2)
      .map((parts) => [parts[0].trim(), parts.slice(1).join("=").trim()])
  );
  const singles = {};
  for (const [dots, value] of Object.entries(parseMap(extra.singleRules.value))) {
    const normalized = normalizeDots(dots);
    if (normalized) singles[normalized] = value;
  }
  const sequences = [];
  for (const [sequence, value] of Object.entries(parseMap(extra.sequenceRules.value))) {
    const cells = sequence.split(/\s+/).map(normalizeDots).filter(Boolean);
    if (cells.length) sequences.push({ sequence: cells, text: value });
  }
  sequences.sort((a, b) => b.sequence.length - a.sequence.length);
  return {
    capitalPrefixes: extra.capitalPrefixes.value.split(/[\s,]+/).map(normalizeDots).filter(Boolean),
    numberPrefixes: extra.numberPrefixes.value.split(/[\s,]+/).map(normalizeDots).filter(Boolean),
    singleCells: singles,
    sequences
  };
}

function fillRulesForm(rules) {
  extra.capitalPrefixes.value = (rules.capitalPrefixes || []).join(", ");
  extra.numberPrefixes.value = (rules.numberPrefixes || []).join(", ");
  extra.singleRules.value = Object.entries(rules.singleCells || {}).map(([k, v]) => `${k}=${v}`).join("\n");
  extra.sequenceRules.value = (rules.sequences || []).map((item) => `${item.sequence.join(" ")}=${item.text}`).join("\n");
}

function activeRules() {
  try {
    return JSON.parse(localStorage.getItem("rataBrailleRules")) || defaultRules();
  } catch {
    return defaultRules();
  }
}

function translateBraille(value) {
  const rules = activeRules();
  const capitals = new Set(rules.capitalPrefixes || []);
  const numbers = new Set(rules.numberPrefixes || []);
  const numberMap = { a: "1", b: "2", c: "3", d: "4", e: "5", f: "6", g: "7", h: "8", i: "9", j: "0" };
  return value.split("\n").map((line) => {
    const tokens = Array.from(line, (ch) => {
      const code = ch.codePointAt(0) - 0x2800;
      return code >= 0 && code < 64 ? dotsFromLabel(code) : null;
    });
    let output = "", capital = false, number = false;
    for (let i = 0; i < tokens.length;) {
      const dots = tokens[i];
      if (dots === null) { output += line[i]; number = false; i += 1; continue; }
      if (dots === "") { output += " "; number = false; i += 1; continue; }
      const sequence = (rules.sequences || []).find((item) =>
        item.sequence.every((cell, offset) => tokens[i + offset] === cell)
      );
      if (sequence) { output += sequence.text; i += sequence.sequence.length; continue; }
      if (capitals.has(dots)) { capital = true; i += 1; continue; }
      if (numbers.has(dots)) { number = true; i += 1; continue; }
      let text = (rules.singleCells || {})[dots] ?? `~${dots}~`;
      if (number) text = numberMap[text.toLowerCase()] || text;
      if (capital) { text = text.toUpperCase(); capital = false; }
      output += text; i += 1;
    }
    return output.trimEnd();
  }).join("\n");
}

function addHistory(result) {
  const history = JSON.parse(localStorage.getItem("rataBrailleHistory") || "[]");
  history.unshift({ ...result, when: new Date().toISOString() });
  localStorage.setItem("rataBrailleHistory", JSON.stringify(history.slice(0, 30)));
  renderHistory();
}

function renderHistory() {
  const history = JSON.parse(localStorage.getItem("rataBrailleHistory") || "[]");
  extra.historyList.replaceChildren(...history.map((item) => {
    const li = document.createElement("li");
    li.textContent = `${new Date(item.when).toLocaleString()} — ${item.name} — ${item.detections} cells`;
    return li;
  }));
}

function anchorBoxes(width, height) {
  const all = [];
  for (let level = 0; level < ANCHOR_AREAS.length; level += 1) {
    const fmW = Math.ceil(width / Math.pow(2, level + 3));
    const fmH = Math.ceil(height / Math.pow(2, level + 3));
    const gridW = width / fmW;
    const gridH = height / fmH;
    const whs = [];
    for (const ar of ASPECT_RATIOS) {
      const h = Math.sqrt(ANCHOR_AREAS[level] / ar);
      const w = ar * h;
      for (const sr of SCALE_RATIOS) whs.push([w * sr, h * sr]);
    }
    for (let y = 0; y < fmH; y += 1) {
      for (let x = 0; x < fmW; x += 1) {
        for (const [aw, ah] of whs) {
          all.push([(x + 0.5) * gridW, (y + 0.5) * gridH, aw, ah]);
        }
      }
    }
  }
  return all;
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  return inter / Math.max(1e-6, areaA + areaB - inter);
}

function decode(loc, cls, width, height) {
  const anchors = anchorBoxes(width, height);
  const detections = [];
  for (let i = 0; i < anchors.length; i += 1) {
    let bestScore = -Infinity;
    let bestLabel = 0;
    for (let c = 0; c < 64; c += 1) {
      const score = sigmoid(cls[i * 64 + c]);
      if (score > bestScore) {
        bestScore = score;
        bestLabel = c;
      }
    }
    if (bestScore <= CLASS_THRESHOLD) continue;
    const [ax, ay, aw, ah] = anchors[i];
    const dx = loc[i * 4];
    const dy = loc[i * 4 + 1];
    const dw = loc[i * 4 + 2];
    const dh = loc[i * 4 + 3];
    const cx = dx * aw + ax;
    const cy = dy * ah + ay;
    const bw = Math.exp(dw) * aw;
    const bh = Math.exp(dh) * ah;
    detections.push({
      x1: cx - bw / 2,
      y1: cy - bh / 2,
      x2: cx + bw / 2,
      y2: cy + bh / 2,
      score: bestScore,
      label: bestLabel
    });
  }
  detections.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const det of detections) {
    if (kept.every((other) => iou(det, other) <= NMS_THRESHOLD)) kept.push(det);
    if (kept.length >= 1200) break;
  }
  return kept;
}

function lineText(detections) {
  if (!detections.length) return { text: "", braille: "" };
  const sorted = detections.slice().sort((a, b) => ((a.y1 + a.y2) / 2) - ((b.y1 + b.y2) / 2));
  const medianH = sorted.map((d) => d.y2 - d.y1).sort((a, b) => a - b)[Math.floor(sorted.length / 2)] || 12;
  const lines = [];
  for (const det of sorted) {
    const cy = (det.y1 + det.y2) / 2;
    let line = lines.find((ln) => Math.abs(ln.cy - cy) < medianH * 0.8);
    if (!line) {
      line = { cy, chars: [] };
      lines.push(line);
    }
    line.chars.push(det);
    line.cy = (line.cy * (line.chars.length - 1) + cy) / line.chars.length;
  }
  lines.sort((a, b) => a.cy - b.cy);
  const textLines = [];
  const brailleLines = [];
  for (const ln of lines) {
    ln.chars.sort((a, b) => a.x1 - b.x1);
    const avgW = ln.chars.reduce((sum, d) => sum + (d.x2 - d.x1), 0) / ln.chars.length;
    let lastX = null;
    let text = "";
    let brl = "";
    for (const ch of ln.chars) {
      if (lastX !== null && ch.x1 - lastX > avgW * 1.6) {
        text += " ";
        brl += "\u2800";
      }
      text += textFromLabel(ch.label);
      brl += brailleFromLabel(ch.label);
      lastX = ch.x2;
    }
    textLines.push(text);
    brailleLines.push(brl);
  }
  return { text: textLines.join("\n"), braille: brailleLines.join("\n") };
}

function detectionLines(detections) {
  if (!detections.length) return [];
  const sorted = detections.slice().sort((a, b) => ((a.y1 + a.y2) / 2) - ((b.y1 + b.y2) / 2));
  const heights = sorted.map((d) => d.y2 - d.y1).sort((a, b) => a - b);
  const medianH = heights[Math.floor(heights.length / 2)] || 12;
  const lines = [];
  for (const det of sorted) {
    const cy = (det.y1 + det.y2) / 2;
    let line = lines.find((candidate) => Math.abs(candidate.cy - cy) < medianH * 0.8);
    if (!line) { line = { cy, chars: [] }; lines.push(line); }
    line.chars.push(det);
    line.cy = line.chars.reduce((sum, item) => sum + (item.y1 + item.y2) / 2, 0) / line.chars.length;
  }
  for (const line of lines) line.chars.sort((a, b) => a.x1 - b.x1);
  return lines.sort((a, b) => a.cy - b.cy);
}

function splitLineWords(chars) {
  if (!chars.length) return [];
  const avgW = chars.reduce((sum, item) => sum + item.x2 - item.x1, 0) / chars.length;
  const words = [[chars[0]]];
  for (let i = 1; i < chars.length; i += 1) {
    if (chars[i].x1 - chars[i - 1].x2 > avgW * 1.6) words.push([]);
    words[words.length - 1].push(chars[i]);
  }
  return words;
}

function translatedGroup(chars) {
  const braille = chars.map((item) => brailleFromLabel(item.label)).join("");
  return elements.useRules.checked ? translateBraille(braille) : chars.map((item) => textFromLabel(item.label)).join("");
}

function drawDetections(detections, modelW, modelH) {
  if (!sourceImage) return;
  drawSourceToCanvas(sourceImage);
  const sx = elements.canvas.width / modelW;
  const sy = elements.canvas.height / modelH;
  ctx.lineWidth = 2;
  ctx.font = `${Math.max(8, Math.min(72, Number(elements.labelSize.value) || 15))}px Segoe UI`;
  ctx.strokeStyle = elements.boxColor.value;
  ctx.fillStyle = elements.labelColor.value;
  const scaled = detections.map((det) => ({
    ...det,
    x1: det.x1 * sx, y1: det.y1 * sy, x2: det.x2 * sx, y2: det.y2 * sy
  }));
  for (const det of scaled) {
    const x = det.x1;
    const y = det.y1;
    const w = det.x2 - det.x1;
    const h = det.y2 - det.y1;
    if (elements.showBoxes.checked) ctx.strokeRect(x, y, w, h);
    if (elements.showLabels.checked && elements.labelPlacement.value === "letter") {
      ctx.fillText(textFromLabel(det.label), x + 3, y + h + Number(elements.labelSize.value || 15));
    }
  }
  if (!elements.showLabels.checked || elements.labelPlacement.value === "letter") return;
  for (const line of detectionLines(scaled)) {
    const groups = elements.labelPlacement.value === "word" ? splitLineWords(line.chars) : [line.chars];
    for (const group of groups) {
      const first = group[0];
      const bottom = Math.max(...group.map((item) => item.y2));
      ctx.fillText(translatedGroup(group), first.x1 + 3, bottom + Number(elements.labelSize.value || 15));
    }
  }
}

function safeStem(name) {
  return (name || "rata_result").replace(/\.[^.]+$/, "").replace(/[^A-Za-z0-9._-]+/g, "_");
}

function storeCurrentResult(detections, text) {
  const name = currentFileName || `camera_${Date.now()}.jpg`;
  const item = {
    id: crypto.randomUUID?.() || `${Date.now()}-${Math.random()}`,
    name,
    detections: detections.length,
    translated: text.text,
    braille: text.braille,
    image: elements.canvas.toDataURL("image/jpeg", 0.9)
  };
  resultCollection.push(item);
  renderCollection();
}

function renderCollection() {
  extra.paperGallery.replaceChildren(...resultCollection.map((item) => {
    const card = document.createElement("article");
    card.className = "paper-card";
    const title = document.createElement("h3");
    title.textContent = `${item.name} — ${item.detections} cells`;
    const image = document.createElement("img");
    image.src = item.image;
    image.alt = `Recognized Braille paper ${item.name}`;
    const braille = document.createElement("textarea");
    braille.value = item.braille;
    braille.setAttribute("aria-label", `Braille for ${item.name}`);
    braille.addEventListener("input", () => { item.braille = braille.value; });
    const translated = document.createElement("textarea");
    translated.value = item.translated;
    translated.setAttribute("aria-label", `Translation for ${item.name}`);
    translated.addEventListener("input", () => { item.translated = translated.value; });
    const load = document.createElement("button");
    load.textContent = "Open for Editing";
    load.addEventListener("click", async () => {
      sourceImage = new Image();
      sourceImage.src = item.image;
      await sourceImage.decode();
      currentFileName = item.name;
      drawSourceToCanvas(sourceImage);
      elements.braille.value = item.braille;
      elements.translated.value = item.translated;
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
    card.append(title, image, braille, translated, load);
    return card;
  }));
}

async function runOcr() {
  if (!sourceImage) {
    setStatus("Load an image first");
    return;
  }
  try {
    setStatus("Loading ONNX model...");
    const session = await getSession();
    const prepared = preprocess(sourceImage);
    setStatus(`Running ${prepared.width}x${prepared.height}...`);
    const started = performance.now();
    const outputs = await session.run({ [EXPECTED_INPUT]: prepared.tensor });
    const locOutput = outputs.loc_pred;
    const clsOutput = outputs.cls_pred;
    if (!locOutput || !clsOutput) {
      throw new Error("The selected ONNX model did not return loc_pred and cls_pred.");
    }
    const loc = locOutput.data;
    const cls = clsOutput.data;
    const detections = decode(loc, cls, prepared.width, prepared.height);
    const text = lineText(detections);
    if (elements.useRules.checked) text.text = translateBraille(text.braille);
    drawDetections(detections, prepared.width, prepared.height);
    elements.translated.value = text.text;
    elements.braille.value = text.braille;
    elements.info.textContent = [
      `language: ${elements.language.value}`,
      `runtime: ONNX Runtime Web (WASM)`,
      `model: ${MODEL_PARTS.length} uploaded parts (reassembled in memory)`,
      `checkpoint_fallback: disabled`,
      `input: ${prepared.width}x${prepared.height}`,
      `loc_pred: ${locOutput.dims.join("x")}`,
      `cls_pred: ${clsOutput.dims.join("x")}`,
      `detections_after_nms: ${detections.length}`,
      `elapsed_ms: ${Math.round(performance.now() - started)}`
    ].join("\n");
    setStatus(`Done: ${detections.length} detections`);
    addHistory({ name: currentFileName || "camera capture", detections: detections.length });
    storeCurrentResult(detections, text);
    return { detections, text };
  } catch (error) {
    console.error(error);
    const message = String(error?.message || error || "Unknown error");
    setStatus(`OCR failed: ${message}`);
    elements.info.textContent = String(error?.stack || error);
  }
}

function canvasPoint(event) {
  const rect = elements.canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * elements.canvas.width / rect.width,
    y: (event.clientY - rect.top) * elements.canvas.height / rect.height
  };
}

function startDraw(event) {
  pushUndo();
  drawing = true;
  lastPoint = canvasPoint(event);
}

function moveDraw(event) {
  if (!drawing || !lastPoint) return;
  const point = canvasPoint(event);
  ctx.strokeStyle = elements.penColor.value;
  ctx.lineWidth = Number(elements.penSize.value);
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(lastPoint.x, lastPoint.y);
  ctx.lineTo(point.x, point.y);
  ctx.stroke();
  lastPoint = point;
}

function endDraw() {
  drawing = false;
  lastPoint = null;
}

function rotateCanvas(degrees) {
  pushUndo();
  const radians = degrees * Math.PI / 180;
  const old = document.createElement("canvas");
  old.width = elements.canvas.width;
  old.height = elements.canvas.height;
  old.getContext("2d").drawImage(elements.canvas, 0, 0);
  const sin = Math.abs(Math.sin(radians));
  const cos = Math.abs(Math.cos(radians));
  const newW = Math.ceil(old.width * cos + old.height * sin);
  const newH = Math.ceil(old.width * sin + old.height * cos);
  elements.canvas.width = newW;
  elements.canvas.height = newH;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, newW, newH);
  ctx.translate(newW / 2, newH / 2);
  ctx.rotate(radians);
  ctx.drawImage(old, -old.width / 2, -old.height / 2);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
}

elements.file.addEventListener("change", async () => {
  selectedFiles = Array.from(elements.file.files || []);
  const file = selectedFiles[0];
  if (!file) return;
  currentFileName = file.name;
  sourceImage = await readImageFromFile(file);
  undoStack = [];
  redoStack = [];
  drawSourceToCanvas(sourceImage);
  setStatus(`Loaded ${file.name}`);
});

elements.folder.addEventListener("change", async () => {
  selectedFiles = Array.from(elements.folder.files || []).filter((file) => file.type.startsWith("image/"));
  if (!selectedFiles.length) { setStatus("No Braille images found in that folder"); return; }
  const file = selectedFiles[0];
  currentFileName = file.webkitRelativePath || file.name;
  sourceImage = await readImageFromFile(file);
  drawSourceToCanvas(sourceImage);
  setStatus(`Loaded folder with ${selectedFiles.length} image(s)`);
  extra.batchStatus.textContent = `${selectedFiles.length} folder image(s) ready for Run All Images.`;
});

elements.runBatch.addEventListener("click", async () => {
  if (!selectedFiles.length) { setStatus("Load one or more images first"); return; }
  elements.runBatch.disabled = true;
  try {
    for (let i = 0; i < selectedFiles.length; i += 1) {
      const file = selectedFiles[i];
      extra.batchStatus.textContent = `Processing ${i + 1} of ${selectedFiles.length}: ${file.name}`;
      sourceImage = await readImageFromFile(file);
      currentFileName = file.name;
      drawSourceToCanvas(sourceImage);
      await runOcr();
    }
    extra.batchStatus.textContent = `Finished ${selectedFiles.length} image(s).`;
  } finally {
    elements.runBatch.disabled = false;
  }
});

elements.run.addEventListener("click", runOcr);
elements.canvas.addEventListener("pointerdown", startDraw);
elements.canvas.addEventListener("pointermove", moveDraw);
window.addEventListener("pointerup", endDraw);
elements.undo.addEventListener("click", () => {
  if (!undoStack.length) return;
  redoStack.push(ctx.getImageData(0, 0, elements.canvas.width, elements.canvas.height));
  restoreImageData(undoStack.pop());
});
elements.redo.addEventListener("click", () => {
  if (!redoStack.length) return;
  undoStack.push(ctx.getImageData(0, 0, elements.canvas.width, elements.canvas.height));
  restoreImageData(redoStack.pop());
});
elements.clear.addEventListener("click", () => sourceImage && drawSourceToCanvas(sourceImage));
elements.rotateLeft.addEventListener("click", () => rotateCanvas(-90));
elements.rotateRight.addEventListener("click", () => rotateCanvas(90));
elements.rotateAngle.addEventListener("click", () => rotateCanvas(Number(elements.angle.value || 0) % 360));
elements.save.addEventListener("click", () => {
  const link = document.createElement("a");
  link.download = "rata_braille_marked.png";
  link.href = elements.canvas.toDataURL("image/png");
  link.click();
});

extra.translateEdited.addEventListener("click", () => {
  elements.translated.value = translateBraille(elements.braille.value);
  setStatus("Edited Braille translated");
});

extra.copyText.addEventListener("click", async () => {
  await navigator.clipboard.writeText(elements.translated.value);
  setStatus("Translated text copied");
});

extra.saveText.addEventListener("click", () => {
  downloadBlob(`${(currentFileName || "rata_result").replace(/\.[^.]+$/, "")}.marked.txt`, elements.translated.value);
});

extra.saveBraille.addEventListener("click", () => {
  downloadBlob(`${(currentFileName || "rata_result").replace(/\.[^.]+$/, "")}.marked.brl`, elements.braille.value);
});

extra.zoomIn.addEventListener("click", () => {
  canvasZoom = Math.min(5, canvasZoom * 1.25);
  elements.canvas.style.width = `${Math.round(elements.canvas.width * canvasZoom)}px`;
});

extra.zoomOut.addEventListener("click", () => {
  canvasZoom = Math.max(0.15, canvasZoom * 0.8);
  elements.canvas.style.width = `${Math.round(elements.canvas.width * canvasZoom)}px`;
});

extra.addNote.addEventListener("click", () => {
  const note = extra.noteText.value.trim();
  if (!note) return;
  pushUndo();
  const weight = extra.noteBold.checked ? "bold " : "";
  const style = extra.noteItalic.checked ? "italic " : "";
  ctx.font = `${style}${weight}24px Segoe UI`;
  ctx.fillStyle = elements.penColor.value;
  ctx.fillText(note, 24, 38);
  setStatus("Text note added");
});

extra.useEditedPaper.addEventListener("click", async () => {
  sourceImage = new Image();
  sourceImage.src = elements.canvas.toDataURL("image/png");
  await sourceImage.decode();
  currentFileName = `${safeStem(currentFileName)}_edited.png`;
  drawSourceToCanvas(sourceImage);
  setStatus("Canvas edits will be used for the next OCR run");
});

extra.downloadAll.addEventListener("click", () => {
  if (!resultCollection.length) { setStatus("No processed papers to download"); return; }
  const url = URL.createObjectURL(collectionArchive());
  const link = document.createElement("a");
  link.href = url;
  link.download = `rata_braille_results_${new Date().toISOString().slice(0, 10)}.zip`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  setStatus(`Downloaded ${resultCollection.length} paper result(s)`);
});

extra.shareResults.addEventListener("click", async () => {
  if (!resultCollection.length) { setStatus("No processed papers to share"); return; }
  const file = new File([collectionArchive()], "rata_braille_results.zip", { type: "application/zip" });
  try {
    if (navigator.share && (!navigator.canShare || navigator.canShare({ files: [file] }))) {
      await navigator.share({ title: "RATA Braille Recognition Results", text: "Braille recognition results", files: [file] });
      setStatus("Results shared");
    } else {
      const url = URL.createObjectURL(file);
      const link = document.createElement("a");
      link.href = url;
      link.download = file.name;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      setStatus("Sharing is unavailable here; results were downloaded instead");
    }
  } catch (error) {
    if (error.name !== "AbortError") setStatus(`Sharing failed: ${error.message}`);
  }
});

extra.clearCollection.addEventListener("click", () => {
  resultCollection = [];
  renderCollection();
  setStatus("Paper collection cleared");
});

extra.saveRules.addEventListener("click", () => {
  localStorage.setItem("rataBrailleRules", JSON.stringify(rulesFromForm()));
  setStatus("Custom Braille rules saved");
});

extra.resetRules.addEventListener("click", () => {
  const rules = defaultRules();
  fillRulesForm(rules);
  localStorage.setItem("rataBrailleRules", JSON.stringify(rules));
  setStatus("English rules restored");
});

extra.exportRules.addEventListener("click", () => {
  downloadBlob("rata_braille_rules.json", JSON.stringify(rulesFromForm(), null, 2), "application/json");
});

extra.importRules.addEventListener("change", async () => {
  const file = extra.importRules.files?.[0];
  if (!file) return;
  try {
    const rules = JSON.parse(await file.text());
    fillRulesForm(rules);
    localStorage.setItem("rataBrailleRules", JSON.stringify(rules));
    setStatus("Rules imported");
  } catch (error) {
    setStatus(`Rules import failed: ${error.message}`);
  }
});

extra.clearHistory.addEventListener("click", () => {
  localStorage.removeItem("rataBrailleHistory");
  renderHistory();
  setStatus("History cleared");
});

extra.savePreferences.addEventListener("click", () => {
  localStorage.setItem("rataBraillePreferences", JSON.stringify({
    language: elements.language.value,
    enhance: elements.enhance.checked,
    useRules: elements.useRules.checked,
    showBoxes: elements.showBoxes.checked,
    boxColor: elements.boxColor.value,
    showLabels: elements.showLabels.checked,
    labelPlacement: elements.labelPlacement.value,
    labelColor: elements.labelColor.value,
    labelSize: elements.labelSize.value,
    penColor: elements.penColor.value,
    penSize: elements.penSize.value
  }));
  setStatus("Preferences saved");
});

extra.resetPreferences.addEventListener("click", () => {
  localStorage.removeItem("rataBraillePreferences");
  elements.language.value = "EN";
  elements.enhance.checked = false;
  elements.useRules.checked = false;
  elements.showBoxes.checked = true;
  elements.boxColor.value = "#1f9d55";
  elements.showLabels.checked = true;
  elements.labelPlacement.value = "letter";
  elements.labelColor.value = "#111111";
  elements.labelSize.value = "15";
  elements.penColor.value = "#d7263d";
  elements.penSize.value = "6";
  setStatus("Preferences reset");
});

async function openCamera() {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
    extra.cameraVideo.srcObject = cameraStream;
    extra.cameraDialog.showModal();
  } catch (error) {
    setStatus(`Camera failed: ${error.message}`);
  }
}

function stopCamera() {
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  extra.cameraVideo.srcObject = null;
  extra.cameraDialog.close();
}

async function captureCamera(runAfter) {
  if (!extra.cameraVideo.videoWidth) return;
  const capture = document.createElement("canvas");
  capture.width = extra.cameraVideo.videoWidth;
  capture.height = extra.cameraVideo.videoHeight;
  capture.getContext("2d").drawImage(extra.cameraVideo, 0, 0);
  sourceImage = new Image();
  sourceImage.src = capture.toDataURL("image/jpeg", 0.95);
  await sourceImage.decode();
  currentFileName = `camera_${new Date().toISOString().replace(/[:.]/g, "-")}.jpg`;
  drawSourceToCanvas(sourceImage);
  stopCamera();
  setStatus("Camera image captured");
  if (runAfter) await runOcr();
}

elements.openCamera.addEventListener("click", openCamera);
extra.captureCamera.addEventListener("click", () => captureCamera(false));
extra.captureAndRun.addEventListener("click", () => captureCamera(true));
extra.closeCamera.addEventListener("click", stopCamera);

fillRulesForm(activeRules());
renderHistory();
try {
  const preferences = JSON.parse(localStorage.getItem("rataBraillePreferences") || "{}");
  if (preferences.language) elements.language.value = preferences.language;
  elements.enhance.checked = Boolean(preferences.enhance);
  elements.useRules.checked = Boolean(preferences.useRules);
  if (preferences.showBoxes !== undefined) elements.showBoxes.checked = Boolean(preferences.showBoxes);
  if (preferences.boxColor) elements.boxColor.value = preferences.boxColor;
  if (preferences.showLabels !== undefined) elements.showLabels.checked = Boolean(preferences.showLabels);
  if (preferences.labelPlacement) elements.labelPlacement.value = preferences.labelPlacement;
  if (preferences.labelColor) elements.labelColor.value = preferences.labelColor;
  if (preferences.labelSize) elements.labelSize.value = preferences.labelSize;
  if (preferences.penColor) elements.penColor.value = preferences.penColor;
  if (preferences.penSize) elements.penSize.value = preferences.penSize;
} catch {
  localStorage.removeItem("rataBraillePreferences");
}

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => navigator.serviceWorker.register("sw.js").catch(console.warn));
}

if (window.location.protocol === "file:") {
  setStatus("Start required: open START_APP.cmd");
  elements.info.textContent =
    "Browsers do not allow ONNX Runtime Web modules to load from file:/// addresses.\n\n" +
    "Close this page, then double-click START_APP.cmd in this folder.";
  elements.run.disabled = true;
}
