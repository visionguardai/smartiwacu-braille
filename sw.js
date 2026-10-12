const CACHE_NAME = "rata-braille-onnx-v9";
const CORE_ASSETS = [
  "./",
  "./index.html",
  "./styles.css",
  "./app.js",
  "./manifest.webmanifest",
  "./ort.min.js",
  "./ort-wasm-simd-threaded.wasm",
  "./ort-wasm-simd-threaded.mjs",
  "./ort-wasm-simd-threaded.asyncify.wasm.part001",
  "./ort-wasm-simd-threaded.asyncify.wasm.part002",
  "./ort-wasm-simd-threaded.asyncify.mjs",
  "./ort-wasm-simd-threaded.jsep.wasm.part001",
  "./ort-wasm-simd-threaded.jsep.wasm.part002",
  "./ort-wasm-simd-threaded.jsep.mjs",
  "./ort-wasm-simd-threaded.jspi.wasm",
  "./ort-wasm-simd-threaded.jspi.mjs",
  "./model.onnx.part001",
  "./model.onnx.part002",
  "./model.onnx.part003",
  "./model.onnx.part004"
  ,"./model.onnx.part005",
  "./model.onnx.part006",
  "./model.onnx.part007",
  "./model.onnx.part008"
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(CORE_ASSETS)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))))
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  event.respondWith(
    caches.match(event.request).then((cached) => cached || fetch(event.request))
  );
});
