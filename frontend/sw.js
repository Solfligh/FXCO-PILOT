/* sw.js — FXCO-PILOT PWA */

// Change this to force update when you deploy
const VERSION = "fxco-pwa-v3";
const CACHE_NAME = `fxco-cache-${VERSION}`;

// Only cache the *pages* you want available offline.
// Do NOT cache maintenance.html as a global fallback.
const PRECACHE = [
  "/",
  "/index.html",
  "/result.html",
  "/terms.html",
  "/privacy.html",
  "/manifest.webmanifest",
  "/favicon.ico",
  "/apple-touch-icon.png",
  "/icons/icon-192.png",
  "/icons/icon-512.png",
  "/icons/icon-512-maskable.png"
];

// Never let SW "fallback" to HTML for these asset types
function isAssetRequest(url) {
  return (
    url.pathname === "/favicon.ico" ||
    url.pathname === "/manifest.webmanifest" ||
    url.pathname === "/sw.js" ||
    url.pathname === "/apple-touch-icon.png" ||
    url.pathname.startsWith("/icons/") ||
    url.pathname.startsWith("/images/") ||
    /\.(png|jpg|jpeg|webp|svg|ico|css|js|json|webmanifest)$/i.test(url.pathname)
  );
}

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(PRECACHE))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.map((k) => (k.startsWith("fxco-cache-") && k !== CACHE_NAME) ? caches.delete(k) : null))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const req = event.request;

  // Only handle GET
  if (req.method !== "GET") return;

  const url = new URL(req.url);

  // Let API go straight to network
  if (url.pathname.startsWith("/api/")) return;

  // ✅ Assets: NETWORK-FIRST (never return HTML fallback)
  if (isAssetRequest(url)) {
    event.respondWith(
      fetch(req).catch(() => caches.match(req))
    );
    return;
  }

  // ✅ Navigations (pages): try network, then cached index.html as fallback
  if (req.mode === "navigate") {
    event.respondWith(
      fetch(req)
        .then((res) => res)
        .catch(() => caches.match("/index.html"))
    );
    return;
  }

  // Other requests: cache-first
  event.respondWith(
    caches.match(req).then((cached) => cached || fetch(req))
  );
});
