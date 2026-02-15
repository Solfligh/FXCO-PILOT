/* sw.js — FXCO-PILOT PWA */

const VERSION = "fxco-pwa-v5";
const CACHE_NAME = `fxco-cache-${VERSION}`;

// Cache essentials only (do NOT cache maintenance.html as a fallback)
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

function isAssetPath(pathname) {
  return (
    pathname === "/favicon.ico" ||
    pathname === "/manifest.webmanifest" ||
    pathname === "/sw.js" ||
    pathname === "/apple-touch-icon.png" ||
    pathname.startsWith("/icons/") ||
    pathname.startsWith("/images/") ||
    /\.(png|jpg|jpeg|webp|svg|ico|css|js|json|webmanifest)$/i.test(pathname)
  );
}

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(PRECACHE)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.map((k) => (k.startsWith("fxco-cache-") && k !== CACHE_NAME ? caches.delete(k) : null))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);

  // Don’t SW-hijack API calls
  if (url.pathname.startsWith("/api/")) return;

  // ✅ Assets: network-first (never return HTML fallback)
  if (isAssetPath(url.pathname)) {
    event.respondWith(fetch(req).catch(() => caches.match(req)));
    return;
  }

  // ✅ Navigations: network-first; offline fallback to cached index.html
  if (req.mode === "navigate") {
    event.respondWith(fetch(req).catch(() => caches.match("/index.html")));
    return;
  }

  // Everything else: cache-first
  event.respondWith(caches.match(req).then((cached) => cached || fetch(req)));
});
