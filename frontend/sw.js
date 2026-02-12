/* FXCO-PILOT Service Worker (simple + safe) */
const CACHE_NAME = "fxco-pilot-v1";

// Add any other static assets you want pre-cached:
const PRECACHE_URLS = [
  "/",
  "/index.html",
  "/maintenance.html",
  "/manifest.webmanifest"
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(PRECACHE_URLS))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.map((k) => (k !== CACHE_NAME ? caches.delete(k) : null)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  const url = new URL(req.url);

  // Don’t cache API calls to your Render backend
  if (url.pathname.startsWith("/api/")) return;

  // Only handle GET requests
  if (req.method !== "GET") return;

  event.respondWith(
    caches.match(req).then((cached) => {
      if (cached) return cached;

      return fetch(req)
        .then((res) => {
          // Cache successful same-origin responses
          if (res.ok && url.origin === self.location.origin) {
            const copy = res.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(req, copy));
          }
          return res;
        })
        .catch(() => {
          // If offline and requesting a page, fall back to index
          if (req.headers.get("accept")?.includes("text/html")) {
            return caches.match("/index.html");
          }
          throw new Error("Offline");
        });
    })
  );
});
