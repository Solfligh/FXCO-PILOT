/* sw.js — FXCO-PILOT PWA */

const VERSION = "fxco-pwa-v6"; // bump version to force refresh
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
  "/icons/icon-512-maskable.png",
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

function isHtmlResponse(res) {
  try {
    const ct = (res.headers.get("content-type") || "").toLowerCase();
    return ct.includes("text/html");
  } catch {
    return false;
  }
}

self.addEventListener("install", (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE_NAME);

    // Using addAll is fine, but we do it explicitly so failures are clearer.
    await Promise.all(
      PRECACHE.map(async (url) => {
        try {
          const res = await fetch(url, { cache: "reload" });
          if (res.ok) await cache.put(url, res);
        } catch {
          // If offline on install, just skip — app still works with network later.
        }
      })
    );
  })());

  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(
      keys.map((k) => (k.startsWith("fxco-cache-") && k !== CACHE_NAME ? caches.delete(k) : null))
    );
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);

  // Don’t SW-hijack API calls
  if (url.pathname.startsWith("/api/")) return;

  // ✅ Assets: network-first (never return HTML as an "image" / "asset")
  if (isAssetPath(url.pathname)) {
    event.respondWith((async () => {
      try {
        const res = await fetch(req);

        // If server returns HTML for an asset request (usually SPA rewrite / missing file),
        // fall back to a real icon instead of showing index.html.
        if (res && res.ok && isHtmlResponse(res)) {
          const fallback =
            (await caches.match(url.pathname)) ||
            (await caches.match("/apple-touch-icon.png")) ||
            (await caches.match("/favicon.ico"));

          return (
            fallback ||
            new Response("", { status: 404, statusText: "Asset not found" })
          );
        }

        return res;
      } catch {
        // Offline: serve cached asset, else icon fallback
        const cached =
          (await caches.match(req)) ||
          (await caches.match(url.pathname)) ||
          (await caches.match("/apple-touch-icon.png")) ||
          (await caches.match("/favicon.ico"));

        return cached || new Response("", { status: 504, statusText: "Offline" });
      }
    })());
    return;
  }

  // ✅ Navigations: network-first; offline fallback to cached index.html
  if (req.mode === "navigate") {
    event.respondWith((async () => {
      try {
        return await fetch(req);
      } catch {
        return (await caches.match("/index.html")) || new Response("Offline", { status: 503 });
      }
    })());
    return;
  }

  // Everything else: cache-first
  event.respondWith((async () => {
    const cached = await caches.match(req);
    if (cached) return cached;
    return fetch(req);
  })());
});
