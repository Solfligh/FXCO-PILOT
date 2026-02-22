/* analytics.js — PostHog (FXCO-Pilot)
   Put this file in: frontend/public/analytics.js
*/

(function () {
  // Toggle debug logs in console (set to false after confirming)
  const DEBUG = true;

  // ✅ Your PostHog Project API Key
  const POSTHOG_KEY = "phc_qPTx6s4nQRSxDY6bPdZ072hBw7a62uqtSOOCJywP4Oi";

  // ✅ IMPORTANT:
  // Your dashboard is on us.posthog.com -> ingestion must be US:
  // api_host: https://us.i.posthog.com
  const POSTHOG_API_HOST = "https://us.i.posthog.com";

  // ✅ Load script from US assets CDN (reliable for US region)
  const POSTHOG_ASSET_HOST = "https://us-assets.i.posthog.com";

  function log() {
    if (!DEBUG) return;
    try {
      // eslint-disable-next-line no-console
      console.log.apply(console, ["[PostHog]", ...arguments]);
    } catch {}
  }

  // Optional: identify user if you have an email/user id stored
  function getUserId() {
    try {
      return localStorage.getItem("fxco_user_email") || localStorage.getItem("fxco_user_id") || "";
    } catch {
      return "";
    }
  }

  function getTenant() {
    try {
      return localStorage.getItem("fxco_tenant") || "";
    } catch {
      return "";
    }
  }

  function safeJsonParse(s) {
    try {
      return JSON.parse(s);
    } catch {
      return null;
    }
  }

  function getLastReportMeta() {
    // Try to extract minimal meta without reading huge data
    try {
      const raw =
        sessionStorage.getItem("fxcoReport") ||
        localStorage.getItem("fxcoReport_last_v1") ||
        "";
      const obj = typeof raw === "string" ? safeJsonParse(raw) : raw;
      const a = obj?.analysis || obj?.result || obj?.raw?.analysis || {};
      return {
        pair_type: obj?.pair_type || obj?.pairType || "",
        timeframe: obj?.timeframe || "",
        chart_tf: obj?.chart_tf || obj?.chartTf || "",
        tier: (a?.decision_tier || a?.tier || a?.decisionTier || "").toString(),
        score: Number.isFinite(Number(a?.score)) ? Number(a.score) : null,
        confidence: Number.isFinite(Number(a?.confidence)) ? Number(a.confidence) : null
      };
    } catch {
      return {};
    }
  }

  // -------------------------
  // Safe queue (so tracking works even before load completes)
  // -------------------------
  const _queue = [];
  function enqueueCapture(event, props) {
    _queue.push({ event, props: props || {} });
  }
  function flushQueue() {
    try {
      if (!window.posthog || !window.posthog.__loaded) return;
      while (_queue.length) {
        const item = _queue.shift();
        window.posthog("capture", item.event, item.props);
      }
    } catch {}
  }

  // Load PostHog
  function loadPostHog(cb) {
    if (!POSTHOG_KEY || POSTHOG_KEY.includes("PASTE_")) {
      // Don’t break the app if not configured
      window.fxTrack = function () {};
      window.fxIdentify = function () {};
      log("Not configured: POSTHOG_KEY missing.");
      return;
    }

    if (window.posthog && window.posthog.__loaded) {
      log("Already loaded.");
      cb && cb();
      return;
    }

    // Minimal PostHog queue function before script loads
    (function (d, w) {
      w.posthog =
        w.posthog ||
        function () {
          (w.posthog.q = w.posthog.q || []).push(arguments);
        };
      w.posthog.q = w.posthog.q || [];
      w.posthog.__loaded = false;

      var s = d.createElement("script");
      s.async = true;
      s.src = POSTHOG_ASSET_HOST.replace(/\/$/, "") + "/static/array.js";
      s.onload = function () {
        w.posthog.__loaded = true;
        log("Script loaded:", s.src);
        try {
          flushQueue();
        } catch {}
        cb && cb();
      };
      s.onerror = function () {
        log("Failed to load script:", s.src);
      };
      d.head.appendChild(s);
      log("Loading script:", s.src);
    })(document, window);

    // Init
    window.posthog("init", POSTHOG_KEY, {
      api_host: POSTHOG_API_HOST,
      autocapture: false,
      capture_pageview: true,
      capture_pageleave: true,
      persistence: "localStorage",
      disable_session_recording: true // turn on later if you want
    });

    log("Init called:", { api_host: POSTHOG_API_HOST });
  }

  function identifyIfPossible() {
    try {
      const uid = getUserId();
      if (uid && window.posthog) {
        window.posthog("identify", uid, {
          tenant: getTenant() || undefined
        });
        log("Identify:", uid);
      }
    } catch {}
  }

  function track(event, props) {
    try {
      const base = {
        app: "fxco-pilot",
        tenant: getTenant() || undefined
      };
      const payload = Object.assign(base, props || {});

      // If not ready yet, queue and flush later
      if (!window.posthog || !window.posthog.__loaded) {
        enqueueCapture(event, payload);
        log("Queued:", event, payload);
        return;
      }

      window.posthog("capture", event, payload);
      log("Captured:", event, payload);
    } catch {}
  }

  // Public helpers
  window.fxIdentify = function () {
    identifyIfPossible();
  };

  window.fxTrack = function (event, props) {
    track(event, props);
  };

  // Boot
  loadPostHog(function () {
    identifyIfPossible();

    // Helpful default context: last report meta (if any)
    const meta = getLastReportMeta();
    if (meta && (meta.pair_type || meta.timeframe || meta.chart_tf)) {
      track("fxco_context_seen", meta);
    }

    // A guaranteed test event (removes "Waiting for events")
    track("fxco_loaded", {
      page: location.pathname,
      title: document.title
    });
  });
})();
