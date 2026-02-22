/* analytics.js — PostHog (FXCO-Pilot)
   Put this file in: frontend/public/analytics.js
*/

(function () {
  // ✅ 1) Create a PostHog project and paste your values here:
  // - POSTHOG_KEY: Project API Key
  // - POSTHOG_HOST: usually "https://app.posthog.com" (cloud) or your self-host URL
  const POSTHOG_KEY = "PASTE_YOUR_POSTHOG_PROJECT_API_KEY_HERE";
  const POSTHOG_HOST = "https://app.posthog.com";

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

  // Load PostHog
  function loadPostHog(cb) {
    if (!POSTHOG_KEY || POSTHOG_KEY.includes("PASTE_")) {
      // Don’t break the app if not configured
      window.fxTrack = function () {};
      window.fxIdentify = function () {};
      return;
    }

    if (window.posthog && window.posthog.__loaded) {
      cb && cb();
      return;
    }

    // Minimal PostHog snippet (no autocapture by default)
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
      s.src = POSTHOG_HOST.replace(/\/$/, "") + "/static/array.js";
      s.onload = function () {
        w.posthog.__loaded = true;
        cb && cb();
      };
      d.head.appendChild(s);
    })(document, window);

    window.posthog("init", POSTHOG_KEY, {
      api_host: POSTHOG_HOST,
      autocapture: false,
      capture_pageview: true,
      capture_pageleave: true,
      persistence: "localStorage",
      disable_session_recording: true // turn on later if you want
    });
  }

  function identifyIfPossible() {
    try {
      const uid = getUserId();
      if (uid && window.posthog) {
        window.posthog("identify", uid, {
          tenant: getTenant() || undefined
        });
      }
    } catch {}
  }

  function track(event, props) {
    try {
      if (!window.posthog || !window.posthog.__loaded) return;
      const base = {
        app: "fxco-pilot",
        tenant: getTenant() || undefined
      };
      window.posthog("capture", event, Object.assign(base, props || {}));
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
  });
})();
