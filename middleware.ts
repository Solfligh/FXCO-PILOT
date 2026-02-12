import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

/**
 * FXCO-PILOT Maintenance Mode Middleware
 *
 * Enable by setting:
 *   MAINTENANCE_MODE=1   (or "true")
 *
 * Optional bypass:
 *   MAINTENANCE_BYPASS_TOKEN=some-secret
 * Then visit any page with:
 *   ?bypass=some-secret
 * (sets a cookie for 7 days)
 */

const COOKIE_NAME = "fxco_maintenance_bypass";

function isEnabled() {
  const v = (process.env.MAINTENANCE_MODE || "").toLowerCase().trim();
  return v === "1" || v === "true" || v === "yes" || v === "on";
}

function getBypassToken() {
  return (process.env.MAINTENANCE_BYPASS_TOKEN || "").trim();
}

export function middleware(req: NextRequest) {
  // If not enabled, do nothing
  if (!isEnabled()) return NextResponse.next();

  const url = req.nextUrl;
  const pathname = url.pathname;

  // Always allow these paths through (don't break Next.js or APIs)
  const allowList = [
    "/maintenance",
    "/api", // backend routes
    "/_next", // Next.js assets
    "/favicon.ico",
    "/robots.txt",
    "/sitemap.xml",
  ];

  if (allowList.some((p) => pathname === p || pathname.startsWith(p + "/"))) {
    return NextResponse.next();
  }

  // Allow common static file requests (images/fonts/etc in /public)
  if (pathname.match(/\.(png|jpg|jpeg|webp|gif|svg|ico|txt|xml|json|woff|woff2|ttf|eot|map)$/i)) {
    return NextResponse.next();
  }

  // Bypass handling (optional)
  const bypassToken = getBypassToken();
  if (bypassToken) {
    const hasCookie = req.cookies.get(COOKIE_NAME)?.value === bypassToken;
    const queryBypass = url.searchParams.get("bypass") === bypassToken;

    if (hasCookie) {
      return NextResponse.next();
    }

    if (queryBypass) {
      const res = NextResponse.next();
      res.cookies.set({
        name: COOKIE_NAME,
        value: bypassToken,
        path: "/",
        httpOnly: true,
        sameSite: "lax",
        secure: true,
        maxAge: 60 * 60 * 24 * 7, // 7 days
      });

      // clean the URL (remove bypass param)
      const cleanUrl = new URL(req.url);
      cleanUrl.searchParams.delete("bypass");
      return NextResponse.redirect(cleanUrl, { headers: res.headers });
    }
  }

  // Redirect everything else to maintenance
  const maintenanceUrl = new URL("/maintenance", req.url);
  return NextResponse.redirect(maintenanceUrl);
}

/**
 * Match all routes except:
 * - static files in /_next
 * - common public files
 *
 * We still do our own allowlist checks above for safety.
 */
export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
