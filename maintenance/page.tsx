import Link from "next/link";

export const metadata = {
  title: "FXCO-PILOT Maintenance",
  description:
    "FXCO-PILOT is temporarily offline while we deploy improvements and system upgrades.",
};

export default function MaintenancePage() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-black text-white">
      <div className="mx-auto flex min-h-screen max-w-3xl flex-col items-center justify-center px-6 py-16 text-center">
        
        {/* Status Badge */}
        <div className="inline-flex items-center gap-2 rounded-full border border-amber-500/30 bg-amber-500/10 px-4 py-2 text-xs font-semibold text-amber-400">
          <span className="h-2 w-2 rounded-full bg-amber-400 animate-pulse" />
          FXCO-PILOT Maintenance Mode
        </div>

        {/* Heading */}
        <h1 className="mt-6 text-4xl font-semibold tracking-tight sm:text-5xl">
          We’re upgrading your trade validation engine.
        </h1>

        {/* Description */}
        <p className="mt-4 text-base leading-relaxed text-slate-300">
          FXCO-PILOT is temporarily offline while we improve performance,
          optimize trade validation logic, and strengthen system reliability.
          <br />
          We’ll be back shortly — sharper and faster.
        </p>

        {/* Info Card */}
        <div className="mt-10 w-full rounded-2xl border border-slate-800 bg-slate-900/60 p-6 text-left backdrop-blur">
          <div className="grid gap-4 sm:grid-cols-2">
            
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
              <p className="text-sm font-semibold text-white">What’s happening?</p>
              <p className="mt-1 text-sm text-slate-400">
                System improvements and infrastructure updates.
              </p>
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
              <p className="text-sm font-semibold text-white">Estimated downtime</p>
              <p className="mt-1 text-sm text-slate-400">
                Short-term. We’re actively deploying updates.
              </p>
            </div>

          </div>

          <div className="mt-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="text-xs text-slate-500">
              Thank you for your patience. Precision takes discipline.
            </div>

            <div className="flex gap-3">
              <Link
                href="/"
                className="inline-flex items-center justify-center rounded-xl border border-slate-700 bg-slate-800 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-700"
              >
                Refresh
              </Link>

              <a
                href="mailto:support@fxco-pilot.com"
                className="inline-flex items-center justify-center rounded-xl bg-amber-500 px-4 py-2 text-sm font-semibold text-black transition hover:bg-amber-400"
              >
                Contact Support
              </a>
            </div>
          </div>
        </div>

        <p className="mt-10 text-xs text-slate-600">
          © {new Date().getFullYear()} FXCO-PILOT
        </p>
      </div>
    </main>
  );
}
