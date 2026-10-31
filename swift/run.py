# SPDX-License-Identifier: AGPL-3.0-only
"""Reverse-engineer a mesh/scan into validated CAD with Swift 1.5.

    python swift/run.py part.stl [--mode auto|cad|scan] [--work DIR] [--max-turns 60] [--no-swift] [--headless-browser]

auto: a CAD-exported mesh (exact surfaces) goes the exact-reading way, headless; a real scan opens
the full ScanRuler page (fits with sigma evidence). Either way the model is held to the independent
validator, and only a PASS is delivered as <part>_CAD.step; otherwise best_attempt_CAD.step, marked so.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent import Agent  # noqa: E402
from autopilot import first_pass  # noqa: E402
from validate import summary  # noqa: E402
from workbench import Workbench, clip  # noqa: E402


def looks_like_cad(work):
    """A CAD tessellation: big planes fit to ~1e-5 mm and few faces per triangle (analyze.py's faces.json)."""
    f = json.loads((work / "data" / "faces.json").read_text())["faces"]
    n_tris = len(np.load(work / "data" / "faces.npy"))
    total = sum(x["area"] for x in f) or 1
    res = [x["res"] for x in f if x["kind"] == "plane" and x["area"] > 0.01 * total and x.get("res") is not None]
    return bool(res) and float(np.median(res)) < 1e-4 and len(f) < max(50, n_tris / 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stl")
    ap.add_argument("--mode", choices=["auto", "cad", "scan"], default="auto")
    ap.add_argument("--work")
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--no-swift", action="store_true", help="deterministic pass only (CAD meshes)")
    ap.add_argument("--headless-browser", action="store_true", help="scan mode: do not show the ScanRuler window")
    a = ap.parse_args()

    stl = Path(a.stl).resolve()
    work = Path(a.work).resolve() if a.work else stl.with_name(stl.stem + "_work")
    logf = open(work.mkdir(parents=True, exist_ok=True) or work / "run.log", "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n"); logf.flush()

    t0 = time.time()
    wb = Workbench(stl, work)
    mode, report, failed = a.mode, "", True
    try:
        analysis = wb.script("analyze")
        log(analysis)
        if mode == "auto":
            mode = "cad" if looks_like_cad(work) else "scan"
        log(f"[run] mode: {mode}")
        if mode == "cad":
            files, _ = first_pass(wb, log=lambda s: log(clip(s, 3000)))
            check = wb.check("cad")
            log(check)
            failed = wb.last_validation["verdict"] != "PASS"
            report = f"Deterministic exact-reading pass: {', '.join(files)}."
            if failed and not a.no_swift:
                agent = Agent(wb, "cad", log=log, max_turns=a.max_turns)
                msg = (f"Part: {stl.name}. analyze.py said:\n{clip(analysis, 5000)}\n\nThe deterministic first pass built "
                       f"{files} (main axis {wb.main_axis}, body \"{wb.body}\") and the validator says:\n{clip(agent.t_check(), 6000)}\n\n"
                       "Fix it until the validator passes.")
                report, failed = agent.run(msg)
        else:
            wb.close()
            wb = Workbench(stl, work, page=True, show_browser=not a.headless_browser)
            agent = Agent(wb, "scan", log=log, max_turns=a.max_turns)
            msg = (f"Scan: {stl.name}, opened in ScanRuler. A first look by the exact-reading analyzer (it fragments on noise, "
                   f"so only the box, frame and big planes mean anything):\n{clip(analysis, 2500)}\n\n"
                   "Start with the guide index, then work the scan-to-cad flow until the validator passes.")
            report, failed = agent.run(msg)
            failed = failed or not agent.passed

        r = wb.last_validation
        if not failed and r and r["verdict"] == "PASS":
            log(wb.deliver(stl.stem))
            status = f"PASS: {stl.stem}_CAD.step"
        else:
            status = "NOT VALIDATED: best attempt in best_attempt_CAD.step"
        md = [f"# {stl.stem}: reverse-engineering report", "", f"**{status}** (mode {mode}, {time.time() - t0:.0f} s)", "",
              "## Validation (independent)", "```", summary(r) if r else "no model was checked", "```", "",
              "## How it was built", report or "(no report)"]
        (work / f"{stl.stem}_REPORT.md").write_text("\n".join(md), encoding="utf-8")
        log(f"[run] {status} -> {work}")
        return 0 if status.startswith("PASS") else 1
    finally:
        wb.close()
        logf.close()


if __name__ == "__main__":
    sys.exit(main())
