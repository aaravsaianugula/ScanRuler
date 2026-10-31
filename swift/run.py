# SPDX-License-Identifier: AGPL-3.0-only
"""Reverse-engineer a mesh/scan into validated CAD with Swift 1.5.

    python swift/run.py part.stl [--mode auto|cad|scan] [--work DIR] [--max-turns 60] [--no-swift]
                                 [--headless-browser] [--live] [--events]

auto: a CAD-exported mesh (exact surfaces) goes the exact-reading way; a real scan is measured in the
ScanRuler page (fits with sigma evidence) by Swift. Either way the model is held to the independent
validator, and only a PASS is delivered as <part>_CAD.step; otherwise best_attempt_CAD.step, marked so.

--events  JSON lines on stdout (phase, log, page, turn, reasoning, content, tool, tool_result, image,
          validation, done, error) for the desktop app; the human log still goes to <work>/run.log.
--live    run in the ScanRuler page and hand its pairing link to the host (event "page") instead of
          opening a browser: the desktop app shows it as the live view, CAD meshes included.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent import MODEL, Agent  # noqa: E402
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
    ap.add_argument("--live", action="store_true", help="hand the ScanRuler page to the host (desktop app) as the live view")
    ap.add_argument("--events", action="store_true", help="JSON lines on stdout for the desktop app")
    a = ap.parse_args()

    stl = Path(a.stl).resolve()
    work = Path(a.work).resolve() if a.work else stl.with_name(stl.stem + "_work")
    work.mkdir(parents=True, exist_ok=True)
    logf = open(work / "run.log", "a", encoding="utf-8")

    def emit(kind, **kw):
        if a.events:
            print(json.dumps({"type": kind, "t": round(time.time(), 3), **kw}, default=float), flush=True)

    def log(s):
        logf.write(s + "\n"); logf.flush()
        if a.events:
            emit("log", text=s)
        else:
            print(s, flush=True)

    def phase(name, label):
        emit("phase", phase=name, label=label)
        log(f"[run] phase {name}: {label}")

    t0 = time.time()
    emit("start", stl=str(stl), work=str(work), model=MODEL, swift=not a.no_swift)
    wb = None
    try:
        phase("open", "Opening ScanRuler")
        on_link = (lambda url: emit("page", url=url)) if a.live else None
        wb = Workbench(stl, work, page=a.live, on_link=on_link)
        wb.emit, wb.narrate = emit, True
        phase("analyze", "Reading the mesh")
        analysis = wb.script("analyze")
        log(analysis)
        mode = a.mode if a.mode != "auto" else ("cad" if looks_like_cad(work) else "scan")
        emit("mode", mode=mode)
        log(f"[run] mode: {mode}")
        report, failed, agent = "", True, None

        if mode == "cad":
            phase("autopilot", "Exact reading: slabs, holes, blends")
            files, _ = first_pass(wb, log=lambda s: log(clip(s, 3000)))
            phase("check", "Validating against the mesh")
            check = wb.check("cad")
            log(check)
            emit("validation", result=wb.last_validation, summary=check)
            failed = wb.last_validation is None or wb.last_validation["verdict"] != "PASS"
            report = f"Deterministic exact-reading pass: {', '.join(files)}."
            if failed and not a.no_swift:
                phase("swift", "Swift is repairing what failed")
                wb.narrate = False
                agent = Agent(wb, "cad", log=log, max_turns=a.max_turns, emit=emit)
                msg = (f"Part: {stl.name}. analyze.py said:\n{clip(analysis, 5000)}\n\nThe deterministic first pass built "
                       f"{files} (main axis {wb.main_axis}, body \"{wb.body}\") and the validator says:\n{clip(agent.t_check(), 6000)}\n\n"
                       "Fix it until the validator passes.")
                report, failed = agent.run(msg)
        elif a.no_swift:
            report = "A real scan needs Swift: the exact-reading pass works on CAD-exported meshes only."
            log(f"[run] {report}")
        else:
            if not a.live:          # the headless engine cannot fit scans: reopen as the page
                wb.close()
                wb = Workbench(stl, work, page=True, show_browser=not a.headless_browser)
            wb.narrate = False
            phase("swift", "Swift is reverse-engineering the scan")
            agent = Agent(wb, "scan", log=log, max_turns=a.max_turns, emit=emit)
            msg = (f"Scan: {stl.name}, opened in ScanRuler (millimetres).\n{chr(10).join(analysis.splitlines()[1:4])}\n\n"
                   "Start with the guide index, then work the scan-to-cad flow until the validator passes.")
            report, failed = agent.run(msg)
            failed = failed or not agent.passed

        r = wb.last_validation
        phase("deliver", "Writing the STEP and the report")
        if not failed and r and r["verdict"] == "PASS":
            log(wb.deliver(stl.stem))
            status, step = "PASS", work / f"{stl.stem}_CAD.step"
        else:
            status, step = "NOT VALIDATED", work / "best_attempt_CAD.step"
        md = [f"# {stl.stem}: reverse-engineering report", "",
              f"**{status}**: {step.name} (mode {mode}, {time.time() - t0:.0f} s)", "",
              "## Validation (independent)", "```", summary(r) if r else "no model was checked", "```", "",
              "## How it was built", report or "(no report)"]
        report_path = work / f"{stl.stem}_REPORT.md"
        report_path.write_text("\n".join(md), encoding="utf-8")
        log(f"[run] {status} -> {work}")
        emit("done", status=status, work=str(work), step=str(step) if step.exists() else None,
             report=str(report_path), seconds=round(time.time() - t0), validation=r)
        return 0 if status == "PASS" else 1
    except Exception as e:
        log(traceback.format_exc())
        emit("error", text=f"{type(e).__name__}: {e}")
        return 2
    finally:
        if wb:
            wb.close()
        logf.close()


if __name__ == "__main__":
    sys.exit(main())
