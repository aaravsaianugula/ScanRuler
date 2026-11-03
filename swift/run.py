# SPDX-License-Identifier: AGPL-3.0-only
"""Reverse-engineer a mesh/scan into validated CAD with Swift 1.5.

    python swift/run.py part.stl [--mode auto|cad|scan|organic] [--work DIR] [--max-turns 60] [--no-swift]
                                 [--headless-browser] [--live] [--events]

auto: a free-form model (figurine, sculpture: few planes or cylinders) goes the organic way - repaired into a
closed solid, written as a faceted STEP; a CAD-exported mesh (exact surfaces) goes the exact-reading way; a real scan is measured in the
ScanRuler page (fits with sigma evidence) by Swift. Either way the model is held to the independent
validator, and only a PASS is delivered as <part>_CAD.step; otherwise best_attempt_CAD.step, marked so.

--events  JSON lines on stdout (phase, log, page, turn, reasoning, content, tool, tool_result, image,
          validation, done, error) for the desktop app; the human log still goes to <work>/run.log.
--live    run in the ScanRuler page and hand its pairing link to the host (event "page") instead of
          opening a browser: the desktop app shows it as the live view, CAD meshes included.
"""
import argparse
import importlib.util
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

# module: pip package, for every route (the organic fallback closes any mesh: igl, pymeshlab; mech: OCP; bodies: manifold3d)
NEEDS = {"scipy": "scipy", "matplotlib": "matplotlib", "igl": "libigl", "pymeshlab": "pymeshlab", "manifold3d": "manifold3d", "OCP": "cadquery-ocp"}


def looks_like_cad(work):
    """A CAD tessellation: big planes fit to ~1e-5 mm and few faces per triangle (analyze.py's faces.json)."""
    f = json.loads((work / "data" / "faces.json").read_text())["faces"]
    n_tris = len(np.load(work / "data" / "faces.npy"))
    total = sum(x["area"] for x in f) or 1
    res = [x["res"] for x in f if x["kind"] == "plane" and x["area"] > 0.01 * total and x.get("res") is not None]
    return bool(res) and float(np.median(res)) < 1e-4 and len(f) < max(50, n_tris / 20)


def surface_shares(stl, samples=40000, candidates=1500, deg=8):
    """How much of the surface faces a few directions (planes: the six most populated, a face and its
    opposite as one) and the most of it on one great circle (cylinders round one axis), by area.
    Machined parts and their scans: planes >= 0.58 or cylinders >= 0.68; figurines: about 0.2-0.3 each."""
    from meshio_min import read_stl
    t = read_stl(stl)
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    a = np.linalg.norm(n, axis=1)
    keep = a > 0
    t, n, a = t[keep], n[keep] / a[keep, None], a[keep]
    rng = np.random.default_rng(0)
    pick = rng.choice(len(n), samples, p=a / a.sum())
    s, mid = n[pick], t[pick].mean(1)
    c = s[rng.choice(len(s), candidates, replace=False)]
    cos, left, planes, flat = np.cos(np.radians(deg)), np.ones(len(s), bool), 0.0, None
    for _ in range(6):
        best = c[(np.abs(c @ s[left].T) > cos).sum(1).argmax()]
        cover = left & (np.abs(s @ best) > cos)
        planes += cover.mean()
        left &= ~cover
        if flat is None:
            # the most populated flat level along it - a figurine's base, a part's biggest face - for measuring
            h = mid[cover] @ best
            level = np.bincount(np.round(h - h.min()).astype(int)).argmax() + h.min()
            on = np.abs(h - level) < 1
            nrm = best * np.sign((s[cover][on] @ best).sum())
            flat = dict(normal=nrm.round(3).tolist(), point=np.median(mid[cover][on], axis=0).round(2).tolist(), share=round(float(on.mean() * cover.mean()), 3))
    return float(planes), float((np.abs(c @ s.T) < np.sin(np.radians(deg))).mean(1).max()), flat


BAMBU_FACES = 5000      # a STEP Bambu Studio opens in about a minute
# Bambu Studio meshes a STEP at 0.003 mm (its default): a few hundred triangles per B-spline span however small,
# so a surfaced figurine's thousands of patches come to millions (Kurisu, 9.4k patches: 34M) - past this, the 3MF
BAMBU_TRIANGLES = 2_000_000

ORGANIC_REPORT = ("Organic route: islands removed and holes filled in ScanRuler, closed into a manifold solid (winding number + "
                  "marching cubes, sealed crevices kept as cavities), decimated keeping it closed (MeshLab), written as a faceted STEP solid.")


def organic_autopilot(wb, phase, log, emit):
    """The organic route without Swift: repair and close, the bodies shape is sure of, the check. True when it failed."""
    phase("organic", "Repairing the scan into a solid")
    log(wb.organic())
    if getattr(wb, "whole", None):          # bodies without Swift: pieces and rings, what shape says for sure
        phase("bodies", "Finding the bodies")
        try:
            log(wb.auto_bodies())
        except Exception:       # finding bodies never costs the delivery: the solid stays one body
            log("[run] finding the bodies failed, the solid stays one body:\n" + traceback.format_exc())
    phase("check", "Validating the solid against the scan")
    check = wb.check("organic")
    log(check)
    emit("validation", result=wb.last_validation, summary=check)
    return wb.last_validation is None or wb.last_validation["verdict"] != "PASS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stl")
    ap.add_argument("--mode", choices=["auto", "cad", "scan", "organic"], default="auto")
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
        log(f"[run] phase {name}: {label} ({time.time() - t0:.0f} s in)")

    t0 = time.time()
    emit("start", stl=str(stl), work=str(work), model=MODEL, swift=not a.no_swift)
    wb = None
    try:
        missing = [pip for mod, pip in NEEDS.items() if importlib.util.find_spec(mod) is None]
        if missing:     # said up front, not as an ImportError an hour into a run
            raise RuntimeError(f"Python packages missing in {sys.executable}: {' '.join(missing)}. Install them with: "
                               f"\"{sys.executable}\" -m pip install -r \"{Path(__file__).with_name('requirements.txt')}\"")
        phase("open", "Opening ScanRuler")
        on_link = (lambda url: emit("page", url=url)) if a.live else None
        wb = Workbench(stl, work, page=a.live, on_link=on_link)
        wb.emit, wb.narrate = emit, True
        phase("analyze", "Reading the mesh")
        n_tris = (stl.stat().st_size - 84) // 50
        planes, cylinders, flat = surface_shares(wb.input)
        organic = planes < 0.4 and cylinders < 0.45
        log(f"[run] surface on planes {planes:.2f}, on cylinders {cylinders:.2f}: " + ("free-form (organic)" if organic else "engineered"))
        mech_step, report = None, ""
        if a.mode in ("cad", "scan") or (a.mode == "auto" and not organic):
            # deterministic first: planes, cylinders, cones and rounds read off the mesh into one exact solid; a pass
            # needs neither the exact-reading analyzer (minutes on a dense scan) nor Swift
            phase("mech", "Fitting planes, cylinders, cones and rounds")
            try:
                mech_step, mrep = wb.mech(log=log)
                if mech_step:
                    report = (f"Mech route: {mrep['solid']['faces']} analytic faces {mrep['solid']['kinds']}, "
                              f"{mrep['step_kb']} kB, validated against the mesh (scan mode).")
                    emit("validation", result=wb.last_validation, summary=summary(wb.last_validation))
            except Exception:
                log("[run] the mech route failed:\n" + traceback.format_exc())
        if mech_step:
            analysis = f"{n_tris} triangles, engineered: read by the mech route\n\n\n"
            mode = a.mode if a.mode != "auto" else "scan"
        elif a.mode == "auto" and organic:
            analysis = f"{n_tris} triangles, free-form: {planes:.0%} of the surface on planes, {cylinders:.0%} on cylinders of one axis\n\n\n"
            mode = "organic"
        elif n_tris > 400_000 and a.mode != "cad":
            # shortcut: the exact-reading analyzer takes minutes on a scan this big and only feeds CAD meshes;
            # a CAD export that dense would need --mode cad
            analysis = f"{n_tris} triangles: a scan (exact reading skipped)\n\n\n"
            mode = a.mode if a.mode != "auto" else "scan"
        else:
            analysis = wb.script("analyze")
            mode = a.mode if a.mode != "auto" else ("cad" if looks_like_cad(work) else "scan")
        log(analysis)
        emit("mode", mode=mode)
        log(f"[run] mode: {mode}")
        failed, agent = not mech_step, None
        paged, tried_organic = a.live, False

        if mech_step:
            pass
        elif mode == "cad":
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
        elif mode == "organic" and (a.no_swift or a.mode == "organic"):
            if not a.live:
                wb.close()
                wb = Workbench(stl, work, page=True, show_browser=not a.headless_browser)
            wb.emit, wb.narrate = emit, True
            failed, tried_organic = organic_autopilot(wb, phase, log, emit), True
            report = ORGANIC_REPORT
        elif a.no_swift:
            report = "Without Swift, a scan is not read into features: the exact-reading pass works on CAD-exported meshes only."
            log(f"[run] {report}")
        else:
            if not a.live:          # the headless engine cannot fit scans: reopen as the page
                wb.close()
                wb = Workbench(stl, work, page=True, show_browser=not a.headless_browser)
                paged = True
            wb.emit, wb.narrate = emit, False
            phase("swift", "Swift is reverse-engineering the scan")
            agent = Agent(wb, "scan", log=log, max_turns=a.max_turns, emit=emit)
            msg = (f"Scan: {stl.name}, opened in ScanRuler (millimetres).\n{chr(10).join(analysis.splitlines()[1:4])}\n\n"
                   "Start with the guide index, then work the scan-to-cad flow until the validator passes.")
            if mode == "organic":
                msg = (f"Scan: {stl.name}, opened in ScanRuler (millimetres). {analysis.splitlines()[0]}: classified organic. "
                       "Look at it, call organic_route, decide by looking which parts are separate bodies (propose_bodies, then group_bodies to fix what the picture shows wrong), then measure the analytic features it has with their sigma. "
                       f"The biggest flat level of its surface (a base, if it stands on one) has normal {flat['normal']} through "
                       f"{flat['point']} ({flat['share']:.0%} of the surface): that normal is the up/down axis, and that is where to start.")
            report, failed = agent.run(msg)
            failed = failed or not agent.passed

        # a run ends NOT VALIDATED only when every route failed: the organic autopilot closes any scan or mesh into a solid
        if failed and not tried_organic:
            log("[run] no pass yet: delivering the autopilot's organic solid")
            try:
                if not paged:
                    wb.close()
                    wb = Workbench(stl, work, page=True, show_browser=not a.headless_browser)
                wb.emit, wb.narrate = emit, True
                if not organic_autopilot(wb, phase, log, emit):
                    failed, report = False, (report + "\n\n" if report else "") + "No feature pass; delivered instead: " + ORGANIC_REPORT
            except Exception:   # the safety net never costs the run what it already has
                log("[run] the organic fallback failed:\n" + traceback.format_exc())

        r = wb.last_validation
        organic = bool(r and r.get("mode") == "organic")
        if organic and mode != "organic":       # delivered by the organic fallback
            mode = "organic"
            emit("mode", mode=mode)
        phase("deliver", "Writing the STEP and the report")
        if not failed and r and r["verdict"] == "PASS" and organic:
            status, step = "PASS", wb.organic_step
        elif not failed and r and r["verdict"] == "PASS":
            if not mech_step:
                log(wb.deliver(stl.stem))
            status, step = "PASS", work / f"{stl.stem}_CAD.step"
        elif organic:
            status, step = "NOT VALIDATED", wb.organic_step
        else:
            status, step = "NOT VALIDATED", work / "best_attempt_CAD.step"
        # the CAD deliverable: the solid as smooth B-spline surfaces (not a triangle a face) when they pass the
        # same validator against the original mesh; else the faceted solid stays the deliverable
        smooth_r = None
        if status == "PASS" and organic:
            phase("surface", "Fitting smooth CAD surfaces over the solid")
            try:
                smooth, srep, smooth_r = wb.smooth()
                log(srep)
                report = (report + "\n\n" if report else "") + "Smooth CAD surfaces:\n" + srep
                if smooth:
                    step = smooth
                else:
                    report += f"\nThe smooth surfaces did not pass: the deliverable stays the faceted solid {step.name}."
            except Exception:
                log("[run] surfacing failed, the faceted solid stays the deliverable:\n" + traceback.format_exc())
        # the STEP read as Bambu Studio reads it: a slicer gets the solid when it opens whole, else the 3MF
        imported = None
        if step.exists():
            phase("import", "Opening the STEP the way Bambu Studio does")
            try:
                from step_faceted import check_import
                # past BAMBU_FACES, or smooth B-spline surfaces of a scanned shape (a slicer meshes those at 0.003 mm:
                # tens of millions of triangles, measured), it goes to the slicer as the 3MF anyway: read it with a
                # coarse mesh, not Bambu's
                text = step.read_text(encoding="ascii", errors="ignore")
                many = text.count("ADVANCED_FACE(") > BAMBU_FACES or "B_SPLINE_SURFACE" in text
                imported = check_import(step, deflection=0.1 if many else 0.003)
                if many:
                    imported["triangles"], imported["heavy"] = None, True
            except Exception:
                log("[run] the import check failed:\n" + traceback.format_exc())
            log(f"[run] STEP import check: {imported}")
        mf = wb.organic_step.with_suffix(".3mf") if organic else step.with_suffix(".3mf")
        # Bambu Studio 2.x (OCCT 7.6) takes ~35 s + 5 ms a face (measured: 20k faces 131 s, 40k 227 s) and meshes
        # B-splines into millions of triangles: a figurine goes to the slicer as its 3MF, the same closed solid; a
        # CAD STEP of a few hundred faces as itself
        light = (imported and not imported.get("heavy") and imported.get("faces", 0) <= BAMBU_FACES
                 and (imported.get("triangles") or 0) <= BAMBU_TRIANGLES)
        slicer = step if imported and imported["ok"] and light else mf if mf.exists() else None
        md = [f"# {stl.stem}: reverse-engineering report", "",
              f"**{status}**: {step.name} (mode {mode}, {time.time() - t0:.0f} s)"
              + (f"; for a slicer, the same triangles: {mf.name}" if mf.exists() else ""), "",
              "## Validation (independent, against the original mesh)", "```", summary(r) if r else "no model was checked", "```", "",
              *(["## Smooth surfaces against the original mesh", "```", summary(smooth_r), "```", ""] if smooth_r else []),
              "## STEP opened as Bambu Studio opens it", (f"{'OK' if imported['ok'] else 'PROBLEMS'}: {imported['solids']} solids, "
              f"{imported.get('faces', 0)} faces, " + (f"{imported['triangles']} triangles at Bambu's 0.003 mm, " if imported.get("triangles") else "") + f"{imported['seconds']} s here"
              + ("" if light else f" (too heavy for a slicer: it gets {mf.name}, the same solid as triangles)") + "".join(f"\n- {p}" for p in imported["problems"])) if imported
              else "not checked (no OCCT in this Python)", "",
              "## How it was built", report or "(no report)"]
        if organic and getattr(wb, "bodies", None):
            md += ["", "## Bodies (a named solid each in the STEP)"] + [f"- {b['name']}: {b['volume']} mm3, box {b['lo']}..{b['hi']}" for b in wb.bodies]
        report_path = work / f"{stl.stem}_REPORT.md"
        report_path.write_text("\n".join(md), encoding="utf-8")
        log(f"[run] {status} -> {work}")
        emit("done", status=status, work=str(work), step=str(step) if step.exists() else None,
             print=str(slicer) if slicer else None, imported=imported,
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
