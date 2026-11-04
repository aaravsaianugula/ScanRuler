# SPDX-License-Identifier: AGPL-3.0-only
"""The reverse-engineering workbench: the skill's exact-reading scripts, headless ScanRuler
(over scanruler-mcp) and the independent validator, as plain calls. The Swift agent's tools
and the deterministic autopilot both go through this."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from mcp_client import REPO, ScanRulerMCP
from meshio_min import read_stl
from validate import summary, validate, validate_organic

SCRIPTS = REPO / "plugins" / "reverse" / "skills" / "cad-stl-reverse-engineering" / "scripts"
ALLOWED = {"analyze", "frame", "slabs", "revolve", "holes", "blends", "compare", "helix", "profile", "rz", "centers",
           "two_way", "slice", "vslice"}


def clip(s, n=6000):
    """Head and tail of a long output: the model's context is 49k tokens, shared."""
    return s if len(s) <= n else s[: n // 3] + f"\n... [{len(s) - n} chars cut] ...\n" + s[-2 * n // 3:]


class Workbench:
    def __init__(self, stl, work=None, python=sys.executable, page=False, show_browser=True, on_link=None):
        """page: False = headless engine; True = the ScanRuler page, opened in Edge/Chrome (page.mjs),
        or, with on_link, handed to a host that shows it (the desktop app's live view)."""
        self.on_link = on_link
        self.stl = Path(stl).resolve()
        self.work = Path(work).resolve() if work else self.stl.with_name(self.stl.stem + "_work")
        (self.work / "data").mkdir(parents=True, exist_ok=True)
        self.input = self.work / "input.stl"           # never modify the original
        if not self.input.exists():
            shutil.copyfile(self.stl, self.input)
        self.python = python
        self.browser = None
        self.page = page
        if page:
            self._open_page(show_browser)
        else:
            self.mcp = ScanRulerMCP(self.work)
        self.built = []                                 # step files of the current build, in order
        self.emit = lambda *a, **k: None                # the desktop app's event stream (run.py --events)
        self.narrate = False                            # report scripts/builds as actions (the autopilot's; Swift's own calls report themselves)
        self.last_validation = None
        self.main_axis, self.body = "z", "Z slab 0"

    def _open_page(self, show):
        """The full app (scan fitting needs the page): the local build served by scanruler-mcp,
        opened in Edge/Chrome and paired, the scan opened in it."""
        import secrets
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
        token = secrets.token_urlsafe(16)
        cfg = self.work / ".mcp-config"; cfg.mkdir(exist_ok=True)
        self.mcp = ScanRulerMCP(self.work, headless=False, extra=["--serve", str(REPO / "dist"), "--port", str(port)],
                                env={"SCANRULER_MCP_TOKEN": token, "SCANRULER_MCP_CONFIG_DIR": str(cfg)})
        link = f"http://127.0.0.1:{port}/#agent={port}:{token}"
        for attempt in range(2):        # a busy machine can keep a browser from pairing in time: start it again once
            if self.on_link:
                self.on_link(link)
            else:
                if self.browser:
                    self.browser.kill()
                self.browser = subprocess.Popen(["node", str(REPO / "swift" / "page.mjs"), link] + ([] if show else ["--headless"]),
                                                stdin=subprocess.PIPE, cwd=str(REPO), text=True)
            txt, _, err = self.mcp.call("scanruler_wait", {"for": "page", "timeoutSeconds": 180})
            if not err and '"connected":true' in txt.replace(" ", ""):
                break
        else:
            raise RuntimeError(f"the ScanRuler page did not connect: {txt[:500]}")
        txt, err = self.batch([{"tool": "scan_open", "arguments": {"path": str(self.input), "units": "mm"}}])
        if err or '"ok":false' in txt:
            raise RuntimeError(f"scan_open failed: {txt[:800]}")

    def close(self):
        self.mcp.close()
        if self.browser:
            try:
                self.browser.stdin.close(); self.browser.wait(timeout=15)
            except Exception:
                self.browser.kill()

    # -- scripts ---------------------------------------------------------
    def _act(self, name, args, run):
        if not self.narrate:
            return run()
        self.emit("tool", name=name, args=json.dumps(args))
        out = run()
        self.emit("tool_result", name=name, text=str(out)[:1500])
        return out

    def script(self, name, args=()):
        return self._act("run_script", {"name": name, "args": [str(a) for a in args]}, lambda: self._script(name, args))

    def _script(self, name, args=()):
        name = name.removesuffix(".py")
        if name not in ALLOWED:
            return f"refused: {name} is not one of {sorted(ALLOWED)}"
        args = [str(a) for a in args]
        if name in ("analyze", "frame") and not any(a.endswith(".stl") for a in args):
            args.insert(0, str(self.input))
        if name == "compare" and not any(a.endswith(".stl") for a in args):
            args = [str(self.input), str(self.work / "model.stl"), *args]
        try:
            p = subprocess.run([self.python, "-I", str(SCRIPTS / f"{name}.py"), *args], cwd=self.work,
                               capture_output=True, text=True, timeout=900, encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            return f"{name}: timed out after 900 s"
        out = p.stdout + (("\nSTDERR:\n" + p.stderr[-3000:]) if p.returncode else "")
        if name == "frame" and "local.stl" in out and (self.work / "data" / "local.stl").exists():
            out += "\n[workbench] the part is tilted: analyze data/local.stl and build on it; add the printed moveBody last."
        return clip(f"$ {name}.py {' '.join(args)}  (exit {p.returncode})\n{out}")

    # -- ScanRuler -------------------------------------------------------
    def batch(self, calls=None, file=None, **opts):
        args = dict(opts)
        if file:
            args["file"] = str((self.work / file).resolve())
        else:
            args["calls"] = calls
        txt, _, err = self.mcp.call("scanruler_batch", args)
        return txt, err

    def rebuild(self, files):
        """Clear the part and run the step files in order. Returns a short report."""
        return self._act("rebuild", {"files": list(files)}, lambda: self._rebuild(files))

    def _rebuild(self, files):
        self.batch([{"tool": "reverse_feature_clear", "arguments": {}}])
        self.built, report = [], []
        for f in files:
            path = (self.work / f)
            if not path.exists():
                report.append(f"{f}: no such file"); continue
            if not json.loads(path.read_text()).get("steps", True):
                report.append(f"{f}: no steps"); continue
            txt, err = self.batch(file=str(path), stopOnError=False, rollback=False)
            try:
                res = json.loads(txt)
                bad = [r for r in res.get("results", []) if not r.get("ok")]
                notes = [r["result"].get("note") for r in res.get("results", []) if r.get("ok") and isinstance(r.get("result"), dict) and r["result"].get("note")]
                report.append(f"{f}: {res.get('ran', '?')} steps, {len(bad)} failed" +
                              "".join(f"\n   FAILED {b.get('tool')}: {json.dumps(b.get('error'))[:400]}" for b in bad[:8]) +
                              "".join(f"\n   note: {str(n)[:300]}" for n in notes[:5]))
            except json.JSONDecodeError:
                report.append(f"{f}: {txt[:600]}")
            self.built.append(f)
        txt, _ = self.batch([{"tool": "reverse_feature_list", "arguments": {}}])
        return "\n".join(report) + (f"\nfeatures built: {txt.count(chr(34) + 'kind' + chr(34))}" if txt else "")

    def export_model(self, chord=0.005):
        txt, err = self.batch([{"tool": "reverse_export_stl", "arguments": {"chord": chord, "path": str(self.work / "model.stl")}}])
        if err or '"ok":false' in txt:
            return f"export failed: {txt[:800]}"
        return "model.stl written"

    def analyze_model(self):
        """analyze.py on the model mesh, into data_model/, for the validator's dimension check."""
        return self.script("analyze", [str(self.work / "model.stl"), "--data", "data_model"])

    def compare(self):
        return self.script("compare", ["--grid", "0.1", "--faces", "data/faces.pkl", "--plot", "cmp.png",
                                       "--suggest", "data/suggest.json", "--main", self.main_axis, "--body", self.body])

    def check(self, mode="cad", noise=None):
        """Export, analyze the model, run the independent validator. The one judge of done."""
        self.last_validation = None
        if mode == "organic":
            if not getattr(self, "reduced", None) or not self.reduced.exists():
                return "no organic solid yet: run the organic route first"
            r = validate_organic(self.input, self.repaired, self.reduced, closed=self.closed, grid=self.grid,
                                 bodies=getattr(self, "delivered", None))
            self.last_validation = r
            (self.work / "validation.json").write_text(json.dumps(r, indent=1))
            return summary(r)
        msg = self.export_model()
        if not msg.endswith("written"):
            return msg
        # the validator reads the input as it is (world frame) from a data dir of its own: the scripts'
        # data/ may hold a local frame (frame.py) or whatever the last analyze was run on
        if mode == "cad":
            if not (self.work / "data_input" / "faces.json").exists():
                self.script("analyze", [str(self.input), "--data", "data_input"])
            self.analyze_model()
            r = validate(self.input, self.work / "model.stl", self.work / "data_input", self.work / "data_model", mode, noise)
        else:
            ref = self.scan_as_posed()
            if isinstance(ref, str):
                return ref
            r = validate(ref, self.work / "model.stl", None, None, mode, noise)
        self.last_validation = r
        (self.work / "validation.json").write_text(json.dumps(r, indent=1))
        return summary(r)

    def organic(self, max_triangles=150000, grid=None):
        """The organic route (a figurine, a sculpture - no analytic faces to rebuild): repair the scan
        in ScanRuler's mesh tools (islands off, holes filled: small ones curved, the rest flat), close what
        the filler could not (organic_solid: winding number + marching cubes on a grid mm - by default half
        the scan's own spacing, within 1/1650..1/1100 of the part - pieces turned to face out, sealed crevices kept as cavities, sub-grid specks dropped; a scan that
        encloses no volume is thickened in ScanRuler first), reduce it to max_triangles keeping it closed, and
        write it as a closed STEP solid. The
        validator holds it to the scan's noise, whatever the caller asks for. Needs the page."""
        from organic_solid import close_mesh, grid_for, orient, pieces, reduce_mesh
        from meshio_min import write_stl
        from step_faceted import write_step
        stem = self.stl.stem
        self.repaired, self.reduced = self.work / f"{stem}_repaired.stl", self.work / f"{stem}_reduced.stl"
        self.organic_step = self.work / f"{stem}_mesh_solid.step"
        closed = self.work / f"{stem}_closed.stl"
        diag = float(np.linalg.norm(np.ptp(read_stl(self.input).reshape(-1, 3), axis=0)))
        report = []
        self._repair(diag, report)
        # every piece facing out, in the delivered repaired scan too: the validator reads it as written
        scan = np.concatenate(orient(pieces(read_stl(self.repaired))))
        write_stl(self.repaired, scan)
        grid = grid or grid_for(scan)
        solid = close_mesh(scan, grid)
        if not len(solid):
            # the scan encloses no volume - one side of a part, a relief, a face: ScanRuler gives it thickness
            report.append("the repaired scan encloses no volume (a single open sheet): thickening it")
            if self._mesh_step({"tool": "mesh_shell", "arguments": {"mode": "thicken", "side": "in"}}, report) is None or \
                    self._mesh_step({"tool": "mesh_export", "arguments": {"part": "edited", "path": str(self.repaired)}}, report) is None:
                report.append("FAILED: the scan is a single open sheet that encloses no volume, and ScanRuler's thickening did not "
                              "go through (above). Next: organic_route again (a page hiccup clears), or thicken it with scanruler "
                              "mesh_shell {mode: thicken} and then organic_route")
                return "\n".join(report)
            scan = np.concatenate(orient(pieces(read_stl(self.repaired))))
            write_stl(self.repaired, scan)
            solid = close_mesh(scan, grid)
            if not len(solid):
                report.append("FAILED: still no volume after thickening: the thickened scan has no inside a winding number finds "
                              "(sheets crossing each other everywhere?). Next: look at it (body_pieces), remove the stray sheets with "
                              "scanruler mesh_* tools, then organic_route")
                return "\n".join(report)
        write_stl(closed, solid)
        self.closed, self.grid = closed, grid      # the validator tells bridged gaps from them
        report.append(f"closed on a {grid:.3g} mm grid: {len(solid)} triangles, {closed.name}")
        reduced = reduce_mesh(solid, max_triangles)
        report.append(f"reduced to {len(reduced)} triangles (MeshLab quadric, topology kept, thin walls kept from cutting through each other)")
        reduced = self._fewest_faces(reduced, closed, grid, report)
        write_stl(self.reduced, reduced)
        self.look = self.work / f"{stem}_look.stl"         # what the part finders read: evenly coarse
        write_stl(self.look, reduce_mesh(solid, max_triangles, safe=False))
        report.append(f"delivered at {len(reduced)} faces: {self.reduced.name}")
        info = write_step(self.organic_step, read_stl(self.reduced), stem)
        report.append(f"STEP solid: {self.organic_step.name}, {info['faces']} faces, {info['solids']} solids, {info['voids']} cavities"
                      + (f"; {info['faces_left_out']} faces of a cavity no solid holds left out" if info["faces_left_out"] else ""))
        self.whole, self.pieces = self.reduced, None      # what group_bodies starts from, every time
        self.delivered = [(stem, reduced)]                 # the bodies as written: what the validator's solid gate judges
        self.rods, self.rings = [], []
        self._bodies([(stem, read_stl(self.reduced))], [], [])
        return "\n".join(report)

    def mech(self, log=print):
        """The mech route (swift/mech.py, no page, no Swift): the input read as planes, cylinders, cones and rounds,
        built into one exact solid, written to <stem>_CAD.step and judged by the validator's scan mode against
        the input. Returns (step or None, report); a failing STEP is not left behind."""
        import mech
        out = self.work / f"{self.stl.stem}_CAD.step"
        rep = mech.run(self.input, out, log=log)
        self.last_validation = rep.get("validation")
        (self.work / "mech.json").write_text(json.dumps(rep, indent=1, default=str))
        if rep["verdict"] != "PASS":
            out.unlink(missing_ok=True)
            return None, rep
        return out, rep

    def smooth(self, patches=3000, **opts):
        """The delivered bodies as smooth CAD surfaces (autosurface: four-sided B-spline patches with shared
        edges, one watertight solid per body), written to <stem>_solid.step and judged like the faceted solid:
        the STEP read back and meshed by OCCT, through the validator against the original mesh. Returns
        (step or None, report, validation)."""
        from autosurface import kernel_rejects, surface, write_step
        from validate import estimate_noise
        from meshio_min import write_stl
        from step_faceted import check_import, indexed
        out = self.work / f"{self.stl.stem}_solid.step"
        report, bodies = [], []
        # patches refine until the mesh's local mean misses them by a third of the detail check's threshold (the scan's
        # noise, at least 0.05 mm): a precise scan of a machined part is held tighter than a noisy figurine
        tol = 0.35 * max(0.05, estimate_noise(read_stl(self.input), 0.3))
        for name, tris in getattr(self, "delivered", None) or []:
            v, f = indexed(np.asarray(tris, float))
            bodies.append((name, surface(v, f, patches * len(f) // max(sum(len(t) for _, t in self.delivered), 1) + 1,
                                         **{"tol": tol, **opts}, judge=lambda sf: kernel_rejects(sf, self.work / f"_kernel_check_{self.stl.stem}.step"),
                                         log=lambda s, n=name: report.append(f"  {n}: {s}"))))
        if not bodies:
            return None, "no delivered solid to surface", None
        info = write_step(out, bodies)
        report.append(f"smooth STEP: {out.name}, {info['faces']} B-spline faces, {info['solids']} solids")
        # the STEP itself, read back and meshed by OCCT as a CAD program or slicer does (0.05 mm, under the scan noise
        # and any print), judged against the original mesh like the faceted solid: detail lost, deviation, closed solids
        imp = check_import(out, meshes=True, deflection=0.05)
        if imp is None:
            return None, "\n".join(report + ["no OCCT here to read the STEP back: not delivered"]), None
        report.append(f"read back in OCCT: {'OK' if imp['ok'] else 'PROBLEMS ' + '; '.join(imp['problems'])}, "
                      f"{imp['solids']} solids, {imp['seconds']} s")
        pieces = imp["meshes"]
        tess = self.work / f"{self.stl.stem}_solid_readback.stl"
        write_stl(tess, np.concatenate([V[F] for V, F in pieces]))
        r = validate_organic(self.input, self.repaired, tess, closed=self.closed, grid=self.grid,
                             bodies=[(f"solid {i + 1}", V[F]) for i, (V, F) in enumerate(pieces)])
        tess.unlink(missing_ok=True)
        report.append(f"smooth surfaces against the original mesh: {r['verdict']}")
        ok = imp["ok"] and r["verdict"] == "PASS"
        if not ok:              # never left where it could pass for the deliverable
            out.unlink(missing_ok=True)
        return (out if ok else None), "\n".join(report), r

    def _fewest_faces(self, tris, closed, grid, report, floor=20000):
        """A slicer reads a faceted STEP at about a millisecond a face (Bambu Studio: 200k faces, over three
        minutes): halving from the full reduction, the fewest faces that still pass the validator against the
        original mesh, its detail check included. The first halving that fails stops it."""
        from meshio_min import write_stl
        from organic_solid import reduce_mesh
        best, trial = tris, self.work / f"{self.stl.stem}_trial.stl"
        while len(best) // 2 >= floor:
            t = reduce_mesh(best, len(best) // 2)
            if len(t) > 0.9 * len(best):        # thin walls kept dense: this is as few as it goes without cutting through
                break
            write_stl(trial, t)
            r = validate_organic(self.input, self.repaired, trial, closed=closed, grid=grid)
            report.append(f"  {len(t)} faces: {r['verdict']}, detail lost {r['detail']['largest_lost_mm2']} mm2 of "
                          f"{r['detail']['allowed_mm2']} allowed" + ("" if r["verdict"] == "PASS" else f" ({', '.join(k for k, v in r['checks'].items() if not v)}): keeping {len(best)}"))
            if r["verdict"] != "PASS":
                break
            best = t
        trial.unlink(missing_ok=True)
        return best

    def _mesh_step(self, c, report, tries=2):
        """One ScanRuler mesh call, at its best: the answer, {} for "nothing to do", None when it did not go
        through - reported either way. A bridge or page hiccup (no answer to read) is tried once more."""
        for _ in range(tries):
            try:
                txt, err = self.batch([c])      # one at a time: each is a long job on a big scan, and each answer is worth reading
                res = json.loads(txt)["results"][0]
                break
            except Exception as e:
                res = {"ok": False, "error": {"message": f"no answer from ScanRuler ({type(e).__name__}: {str(e)[:300]})"}}
        msg = str((res.get("error") or {}).get("message", ""))
        # "No hole/piece ... smaller than that": nothing to do, not a failure
        ok = res.get("ok") or msg.startswith(("No hole", "No piece"))
        # an answer's status line says what it did; a failure is kept whole, it is what the reader needs
        body = (res.get("result") or {}).get("status") or "" if res.get("ok") else msg
        report.append(f"{c['tool']}: {'ok' if ok else 'did not go through:'} {body[:600]}")
        return res.get("result") or {} if ok else None

    def _repair(self, diag, report):
        """ScanRuler's repair, every step best-effort: one that refuses, finds nothing or errors is reported
        and the next runs. self.repaired is then the edited copy as exported, or, with no edited copy (nothing
        needed repair) or no export, the scan as it is: close_mesh closes what the filler left open."""
        try:      # a second run starts from the scan again (fails harmlessly on a first)
            self.batch([{"tool": "mesh_discard_edited", "arguments": {}}])
        except Exception:
            pass
        # the panel keeps its sizes, so the curved fill is given the size it had the first time
        curved = {"patch": "curved", **({"size": self.hole_size} if getattr(self, "hole_size", None) else {})}
        for c in [{"tool": "mesh_remove_islands", "arguments": {}},
                  {"tool": "mesh_fill_holes", "arguments": curved},
                  {"tool": "mesh_fill_holes", "arguments": {"patch": "flat", "size": round(diag, 1)}}]:
            res = self._mesh_step(c, report)
            if res is not None and c["arguments"] is curved:
                self.hole_size = getattr(self, "hole_size", None) or (res.get("edit") or {}).get("maxSize")
        Path(self.repaired).unlink(missing_ok=True)      # never a copy left from an earlier run
        export = {"tool": "mesh_export", "arguments": {"part": "edited", "path": str(self.repaired)}}
        if self._mesh_step(export, report) is None or not Path(self.repaired).exists():
            shutil.copyfile(self.input, self.repaired)
            report.append(f"carried on from the scan as it is (no edited copy to export): {Path(self.repaired).name}")

    def _bodies(self, bodies, dropped, cuts):
        from bodies import _volume
        indexed_bodies = bodies
        bodies = [(n, b[0][b[1]] if isinstance(b, tuple) else b) for n, b in bodies]
        self.bodies = [dict(name=n, volume=round(abs(_volume(t)), 1), triangles=len(t), lo=t.reshape(-1, 3).min(0).round(1).tolist(),
                            hi=t.reshape(-1, 3).max(0).round(1).tolist()) for n, t in bodies]
        from bodies import draw
        png = self.work / "bodies.png"
        t = np.concatenate([b for _, b in bodies])
        lab = np.repeat(np.arange(len(bodies)), [len(b) for _, b in bodies]) + (1 if len(bodies) > 1 else 0)
        # each name at the body's own surface point nearest its centre (a ring's centre is in its bore)
        notes = [(n, b.reshape(-1, 3)[np.linalg.norm(b.reshape(-1, 3) - b.reshape(-1, 3).mean(0), axis=1).argmin()].tolist()) for n, b in bodies]
        draw(t, lab, notes, png, "bodies as written to the STEP")
        from meshio_min import write_3mf
        write_3mf(self.organic_step.with_suffix(".3mf"), indexed_bodies)      # the same bodies for a slicer, a few MB
        self.emit("bodies", bodies=self.bodies, dropped=dropped, cuts=cuts, image=str(png))

    def body_pieces(self):
        """The organic solid's closed pieces, largest first, and pieces.png (each piece coloured, numbered)."""
        from bodies import describe, survey
        if not getattr(self, "whole", None):
            return "no organic solid yet: run the organic route first"
        self.pieces = survey(read_stl(self.whole), self.grid, self.work / "pieces.png")
        return describe(self.pieces, self.grid) + f"\n\npicture: {self.work / 'pieces.png'} (view_image) - the listed pieces numbered at their box centres"

    def find_parts(self, near=None, within=40.0):
        """Parts the solid may hold fused or apart, found by shape (rings.py): rods (staffs, shafts, pins)
        and flat rings and gears - over the whole solid, or only within within mm of near (a part seen in a
        picture, looked for where it is). Candidates add up over calls, numbered on; a repeat is not added."""
        from bodies import _pieces
        from rings import describe, describe_rods, draw_parts, find, find_rods
        if not getattr(self, "whole", None):
            return "no organic solid yet: run the organic route first"
        v, f, of, *_ = _pieces(read_stl(self.whole))
        t, ot = v[f], of
        look = getattr(self, "look", None)
        if look and Path(look).exists():
            # rings read the evenly decimated copy (flat faces grow whole there); a face's piece is that of the
            # solid's face nearest it. Rods read the solid: they follow its own surface further along
            from scipy.spatial import cKDTree
            t = read_stl(look)
            ot = of[cKDTree(v[f].mean(1)).query(t.mean(1))[1]]
        near_of = lambda x: np.ones(len(x), bool) if near is None else np.linalg.norm(x.mean(1) - np.asarray(near, float), axis=1) < float(within)
        sel, sel_s = near_of(t), near_of(v[f])
        if sel.sum() < 50 or sel_s.sum() < 50:
            return f"nothing of the solid within {within} mm of {near}"
        # rings on a coarse copy too: a thin gear fused to the figure breaks into sub-60 mm2 patches on a fine, noisy
        # mesh (Kurisu: 2 of 5 gears seen only at ~25k faces, not on the 150k look copy nor an 80k solid)
        rods = find_rods(v[f][sel_s], of[sel_s], solid=v[f])
        rings = find(t[sel], ot[sel])
        if len(f) > 30000:
            from organic_solid import reduce_mesh
            from scipy.spatial import cKDTree
            tc = reduce_mesh(v[f], 25000, safe=False)
            oc = of[cKDTree(v[f].mean(1)).query(tc.mean(1))[1]]
            sc = near_of(tc)
            rings += find(tc[sc], oc[sc])
        else:
            rings += find(v[f][sel_s], of[sel_s])
        self.rods, self.rings = getattr(self, "rods", None) or [], getattr(self, "rings", None) or []
        same_rod = lambda a, b: abs(np.dot(a["axis"], b["axis"])) > 0.995 and abs(a["radius"] - b["radius"]) < 0.2 * b["radius"] and \
            np.linalg.norm(np.cross(np.subtract(a["centre"], b["centre"]), b["axis"])) < b["radius"]
        same_ring = lambda a, b: abs(np.dot(a["normal"], b["normal"])) > 0.995 and np.linalg.norm(np.subtract(a["centre"], b["centre"])) < 2
        new_rods = [r for r in rods if not any(same_rod(r, o) for o in self.rods)]
        new_rings = []
        for r in rings:
            if not any(same_ring(r, o) for o in self.rings + new_rings):
                new_rings.append(r)
        for r in new_rods:
            r["id"] = len(self.rods); self.rods.append(r)
        for r in new_rings:
            r["id"] = len(self.rings); self.rings.append(r)
        draw_parts(t, self.rods, self.rings, self.work / "parts.png")
        where = "" if near is None else f" within {within} mm of {list(near)}"
        return (f"new{where}:\n" + describe_rods(new_rods) + "\n" + describe(new_rings)
                + f"\n({len(self.rods)} rods and {len(self.rings)} rings found so far, all drawn)"
                + f"\n\npicture: {self.work / 'parts.png'} - each candidate in its own colour, labelled")

    def propose_bodies(self):
        """The finders' bodies, applied (and drawn) for Swift to look at and change, or delivered as they are
        without Swift: every piece of 1 % of the volume or more that is no sheet a body of its own (the
        largest "Model"), the rest joining the nearest; cut out of the pieces they are fused into: each ring
        or gear seen over 60 % of its turn, each straight rod 15 radii long or more (a staff, a shaft; a
        leg is round too, but not that straight that long) and a stand the solid stands on (at its top face)."""
        from bodies import _pieces, survey
        from rings import find_stand
        whole = read_stl(self.whole)
        self.pieces = survey(whole, self.grid, self.work / "pieces.png")
        self.rods, self.rings = [], []
        self.find_parts()
        big = [p for p in self.pieces if p["share"] >= 0.01 and p["thickness"] >= 2 * self.grid]
        groups = [{"name": "Model" if k == 0 else f"Part {k + 1}", "pieces": [p["id"]]} for k, p in enumerate(big)]
        fused = lambda ids, vol: [i for i in ids if self.pieces[i]["volume"] > 2 * vol]     # a part that is a piece of its own is a body already
        cuts = []
        for r in (x for x in self.rings if x["coverage"] >= 0.6):
            on = fused(r["pieces"], np.pi * (r["tip"] ** 2 - r["bore"] ** 2) * r["thickness"])
            if on:
                cuts.append({"name": f"Ring {sum(c['kind'] == 'ring' for c in cuts) + 1}", "kind": "ring", "ring": r["id"], "pieces": on})
        for r in (x for x in self.rods if x["length"] >= 15 * x["radius"] and x["coverage"] >= 0.8):
            on = fused(r["pieces"], np.pi * r["radius"] ** 2 * r["length"])
            if on:
                name = f"Rod {sum(c['kind'] == 'rod' for c in cuts) + 1}"
                cuts.append({"name": name, "kind": "rod", "rod": r["id"], "pieces": on})
                cuts += [{"name": name, "kind": "box", "centre": hd["centre"], "size": hd["size"], "axes": hd["axes"], "pieces": on}
                         for hd in r.get("heads", []) if hd["clean"]]
        v, f, of, *_ = _pieces(whole)
        stand = find_stand(v[f][of == 0]) if big else None
        if stand:
            cuts.append({"name": "Stand", "kind": "plane", "point": stand["point"], "normal": stand["normal"], "pieces": [big[0]["id"]]})
        self.proposal = dict(groups=groups, cuts=cuts, stand=stand)
        out = self.group_bodies(groups, (), "nearest", cuts)
        todo = ["rename every body for what it is (Model -> the figure's or part's name, Rod 1 -> staff, shaft, pin ...)"]
        for r in self.rods:
            for hd in r.get("heads", []):
                if not hd["clean"]:
                    todo.append(f"rod {r['id']} runs into more material at {hd['at']}: if the picture shows its head there (an ornament, a "
                                f"knob), add {{kind: box, centre: {hd['centre']}, size: {hd['size']}, axes: {hd['axes']}}} under the rod's name, "
                                "shrunk to the head if it takes more (hair, a hand)")
        todo.append("look for any part a person would call separate that no finder took, and cut it")
        out += "\n\nCHECK, then call group_bodies with the call above edited:\n" + "\n".join(f"- {x}" for x in todo)
        return (f"proposed by the finders ({len(groups)} pieces as bodies, {len(cuts)} cuts"
                + (f"; stand found: top face at {stand['point']}, {stand['thickness']} mm thick" if stand else "; no stand found")
                + "):\n" + json.dumps(dict(groups=groups, cuts=cuts)) + "\n" + out)

    def auto_bodies(self):
        return self.propose_bodies()

    def group_bodies(self, groups, drop=(), rest="touching", splits=()):
        """Named bodies from the organic solid's pieces (bodies.group): the STEP gets a product per body, and the
        validator then judges the pieces kept (dropped ones are gone from the solid; splits change no surface)."""
        from bodies import group, survey
        from meshio_min import write_stl
        from step_faceted import write_step
        if not getattr(self, "whole", None):
            return "no organic solid yet: run the organic route first"
        whole = read_stl(self.whole)
        self.pieces = self.pieces or survey(whole, self.grid)
        # the finders' ring and rod cuts stay whatever the call says: Swift, misreading a picture, left Kurisu's two fused
        # gears in the figure even when warned. A cut of a piece no group keeps is not added
        key = lambda c: (c.get("kind"), c.get("ring"), c.get("rod"))
        have, kept_ids = {key(s) for s in splits}, {i for g in groups for i in g.get("pieces", [])}
        names = {str(g.get("name", "")).strip() for g in groups}
        found = (getattr(self, "proposal", None) or {}).get("cuts", [])
        lost = {c["name"] for c in found if c.get("kind") in ("ring", "rod") and key(c) not in have and set(c["pieces"]) <= kept_ids}
        added = [dict(c, name=c["name"] + (" (finder)" if c["name"] in names else "")) for c in found
                 if c["name"] in lost and c.get("kind") != "plane"]
        splits = list(splits) + added
        try:
            bodies, kept, dropped, cuts = group(whole, self.pieces, groups, drop, rest, splits, getattr(self, 'rings', None) or (),
                                                getattr(self, 'rods', None) or ())
        except (ValueError, KeyError, TypeError) as e:
            return f"refused: {e}"
        # a slicer drops a body that is open or inside out (Kurisu's knife-cut ring fragments printed nothing):
        # every body must be closed and face out, or the solid goes as the one body it is, uncut
        from validate import edge_counts, volume
        tri = lambda b: b[0][b[1]] if isinstance(b, tuple) else b
        broken = [n for n, b in bodies if edge_counts(tri(b)) != (0, 0) or volume(tri(b)) <= 0]
        if broken:
            bodies, cuts, kept = [(self.stl.stem, kept)], [], kept
        self.reduced = self.work / f"{self.stl.stem}_bodies.stl"
        write_stl(self.reduced, kept)
        info = write_step(self.organic_step, None, bodies=bodies)
        self._bodies(bodies, dropped, cuts)
        self.delivered = [(n, tri(b)) for n, b in bodies]
        warn = [f"KEPT the finders' cut {c['name']} ({c['kind']} {c.get('ring', c.get('rod'))}) the call left out: that part stays a body "
                "of its own (to name it, cut the same ring or rod under the name you want)" for c in added if c.get("kind") != "box"]
        return "\n".join(([f"REFUSED the split: bodies {', '.join(map(str, broken[:8]))} came out open or inside out, so the solid "
                           "is written as one body"] if broken else []) + warn + [f"STEP {self.organic_step.name}: {info['products']} bodies, {info['solids']} solids, {info['voids']} cavities; "
                          f"picture: {self.work / 'bodies.png'} (view_image: is each part one body, and nothing of one in another?)"]
                         + [f"  {b['name']}: {b['volume']} mm3, box {b['lo']}..{b['hi']}" for b in self.bodies]
                         + [f"  {c['name']}: {c['kind']} cut from pieces {c['pieces']}, {c['volume']} mm3"
                            + (f" ({c['torn_off_joined']} mm3 torn off the rest joined it)" if c["torn_off_joined"] else "") for c in cuts]
                         + ["the validator judges the kept pieces uncut: a cut changes no surface, so a FAIL now comes from what was "
                            "dropped (a dropped sheet of scanned surface leaves that scan uncovered: keep it with rest touching or nearest)"]
                         + [f"dropped {len(dropped)} pieces, {sum(d['volume'] for d in dropped):.1f} mm3, {sum(d['area'] for d in dropped):.0f} mm2 of surface: "
                            + ", ".join(f"{d['id']} ({d['volume']} mm3, {d['why']})" for d in dropped[:12]) + (" ..." if len(dropped) > 12 else "")])

    def scan_as_posed(self):
        """The scan in the pose the model is built in (after align_*), proven to be the original scan
        moved rigidly: same triangles, Kabsch residual ~0. Anything else (smoothed, trimmed) is refused."""
        import numpy as np
        from meshio_min import read_stl
        path = self.work / "scan_posed.stl"
        txt, err = self.batch([{"tool": "export_stl", "arguments": {"path": str(path)}}])
        if err or '"ok":false' in txt:
            return f"could not export the scan in its pose: {txt[:500]}"
        A, B = read_stl(self.input), read_stl(path)
        # the exporter may reorder triangles: compare what a rigid move keeps, order-free - every edge length
        edges = lambda T: np.sort(np.linalg.norm(T - np.roll(T, 1, axis=1), axis=2).ravel())
        if A.shape != B.shape or np.abs(edges(A) - edges(B)).max() > 1e-4:
            return "refused: the scan in ScanRuler is not the input scan moved rigidly (it was edited). Undo the mesh edits."
        return path

    def deliver(self, name=None):
        """STEP with read-back verify, body check, feature list; refuses without a passing validation."""
        name = name or self.stl.stem
        txt, _ = self.batch([
            {"tool": "reverse_body_check", "arguments": {} if self.page else {"body": self.body}},
            {"tool": "reverse_export_step", "arguments": {"verify": True, "path": str(self.work / f"{name}_CAD.step")}},
            {"tool": "reverse_feature_list", "arguments": {}},
        ], stopOnError=False)
        (self.work / f"{name}_features.json").write_text(txt)
        try:
            res = json.loads(txt)["results"]
            check, step = ((res[0].get("result") or {}).get("bodies") or [{}])[0], res[1].get("result", {})
        except Exception:
            return f"deliver: unreadable answer {txt[:500]}"
        v = (step or {}).get("verify") or {}
        return json.dumps(dict(step=str(self.work / f"{name}_CAD.step"), step_verify_ok=v.get("ok"), step_problems=v.get("problems"),
                               body_valid=check.get("valid"), body_closed=check.get("closed"), solids=check.get("solids")))[:1500]
