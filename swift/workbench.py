# SPDX-License-Identifier: AGPL-3.0-only
"""The reverse-engineering workbench: the skill's exact-reading scripts, headless ScanRuler
(over scanruler-mcp) and the independent validator, as plain calls. The Swift agent's tools
and the deterministic autopilot both go through this."""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from mcp_client import REPO, ScanRulerMCP
from validate import summary, validate

SCRIPTS = REPO / "plugins" / "reverse" / "skills" / "cad-stl-reverse-engineering" / "scripts"
ALLOWED = {"analyze", "frame", "slabs", "revolve", "holes", "blends", "compare", "helix", "profile", "rz", "centers",
           "two_way", "slice", "vslice"}


def clip(s, n=6000):
    """Head and tail of a long output: the model's context is 49k tokens, shared."""
    return s if len(s) <= n else s[: n // 3] + f"\n... [{len(s) - n} chars cut] ...\n" + s[-2 * n // 3:]


class Workbench:
    def __init__(self, stl, work=None, python=sys.executable, page=False, show_browser=True):
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
        self.browser = subprocess.Popen(["node", str(REPO / "swift" / "page.mjs"), link] + ([] if show else ["--headless"]),
                                        stdin=subprocess.PIPE, cwd=str(REPO), text=True)
        txt, _, err = self.mcp.call("scanruler_wait", {"for": "page", "timeoutSeconds": 120})
        if err or '"connected":true' not in txt.replace(" ", ""):
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
    def script(self, name, args=()):
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
        bodies = re.findall(r'"name":"([^"]+)"[^{}]*"kind":"(?:extrude|revolve)"', txt)
        return "\n".join(report) + (f"\nfeatures built: {len(re.findall(chr(34) + 'kind' + chr(34), txt))}" if txt else "")

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
