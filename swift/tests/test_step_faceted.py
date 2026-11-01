# SPDX-License-Identifier: AGPL-3.0-only
"""The faceted STEP writer, read back by ScanRuler's own CAD kernel (headless engine, OCCT) and, when
installed, native OCCT: a cube and a bench part must come back as one closed, valid solid of the right volume.
    python swift/tests/test_step_faceted.py"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mcp_client import ScanRulerMCP  # noqa: E402
from meshio_min import read_stl  # noqa: E402
from step_faceted import write_step  # noqa: E402
from validate import volume  # noqa: E402

try:        # pip install cadquery-ocp
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    OCP = True
except ImportError:
    OCP = False

V = np.array([[x, y, z] for x in (0, 10) for y in (0, 10) for z in (0, 10)], float)
F = [(0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3), (0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6)]
cases = {"cube": (V[np.array(F)], 1000.0)}
bench = Path(__file__).resolve().parents[3] / ".claude-work" / "tmp" / "bench" / "spacer.stl"
if bench.exists():
    t = read_stl(bench)
    cases["spacer"] = (t, abs(volume(t)))

with tempfile.TemporaryDirectory() as d:
    m = ScanRulerMCP(Path(d))
    try:
        for name, (tris, vol) in cases.items():
            out = Path(d) / f"{name}.step"
            write_step(out, tris, name)
            m.call("scanruler_batch", {"calls": [{"tool": "reverse_feature_clear", "arguments": {}}]})
            txt, _, _ = m.call("scanruler_batch", {"calls": [
                {"tool": "reverse_import_step", "arguments": {"path": str(out)}},
                {"tool": "reverse_body_check", "arguments": {}}]})
            res = json.loads(txt)
            assert not res.get("failed"), (name, txt[:800])
            body = res["results"][1]["result"]["bodies"][0]
            assert body["valid"] and body["closed"] and body["solids"] == 1, (name, body)
            assert abs(body["volume"] - vol) < 1e-3 * vol + 0.01, (name, body["volume"], vol)
            print(name, "ok", body["volume"], body["faces"], "faces")
            if OCP:     # native OCCT is stricter than the engine's: one valid solid, facing out (positive volume)
                r = STEPControl_Reader()
                assert r.ReadFile(str(out)) == IFSelect_RetDone, name
                r.TransferRoots()
                shape, props, solids = r.OneShape(), GProp_GProps(), TopExp_Explorer(r.OneShape(), TopAbs_SOLID)
                BRepGProp.VolumeProperties_s(shape, props)
                n = 0
                while solids.More():
                    n, _ = n + 1, solids.Next()
                assert n == 1 and BRepCheck_Analyzer(shape).IsValid() and abs(props.Mass() - vol) < 1e-3 * vol + 0.01, (name, n, props.Mass())
                print(name, "ok in native OCCT")
    finally:
        m.close()
print("ok")

# a solid with a sealed cavity: one solid, its volume less the cavity's (native OCCT)
if OCP:
    inner = (V[np.array(F)] - 5) * 0.4 + 5
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "hollow.step"
        info = write_step(out, np.concatenate([V[np.array(F)], inner[:, ::-1]]), "hollow")
        assert info["solids"] == 1 and info["voids"] == 1, info
        r = STEPControl_Reader()
        assert r.ReadFile(str(out)) == IFSelect_RetDone
        r.TransferRoots()
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(r.OneShape(), props)
        assert BRepCheck_Analyzer(r.OneShape()).IsValid() and abs(props.Mass() - (1000 - 64)) < 0.01, props.Mass()
        print("hollow ok in native OCCT", props.Mass())
