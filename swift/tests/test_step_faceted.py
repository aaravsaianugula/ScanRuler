# SPDX-License-Identifier: AGPL-3.0-only
"""The faceted STEP writer, read back by ScanRuler's own CAD kernel (headless engine, OCCT): a cube and
a bench part must come back as one closed, valid solid of the right volume.
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
    finally:
        m.close()
print("ok")
