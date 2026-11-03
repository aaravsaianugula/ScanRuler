# SPDX-License-Identifier: AGPL-3.0-only
"""Self-checks for the mech route: primitive fits on noisy samples, and a drilled block end to end (segment,
cell-complex solid, STEP, validator).   python swift/tests/test_mech.py"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mech  # noqa: E402
from meshio_min import write_stl  # noqa: E402

rng = np.random.default_rng(1)
sig = 0.02

# cylinder r=4 along z through (1, 2): axis, centre and radius back within the noise
t, z = rng.uniform(0, 2 * np.pi, 4000), rng.uniform(0, 10, 4000)
N = np.c_[np.cos(t), np.sin(t), 0 * t]
P = np.c_[1 + 4 * np.cos(t), 2 + 4 * np.sin(t), z] + N * rng.normal(0, sig, (4000, 1))
p, ok = mech.best(P, N, sig, 50)
w = p["c"] - np.array([1, 2, 0])
assert ok and p["kind"] == "cylinder" and abs(p["r"] - 4) < 0.005 and abs(abs(p["a"][2]) - 1) < 1e-4, p
assert np.linalg.norm(w - (w @ p["a"]) * p["a"]) < 0.005, p

# 45 deg cone band (a chamfer), apex at (0, 0, 7), opening down: read as a cone, not a sphere
h = rng.uniform(1, 3, 4000)
a = np.array([0, 0, -1.0])
rad = np.c_[np.cos(t), np.sin(t), 0 * t]
N = mech.unit(rad + np.array([0, 0, 1.0]))
P = np.array([0, 0, 7.0]) + h[:, None] * a + h[:, None] * rad + N * rng.normal(0, sig, (4000, 1))
p, ok = mech.best(P, N, sig, 50)
assert ok and p["kind"] == "cone" and abs(np.degrees(p["t"]) - 45) < 0.5 and np.linalg.norm(p["c"] - [0, 0, 7]) < 0.05, p

# a 20 x 10 x 5 block with a 6 mm bore: 6 planes + 1 cylinder, volume and the validator's verdict
from manifold3d import Manifold  # noqa: E402
m = (Manifold.cube([20, 10, 5]) - Manifold.cylinder(7, 3, circular_segments=96).translate([10, 5, -1])).to_mesh()
V, F = np.asarray(m.vert_properties)[:, :3].astype(float), np.asarray(m.tri_verts)
with tempfile.TemporaryDirectory() as d:
    stl = Path(d) / "block.stl"
    write_stl(stl, V[F])
    rep = mech.run(stl, Path(d) / "block.step", log=lambda s: None)
assert rep["verdict"] == "PASS", rep.get("validation", rep)["checks"] if rep.get("validation") else rep
assert rep["solid"]["kinds"] == {"plane": 6, "cylinder": 1}, rep["solid"]
assert abs(rep["solid"]["volume"] - (1000 - np.pi * 9 * 5)) < 0.5, rep["solid"]
print("test_mech: ok")
