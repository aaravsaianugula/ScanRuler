# SPDX-License-Identifier: AGPL-3.0-only
"""close_mesh and enclosing on a scan-like sphere: a hole in it, a stray sheet inside, a flap outside, a
sealed cavity (a small sphere facing in). The solid must come out closed and manifold, the cavity kept (its
walls are scanned surface): the sphere's volume less the cavity's; only the sphere and the cavity are enclosing.
    python swift/tests/test_organic_solid.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from organic_solid import close_mesh  # noqa: E402
from validate import edge_counts, enclosing, sample, volume  # noqa: E402

# a sphere of radius 10 from a subdivided octahedron
T = np.array([[a, b, c] for a in ([1, 0, 0], [-1, 0, 0]) for b in ([0, 1, 0], [0, -1, 0]) for c in ([0, 0, 1], [0, 0, -1])], float)
T = np.array([t if np.linalg.det(t) > 0 else t[::-1] for t in T])
for _ in range(5):
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
    T = np.concatenate([np.stack(x, 1) for x in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
T = 10 * T / np.linalg.norm(T, axis=2, keepdims=True)
sphere = T[T[:, :, 2].mean(1) < 9.5]                        # a hole at the top
sheet = lambda z, r: np.array([[[0, 0, z], [r, 0, z], [0, r, z]], [[0, 0, z], [0, r, z], [-r, 0, z]]])
void = (0.3 * T + [0, 0, -4.5])[:, ::-1]                       # radius 3, facing in
scan = np.concatenate([sphere, sheet(5, 4), sheet(-12, 3), void])   # stray sheet inside, flap outside

solid = close_mesh(scan, h=0.4)
assert edge_counts(solid) == (0, 0), edge_counts(solid)
v = volume(solid)
want = 4 / 3 * np.pi * (1000 - 27)
assert abs(v - want) < 0.015 * want, (v, want)          # dropping the cavity would add 2.8 %
pts, tid = sample(scan, 0.5, return_tri=True)
on = enclosing(scan, pts)
on_sphere = (tid < len(sphere)) | (tid >= len(scan) - len(void))
assert on[on_sphere].mean() > 0.97 and on[~on_sphere].mean() < 0.05, (on[on_sphere].mean(), on[~on_sphere].mean())
print("ok", len(solid), "triangles, volume", round(v, 1))

# the model-type detector run.py's auto mode uses: a sphere is free-form, a cube is engineered
import tempfile  # noqa: E402
from meshio_min import write_stl  # noqa: E402
from run import surface_shares  # noqa: E402
cube = np.array([[[0, 0, 0], [1, 1, 0], [1, 0, 0]], [[0, 0, 0], [0, 1, 0], [1, 1, 0]], [[0, 0, 1], [1, 0, 1], [1, 1, 1]], [[0, 0, 1], [1, 1, 1], [0, 1, 1]],
                 [[0, 0, 0], [1, 0, 0], [1, 0, 1]], [[0, 0, 0], [1, 0, 1], [0, 0, 1]], [[0, 1, 0], [1, 1, 1], [1, 1, 0]], [[0, 1, 0], [0, 1, 1], [1, 1, 1]],
                 [[0, 0, 0], [0, 0, 1], [0, 1, 1]], [[0, 0, 0], [0, 1, 1], [0, 1, 0]], [[1, 0, 0], [1, 1, 1], [1, 0, 1]], [[1, 0, 0], [1, 1, 0], [1, 1, 1]]], float)
with tempfile.TemporaryDirectory() as d:
    for name, mesh, organic in (("sphere", sphere, True), ("cube", cube, False)):
        write_stl(Path(d) / f"{name}.stl", mesh)
        planes, cylinders, _ = surface_shares(Path(d) / f"{name}.stl")
        assert (planes < 0.4 and cylinders < 0.45) == organic, (name, planes, cylinders)
print("ok surface_shares")

# the organic validator's samples stand for equal areas, however uneven the triangles (a reduced mesh)
from validate import sample_area  # noqa: E402
big = np.array([[[0, 0, 0], [10, 0, 0], [0, 10, 0]]], float)                     # 50 mm2
tiny = np.array([[[20 + i % 100 * 0.1, i // 100 * 0.1, 0], [20.1 + i % 100 * 0.1, i // 100 * 0.1, 0], [20 + i % 100 * 0.1, 0.1 + i // 100 * 0.1, 0]]
                 for i in range(10000)], float)                                     # 10,000 x 0.005 = 50 mm2
_, tid = sample_area(np.concatenate([big, tiny]), 20000)
assert abs((tid == 0).mean() - 0.5) < 0.02, (tid == 0).mean()
print("ok sample_area")
