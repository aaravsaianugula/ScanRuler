# SPDX-License-Identifier: AGPL-3.0-only
"""Auto-surfacing a torus (a handle, so cells must be cut into disks): the B-spline STEP must read back
in OCCT as one valid closed solid whose tessellation stays within 0.1 mm of the true torus, with the
torus's volume.
    python swift/tests/test_autosurface.py"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autosurface import surface, write_step  # noqa: E402
from step_faceted import check_import  # noqa: E402
from validate import volume  # noqa: E402

R, r, nu, nv = 20.0, 6.0, 120, 40
u, w = np.meshgrid(np.linspace(0, 2 * np.pi, nu, endpoint=False), np.linspace(0, 2 * np.pi, nv, endpoint=False), indexing="ij")
V = np.stack([(R + r * np.cos(w)) * np.cos(u), (R + r * np.cos(w)) * np.sin(u), r * np.sin(w)], -1).reshape(-1, 3)
i, j = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
a, b, c, d = i * nv + j, (i + 1) % nu * nv + j, (i + 1) % nu * nv + (j + 1) % nv, i * nv + (j + 1) % nv
F = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
if volume(V[F]) < 0:
    F = F[:, ::-1]

sf = surface(V, F, patches=300, log=lambda s: None)
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "torus.step"
    info = write_step(path, [("torus", sf)])
    imp = check_import(path, meshes=True)
if imp is None:
    print("autosurface: no OCCT here, STEP not read back")
    sys.exit(0)
assert imp["ok"] and imp["valid"] and imp["solids"] == 1, imp["problems"]
tv, tf = imp["meshes"][0]
q = np.hypot(tv[:, 0], tv[:, 1])
dev = np.abs(np.hypot(q - R, tv[:, 2]) - r)
assert dev.max() < 0.1, dev.max()
vol, true = volume(tv[tf]), 2 * np.pi ** 2 * R * r * r
assert abs(vol - true) < 0.005 * true, (vol, true)
print(f"autosurface ok: {info['faces']} B-spline faces, max off the torus {dev.max():.3f} mm, volume {vol:.0f} of {true:.0f}")
