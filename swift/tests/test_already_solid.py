# SPDX-License-Identifier: AGPL-3.0-only
"""already_solid and tunnels(closed_only): a closed sphere is a solid as it is, one with a hole is not; a
torus brings one handle of its own, an open sheet none.
    python swift/tests/test_already_solid.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from organic_solid import already_solid  # noqa: E402
from validate import tunnels  # noqa: E402

T = np.array([[a, b, c] for a in ([1, 0, 0], [-1, 0, 0]) for b in ([0, 1, 0], [0, -1, 0]) for c in ([0, 0, 1], [0, 0, -1])], float)
T = np.array([t if np.linalg.det(t) > 0 else t[::-1] for t in T])
for _ in range(4):
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
    T = np.concatenate([np.stack(x, 1) for x in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
sphere = 10 * T / np.linalg.norm(T, axis=2, keepdims=True)
assert already_solid(sphere)
assert not already_solid(sphere[:, ::-1])                    # inside out
assert not already_solid(sphere[sphere[:, :, 2].mean(1) < 9.5])  # a hole

n, m = 48, 24
u, v = np.meshgrid(np.linspace(0, 2 * np.pi, n, endpoint=False), np.linspace(0, 2 * np.pi, m, endpoint=False), indexing="ij")
P = np.stack([(10 + 3 * np.cos(v)) * np.cos(u), (10 + 3 * np.cos(v)) * np.sin(u), 3 * np.sin(v)], -1)
i, j = np.meshgrid(np.arange(n), np.arange(m), indexing="ij")
a, b, c, d = P[i, j], P[(i + 1) % n, j], P[(i + 1) % n, (j + 1) % m], P[i, (j + 1) % m]
torus = np.concatenate([np.stack([a, b, c], -2).reshape(-1, 3, 3), np.stack([a, c, d], -2).reshape(-1, 3, 3)])
assert tunnels(torus) == 1 and tunnels(torus, closed_only=True) == 1
assert already_solid(torus) or already_solid(torus[:, ::-1])
open_torus = torus[np.abs(torus[:, :, 0].mean(1)) > 2]          # cut through: two open tubes
assert tunnels(open_torus, closed_only=True) == 0
assert tunnels(np.concatenate([torus + [40, 0, 0], open_torus]), closed_only=True) == 1
print("ok")
