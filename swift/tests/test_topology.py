# SPDX-License-Identifier: AGPL-3.0-only
"""Self-checks for topology.simplify on voxel shapes: a pinhole is plugged, a thin handle cut, a real torus and a
bottle-shaped crevice kept, a thin loose ring cut once rather than lost.   python swift/tests/test_topology.py"""
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import topology as T  # noqa: E402

box, six = ndimage.generate_binary_structure(3, 3), ndimage.generate_binary_structure(3, 1)


def topo(X):
    """(genus, pieces, cavities) of a voxel solid: 26-connected inside, 6-connected outside."""
    comp, cav = ndimage.label(X, box)[1], ndimage.label(~X, six)[1] - 1
    return comp + cav - T.euler(X), comp, cav


z, y, x = np.mgrid[:64, :64, :64] - 32.0
ball = x ** 2 + y ** 2 + z ** 2 < 20 ** 2
torus = lambda R, a, zc=0.0: (np.sqrt(x ** 2 + y ** 2) - R) ** 2 + (z - zc) ** 2 < a ** 2
handle = lambda a: ((np.sqrt(x ** 2 + (z - 22) ** 2) - 7) ** 2 + y ** 2) < a ** 2
cases = {   # shape: (topology after, most voxels changed)
    "pinhole torus": (torus(10, 9.2), (0, 1, 0), 10),
    "thin handle": (ball | handle(1.6), (0, 1, 0), 20),
    "real torus": (torus(15, 6), (1, 1, 0), 0),
    "bottle crevice": (ball & ~((x ** 2 + y ** 2 + z ** 2 < 36) | ((x ** 2 + y ** 2 < 1.6 ** 2) & (z > 0))), (0, 1, 0), 0),
    "loose thin ring": (torus(15, 1.6) | ball & (z < -12), (0, 2, 0), 20),
}
for name, (X, want, most) in cases.items():
    add, cut = T.simplify(X, 2, log=lambda s: None)
    Y = X.copy()
    Y.reshape(-1)[add], Y.reshape(-1)[cut] = True, False
    assert topo(Y) == want, (name, topo(X), topo(Y))
    assert len(add) + len(cut) <= most, (name, len(add), len(cut))

assert all(len(a) == 0 for a in T.simplify(np.zeros((8, 8, 8), bool), log=lambda s: None))   # no solid at all

# the simple-point test on its own: a half-space's face voxel and a rod's end go, a rod's middle does not
nb = lambda f: np.array([sum(1 << b for b, o in enumerate(T.OFF) if f(*o))], np.uint32)
assert T.simple(nb(lambda i, j, k: i < 0))[0] and T.simple(nb(lambda i, j, k: i < 0 and j == k == 0))[0]
assert not T.simple(nb(lambda i, j, k: j == k == 0))[0]
print("test_topology: ok")
