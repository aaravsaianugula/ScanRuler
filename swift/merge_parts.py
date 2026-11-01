# SPDX-License-Identifier: AGPL-3.0-only
"""One mesh from a folder of OBJ parts (a model exported from a game or viewer: a body, hair, accessories),
for the organic route: exact duplicate parts dropped, strip stitches (zero-area triangles) dropped, every
part turned to face out - exporters mix clockwise and counter-clockwise winding.

    python swift/merge_parts.py <folder of .obj> <out.stl>
"""
import glob
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meshio_min import write_stl  # noqa: E402
from organic_solid import orient  # noqa: E402


def read_obj(path):
    v, f = [], []
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("v "):
            v.append(line.split()[1:4])
        elif line.startswith("f "):
            idx = [int(t.split("/")[0]) for t in line.split()[1:]]
            f += [[idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1)]
    v, f = np.array(v, float), np.array(f)
    f = np.where(f < 0, f + len(v), f - 1)
    return v[f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])]]


def merge(folder):
    seen, parts = set(), []
    for p in sorted(glob.glob(os.path.join(folder, "*.obj"))):
        t = read_obj(p)
        h = hashlib.md5(np.round(np.sort(t.reshape(-1, 9), axis=0), 5).tobytes()).hexdigest()
        if len(t) and h not in seen:
            seen.add(h)
            parts.append(t)
    return orient(parts)


if __name__ == "__main__":
    parts = merge(sys.argv[1])
    tris = np.concatenate(parts)
    write_stl(sys.argv[2], tris)
    print(f"{len(parts)} parts, {len(tris)} triangles -> {sys.argv[2]}")
