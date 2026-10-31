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

import igl
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meshio_min import write_stl  # noqa: E402
from validate import volume  # noqa: E402


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


def orient(parts):
    """A part that encloses volume faces out when its volume about its own centre is positive. A thin shell
    (a fringe of hair, a ribbon) encloses next to none: it faces away from the body the others make."""
    sure = lambda t: abs(volume(t - t.reshape(-1, 3).mean(0))) > 0.02 * np.prod(np.ptp(t.reshape(-1, 3), axis=0))
    out = [t if not sure(t) or volume(t - t.reshape(-1, 3).mean(0)) > 0 else t[:, ::-1] for t in parts]
    body = np.concatenate([t for t, p in zip(out, parts) if sure(p)])
    v, inv = np.unique(body.reshape(-1, 3), axis=0, return_inverse=True)
    f = inv.reshape(-1, 3).astype(np.int64)
    for i, t in enumerate(out):
        if sure(parts[i]):
            continue
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-20
        c = t.mean(1)
        into_front = (igl.fast_winding_number(v, f, c + 2 * n) > 0.5).sum()
        into_back = (igl.fast_winding_number(v, f, c - 2 * n) > 0.5).sum()
        if into_front > into_back:
            out[i] = t[:, ::-1]
    return out


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
