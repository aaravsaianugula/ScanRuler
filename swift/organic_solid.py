# SPDX-License-Identifier: AGPL-3.0-only
"""A closed, manifold solid from a repaired scan that is not quite one (rims the hole filler could not
close, sheets crossing at non-manifold edges): inside is where the scan's generalized winding number
is over 1/2, the surface is the exact distance to the scan, marching cubes on a grid of step h.
Needs libigl and pymeshlab (pip install libigl pymeshlab)."""
import igl
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SIGNED = igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER


def indexed(tris):
    v, inv = np.unique(tris.reshape(-1, 3), axis=0, return_inverse=True)
    return v, inv.reshape(-1, 3).astype(np.int64)


def pieces(tris):
    """The connected pieces of a triangle soup (by shared vertex)."""
    v, f = indexed(tris)
    comp = connected_components(coo_matrix((np.ones(3 * len(f)), (f.ravel(), np.roll(f, 1, axis=1).ravel())),
                                           shape=(len(v), len(v))), directed=False)[1][f[:, 0]]
    order = np.argsort(comp, kind="stable")
    return np.split(tris[order], np.flatnonzero(np.diff(comp[order])) + 1)


def orient(parts):
    """Every piece facing out - exporters and scan merges mix windings, and the winding number needs it.
    A piece that encloses volume faces out when its volume about its own centre is positive, unless it lies
    wholly inside the others: then it is a cavity and faces in. A thin shell (a fringe of hair, a ribbon, a
    flap) encloses next to none: it faces away from the body the others make."""
    def vol(t):
        c = t - t.reshape(-1, 3).mean(0)
        return float(np.einsum("ij,ij->i", c[:, 0], np.cross(c[:, 1], c[:, 2])).sum() / 6)
    sure = [abs(vol(t)) > 0.02 * np.prod(np.ptp(t.reshape(-1, 3), axis=0)) for t in parts]
    rng = np.random.default_rng(0)

    def cavity(i):
        rest = [t for j, t in enumerate(parts) if j != i and sure[j] and vol(t) > 0]
        if not rest:
            return False
        V, F = indexed(np.concatenate(rest))
        p = parts[i].reshape(-1, 3)
        return bool((igl.fast_winding_number(V, F, p[rng.choice(len(p), min(20, len(p)), replace=False)]) > 0.5).all())
    out = [t[:, ::-1] if ok and vol(t) < 0 and not cavity(i) else t for i, (t, ok) in enumerate(zip(parts, sure))]
    if not any(sure) or all(sure):
        return out
    V, F = indexed(np.concatenate([t for t, ok in zip(out, sure) if ok]))
    for i, t in enumerate(out):
        if not sure[i]:
            n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-20
            c = t.mean(1)
            if (igl.fast_winding_number(V, F, c + 2 * n) > 0.5).sum() > (igl.fast_winding_number(V, F, c - 2 * n) > 0.5).sum():
                out[i] = t[:, ::-1]
    return out


def grid_for(tris):
    """The grid step: about 1/1100 of the part's diagonal (0.3 mm on a 33 cm figurine), never under 0.05 mm,
    and coarse enough that the grid stays under ~250 M points (a few minutes)."""
    ext = np.ptp(tris.reshape(-1, 3), axis=0)
    return float(max(0.05, np.linalg.norm(ext) / 1100, (np.prod(ext + 1) / 2.5e8) ** (1 / 3)))


def close_mesh(tris, h=None, slab=24):
    """Triangles of the closed surface (none when the scan encloses no volume: a single open sheet).
    Pieces are turned to face out first. Marching cubes runs in z slabs sharing one layer (memory), and
    the slabs are welded on it: both compute those vertices from the same grid values."""
    h = h or grid_for(tris)
    V, F = indexed(np.concatenate(orient(pieces(tris))))
    lo = V.min(0) - 2 * h
    n = np.ceil((V.max(0) + 2 * h - lo) / h).astype(int) + 1
    X, Y = np.meshgrid(lo[0] + h * np.arange(n[0]), lo[1] + h * np.arange(n[1]), indexing="xy")   # x fastest, as igl wants
    layer, per = np.c_[X.ravel(), Y.ravel()], n[0] * n[1]
    verts, faces, base = [], [], 0
    for z0 in range(0, n[2] - 1, slab):
        z = lo[2] + h * np.arange(z0, min(n[2], z0 + slab + 1))
        P = np.c_[np.tile(layer, (len(z), 1)), np.repeat(z, per)]
        s, *_ = igl.signed_distance(P, V, F, SIGNED, -3 * h, 3 * h)
        far = ~np.isfinite(s)                       # past the band igl gives no sign: the winding number does
        s[far] = np.where(igl.fast_winding_number(V, F, P[far]) > 0.5, -3 * h, 3 * h)
        # a grid point on the surface puts several vertices on it, and the weld below would merge them
        s = np.where(np.abs(s) < 1e-4 * h, np.where(s < 0, -1e-4 * h, 1e-4 * h), s)
        mv, mf, _ = igl.marching_cubes(s, P, int(n[0]), int(n[1]), len(z), 0.0)
        verts.append(mv); faces.append(mf + base); base += len(mv)
    out = np.concatenate(verts)[np.concatenate(faces)]
    v, f = indexed(np.round(out, 9))               # weld the shared layers
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])]
    out = v[f]
    if np.einsum("ij,ij->i", out[:, 0], np.cross(out[:, 1], out[:, 2])).sum() < 0:
        out, f = out[:, ::-1], f[:, ::-1]
    # a shell facing in is a sealed cavity - a crevice where an arm presses on the body, closed at its mouth -
    # and stays: its walls are scanned surface. A speck under three cells across is a bubble where a noisy scan's winding number flickers past 1/2.
    # Thin slivers larger than that follow scanned surface (a base's rim lip) and stay, as pieces of their own
    n = len(v)
    adj = coo_matrix((np.ones(3 * len(f)), (f.ravel(), np.roll(f, 1, axis=1).ravel())), shape=(n, n))
    comp = connected_components(adj, directed=False)[1][f[:, 0]]
    lo = np.full((comp.max() + 1, 3), np.inf)
    hi = -lo
    np.minimum.at(lo, comp, out.min(1))
    np.maximum.at(hi, comp, out.max(1))
    keep = np.linalg.norm(hi - lo, axis=1) >= 3 * h
    return out[keep[comp]]


def reduce_mesh(tris, faces):
    """Quadric decimation to faces triangles that keeps a closed manifold one (MeshLab, preservetopology).
    ScanRuler's mesh.reduce is not used here: on a dense closed scan it leaves non-manifold edges."""
    import pymeshlab
    v, f = indexed(tris)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(v, f))
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(faces), qualitythr=0.5, preserveboundary=True, preservenormal=True,
                                                preservetopology=True, optimalplacement=True, planarquadric=True)
    m = ms.current_mesh()
    return m.vertex_matrix()[m.face_matrix()]
