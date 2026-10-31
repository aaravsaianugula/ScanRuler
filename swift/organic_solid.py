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


def close_mesh(tris, h=0.3, slab=24):
    """Triangles of the closed surface. Marching cubes runs in z slabs sharing one layer (memory), and
    the slabs are welded on it: both compute those vertices from the same grid values."""
    V, F = indexed(tris)
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
    vol = np.einsum("ij,ij->i", out[:, 0], np.cross(out[:, 1], out[:, 2]))
    if vol.sum() < 0:
        out, f, vol = out[:, ::-1], f[:, ::-1], -vol
    # a shell facing in is a sealed void - a pocket where an arm presses on the body - not part of a figurine;
    # a speck under three cells across is a bubble where a noisy scan's winding number flickers past 1/2.
    # Thin slivers larger than that follow scanned surface (a base's rim lip) and stay, as pieces of their own
    n = len(v)
    adj = coo_matrix((np.ones(3 * len(f)), (f.ravel(), np.roll(f, 1, axis=1).ravel())), shape=(n, n))
    comp = connected_components(adj, directed=False)[1][f[:, 0]]
    lo = np.full((comp.max() + 1, 3), np.inf)
    hi = -lo
    np.minimum.at(lo, comp, out.min(1))
    np.maximum.at(hi, comp, out.max(1))
    keep = (np.bincount(comp, weights=vol) > 0) & (np.linalg.norm(hi - lo, axis=1) >= 3 * h)
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
