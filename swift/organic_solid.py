# SPDX-License-Identifier: AGPL-3.0-only
"""A closed, manifold solid from a repaired scan that is not quite one (rims the hole filler could not
close, sheets crossing at non-manifold edges): inside is where the scan's generalized winding number
is over 1/2, the surface is the exact distance to the scan, marching cubes on a grid of step h.
Needs libigl and pymeshlab (pip install libigl pymeshlab)."""
import sys

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


CAVITY_CLEAR = 0.5      # mm a sealed void's wall keeps from the outside at least


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
        p = p[rng.choice(len(p), min(20, len(p)), replace=False)]
        # and clear of their surface: a shell just under another's skin (an anime model's face mask under the head's)
        # is a layer of the outside, not a void - kept inside out it took the face out of Kurisu's solid
        return bool((igl.fast_winding_number(V, F, p) > 0.5).all() and np.sqrt(igl.point_mesh_squared_distance(p, V, F)[0]).min() > CAVITY_CLEAR)
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
    """The grid step: half the scan's median edge - the closing rounds every edge and thin fin by about a cell,
    and a cell under the scan's own spacing has nothing more to follow - kept between 1/1650 and 1/1100 of the
    part's diagonal, never under 0.05 mm, and coarse enough that the grid stays under ~1 G points (about ten
    minutes). A 33 cm figurine scanned at 0.46 mm closes on 0.2 mm (scan->solid rms 0.039, 0.049 on 0.3 mm);
    a coarse 52k-triangle scan stays on 1/1100, where a finer grid bought nothing but triangles."""
    ext = np.ptp(tris.reshape(-1, 3), axis=0)
    diag = np.linalg.norm(ext)
    edge = float(np.median(np.linalg.norm(tris - np.roll(tris, 1, axis=1), axis=2)))
    return float(max(0.05, diag / 1650, min(diag / 1100, edge / 2.25), (np.prod(ext + 1) / 1e9) ** (1 / 3)))


def sheet_faces(V, F, h):
    """Which faces of an oriented scan are thin surface with no solid behind them (a face, a hand, hair: an open
    sheet): the winding number a step and a half behind the face is not clearly inside. Surface with solid
    behind it bounds the body already, however noisy in front (a fold's outer layer), and is not thickened.
    One face's test flickers on noisy surface that does bound the body (its normal off by the noise, a thin
    spot), and each such face got a plate standing 0.4 mm off the scan: it is asked along its own and its
    smoothed normal, and a face that says sheet is one only when most of the surface within 2h does (Frieren:
    45 % of the measured scan lay under plates, 8 % so; relief_sheet's real sheets stay). The vote only takes
    sheet off: giving it to bounding faces among sheets buried them under a plate (relief_sheet p99 0.21)."""
    from scipy.sparse import csr_matrix
    from scipy.spatial import cKDTree
    t = V[F]
    c = t.mean(1)
    nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    nrm /= 2 * area[:, None] + 1e-20
    ns = np.nan_to_num(igl.per_vertex_normals(V, F))[F].mean(1)
    ns /= np.linalg.norm(ns, axis=1, keepdims=True) + 1e-20
    raw = (igl.fast_winding_number(V, F, c - 1.5 * h * nrm) <= 0.75) & (igl.fast_winding_number(V, F, c - 1.5 * h * ns) <= 0.75)
    i, j = cKDTree(c).query_pairs(2 * h, output_type="ndarray").T
    near = csr_matrix((np.ones(2 * len(i)), (np.r_[i, j], np.r_[j, i])), shape=(len(F), len(F)))
    return raw & (near @ (area * raw) + area * raw >= 0.5 * (near @ area + area))


def plate_radius(h):
    """Half the thickness a thin scanned surface is given: 0.8 mm plates at least (two 0.4 mm nozzle lines)."""
    return max(1.5 * h, 0.4)


def close_mesh(tris, h=None, slab=24, handles=3, log=lambda s: print(s, file=sys.stderr)):
    """Triangles of the closed surface (none when the scan encloses no volume: a single open sheet).
    Pieces are turned to face out first. The distance is sampled in z slabs (memory) into one grid; tunnels
    and handles under about 2*handles cells across are taken off it (topology.py: a patch layout needs the
    few a figurine really has, not the hundreds a noisy closing leaves); marching cubes runs in slabs sharing
    one layer, welded on it: both compute those vertices from the same grid values."""
    h = h or grid_for(tris)
    V, F = indexed(np.concatenate(orient(pieces(tris))))
    # thin scanned surface with no clean inside and outside (hair, a cloak's edge, a scan's open flaps): the
    # winding number hovers at 1/2 beside it and the closing comes out as lace (Kurisu: 1522 tunnels). Such
    # surface becomes a plate, at least 0.8 mm thick, printable with a 0.4 mm nozzle, centred on it: one laid
    # behind it (offset along vertex normals) buried stacked hair layers (Kurisu) and, where those normals
    # scatter on a noisy sheet, fell short of the scan (relief_sheet). It reaches only where the sheet is the
    # nearest scan: where it meets surface that bounds the body it stops there, half a cell over it, not 0.4 mm
    sheet = sheet_faces(V, F, h)
    r = plate_radius(h)
    band = max(3 * h, r + 2 * h)
    lo = V.min(0) - band - h
    n = np.ceil((V.max(0) + band + h - lo) / h).astype(int) + 1
    X, Y = np.meshgrid(lo[0] + h * np.arange(n[0]), lo[1] + h * np.arange(n[1]), indexing="xy")   # x fastest, as igl wants
    layer, per = np.c_[X.ravel(), Y.ravel()], n[0] * n[1]
    grid = np.empty((n[2], n[1], n[0]), np.float32)
    for z0 in range(0, n[2], slab):
        z = lo[2] + h * np.arange(z0, min(n[2], z0 + slab))
        P = np.c_[np.tile(layer, (len(z), 1)), np.repeat(z, per)]
        s, I, C, _ = igl.signed_distance(P, V, F, SIGNED, -band, band)
        far = ~np.isfinite(s)                       # past the band igl gives no sign: the winding number does
        s[far] = np.where(igl.fast_winding_number(V, F, P[far]) > 0.5, -band, band)
        # igl signs by |w|: in front of an open sheet's hollow (a relief's dip) w nears -1 and it says inside
        neg = np.flatnonzero(~far & (s < 0))
        s[neg[igl.fast_winding_number(V, F, P[neg]) <= 0.5]] *= -1
        # and s is d (1 - 2|w|), not d: where w stays near 1/2 beside the surface (a crevice, a noisy layer)
        # marching cubes put it up to half a cell off the scan (Frieren: 5-10 mm2 patches 0.1 mm proud)
        s[~far] = np.copysign(np.linalg.norm(P[~far] - C[~far], axis=1), s[~far])
        near = np.flatnonzero(np.abs(s) < r)
        on = near[sheet[I[near]]]
        s[on] = np.minimum(s[on], np.abs(s[on]) - r)
        # other scan nearer: the plate tapers to half a cell there, so its rim holds (Frieren: 141 tunnels
        # cut off sheer, 109 so; a whole cell gave 58 but stood proud of the scan, detail lost)
        rest = near[~sheet[I[near]]]
        if len(rest) and sheet.any():
            s[rest] = np.minimum(s[rest], np.sqrt(igl.point_mesh_squared_distance(P[rest], V, F[sheet])[0]) - h / 2)
        # a grid point on the surface puts several vertices on it, and the weld below would merge them
        s = np.where(np.abs(s) < 1e-4 * h, np.where(s < 0, -1e-4 * h, 1e-4 * h), s)
        grid[z0:z0 + len(z)] = s.reshape(len(z), n[1], n[0])
    if handles:
        import topology
        try:
            add, cut = topology.simplify(grid < 0, handles, log=log)
            g = grid.reshape(-1)
            g[add], g[cut] = -0.5 * h, 0.5 * h         # a plug or a cut about a cell thick
        except MemoryError:     # its labels take ~3x the grid: a solid with its small handles beats none
            log(f"[organic] not enough memory to take small handles off a {grid.size / 1e6:.0f} M-point grid: kept as closed")
    verts, faces, base = [], [], 0
    for z0 in range(0, n[2] - 1, slab):
        z = lo[2] + h * np.arange(z0, min(n[2], z0 + slab + 1))
        P = np.c_[np.tile(layer, (len(z), 1)), np.repeat(z, per)]
        mv, mf, _ = igl.marching_cubes(grid[z0:z0 + len(z)].reshape(-1).astype(float), P, int(n[0]), int(n[1]), len(z), 0.0)
        verts.append(mv); faces.append(mf + base); base += len(mv)
    out = np.concatenate(verts)[np.concatenate(faces)]
    v, f = indexed(np.round(out, 9))               # weld the shared layers
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])]
    out = v[f]
    if np.einsum("ij,ij->i", out[:, 0], np.cross(out[:, 1], out[:, 2])).sum() < 0:
        out, f = out[:, ::-1], f[:, ::-1]
    # a shell facing in is a sealed cavity - a crevice where an arm presses on the body, closed at its mouth -
    # and stays: its walls are scanned surface; but one under two cells thick (2 volume / area) is a sliver
    # where the distance hovers at zero, and reduction would push its walls through the body's: filled.
    # A speck under eight cells across is a bubble where a noisy scan's winding number flickers past 1/2, or
    # one caught between two scan layers a fraction of a mm apart (Frieren: some 200, loose debris in a print).
    # Thin slivers larger than that follow scanned surface (a base's rim lip) and stay, as pieces of their own
    n = len(v)
    adj = coo_matrix((np.ones(3 * len(f)), (f.ravel(), np.roll(f, 1, axis=1).ravel())), shape=(n, n))
    comp = connected_components(adj, directed=False)[1][f[:, 0]]
    lo = np.full((comp.max() + 1, 3), np.inf)
    hi = -lo
    np.minimum.at(lo, comp, out.min(1))
    np.maximum.at(hi, comp, out.max(1))
    vol = np.bincount(comp, weights=np.einsum("ij,ij->i", out[:, 0], np.cross(out[:, 1], out[:, 2])) / 6)
    area = np.bincount(comp, weights=0.5 * np.linalg.norm(np.cross(out[:, 1] - out[:, 0], out[:, 2] - out[:, 0]), axis=1))
    keep = (np.linalg.norm(hi - lo, axis=1) >= 8 * h) & ((vol > 0) | (-2 * vol / area >= 2 * h))
    # nothing floats in a print: a separate piece under 0.5 % of the volume is debris (a bubble between scan
    # layers, a stray flap), not a part; if it was scanned surface the validator's detail check says so
    keep &= (vol < 0) | (vol >= 0.005 * vol[vol > 0].sum())
    return out[keep[comp]]


def self_intersecting(v, f):
    """Which faces cut through another face of the mesh (MeshLab)."""
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(v, f))
    ms.compute_selection_by_self_intersections_per_face()
    return ms.current_mesh().face_selection_array()


def reduce_mesh(tris, faces, passes=12, safe=True):
    """Quadric decimation to about faces triangles that keeps a closed manifold one (MeshLab, preservetopology).
    ScanRuler's mesh.reduce is not used here: on a dense closed scan it leaves non-manifold edges.
    Decimation knows nothing of the other side of a thin wall (a hair tip, a finger on the staff): collapses
    there push one wall through the other, and a small shell can turn inside out - surface showing its back.
    Where that happens the dense mesh around it is kept out of the decimation and it runs again.
    safe=False: one plain pass, evenly coarse, for the part finders (rings.py), whose flat-face growing a
    dense kept patch breaks up (a gear lost on Kurisu); never delivered."""
    import pymeshlab
    from scipy.spatial import cKDTree
    if safe and len(tris) > 6 * faces:      # in stages: what is kept out of the last one is then 3x coarser than the closing's cells
        tris = reduce_mesh(tris, 3 * faces, passes)
    v, f = indexed(tris)
    keep, tree, best = np.zeros(len(v), bool), cKDTree(v), None
    tries = np.zeros(len(v), int)          # how often a spot came back: its keep widens only there
    signed = lambda t: np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum()
    cavities = [(p.reshape(-1, 3).min(0), p.reshape(-1, 3).max(0)) for p in pieces(tris) if signed(p) < 0]
    for _ in range(passes):
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(v, f, f_scalar_array=(~keep[f].any(1)).astype(float)))
        ms.compute_selection_by_condition_per_face(condselect="fq > 0.5")
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(faces), qualitythr=0.5, preserveboundary=True, preservenormal=True,
                                                    preservetopology=True, optimalplacement=True, planarquadric=True, selected=True)
        m = ms.current_mesh()
        rv, rf = m.vertex_matrix(), m.face_matrix()
        out = rv[rf]
        if not safe:
            return out
        bad = list(out[self_intersecting(rv, rf)])
        # a shell turned inside out: negative, and not where the dense mesh has a cavity
        bad += [p for p in pieces(out) if signed(p) < 0 and not any(((p.mean((0, 1)) >= lo) & (p.mean((0, 1)) <= hi)).all() for lo, hi in cavities)]
        if not bad:
            return out
        if best is None or len(bad) < best[0]:
            best = (len(bad), out)
        before = tries.copy()
        for t in bad:      # a face, or a whole shell; twice as wide where it comes back: at worst all is kept, and the dense mesh is clean
            p = t.reshape(-1, 3)
            r = np.linalg.norm(t.reshape(-1, 3, 3) - np.roll(t.reshape(-1, 3, 3), 1, axis=1), axis=2).max()
            n = before[tree.query(p.mean(0))[1]]
            ball = tree.query_ball_point(p.mean(0), max(min(3.0, r) * 2 ** n, np.ptp(p, axis=0).max()))
            keep[ball] = True
            tries[ball] = np.maximum(tries[ball], n + 1)
    # a few spots left beat no solid: the cleanest pass is delivered and the validator judges it
    print(f"[organic] reduce_mesh: {best[0]} faces or shells still cut through or turned after {passes} passes; delivering the cleanest pass", file=sys.stderr)
    return best[1]
