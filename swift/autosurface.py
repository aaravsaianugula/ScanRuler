# SPDX-License-Identifier: AGPL-3.0-only
"""Smooth CAD surfaces over a closed solid mesh, the way reverse-engineering suites auto-surface a scan:
the surface is cut into four-sided patches, each a bicubic B-spline whose boundary curves are shared with
its neighbours, written as one watertight STEP B-rep of smooth faces (not a triangle per face).

Layout: geodesic Voronoi cells on the mesh (farthest-point seeds grown over the faces; crossing a fold
costs extra, so cells end on creases and crowd where the shape bends) that are disks with >= 3 corners.
Each cell is mapped onto a regular polygon (harmonic map) and cut into a quad per corner (centre, side
midpoints, corner). Samples come through that map, so they lie on the mesh itself, also on thin plates
and around tunnels where projecting onto a coarse layout picks the wrong side. Cells whose patches miss
the mesh by more than the tolerance get another seed, and the layout is grown again.

    python swift/autosurface.py solid.stl out.step [--patches 3000] [--n 6]
"""
import argparse
import datetime
import sys
import time
from pathlib import Path

import igl
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import cho_factor, cho_solve
from scipy import sparse
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meshio_min import read_stl  # noqa: E402
from step_faceted import indexed, num, solid_pieces  # noqa: E402

DEG = 3


def knots(n):
    return np.concatenate([[0.0] * DEG, np.linspace(0, 1, n - DEG + 1), [1.0] * DEG])


def basis(x, n):
    return BSpline.design_matrix(np.clip(x, 0, 1), knots(n), DEG).toarray()


def second_diff(n):
    return np.diff(np.eye(n), 2, axis=0)


def fitter(A, D, fixed, lam):
    """x_free = R1 @ samples + R2 @ x_fixed: the least-squares control points with the fixed ones (shared
    boundary) given, plus lam x a bending penalty. Every patch has the same A, so this is solved once."""
    free = np.setdiff1d(np.arange(A.shape[1]), fixed)
    lam = lam * np.trace(A.T @ A) / max(np.trace(D.T @ D), 1e-12)
    Au, Ak, Du, Dk = A[:, free], A[:, fixed], D[:, free], D[:, fixed]
    lhs = Au.T @ Au + lam * Du.T @ Du
    R = np.linalg.solve(lhs, np.hstack([Au.T, -(Au.T @ Ak + lam * Du.T @ Dk)]))
    return free, R[:, :A.shape[0]], R[:, A.shape[0]:]


def solver(A, D, fixed, lam):
    """fit(samples, x_fixed) -> the free control points, as fitter's, for a patch fitted once."""
    free = np.setdiff1d(np.arange(A.shape[1]), fixed)
    lam = lam * np.trace(A.T @ A) / max(np.trace(D.T @ D), 1e-12)
    Au, Ak, Du, Dk = A[:, free], A[:, fixed], D[:, free], D[:, fixed]
    cf = cho_factor(Au.T @ Au + lam * Du.T @ Du)
    Ru = Au.T @ Ak + lam * Du.T @ Dk
    return free, lambda X, xk: cho_solve(cf, Au.T @ X - Ru @ xk)


def taubin(V, F, it, lam=0.5, mu=-0.53):
    """Taubin smoothing (no shrinking): the scan's noise and the facets' creases off, its shape kept."""
    A = igl.adjacency_matrix(F).astype(float)
    W = diags(1 / np.maximum(np.asarray(A.sum(1)).ravel(), 1)) @ A
    X = np.array(V, float)
    for _ in range(it):
        for k in (lam, mu):
            X = X + k * (W @ X - X)
    return X


def split4(V, F):
    """Each triangle into four at its edge midpoints (same surface, finer): face f's children are 4f..4f+3,
    4f+3 the middle one."""
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    und, inv = np.unique(np.sort(E, axis=1), axis=0, return_inverse=True)
    ab, bc, ca = (len(V) + inv.ravel()).reshape(3, -1)
    a, b, c = F.T
    F2 = np.stack([np.stack([a, ab, ca], 1), np.stack([ab, b, bc], 1), np.stack([ca, bc, c], 1),
                   np.stack([ab, bc, ca], 1)], 1).reshape(-1, 3)
    return np.vstack([V, V[und].mean(1)]), F2


def face_graph(V, F, bend):
    """Faces joined across edges; the step costs the centroid distance plus bend mm per radian of turn."""
    TT = igl.triangle_triangle_adjacency(F)[0]
    c = V[F].mean(1)
    n = igl.per_face_normals(V, F, np.array([0.0, 0, 1]))
    i = np.repeat(np.arange(len(F)), 3)
    j = TT.ravel()
    ok = j >= 0
    i, j = i[ok], j[ok]
    w = np.linalg.norm(c[i] - c[j], axis=1) + bend * np.arccos(np.clip((n[i] * n[j]).sum(1), -1, 1)) + 1e-9
    return csr_matrix((w, (i, j)), shape=(len(F), len(F))), TT


def farthest(G, k, seeds):
    seeds = list(seeds)
    d = dijkstra(G, indices=seeds, min_only=True) if seeds else np.full(G.shape[0], np.inf)
    while len(seeds) < k:
        s = int(np.argmax(d))           # inf first: a piece with no seed yet
        if d[s] <= 0:
            break
        seeds.append(s)
        d = np.minimum(d, dijkstra(G, indices=s, min_only=True, limit=d[s] if np.isfinite(d[s]) else np.inf))
    return seeds


def grow(G, seeds, TT=None):
    d, _, src = dijkstra(G, indices=seeds, min_only=True, return_predecessors=True)
    lab = np.argsort(seeds)[np.searchsorted(np.sort(seeds), src)]
    for _ in range(4 if TT is not None else 0):     # a face with two sides on one other cell joins it:
        nb = lab[TT]                                # fewer stair steps along the borders
        two = np.where(nb[:, 0] == nb[:, 1], nb[:, 0], np.where(nb[:, 1] == nb[:, 2], nb[:, 1], nb[:, 2]))
        flip = (two != lab) & (((nb == two[:, None]).sum(1)) >= 2)
        flip[seeds] = False
        if not flip.any():
            break
        lab = np.where(flip, two, lab)
    return lab, d


def centred(G, TT, lab, seeds):
    """Lloyd step: each cell's seed moves to its face farthest from the cell's border."""
    border = np.flatnonzero((lab[TT] != lab[:, None]).any(1))
    d = dijkstra(G, indices=border, min_only=True)
    o = np.lexsort((-d, lab))
    first = o[np.r_[0, np.flatnonzero(np.diff(lab[o])) + 1]]
    return [int(f) for f in first]


def cells(F, TT, lab, nv):
    """Each cell's border loop (vertices in order, the cell on the left as seen from outside), the cell
    across each loop edge and the loop's corners; the cells that are not disks with >= 2 corners, and of
    those the disks with none (a cell inside one other)."""
    nf, k = len(F), lab.max() + 1
    fi = np.repeat(np.arange(nf), 3)
    ej = np.tile(np.arange(3), nf)
    other = lab[TT.ravel()]
    b = other != lab[fi]
    a, z, r, o = F[fi[b], ej[b]], F[fi[b], (ej[b] + 1) % 3], lab[fi[b]], other[b]
    nfc = np.bincount(lab, minlength=k)
    nvc = np.bincount(np.unique(lab[fi] * nv + F.ravel()) // nv, minlength=k)
    nb = np.bincount(r, minlength=k)
    chi = nvc - (3 * nfc + nb) // 2 + nfc
    order = np.argsort(r, kind="stable")
    splits = np.split(order, np.cumsum(nb)[:-1])
    out, bad, few = [None] * k, [], set()
    for c in range(k):
        idx = splits[c]
        if chi[c] != 1 or len(idx) < 3:
            bad.append(c)
            continue
        nxt = {}
        for e in idx:
            if a[e] in nxt:
                break
            nxt[a[e]] = e
        else:
            loop, e0 = [], idx[0]
            e = e0
            while True:
                loop.append(e)
                e = nxt.get(z[e])
                if e is None or e == e0 or len(loop) > len(idx):
                    break
            if e == e0 and len(loop) == len(idx):
                loop = np.array(loop)
                nb_ = o[loop]
                corner = np.flatnonzero(nb_ != np.roll(nb_, 1))
                if len(corner) >= 2:            # two corners: a lens, two quads
                    out[c] = dict(v=a[loop], nbr=nb_, corner=corner)
                    continue
                few.add(c)
        bad.append(c)
    return out, bad, few


def harmonic(Vl, Fl, bi, buv):
    """Interior of a disk mapped with the boundary held on the polygon: cotangent weights, or uniform
    (Tutte, always an embedding) when cotangent weights fold a triangle."""
    n = len(Vl)
    inner = np.setdiff1d(np.arange(n), bi)
    for L in (-igl.cotmatrix(Vl, Fl), None):
        if L is None:
            A = igl.adjacency_matrix(Fl).astype(float)
            L = diags(np.asarray(A.sum(1)).ravel()) - A
        L = L.tocsr()
        uv = np.zeros((n, 2))
        uv[bi] = buv
        if len(inner):
            Lii = L[inner][:, inner]
            rhs = -(L[inner][:, bi] @ buv)
            uv[inner] = spsolve(Lii.tocsc(), rhs).reshape(len(inner), 2)
        e1, e2 = uv[Fl[:, 1]] - uv[Fl[:, 0]], uv[Fl[:, 2]] - uv[Fl[:, 0]]
        if (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0] > -1e-12).all():
            return uv
    return uv


def chart_map(uv, Fl, X):
    """Points of the disk -> 3D through the map: barycentric in the param triangle that holds the point,
    found among all the cell's triangles (a nearest-few search misses in a squeezed map and extrapolates);
    a point just outside the inscribed polygon takes its border triangle, a sliver off."""
    a, b, c = uv[Fl[:, 0]], uv[Fl[:, 1]], uv[Fl[:, 2]]
    v0, v1 = b - a, c - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    ok = np.abs(den) > 1e-14
    den = np.where(ok, den, 1)

    def at(q):
        q = np.asarray(q, float)
        best, l1b, l2b, sc = (np.zeros(len(q), np.int64), np.zeros(len(q)), np.zeros(len(q)), np.full(len(q), -np.inf))
        step = max(1, 4_000_000 // max(len(Fl), 1))
        for s0 in range(0, len(q), step):
            v2 = q[s0:s0 + step, None] - a
            l1 = (v2[..., 0] * v1[:, 1] - v1[:, 0] * v2[..., 1]) / den
            l2 = (v0[:, 0] * v2[..., 1] - v2[..., 0] * v0[:, 1]) / den
            score = np.where(ok, np.minimum(np.minimum(1 - l1 - l2, l1), l2), -np.inf)
            t = np.argmax(score, axis=1)
            r = np.arange(len(t))
            best[s0:s0 + step], l1b[s0:s0 + step], l2b[s0:s0 + step], sc[s0:s0 + step] = t, l1[r, t], l2[r, t], score[r, t]
        l0 = 1 - l1b - l2b
        return l0[:, None] * X[Fl[best, 0]] + l1b[:, None] * X[Fl[best, 1]] + l2b[:, None] * X[Fl[best, 2]]
    return at


def arc_length(P):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    return s


def resample(P, t):
    s = arc_length(P)
    s = s / s[-1] if s[-1] > 0 else np.linspace(0, 1, len(P))
    return np.stack([np.interp(t, s, P[:, k]) for k in range(3)], 1)


def surface(V, F, patches=3000, n=6, bend=10.0, tol=0.1, rounds=6, lam=0.1, spans=3, small=0.1, denoise=0,
            judge=None, log=print):
    """Patches over the closed, outward mesh (V, F). Returns dict(vertices, edges, faces, ...) for
    write_step: vertices (k,3); edges [(v0, v1, cps (n,3))]; faces [(cps (n,n,3), [(edge, forward) x4])]."""
    V, F = np.asarray(V, float), np.asarray(F, np.int64)
    # closed pieces: solids and the sealed cavities (voids) inside them, every piece turned to face out of itself
    F, comp, _, solids, voids, _ = solid_pieces(V, F)
    area = igl.doublearea(V, F).sum() / 2
    want = max(int(patches / 6), 4)
    fa0 = igl.doublearea(V, F) / 2
    m = 3 * n                                   # samples per patch side
    t_curve = np.linspace(0, 1, m)
    Ac = basis(t_curve, n)
    cfree, cR1, cR2 = fitter(Ac, second_diff(n), np.array([0, n - 1]), lam)

    def layout(seeds, rnd, G, TT, G0, fa):
        """Grow cells from the seeds until every one is a disk with >= 2 corners: (seeds, lab, dist, cells),
        or None when 150 tries do not get there."""
        merges = 0
        for tries in range(150):
            # every piece keeps a seed: a piece left without one (its seed dropped as a sliver or a lone face)
            # has no cell to grow and its faces would land in some cell of another piece
            have = set(comp[seeds].tolist()) if seeds else set()
            seeds = seeds + [int(np.flatnonzero(comp == pc)[0]) for pc in np.unique(comp) if pc not in have]
            if rnd == 0 and tries < 2:          # centred once: later rounds only add seeds, so cells
                seeds = list(dict.fromkeys(centred(G, TT, grow(G, seeds)[0], seeds)))   # away from a miss stay
            lab, dist = grow(G, seeds, TT)
            cl, bad, few = cells(F, TT, lab, len(V))
            if not bad:
                # a sliver cell makes slivers of patches (a CAD mesher spends thousands of triangles on each):
                # it goes to its neighbours, a few times over, at the first layout only (later a refining
                # seed's small cell is wanted)
                tiny = np.flatnonzero(np.bincount(lab, fa, len(seeds)) < small * area / want)
                if not len(tiny) or merges >= 3 or rnd > 0 or tries > 50:
                    return seeds, lab, dist, cl
                merges += 1
                seeds = [x for i, x in enumerate(seeds) if i not in set(tiny.tolist())]
                continue
            drop = set()
            for c in bad:                       # split a cell that is no disk at its face farthest from its seed
                fc = np.flatnonzero(lab == c)   # along the surface (the fold cost would pick a lone face on a
                far = seeds[c]                  # sharp edge that never grows a cell); one inside another goes to it
                if len(fc) > 1:
                    d0 = dijkstra(G0, indices=seeds[c], min_only=True)
                    far = int(fc[np.argmax(np.where(np.isfinite(d0[fc]), d0[fc], -1))])
                if c in few or far == seeds[c]:
                    drop.add(seeds[c])
                else:
                    seeds.append(far)
            seeds = [x for x in dict.fromkeys(seeds) if x not in drop]
        return None

    G0 = face_graph(V, F, 0.0)[0]
    for b in (bend, bend / 2, 0.0):             # a ring around a thin strand may never split where folds cost
        G, TT = face_graph(V, F, b)             # much: then less, and plain distance cuts any ring
        seeds = farthest(G, want, [])
        first = layout(seeds, 0, G, TT, G0, fa0)
        if first:
            break
        log(f"no disk layout with fold cost {b}: trying less")
    else:
        raise RuntimeError("no disk layout")
    fa = fa0
    Xs = taubin(V, F, denoise)

    def finish(out):
        piece = comp[np.asarray(kept[0])[[f[2] for f in out["faces"]]]]   # the seeds of the cells built
        out.update(cells=len(kept[2]), area=area, shell=piece, solids=[int(c) for c in solids],
                   voids={int(k): [int(c) for c in cs] for k, cs in voids.items()})
        return out

    for rnd in range(rounds + 4):               # past rounds: only pinched cells merged away
        res = first if rnd == 0 else layout(seeds, rnd, G, TT, G0, fa)
        if res is None:
            log(f"round {rnd}: no disk layout with the added seeds; the last round's patches stand")
            seeds, lab, cl = kept
            break
        seeds, lab, dist, cl = res
        kept = (list(seeds), lab, cl)
        out = build(V, F, lab, cl, n, t_curve, cR1, cR2, lam, spans, Xs)
        err, where = deviation(V, F, lab, out)
        worst = np.flatnonzero(err > tol)
        log(f"round {rnd}: {len(cl)} cells, {len(out['faces'])} patches, fit max {err.max():.3f} mm, "
            f"p95 {np.percentile(err, 95):.3f}, cells over {tol}: {len(worst)}")
        pinched = set(np.flatnonzero(err >= 1e3).tolist())
        done = not len(worst) or rnd >= rounds
        if not pinched and done and judge is not None:     # the CAD kernel's own check, last
            pinched = set(judge(finish(out)))
            if pinched:
                log(f"round {rnd}: the CAD kernel rejects the patches of {len(pinched)} cells: merged away")
        if not pinched and done:
            break
        cen = V[F].mean(1)
        for c in worst:                         # another seed at the cell's face nearest to where it misses
            if c in pinched or rnd >= rounds:
                continue
            fc = np.flatnonzero(lab == c)
            near = int(fc[np.argmin(((cen[fc] - where[c]) ** 2).sum(1))])
            seeds.append(near if near != seeds[c] else int(fc[np.argmax(dist[fc])]))
        # a pinched cell (two of its patch's sides touching) goes to its neighbours: another seed in a sliver
        # only makes two slivers
        seeds = [x for i, x in enumerate(dict.fromkeys(seeds)) if i not in pinched]
    return finish(out)


def merged(*ts):
    """The knot vector holding each of ts (every knot at its highest multiplicity)."""
    u = np.unique(np.concatenate(ts))
    return np.repeat(u, [max(int((t == x).sum()) for t in ts) for x in u])


def flipped(t):
    return np.round(1 - t[::-1], 12)


def spread(t, sp):
    """At least sp spans: a knot into the middle of the longest span until there are."""
    u = np.unique(t)
    while len(u) - 1 < sp:
        i = int(np.argmax(np.diff(u)))
        t = np.sort(np.r_[t, round((u[i] + u[i + 1]) / 2, 12)])
        u = np.unique(t)
    return t


def refined(t, c, target):
    """Control points of the B-spline (t, c) with its knots raised to target: the same curve, exactly."""
    b = BSpline(t, c, DEG)
    for x in np.unique(target):
        more = int((target == x).sum() - (b.t == x).sum())
        if more > 0:
            b = b.insert_knot(x, more)
    return b.c


def evaluate(G, U, V, u, v):
    """Points of the surface (control net G, knots U, V) at the param pairs (u[i], v[i])."""
    Bu = BSpline.design_matrix(np.clip(u, 0, 1), U, DEG).toarray()
    Bv = BSpline.design_matrix(np.clip(v, 0, 1), V, DEG).toarray()
    return np.einsum("ai,ijk,aj->ak", Bu, G, Bv)


CORNER = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], float)     # a patch's sides, walked ccw from (0, 0)
STEP = np.array([(1, 0), (0, 1), (-1, 0), (0, -1)], float)


def face_meshes(sf, g=10):
    """Each patch triangulated in its own (u, v): its border points are its edges' own samples, so a
    neighbour that shares an edge (whole, or as part of a longer side) has exactly the same points there,
    and a grid inside (finer for a patch of many spans). [(points, triangles)] per patch."""
    tc = np.linspace(0, 1, g + 1)
    ept = []
    for _, _, cps, kt in sf["edges"]:
        p = BSpline(kt, cps, DEG)(tc)
        p[0], p[-1] = cps[0], cps[-1]
        ept.append(p)
    out = []
    for G, sides, _, U, V, fr in sf["faces"]:
        uv, xyz = [], []
        for q in range(4):
            for h, (e, f) in enumerate(sides[q]):
                w = fr[q][h] + tc[:-1] * (fr[q][h + 1] - fr[q][h])
                uv.append(CORNER[q] + w[:, None] * STEP[q])
                xyz.append((ept[e] if f else ept[e][::-1])[:-1])
        gu = g * max(1, -(-(len(np.unique(U)) - 1) // 3))
        gv = g * max(1, -(-(len(np.unique(V)) - 1) // 3))
        iu, iv = [x.ravel() for x in np.meshgrid(np.linspace(0, 1, gu + 1)[1:-1], np.linspace(0, 1, gv + 1)[1:-1],
                                                 indexing="ij")]
        uv.append(np.stack([iu, iv], 1))
        xyz.append(evaluate(G, U, V, iu, iv))
        uv, xyz = np.vstack(uv), np.vstack(xyz)
        tri = Delaunay(uv).simplices
        a, b, c = uv[tri[:, 0]], uv[tri[:, 1]], uv[tri[:, 2]]
        cr = (b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0]
        tri = np.where((cr < 0)[:, None], tri[:, ::-1], tri)[np.abs(cr) > 1e-14]
        out.append((xyz, tri))
    return out


def tessellate(sf, g=12):
    """The patches as one welded triangle mesh per solid (its cavities' shells turned inward),
    [(vertices, faces)]: neighbours share their edges' sample points exactly, so the weld closes the shell."""
    parts = face_meshes(sf, g)
    shell = np.asarray(sf["shell"])
    out = []
    for s in sf["solids"]:
        voids = sf["voids"].get(s, [])
        t = np.concatenate([x[tr] for (x, tr), sh in zip(parts, shell) if sh == s] +
                           [x[tr][:, ::-1] for (x, tr), sh in zip(parts, shell) if sh in voids])
        v, inv = np.unique(np.round(t.reshape(-1, 3), 6), axis=0, return_inverse=True)
        f = inv.reshape(-1, 3)
        out.append((v, f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]))
    return out


def deviation(V, F, lab, sf):
    """Per cell, how far its patches miss the mesh the way the detail check sees it (mm), and where: the
    mesh's signed distance to the patches averaged over each vertex's neighbourhood (scan noise cancels out,
    a missed feature does not), or a quarter of the farthest a patch strays off the mesh, whichever is more."""
    pieces = face_meshes(sf, 6)                 # a triangle soup is enough for the winding number's sign
    off = np.cumsum([0] + [len(v) for v, _ in pieces])
    tv = np.vstack([v for v, _ in pieces])
    tf = np.vstack([f + o for (_, f), o in zip(pieces, off)])
    d = igl.signed_distance(V, tv, tf, igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER)[0]
    A = igl.adjacency_matrix(F) + sparse.eye(len(V))
    deg = np.asarray(A.sum(1)).ravel()
    for _ in range(2):                          # ~ the 2-ring mean: the validator's 24 nearest
        d = (A @ d) / deg
    nc = lab.max() + 1
    err, where = np.zeros(nc), np.zeros((nc, 3))
    vcell = np.empty(len(V), np.int64)
    vcell[F.ravel()] = np.repeat(lab, 3)
    o = np.lexsort((-np.abs(d), vcell))
    first = o[np.r_[0, np.flatnonzero(np.diff(vcell[o])) + 1]]
    err[vcell[first]], where[vcell[first]] = np.abs(d[first]), V[first]
    pts = tv
    pc = np.repeat([f[2] for f in sf["faces"]], [len(x) for x, _ in pieces])
    stray = 0.25 * np.sqrt(igl.point_mesh_squared_distance(pts, V, F)[0])
    o = np.lexsort((-stray, pc))
    first = o[np.r_[0, np.flatnonzero(np.diff(pc[o])) + 1]]
    more = stray[first] > err[pc[first]]
    err[pc[first[more]]], where[pc[first[more]]] = stray[first[more]], pts[first[more]]
    # a patch whose opposite sides touch is pinched (a CAD kernel rejects its self-intersecting border): its
    # cell gets another seed like a miss
    tc = np.linspace(0, 1, 9)
    ept = [BSpline(kt, cps, DEG)(tc) for _, _, cps, kt in sf["edges"]]
    for G, sides, c, *_ in sf["faces"]:
        run = [np.vstack([ept[e] for e, _ in sd]) for sd in sides]
        size = max(np.ptp(np.vstack(run), 0).max(), 1e-9)
        for a, b in ((0, 2), (1, 3)):
            d = np.sqrt(((run[a][:, None] - run[b][None]) ** 2).sum(-1))
            if d.min() < 0.002 * size:
                i, j = np.unravel_index(np.argmin(d), d.shape)
                err[c], where[c] = max(err[c], 1e3), run[a][i]
    return err, where


def piece(t, c, a, b):
    """The B-spline (t, c) between params a and b (b < a: walked backwards) as a curve of its own on [0, 1]:
    the same points exactly (knots inserted at a and b, the control points between taken)."""
    snap = lambda x: t[np.argmin(np.abs(t - x))] if np.abs(t - x).min() < 1e-9 else x   # a knot a hair off: the knot
    lo, hi = snap(min(a, b)), snap(max(a, b))
    bs = BSpline(t, c, DEG)
    for x in (lo, hi):
        more = DEG - int((bs.t == x).sum())
        if more > 0:
            bs = bs.insert_knot(x, more)
    T = bs.t
    k0 = max(int(np.searchsorted(T, lo, "left")) - 1, 0) if lo > T[0] else 0
    K = np.r_[[lo] * (DEG + 1), T[(T > lo) & (T < hi)], [hi] * (DEG + 1)]
    C = bs.c[k0:k0 + len(K) - DEG - 1]
    K = np.round((K - lo) / (hi - lo), 12)
    return (K, C) if b > a else (flipped(K), C[::-1])


def joined(pieces, lens):
    """Curves (knots on [0, 1], control points) end to end, each over a share of [0, 1] by lens: a triple
    knot (C0) at each join. Returns knots, control points, the joins' fractions."""
    fr = np.round(np.r_[0, np.cumsum(lens)] / np.sum(lens), 12)
    t, cs = [0.0] * (DEG + 1), []
    for i, (K, C) in enumerate(pieces):
        t += list(fr[i] + K[DEG + 1:-DEG - 1] * (fr[i + 1] - fr[i]))
        t += [fr[i + 1]] * (DEG if i < len(pieces) - 1 else DEG + 1)
        cs.append(C if i == 0 else C[1:])
    return np.round(np.array(t), 12), np.vstack(cs), fr


def build(V, F, lab, cl, n, t_curve, cR1, cR2, lam, sp, X=None):
    """The patches over the cell layout. Cell borders are cut where cells meet into runs, each run into
    two halves at its midpoint, and every half is an edge. Halves that run straight on through a node (a
    run's midpoint, or a cell corner where one of its cells' border does not turn) are one smooth chain
    curve, so a patch side made of several edges is still one smooth curve; each edge is its piece of
    its chain, exactly. A cell with four or more corners where its border turns is one patch on a square;
    with three, a patch per corner around a centre. The layout follows V; the surfaces are fitted to X
    (V's points, denoised: a fitted surface that keeps the scan's ripple costs a CAD mesher millions of triangles)."""
    X = V if X is None else X
    verts, vid = [], {}

    def vertex(key, p):
        if key not in vid:
            vid[key] = len(verts)
            verts.append(np.asarray(p, float))
        return vid[key]

    edges, spokes, chains, info = [], [], [], {}     # info[e]: (chain, ta, tb), e = chain[ta..tb]

    def curve(v0, v1, pts):
        """A B-spline edge (its own chain) through the samples (at t_curve) from vertex v0 to v1."""
        fixed = np.stack([verts[v0], verts[v1]])
        cps = np.empty((n, 3))
        cps[[0, n - 1]] = fixed
        cps[1:n - 1] = cR1 @ pts + cR2 @ fixed
        chains.append((knots(n), cps))
        edges.append((v0, v1, cps, knots(n)))
        info[len(edges) - 1] = (len(chains) - 1, 0.0, 1.0)
        return len(edges) - 1

    def fit_net(U, V_, X, G):
        """The patch's inner control points fitted to samples X (at the grid S x T it returns first when
        X is None), its border G[rim] held; a bending penalty keeps it fair. Sparse: a patch's knots are its own."""
        nu, nv = len(U) - DEG - 1, len(V_) - DEG - 1
        su, sv = (np.arange(3 * nu) + 0.5) / (3 * nu), (np.arange(3 * nv) + 0.5) / (3 * nv)
        if X is None:
            S, T = np.meshgrid(su, sv, indexing="ij")
            return S.ravel(), T.ravel()
        A = sparse.kron(BSpline.design_matrix(su, U, DEG), BSpline.design_matrix(sv, V_, DEG)).tocsc()
        d2 = lambda k: sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(k - 2, k))
        D = sparse.vstack([sparse.kron(d2(nu), sparse.eye(nv)), sparse.kron(sparse.eye(nu), d2(nv))]).tocsc()
        gi, gj = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
        rim = ((gi == 0) | (gi == nu - 1) | (gj == 0) | (gj == nv - 1)).ravel()
        free, fixed = np.flatnonzero(~rim), np.flatnonzero(rim)
        flat = G.reshape(nu * nv, 3)
        Au, Ak, Du, Dk = A[:, free], A[:, fixed], D[:, free], D[:, fixed]
        w0 = lam * A.multiply(A).sum() / D.multiply(D).sum()
        lo, hi = np.minimum(X.min(0), flat[fixed].min(0)), np.maximum(X.max(0), flat[fixed].max(0))
        pad = 0.2 * np.linalg.norm(hi - lo)
        # a skewed patch can leave its middle free and the fit flies off: when a control point leaves the
        # samples' box by a fifth of its size, a stiffer fit instead
        for w in (w0, 100 * w0):
            rhs = Au.T @ X - (Au.T @ Ak + w * (Du.T @ Dk)) @ flat[fixed]
            flat[free] = spsolve((Au.T @ Au + w * (Du.T @ Du)).tocsc(), rhs).reshape(len(free), 3)
            if ((flat[free] >= lo - pad) & (flat[free] <= hi + pad)).all():
                break
        return flat.reshape(nu, nv, 3)

    def composite(side, lens):
        """A side of one or more edges as one curve: edges on one chain, in a row, are that chain's piece
        (smooth); a change of chain is a C0 join. Returns knots, control points, the edges' fractions."""
        groups = []
        for (e, f), ln in zip(side, lens):
            ch, ta, tb = info[e]
            s0, s1 = (ta, tb) if f else (tb, ta)
            if groups and groups[-1][0] == ch and abs(groups[-1][2] - s0) < 1e-12:
                groups[-1][2] = s1
                groups[-1][3] += ln
                groups[-1][4].append(s0)
            else:
                groups.append([ch, s0, s1, ln, [s0]])
        K, C, gfr = joined([piece(*chains[g[0]], g[1], g[2]) for g in groups], [g[3] for g in groups])
        fr = [gfr[i] + (x - g[1]) / (g[2] - g[1]) * (gfr[i + 1] - gfr[i]) for i, g in enumerate(groups) for x in g[4]]
        return K, C, np.round(np.r_[fr, 1.0], 12)

    nv_ = len(V)
    nrm = igl.per_vertex_normals(V, F)
    ring = lambda x: np.stack([np.cos(x), np.sin(x)], -1)
    line = lambda m: (lambda x: np.asarray(x)[..., None] * m)

    # pass 1: every cell's border runs, and how far its border turns at each corner
    runs, cdata = {}, []
    for c, cell in enumerate(cl):
        loop, corner = cell["v"], cell["corner"]
        k, L = len(corner), len(loop)
        P = V[loop]
        s = np.r_[0, np.cumsum(np.linalg.norm(np.roll(P, -1, 0) - P, axis=1))]
        per = s[-1]
        rk, rdir, rlen = [], [], []
        for j in range(k):
            a, z = corner[j], corner[(j + 1) % k]
            path = loop[np.arange(a, a + ((z - a) % L or L) + 1) % L]
            key = min(min(p, q) * nv_ + max(p, q) for p, q in zip(path[:-1], path[1:]))
            if key not in runs:                 # first cell to reach it sets its direction
                Q = X[path]
                sq = arc_length(Q)
                mid = resample(Q, np.array([0.5]))[0]
                hl = sq[-1] / 2
                h0 = np.vstack([Q[sq < hl], mid]) if (sq < hl).any() else np.vstack([Q[0], mid])
                h1 = np.vstack([mid, Q[sq > hl]]) if (sq > hl).any() else np.vstack([mid, Q[-1]])
                runs[key] = dict(start=int(path[0]), end=int(path[-1]), pts=(h0, h1), mid=mid)
            rk.append(key)
            rdir.append(runs[key]["start"] == path[0])
            rlen.append((s[z] - s[a]) % per or per)

        def turn(j):
            """How far the border turns left at corner j (rad): ~0 where it runs straight on."""
            a = corner[j]
            dl = min(0.3 * min(rlen[j - 1], rlen[j]), 1.0)
            back, fwd = (s[a] - s[:L]) % per, (s[:L] - s[a]) % per
            ip = int(np.argmin(np.abs(back - dl) + (back == 0) * 1e9))
            ix = int(np.argmin(np.abs(fwd - dl) + (fwd == 0) * 1e9))
            din, dout = P[a] - P[ip], P[ix] - P[a]
            return np.arctan2(nrm[loop[a]] @ np.cross(din, dout), din @ dout)
        th = np.array([turn(j) for j in range(k)])
        # patch corners: where the border turns (a corner where it runs straight on is a corner of the
        # neighbours only: a patch corner there would be a flat 180 degree pinch); at least three
        real = list(range(k)) if k == 2 else [j for j in range(k) if th[j] >= np.radians(30)]
        for j in np.argsort(-th):
            if len(real) >= 3:
                break
            if j not in real:
                real.append(int(j))
        cdata.append(dict(loop=loop, corner=corner, s=s, per=per, rk=rk, rdir=rdir, rlen=rlen, th=th, real=real))

    # chains: at a corner where a cell's border runs straight on, its two halves there continue each other
    half_ends = {}
    for key, r in runs.items():
        half_ends[key, 0] = (("c", r["start"]), ("m", key))
        half_ends[key, 1] = (("m", key), ("c", r["end"]))
    link, claim = {}, {}
    for key in runs:
        link[("m", key), (key, 0)], link[("m", key), (key, 1)] = (key, 1), (key, 0)
    for c, d in enumerate(cdata):
        k = len(d["corner"])
        for j in set(range(k)) - set(d["real"]):
            node = ("c", int(d["loop"][d["corner"][j]]))
            if node not in claim or abs(d["th"][j]) < claim[node][0]:
                rp, rn = d["rk"][j - 1], d["rk"][j]
                claim[node] = (abs(d["th"][j]), (rp, 1 if d["rdir"][j - 1] else 0), (rn, 0 if d["rdir"][j] else 1))
    for node, (_, ha, hb) in claim.items():
        link[node, ha], link[node, hb] = hb, ha

    half_edge, done = {}, set()
    for h0 in half_ends:
        if h0 in done:
            continue
        seq, seen = [(h0, True)], {h0}
        for forward in (False, True):        # grow the chain backwards from h0's start, then on from its end
            x, node = h0, half_ends[h0][1 if forward else 0]
            while (node, x) in link and link[node, x] not in seen:
                y = link[node, x]
                a, b = half_ends[y]
                along = (a == node) == forward  # y walked in its own direction
                seen.add(y)
                if forward:
                    seq.append((y, along))
                else:
                    seq.insert(0, (y, along))
                node = (b if along else a) if forward else (a if along else b)
                x = y
        done |= seen
        pts, nodes, tn = [], [], [0.0]
        for i, (h, f) in enumerate(seq):
            p = runs[h[0]]["pts"][h[1]]
            p = p if f else p[::-1]
            pts.append(p if i == 0 else p[1:])
            nodes.append(half_ends[h][0 if f else 1])
            tn.append(tn[-1] + arc_length(p)[-1])
        nodes.append(half_ends[seq[-1][0]][1 if seq[-1][1] else 0])
        pts = np.vstack(pts)
        tn = np.array(tn) / tn[-1]
        node_p = np.array([X[x[1]] if x[0] == "c" else runs[x[1]]["mid"] for x in nodes])
        # knots at the nodes and between: every half two spans however short (uniform knots put several
        # nodes in one span of a short half and the interpolation flies off)
        kt = np.r_[[0.0] * DEG, np.sort(np.r_[tn, (tn[:-1] + tn[1:]) / 2]), [1.0] * DEG]
        nc = len(kt) - DEG - 1
        ts = np.sort(np.r_[np.linspace(0, 1, 3 * nc), (tn[:-1] + tn[1:]) / 2])
        A, D, Cn = (BSpline.design_matrix(ts, kt, DEG).toarray(), second_diff(nc),
                    BSpline.design_matrix(tn, kt, DEG).toarray())
        w = lam * np.trace(A.T @ A) / np.trace(D.T @ D)
        kkt = np.block([[A.T @ A + w * D.T @ D, Cn.T], [Cn, np.zeros((len(tn), len(tn)))]])
        cps = np.linalg.solve(kkt, np.vstack([A.T @ resample(pts, ts), node_p]))[:nc]
        chains.append((kt, cps))
        ci = len(chains) - 1
        for i, (h, f) in enumerate(seq):
            ta, tb = (tn[i], tn[i + 1]) if f else (tn[i + 1], tn[i])
            K, C = piece(kt, cps, ta, tb)
            a, b = half_ends[h]
            v0 = vertex(a, X[a[1]] if a[0] == "c" else runs[a[1]]["mid"])
            v1 = vertex(b, X[b[1]] if b[0] == "c" else runs[b[1]]["mid"])
            C[0], C[-1] = verts[v0], verts[v1]
            edges.append((v0, v1, C, K))
            info[len(edges) - 1] = (ci, ta, tb)
            half_edge[h] = len(edges) - 1

    # pass 2: the patches
    faces = []
    for c, d in enumerate(cdata):
        loop, corner, s, per, real = d["loop"], d["corner"], d["s"], d["per"], d["real"]
        k, L = len(corner), len(loop)

        def pos(i):                     # arc position along the border from corner 0
            return (s[i] - s[corner[0]]) % per
        node_v, node_s, half, half_len = [], [], [], []
        for j in range(k):
            key, fw, run = d["rk"][j], d["rdir"][j], d["rlen"][j]
            e0, e1 = half_edge[key, 0], half_edge[key, 1]
            node_v += [vertex(("c", int(loop[corner[j]])), X[loop[corner[j]]]), vid["m", key]]
            node_s += [pos(corner[j]), pos(corner[j]) + run / 2]
            half += [(e0, True), (e1, True)] if fw else [(e1, False), (e0, False)]
            half_len += [run / 2, run / 2]
        nn = 2 * k

        def between(i, j):
            idx = [(i + t) % nn for t in range((j - i) % nn or nn)]
            return [half[x] for x in idx], [half_len[x] for x in idx]

        cid = np.flatnonzero(lab == c)
        vl, Fl = np.unique(F[cid], return_inverse=True)
        Fl = Fl.reshape(-1, 3)
        local = {g_: i for i, g_ in enumerate(vl)}
        bi = np.array([local[g_] for g_ in loop])

        def outline(cn, place):
            """The cell mapped harmonically with loop vertex at fraction w of side q (corner node cn[q] to
            cn[q+1]) held at place(q, w); returns param -> 3D."""
            ends = np.array([node_s[x] for x in cn] + [node_s[cn[0]] + per])
            buv = np.empty((L, 2))
            for i in range(L):
                p = pos(i)
                if p < ends[0] - 1e-12:
                    p += per
                q = min(int(np.searchsorted(ends, p + 1e-12, side="right")) - 1, len(cn) - 1)
                buv[i] = place(q, (p - ends[q]) / (ends[q + 1] - ends[q]))
            return chart_map(harmonic(V[vl], Fl, bi, buv), Fl, X[vl])

        def quad(sides, lens, param):
            """A patch on four sides (each one or more edges, walked ccw from its (0,0) corner), fitted to the
            mesh at param(S, T); its knots hold every side's, so each side is its edges exactly."""
            (k0, c0, f0), (k1, c1, f1), (k2, c2, f2), (k3, c3, f3) = [composite(sd, ln) for sd, ln in zip(sides, lens)]
            U, V_ = spread(merged(k0, flipped(k2)), sp), spread(merged(k1, flipped(k3)), sp)
            nu, nv = len(U) - DEG - 1, len(V_) - DEG - 1
            G = np.empty((nu, nv, 3))
            G[:, 0], G[:, nv - 1] = refined(k0, c0, U), refined(flipped(k2), c2[::-1], U)
            G[nu - 1, :], G[0, :] = refined(k1, c1, V_), refined(flipped(k3), c3[::-1], V_)
            S, T = fit_net(U, V_, None, G)
            G = fit_net(U, V_, param(S, T), G)
            faces.append((G, [list(x) for x in sides], c, U, V_, (f0, f1, f2, f3)))

        def coons(c0, c1, d0, d1):
            """Param points of a Coons patch: c0/c1 the t=0/1 sides, d0/d1 the s=0/1 sides, all along s or t."""
            def at_(S, T):
                k00, k10, k11, k01 = c0(0.0), c0(1.0), c1(1.0), c1(0.0)
                Sx, Tx = S[:, None], T[:, None]
                return ((1 - Tx) * c0(S) + Tx * c1(S) + (1 - Sx) * d0(T) + Sx * d1(T)
                        - (1 - Sx) * (1 - Tx) * k00 - Sx * (1 - Tx) * k10 - Sx * Tx * k11 - (1 - Sx) * Tx * k01)
            return at_

        if k == 2:      # a lens: one patch, its corners and side midpoints at 0, 90, 180, 270 degrees
            h = np.pi / 2
            at = outline([0, 2], lambda q, w: ring(np.pi * (q + w)))
            pm = coons(lambda x: ring(h * np.asarray(x)), lambda x: ring(3 * h - h * np.asarray(x)),
                       lambda x: ring(-h * np.asarray(x)), lambda x: ring(h + h * np.asarray(x)))
            quad([[half[0]], [half[1]], [half[2]], [half[3]]], [[1]] * 4, lambda S, T: at(pm(S, T)))
            continue
        if len(real) >= 4:              # one patch: the four sharpest corners, the border mapped to a square
            cn = [2 * j for j in sorted(sorted(real, key=lambda j: -d["th"][j])[:4])]
            at = outline(cn, lambda q, w: CORNER[q] + w * STEP[q])
            sides, lens = zip(*[between(cn[q], cn[(q + 1) % 4]) for q in range(4)])
            quad(list(sides), list(lens), lambda S, T: at(np.stack([S, T], 1)))
            continue
        # three corners: a patch per corner around a centre (centre, a point on each side, corner)
        cn = [2 * j for j in sorted(real)]
        dA = 2 * np.pi / 3
        at = outline(cn, lambda q, w: ring(dA * (q + w)))
        ends = [node_s[x] for x in cn] + [node_s[cn[0]] + per]
        centre = vertex(("o", c), at(np.zeros((1, 2)))[0])
        mnode, mw, spoke = [], [], []
        for q in range(3):              # the side's node nearest its middle
            inside = [(cn[q] + t) % nn for t in range(1, (cn[(q + 1) % 3] - cn[q]) % nn or nn)]
            ws = [((node_s[x] - ends[q]) % per) / (ends[q + 1] - ends[q]) for x in inside]
            b = int(np.argmin(np.abs(np.array(ws) - 0.5)))
            mnode.append(inside[b])
            mw.append(ws[b])
            pts = at(t_curve[:, None] * ring(dA * (q + ws[b])) * 0.9999)
            pts[-1] = verts[node_v[inside[b]]]
            spoke.append(curve(centre, node_v[inside[b]], pts))
            spokes.append(spoke[-1])
        for q in range(3):              # patch at corner q: centre, side q-1 point, corner q, side q point
            a0, a1 = dA * (q - 1 + mw[q - 1]), dA * (q + mw[q])
            s1, l1 = between(mnode[q - 1], cn[q])
            s2, l2 = between(cn[q], mnode[q])
            pm = coons(line(ring(a0)), lambda x, a1=a1, q=q: ring(a1 + np.asarray(x) * (dA * q - a1)),
                       line(ring(a1)), lambda x, a0=a0, q=q: ring(a0 + np.asarray(x) * (dA * q - a0)))
            quad([[(spoke[q - 1], True)], s1, s2, [(spoke[q], False)]], [[1], l1, l2, [1]],
                 lambda S, T, pm=pm: at(pm(S, T)))
    return dict(vertices=np.array(verts), edges=edges, faces=faces, n=n, spokes=spokes)


def kernel_rejects(sf, path):
    """The cells whose patches OCCT's shape checker rejects, sf written to path and read back; [] without OCCT."""
    try:
        from OCP.BRep import BRep_Tool
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_Reader
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS
    except ImportError:
        return []
    from scipy.spatial import cKDTree
    write_step(path, [("check", sf)])
    r = STEPControl_Reader()
    if r.ReadFile(str(path)) != IFSelect_RetDone or not r.TransferRoots():
        return []
    face_of = getattr(TopoDS, "Face_s", None) or TopoDS.Face
    first = cKDTree(np.array([f[0][0, 0] for f in sf["faces"]]))
    out, e = set(), TopExp_Explorer(r.OneShape(), TopAbs_FACE)
    while e.More():
        f = face_of(e.Current())
        if not BRepCheck_Analyzer(f).IsValid():
            p = BRep_Tool.Surface_s(f).Pole(1, 1)
            out.add(int(sf["faces"][first.query([p.X(), p.Y(), p.Z()])[1]][2]))
        e.Next()
    Path(path).unlink(missing_ok=True)
    return sorted(out)


def knot_text(t):
    u = np.unique(t)
    return "(" + ",".join(str(int((t == x).sum())) for x in u) + ")", "(" + ",".join(num(x, 12) for x in u) + ")"


def write_step(path, bodies):
    """bodies: [(name, surfaced)] from surface(); one MANIFOLD_SOLID_BREP per body (a closed shell)."""
    lines, cnt = [], [0]

    def ent(text):
        cnt[0] += 1
        lines.append(f"#{cnt[0]}={text};")
        return cnt[0]

    def pt(p):
        return ent(f"CARTESIAN_POINT('',({num(p[0], 6)},{num(p[1], 6)},{num(p[2], 6)}))")

    app = ent("APPLICATION_CONTEXT('automotive design')")
    ent(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{app})")
    pctx = ent(f"PRODUCT_CONTEXT('',#{app},'mechanical')")
    dctx = ent(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{app},'design')")
    mm = ent("(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))")
    rad = ent("(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))")
    sr = ent("(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())")
    unc = ent(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-06),#{mm},'distance_accuracy_value','')")
    ctx = ent(f"(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{unc}))"
              f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{mm},#{rad},#{sr}))REPRESENTATION_CONTEXT('',''))")
    o, z, x = pt((0, 0, 0)), ent("DIRECTION('',(0.,0.,1.))"), ent("DIRECTION('',(1.,0.,0.))")
    axis = ent(f"AXIS2_PLACEMENT_3D('',#{o},#{z},#{x})")
    ctx2d = ent("(GEOMETRIC_REPRESENTATION_CONTEXT(2)PARAMETRIC_REPRESENTATION_CONTEXT()REPRESENTATION_CONTEXT('2D SPACE',''))")
    info = dict(faces=0, edges=0, solids=0)
    for bname, sf in bodies:
        bname = str(bname).replace("'", "")
        n = sf["n"]
        vp = [pt(p) for p in sf["vertices"]]
        vx = [ent(f"VERTEX_POINT('',#{i})") for i in vp]
        ecp = [[vp[v0]] + [pt(p) for p in cps[1:-1]] + [vp[v1]] for v0, v1, cps, _ in sf["edges"]]
        on = [[] for _ in ecp]                  # each edge's (surface, start, run) in its two faces' (u, v)
        srfs = []
        for G, sides, _, U, V, fr in sf["faces"]:
            rows = ",".join("(" + ",".join(f"#{pt(p)}" for p in row) + ")" for row in G)
            (mu, ku), (mv, kv2) = knot_text(U), knot_text(V)
            srfs.append(ent(f"B_SPLINE_SURFACE_WITH_KNOTS('',{DEG},{DEG},({rows}),.UNSPECIFIED.,.F.,.F.,.F.,"
                            f"{mu},{mv},{ku},{kv2},.UNSPECIFIED.)"))
            for q in range(4):
                for h, (e, f) in enumerate(sides[q]):
                    a, b = CORNER[q] + fr[q][h] * STEP[q], CORNER[q] + fr[q][h + 1] * STEP[q]
                    on[e].append((srfs[-1], a, b - a) if f else (srfs[-1], b, a - b))
        # the edges' curves in each face's (u, v) as well (segments of iso lines): without them the reader
        # projects every edge onto both faces, most of a B-spline STEP's import time
        ee = []
        for (v0, v1, _, kt), ids, faces_on in zip(sf["edges"], ecp, on):
            mult, kv = knot_text(kt)
            cv = ent(f"B_SPLINE_CURVE_WITH_KNOTS('',{DEG},(" + ",".join(f"#{i}" for i in ids) +
                     f"),.UNSPECIFIED.,.F.,.F.,{mult},{kv},.UNSPECIFIED.)")
            pcs = []
            for srf, p0, run in faces_on:
                size = float(np.hypot(*run))
                cp = ent(f"CARTESIAN_POINT('',({num(p0[0], 12)},{num(p0[1], 12)}))")
                dr = ent(f"DIRECTION('',({num(run[0] / size, 1)},{num(run[1] / size, 1)}))")
                ln = ent(f"LINE('',#{cp},#{ent(f'VECTOR({chr(39)}{chr(39)},#{dr},{num(size, 12)})')})")
                pcs.append(ent(f"PCURVE('',#{srf},#{ent(f'DEFINITIONAL_REPRESENTATION({chr(39)}{chr(39)},(#{ln}),#{ctx2d})')})"))
            sc = ent(f"SURFACE_CURVE('',#{cv},(" + ",".join(f"#{x}" for x in pcs) + "),.CURVE_3D.)")
            ee.append(ent(f"EDGE_CURVE('',#{vx[v0]},#{vx[v1]},#{sc},.T.)"))
        faces = []
        for (_, sides, *_), srf in zip(sf["faces"], srfs):
            oe = [ent(f"ORIENTED_EDGE('',*,*,#{ee[e]},{'.T.' if f else '.F.'})") for sd in sides for e, f in sd]
            bound = ent(f"FACE_OUTER_BOUND('',#{ent('EDGE_LOOP(' + chr(39) * 2 + ',(' + ','.join(f'#{o}' for o in oe) + '))')},.T.)")
            faces.append(ent(f"ADVANCED_FACE('',(#{bound}),#{srf},.T.)"))
        # a solid per closed piece, its sealed cavities as voids (a cavity no solid holds is left out)
        shell = np.asarray(sf["shell"])

        def closed(sh):
            return ent("CLOSED_SHELL('',(" + ",".join(f"#{faces[i]}" for i in np.flatnonzero(shell == sh)) + "))")
        breps = []
        for k, sh in enumerate(sf["solids"]):
            cs = closed(sh)
            vs = [ent(f"ORIENTED_CLOSED_SHELL('',*,#{closed(c)},.F.)") for c in sf["voids"].get(sh, [])]
            label = f"{bname}{'' if k == 0 else f' {k + 1}'}"
            breps.append(ent(f"BREP_WITH_VOIDS('{label}',#{cs},(" + ",".join(f"#{o}" for o in vs) + "))") if vs
                         else ent(f"MANIFOLD_SOLID_BREP('{label}',#{cs})"))
        prod = ent(f"PRODUCT('{bname}','{bname}','',(#{pctx}))")
        pd = ent(f"PRODUCT_DEFINITION('design','',#{ent(f'PRODUCT_DEFINITION_FORMATION({chr(39)}{chr(39)},{chr(39)}{chr(39)},#{prod})')},#{dctx})")
        rep = ent(f"ADVANCED_BREP_SHAPE_REPRESENTATION('{bname}',(" + ",".join(f"#{b}" for b in breps) + f",#{axis}),#{ctx})")
        ent(f"SHAPE_DEFINITION_REPRESENTATION(#{ent(f'PRODUCT_DEFINITION_SHAPE({chr(39)}{chr(39)},{chr(39)}{chr(39)},#{pd})')},#{rep})")
        info["faces"] += len(faces)
        info["edges"] += len(ee)
        info["solids"] += len(breps)
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    head = ["ISO-10303-21;", "HEADER;", "FILE_DESCRIPTION(('B-spline surfaced solid from a repaired scan'),'2;1');",
            f"FILE_NAME('{Path(path).name}','{now}',(''),(''),'scanruler-swift','scanruler-swift','');",
            "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));", "ENDSEC;", "DATA;"]
    Path(path).write_text("\n".join(head + lines + ["ENDSEC;", "END-ISO-10303-21;", ""]), encoding="ascii")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stl")
    ap.add_argument("out")
    ap.add_argument("--patches", type=int, default=3000)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--tol", type=float, default=0.08)
    ap.add_argument("--rounds", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    v, f = indexed(read_stl(a.stl))
    sf = surface(v, f, a.patches, a.n, tol=a.tol, rounds=a.rounds)
    print(f"surfaced in {time.time() - t0:.0f} s")
    print(write_step(a.out, [(Path(a.stl).stem, sf)]))
    from step_faceted import check_import
    print(check_import(a.out))


if __name__ == "__main__":
    main()
