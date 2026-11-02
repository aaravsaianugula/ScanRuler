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
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import spsolve

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


def surface(V, F, patches=3000, n=6, bend=10.0, tol=0.08, rounds=6, lam=1e-3, fine=0, log=print):
    """Patches over the closed, outward mesh (V, F). Returns dict(vertices, edges, faces, ...) for
    write_step: vertices (k,3); edges [(v0, v1, cps (n,3))]; faces [(cps (n,n,3), [(edge, forward) x4])]."""
    V, F = np.asarray(V, float), np.asarray(F, np.int64)
    # closed pieces: solids and the sealed cavities (voids) inside them, every piece turned to face out of itself
    F, comp, _, solids, voids, _ = solid_pieces(V, F)
    area = igl.doublearea(V, F).sum() / 2
    want = max(int(patches / 6), 4)
    seeds = farthest(face_graph(V, F, bend)[0], want, [])
    for _ in range(fine):                       # cell borders run on mesh edges: finer edges, smoother borders
        V, F = split4(V, F)
        seeds = [4 * s + 3 for s in seeds]
    G, TT = face_graph(V, F, bend)
    m = 3 * n                                   # samples per patch side
    t_curve = np.linspace(0, 1, m)
    Ac = basis(t_curve, n)
    cfree, cR1, cR2 = fitter(Ac, second_diff(n), np.array([0, n - 1]), lam)
    s_in = (np.arange(m) + 0.5) / m
    Bs = basis(s_in, n)
    As = np.einsum("ai,bj->abij", Bs, Bs).reshape(m * m, n * n)
    gi, gj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    rim = np.flatnonzero(((gi == 0) | (gi == n - 1) | (gj == 0) | (gj == n - 1)).ravel())
    Ds = np.vstack([np.kron(second_diff(n), np.eye(n)), np.kron(np.eye(n), second_diff(n))])
    sfree, sR1, sR2 = fitter(As, Ds, rim, lam)
    stiff = fitter(As, Ds, rim, 100 * lam)[1:]     # for a quad too skewed to pin its middle: a smooth membrane
    S, T = np.meshgrid(s_in, s_in, indexing="ij")
    S, T = S.ravel(), T.ravel()
    for rnd in range(rounds + 1):
        for tries in range(100):                # grow until every cell is a disk with >= 2 corners
            if rnd == 0 and tries < 2:          # centred once: later rounds only add seeds, so cells
                seeds = list(dict.fromkeys(centred(G, TT, grow(G, seeds)[0], seeds)))   # away from a miss stay
            lab, dist = grow(G, seeds, TT)
            cl, bad, few = cells(F, TT, lab, len(V))
            if not bad:
                break
            drop = set()
            for c in bad:                       # split a cell that is no disk at its face farthest from its seed;
                fc = np.flatnonzero(lab == c)   # one inside another goes to it
                far = int(fc[np.argmax(dist[fc])]) if len(fc) else seeds[c]
                if c in few or far == seeds[c]:
                    drop.add(seeds[c])
                else:
                    seeds.append(far)
            seeds = [x for x in dict.fromkeys(seeds) if x not in drop]
        else:
            raise RuntimeError(f"no disk layout after 100 tries ({len(bad)} cells not disks)")
        out = build(V, F, lab, cl, n, m, t_curve, cR1, cR2, (sR1, sR2), stiff, rim, sfree, As, S, T)
        err, where = deviation(V, F, lab, out)
        worst = np.flatnonzero(err > tol)
        log(f"round {rnd}: {len(cl)} cells, {len(out['faces'])} patches, fit max {err.max():.3f} mm, "
            f"p95 {np.percentile(err, 95):.3f}, cells over {tol}: {len(worst)}")
        if not len(worst) or rnd == rounds:
            break
        cen = V[F].mean(1)
        for c in worst:                         # another seed at the cell's face nearest to where it misses
            fc = np.flatnonzero(lab == c)
            near = int(fc[np.argmin(((cen[fc] - where[c]) ** 2).sum(1))])
            seeds.append(near if near != seeds[c] else int(fc[np.argmax(dist[fc])]))
        seeds = list(dict.fromkeys(seeds))
    piece = comp[np.asarray(seeds)[[f[2] for f in out["faces"]]] // 4 ** fine]
    out.update(cells=len(cl), area=area, shell=piece, solids=[int(c) for c in solids],
               voids={int(k): [int(c) for c in cs] for k, cs in voids.items()})
    return out


def grids(sf, g=10):
    """Each patch evaluated on a g x g grid, as triangles: (points (P, g*g, 3), faces (P*2(g-1)^2, 3))."""
    n = sf["n"]
    B = basis(np.linspace(0, 1, g), n)
    Gs = np.stack([f[0] for f in sf["faces"]])
    pts = np.einsum("ai,bj,pijk->pabk", B, B, Gs).reshape(len(Gs), g * g, 3)
    i, j = np.meshgrid(np.arange(g - 1), np.arange(g - 1), indexing="ij")
    a = (i * g + j).ravel()
    # every cell split from its (0,0) corner, except the two that hold patch corners (0,1) and (1,0): split
    # there it would cut the corner, and at a straight one (a lens's side midpoint) the neighbour cuts the same
    other = ((i == 0) & (j == g - 2) | (i == g - 2) & (j == 0)).ravel()
    one = np.where(other[:, None], np.stack([a, a + g, a + 1], 1), np.stack([a, a + g, a + g + 1], 1))
    two = np.where(other[:, None], np.stack([a + g, a + g + 1, a + 1], 1), np.stack([a, a + g + 1, a + 1], 1))
    quad = np.concatenate([one, two])
    tri = (quad[None] + (np.arange(len(Gs)) * g * g)[:, None, None]).reshape(-1, 3)
    return pts, tri


def tessellate(sf, g=12):
    """The patches as one welded triangle mesh per solid (its cavities' shells turned inward),
    [(vertices, faces)]: neighbours share their edge curves exactly, so their grid points on a shared edge
    coincide and the weld closes the shell."""
    pts, tri = grids(sf, g)
    shell = np.repeat(np.asarray(sf["shell"]), 2 * (g - 1) ** 2)
    out = []
    for s in sf["solids"]:
        t = np.concatenate([pts.reshape(-1, 3)[tri[shell == s]]] +
                           [pts.reshape(-1, 3)[tri[shell == c][:, ::-1]] for c in sf["voids"].get(s, [])])
        v, inv = np.unique(np.round(t.reshape(-1, 3), 6), axis=0, return_inverse=True)
        f = inv.reshape(-1, 3)
        out.append((v, f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]))
    return out


def deviation(V, F, lab, sf):
    """Per cell, the farthest its patches lie from the mesh or the mesh from its patches (mm), and where:
    the point (on the patch or the mesh) at that distance."""
    pts, tri = grids(sf)
    cell = np.array([f[2] for f in sf["faces"]])
    nc = lab.max() + 1
    d_out = np.sqrt(igl.point_mesh_squared_distance(pts.reshape(-1, 3), V, F)[0]).reshape(len(pts), -1)
    err, where = np.zeros(nc), np.zeros((nc, 3))
    k = d_out.argmax(1)
    pm, pp = d_out[np.arange(len(pts)), k], pts[np.arange(len(pts)), k]
    o = np.lexsort((-pm, cell))
    first = o[np.r_[0, np.flatnonzero(np.diff(cell[o])) + 1]]
    err[cell[first]], where[cell[first]] = pm[first], pp[first]
    d_in = np.sqrt(igl.point_mesh_squared_distance(V, pts.reshape(-1, 3), tri)[0])
    vcell = np.empty(len(V), np.int64)
    vcell[F.ravel()] = np.repeat(lab, 3)
    o = np.lexsort((-d_in, vcell))
    first = o[np.r_[0, np.flatnonzero(np.diff(vcell[o])) + 1]]
    more = d_in[first] > err[vcell[first]]
    err[vcell[first[more]]], where[vcell[first[more]]] = d_in[first[more]], V[first[more]]
    return err, where


def build(V, F, lab, cl, n, m, t_curve, cR1, cR2, loose, stiff, rim, sfree, As, S, T):
    verts, vid = [], {}

    def vertex(key, p):
        if key not in vid:
            vid[key] = len(verts)
            verts.append(np.asarray(p, float))
        return vid[key]

    edges, arcs, spokes = [], {}, []

    def curve(v0, v1, pts):
        """A B-spline edge through the samples (at t_curve) from vertex v0 to v1."""
        fixed = np.stack([verts[v0], verts[v1]])
        cps = np.empty((n, 3))
        cps[[0, n - 1]] = fixed
        cps[1:n - 1] = cR1 @ pts + cR2 @ fixed
        edges.append((v0, v1, cps))
        return len(edges) - 1

    nv = len(V)
    faces = []
    for c, cell in enumerate(cl):
        loop, nbr, corner = cell["v"], cell["nbr"], cell["corner"]
        k, L = len(corner), len(loop)
        sides = []
        for j in range(k):                      # the border between corner j and j+1, as cell c walks it
            a, z = corner[j], corner[(j + 1) % k]
            idx = np.arange(a, a + ((z - a) % L or L) + 1) % L
            path = loop[idx]
            key = min(min(p, q) * nv + max(p, q) for p, q in zip(path[:-1], path[1:]))
            if key not in arcs:                 # first cell to reach it sets its direction
                P = V[path]
                s = arc_length(P)
                mid = resample(P, np.array([0.5]))[0]
                u0, u1 = vertex(("c", path[0]), P[0]), vertex(("c", path[-1]), P[-1])
                um = vertex(("m", key), mid)
                half = s[-1] / 2
                h0 = np.vstack([P[s < half], mid]) if (s < half).any() else np.vstack([P[0], mid])
                h1 = np.vstack([mid, P[s > half]]) if (s > half).any() else np.vstack([mid, P[-1]])
                e0 = curve(u0, um, resample(h0, t_curve))
                e1 = curve(um, u1, resample(h1, t_curve))
                arcs[key] = (path[0], e0, e1)
            start, e0, e1 = arcs[key]
            sides.append(((e0, True), (e1, True)) if start == path[0] else ((e1, False), (e0, False)))
        # the cell on a regular k-gon, its border by arc length along each side
        cid = np.flatnonzero(lab == c)
        Fc = F[cid]
        vl, Fl = np.unique(Fc, return_inverse=True)
        Fl = Fl.reshape(-1, 3)
        pos = {g: i for i, g in enumerate(vl)}
        # on the unit circle, not a polygon: three border vertices of one side would lie on a line and the
        # thin border triangles between them would map flat, their part of the mesh lost to the samples
        ang = 2 * np.pi * np.arange(k + 1) / k
        ring = lambda a: np.stack([np.cos(a), np.sin(a)], -1)
        Pk = ring(ang[:k])
        buv = np.empty((L, 2))
        for j in range(k):
            a, z = corner[j], corner[(j + 1) % k]
            idx = np.arange(a, a + ((z - a) % L or L) + 1) % L
            s = arc_length(V[loop[idx]])
            w = (s / s[-1])[:, None]
            buv[idx[:-1]] = ring(ang[j] + w[:-1, 0] * (ang[j + 1] - ang[j]))
        bi = np.array([pos[g] for g in loop])
        uv = harmonic(V[vl], Fl, bi, buv)
        at = chart_map(uv, Fl, V[vl])
        def quad(loop4, q):
            """A patch on four edges (walked ccw from its (0,0) corner), fitted to the mesh at param points q."""
            X = at(q)
            run = [edges[e][2] if f else edges[e][2][::-1] for e, f in loop4]
            G = np.empty((n, n, 3))
            G[:, 0], G[n - 1, :], G[:, n - 1], G[0, :] = run[0], run[1], run[2][::-1], run[3][::-1]
            flat = G.reshape(n * n, 3)
            # a skewed quad (a near-0 or near-180 degree corner) leaves its middle free and the fit flies off:
            # when a control point leaves the samples' box by a fifth of its size, the stiff fit instead
            lo, hi = np.minimum(X.min(0), flat[rim].min(0)), np.maximum(X.max(0), flat[rim].max(0))
            pad = 0.2 * np.linalg.norm(hi - lo)
            for R1, R2 in (loose, stiff):
                flat[sfree] = R1 @ X + R2 @ flat[rim]
                if ((flat[sfree] >= lo - pad) & (flat[sfree] <= hi + pad)).all():
                    break
            faces.append((flat.reshape(n, n, 3), loop4, c))

        def coons(c0, c1, d0, d1):
            """Param points of a Coons patch: c0/c1 the t=0/1 sides, d0/d1 the s=0/1 sides, all along s or t."""
            k00, k10, k11, k01 = c0(0.0), c0(1.0), c1(1.0), c1(0.0)
            Sx, Tx = S[:, None], T[:, None]
            return ((1 - Tx) * c0(S) + Tx * c1(S) + (1 - Sx) * d0(T) + Sx * d1(T)
                    - (1 - Sx) * (1 - Tx) * k00 - Sx * (1 - Tx) * k10 - Sx * Tx * k11 - (1 - Sx) * Tx * k01)

        if k == 2:      # a lens: one patch, its corners and side midpoints at 0, 90, 180, 270 degrees
            h = np.pi / 2
            quad([sides[0][0], sides[0][1], sides[1][0], sides[1][1]],
                 coons(lambda x: ring(h * np.asarray(x)), lambda x: ring(3 * h - h * np.asarray(x)),
                       lambda x: ring(-h * np.asarray(x)), lambda x: ring(h + h * np.asarray(x))))
            continue
        Mk = ring(ang[:k] + np.pi / k)                     # side j midpoint, between corner j and j+1
        centre = vertex(("o", c), at(np.zeros((1, 2)))[0])
        spoke = []
        for j in range(k):                                 # centre -> midpoint of side j
            um = edges[sides[j][0][0]][1] if sides[j][0][1] else edges[sides[j][0][0]][0]
            pts = at(t_curve[:, None] * Mk[j] * 0.9999)
            pts[-1] = verts[um]
            spoke.append(curve(centre, um, pts))
            spokes.append(spoke[-1])
        for j in range(k):                                 # quad at corner j: centre, mid j-1, corner j, mid j
            # Coons patch in the disk: two straight spokes, two arcs of the circle (angle ~ arc length)
            a0, a1 = ang[j] - np.pi / k, ang[j] + np.pi / k
            line = lambda m: (lambda x: np.asarray(x)[..., None] * m)
            quad([(spoke[j - 1], True), sides[j - 1][1], sides[j][0], (spoke[j], False)],
                 coons(line(Mk[j - 1]), lambda x: ring(a1 + np.asarray(x) * (ang[j] - a1)),
                       line(Mk[j]), lambda x: ring(a0 + np.asarray(x) * (ang[j] - a0))))
    return dict(vertices=np.array(verts), edges=edges, faces=faces, n=n, spokes=spokes)


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
        kn = knots(n)
        uk = np.unique(kn)
        mult = "(" + ",".join(str(int((kn == u).sum())) for u in uk) + ")"
        kv = "(" + ",".join(num(u, 9) for u in uk) + ")"
        vp = [pt(p) for p in sf["vertices"]]
        vx = [ent(f"VERTEX_POINT('',#{i})") for i in vp]
        ecp = [[vp[v0]] + [pt(p) for p in cps[1:-1]] + [vp[v1]] for v0, v1, cps in sf["edges"]]
        on = [[] for _ in ecp]                  # each edge's (surface, side, forward) on its two faces
        srfs = []
        for G, loop, _ in sf["faces"]:
            ids = np.empty((n, n), dtype=object)
            run = [ecp[e] if f else ecp[e][::-1] for e, f in loop]
            ids[:, 0] = run[0]
            ids[n - 1, :] = run[1]
            ids[:, n - 1] = run[2][::-1]
            ids[0, :] = run[3][::-1]
            for i in range(1, n - 1):
                for j in range(1, n - 1):
                    ids[i, j] = pt(G[i, j])
            rows = ",".join("(" + ",".join(f"#{x}" for x in row) + ")" for row in ids)
            srfs.append(ent(f"B_SPLINE_SURFACE_WITH_KNOTS('',{DEG},{DEG},({rows}),.UNSPECIFIED.,.F.,.F.,.F.,"
                            f"{mult},{mult},{kv},{kv},.UNSPECIFIED.)"))
            for side, (e, f) in enumerate(loop):
                on[e].append((srfs[-1], side, f))
        # the edges' curves in each face's (u, v) as well (all iso lines): without them the reader projects
        # every edge onto both faces, most of a B-spline STEP's import time
        start = {0: ((0, 0), (1, 0)), 1: ((1, 0), (0, 1)), 2: ((1, 1), (-1, 0)), 3: ((0, 1), (0, -1))}
        lines2d = {}
        for side in range(4):
            for f in (True, False):
                p0, d = start[side]
                if not f:                       # the edge runs the other way: t = 0 at the side's end
                    p0, d = (p0[0] + d[0], p0[1] + d[1]), (-d[0], -d[1])
                cp = ent(f"CARTESIAN_POINT('',({num(p0[0], 1)},{num(p0[1], 1)}))")
                dr = ent(f"DIRECTION('',({num(d[0], 1)},{num(d[1], 1)}))")
                lines2d[side, f] = ent(f"LINE('',#{cp},#{ent(f'VECTOR({chr(39)}{chr(39)},#{dr},1.)')})")
        ee = []
        for (v0, v1, _), ids, faces_on in zip(sf["edges"], ecp, on):
            cv = ent(f"B_SPLINE_CURVE_WITH_KNOTS('',{DEG},(" + ",".join(f"#{i}" for i in ids) +
                     f"),.UNSPECIFIED.,.F.,.F.,{mult},{kv},.UNSPECIFIED.)")
            pcs = [ent(f"PCURVE('',#{sr},#{ent(f'DEFINITIONAL_REPRESENTATION({chr(39)}{chr(39)},(#{lines2d[side, f]}),#{ctx2d})')})")
                   for sr, side, f in faces_on]
            sc = ent(f"SURFACE_CURVE('',#{cv},(" + ",".join(f"#{x}" for x in pcs) + "),.CURVE_3D.)")
            ee.append(ent(f"EDGE_CURVE('',#{vx[v0]},#{vx[v1]},#{sc},.T.)"))
        faces = []
        for (G, loop, _), srf in zip(sf["faces"], srfs):
            oe = [ent(f"ORIENTED_EDGE('',*,*,#{ee[e]},{'.T.' if f else '.F.'})") for e, f in loop]
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
