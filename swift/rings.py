# SPDX-License-Identifier: AGPL-3.0-only
"""Rods (staffs, shafts, pins, tubes: long round cylinders) and flat rings and gears in a solid (a figurine wrapped in gear wheels): two parallel flat faces,
an annulus each, round one axis. find() lists them; knife() is the ring as fitted - bore, teeth (their
profile the median over all teeth, so a tooth the figure hides is the others' shape), thickness - grown by
a margin. bodies.group() cuts a ring body out of the scanned solid with that knife: the body keeps every
scanned detail inside it, the rest keeps every detail outside, and the two together are the solid exactly.

    python swift/rings.py reduced.stl [--rings]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step_faceted import indexed  # noqa: E402

MIN_FACE_MM2 = 60       # smaller flat patches are not looked at


def _frame(n):
    u = np.cross(n, [1, 0, 0] if abs(n[0]) < 0.9 else [0, 1, 0])
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def _circle(p2):
    """Least-squares circle through 2D points (Kasa): centre, radius."""
    A = np.c_[2 * p2, np.ones(len(p2))]
    sol = np.linalg.lstsq(A, (p2 ** 2).sum(1), rcond=None)[0]
    return sol[:2], float(np.sqrt(sol[2] + sol[:2] @ sol[:2]))


def flat_patches(t, tol_deg=6.0, tol_mm=0.75, min_mm2=MIN_FACE_MM2):
    """Connected flat patches, grown triangle to triangle while the normal stays within tol_deg of the
    patch's and the centre within tol_mm of its plane (noisy scans: a neighbour-to-neighbour angle
    shatters a flat face). [(triangles, outward normal, offset, area)], largest first."""
    from collections import deque
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    a = np.linalg.norm(n, axis=1)
    n /= a[:, None] + 1e-20
    a /= 2
    c = t.mean(1)
    v, f = indexed(t, 5)
    he = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    fid = np.tile(np.arange(len(f)), 3)
    key = np.sort(he, 1)
    o = np.lexsort(key.T[::-1])
    same = (key[o][1:] == key[o][:-1]).all(1)
    A, B = fid[o][:-1][same], fid[o][1:][same]
    nb = coo_matrix((np.ones(len(A)), (A, B)), shape=(len(f),) * 2).tocsr()
    nb = nb + nb.T
    if len(f) != len(t):            # indexed() drops collapsed triangles: patches need the caller's numbering
        return []
    cos, owner, out = np.cos(np.radians(tol_deg)), np.full(len(t), -1), []
    for seed in np.argsort(-a):
        if owner[seed] >= 0:
            continue
        nn, d0, q, mem = n[seed].copy(), c[seed] @ n[seed], deque([seed]), [seed]
        owner[seed] = len(out)
        while q:
            k = q.popleft()
            for j in nb.indices[nb.indptr[k]:nb.indptr[k + 1]]:
                if owner[j] < 0 and n[j] @ nn > cos and abs(c[j] @ nn - d0) < tol_mm:
                    owner[j] = len(out); q.append(j); mem.append(j)
            if len(mem) in (20, 200, 2000):          # follow the patch's own plane as it grows
                m = np.array(mem)
                nn = (n[m] * a[m, None]).sum(0); nn /= np.linalg.norm(nn)
                d0 = float(np.average(c[m] @ nn, weights=a[m]))
        m = np.array(mem)
        if a[m].sum() >= min_mm2:
            nn = (n[m] * a[m, None]).sum(0); nn /= np.linalg.norm(nn)
            out.append((m, nn, float(np.average(c[m] @ nn, weights=a[m])), float(a[m].sum())))
        else:
            owner[m] = -2
    return sorted(out, key=lambda x: -x[3])


def _bore(t, m, n, rng, tries=300):
    """The circle bounding a flat patch from inside (a bore's edge): RANSAC over its boundary vertices,
    the circle with most of them on it and none of the patch inside it. (centre 3D, radius, inliers) or None."""
    u, w = _frame(n)
    tri = t[m]
    v, f = indexed(tri, 5)
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1)
    ue, cnt = np.unique(e, axis=0, return_counts=True)
    bv = np.unique(ue[cnt == 1])
    if len(bv) < 12:
        return None
    b2 = np.c_[v[bv] @ u, v[bv] @ w]
    inside = np.c_[tri.mean(1) @ u, tri.mean(1) @ w]
    best, tol = None, 0.4
    for _ in range(tries):
        p = b2[rng.choice(len(b2), 3, replace=False)]
        try:
            c2, R = _circle(p)
        except np.linalg.LinAlgError:
            continue
        if not 2 < R < 1000:
            continue
        on = np.abs(np.linalg.norm(b2 - c2, axis=1) - R) < tol
        if on.sum() < 12 or (np.linalg.norm(inside - c2, axis=1) < R - 2 * tol).mean() > 0.01:
            continue
        # the arc it follows: a quarter of the turn at least, or a bore is not what it is
        th = np.arctan2(*(b2[on] - c2).T[::-1])
        span = len(np.unique(((th + np.pi) / (2 * np.pi) * 36).astype(int) % 36))
        if span >= 9 and (best is None or on.sum() > best[2].sum()):
            best = (c2, R, on)
    if best is None:
        return None
    c2, R = _circle(b2[best[2]])
    d = float(np.median(tri.mean(1) @ n))
    return c2[0] * u + c2[1] * w + d * n, R


def find(tris, of=None):
    """Rings and gears: flat patches whose inner edge is one circle (a bore), the patches of one circle
    joined (a ring the figure breaks into arcs), faces both sides, 0.5-15 mm apart. of: each triangle's
    piece. Returns [{id, centre, normal, thickness, bore, tip, root, teeth, coverage, pieces, area}],
    largest first. A candidate list for the caller to judge, not a verdict."""
    t = np.asarray(tris, float)
    rng = np.random.default_rng(0)
    hyp = []
    for m, n, d, a in flat_patches(t):
        b = _bore(t, m, n, rng)
        if b:
            hyp.append(dict(m=m, n=n, d=d, a=a, c=b[0], R=b[1]))
    # one ring: the same axis line and bore, faces on either side
    groups = []
    for h in hyp:
        for g in groups:
            g0 = g[0]
            off = h["c"] - g0["c"]
            if abs(h["n"] @ g0["n"]) > np.cos(np.radians(4)) and abs(h["R"] - g0["R"]) < max(1.0, 0.03 * g0["R"]) \
                    and np.linalg.norm(off - (off @ g0["n"]) * g0["n"]) < max(1.5, 0.04 * g0["R"]) and abs(off @ g0["n"]) < 15:
                g.append(h)
                break
        else:
            groups.append([h])
    rings = []
    for g in groups:
        n = g[0]["n"]
        up = [h for h in g if h["n"] @ n > 0]
        down = [h for h in g if h["n"] @ n < 0]
        if not up or not down:
            continue
        m1, m2 = np.concatenate([h["m"] for h in up]), np.concatenate([h["m"] for h in down])
        # the ring's faces: the outermost planes of its patches along n (a face's patches can face either way where
        # the scan folds: Kurisu's fused gear read 1.8 mm thick from up-vs-down averages, 4.3 from the planes)
        lv = [h["d"] if h["n"] @ n > 0 else -h["d"] for h in g]
        d1, d2 = max(lv), -min(lv)
        gap = d1 + d2
        if not 0.5 < gap < 15:
            continue
        c = np.average([h["c"] for h in g], axis=0, weights=[h["a"] for h in g])
        R = float(np.average([h["R"] for h in g], weights=[h["a"] for h in g]))
        ring = _shape(t, np.concatenate([m1, m2]), n, c, R, d1 - gap / 2, gap)
        if ring:
            on = np.concatenate([m1, m2])
            rings.append(dict(ring, area=round(sum(h["a"] for h in g), 1),
                              pieces=sorted(set(np.asarray(of)[on].tolist())) if of is not None else []))
    rings.sort(key=lambda r: -r["tip"])
    for k, r in enumerate(rings):
        r["id"] = k
    return rings


def _shape(t, m, n, c, bore, mid, gap):
    """Tip, root and teeth of a ring from its faces' outer edge; None if it is no ring (under a third
    of the turn seen, or nothing outside the bore)."""
    u, w = _frame(n)
    # points along every face edge (a CAD mesh's long triangles have their corners far apart)
    a, b = t[m].reshape(-1, 3), t[m][:, [1, 2, 0]].reshape(-1, 3)
    k = np.minimum(400, np.ceil(np.linalg.norm(b - a, axis=1) / 0.2).astype(int) + 1)
    s = np.concatenate([np.linspace(0, 1, j) for j in k])
    p = np.repeat(a, k, axis=0) * (1 - s)[:, None] + np.repeat(b, k, axis=0) * s[:, None] - c
    p2 = np.c_[p @ u, p @ w]
    r = np.linalg.norm(p2, axis=1)
    th = np.arctan2(p2[:, 1], p2[:, 0])
    bo = ((th + np.pi) / (2 * np.pi) * 720).astype(int) % 720
    outer = np.full(720, np.nan)
    np.fmax.at(outer, bo, r)
    ok = np.isfinite(outer)
    if ok.mean() < 1 / 3 or np.nanmedian(outer) < bore + 0.5:
        return None
    # teeth need not be flat (pointed, bevelled): the outer edge is all surface in the ring's slab, out to a
    # little past the flat rim; the teeth its strongest period, their shape the median over all of them, so
    # a tooth a figure presses on or hides is the others' shape
    rim = float(np.nanpercentile(outer, 95))
    cap = rim + 1.5 * (rim - bore) + 2
    cen = t.mean(1) - c
    ax = cen @ n - (mid - c @ n)
    rc = np.hypot(cen @ u, cen @ w)
    near = (np.abs(ax) < gap / 2 + 0.5) & (rc > bore) & (rc < cap + 3)
    a, b = t[near].reshape(-1, 3), t[near][:, [1, 2, 0]].reshape(-1, 3)
    k = np.minimum(200, np.ceil(np.linalg.norm(b - a, axis=1) / 0.2).astype(int) + 1)
    s_ = np.concatenate([np.linspace(0, 1, j) for j in k]) if len(k) else np.zeros(0)
    q = np.repeat(a, k, axis=0) * (1 - s_)[:, None] + np.repeat(b, k, axis=0) * s_[:, None] - c
    q = q[np.abs(q @ n - (mid - c @ n)) < gap / 2 + 0.3]
    rq = np.hypot(q @ u, q @ w)
    q, rq = q[rq < cap], rq[rq < cap]
    edge = np.full(720, np.nan)
    np.fmax.at(edge, ((np.arctan2(q @ w, q @ u) + np.pi) / (2 * np.pi) * 720).astype(int) % 720, rq)
    outer = np.fmax(outer, edge)
    ok = np.isfinite(outer)
    x = np.where(ok, outer - np.nanmean(outer), 0)
    spec = np.abs(np.fft.rfft(x))
    k = int(6 + spec[6:121].argmax())
    teeth = k if spec[k] > 4 * np.median(spec[6:121]) and np.nanpercentile(outer, 90) - np.nanpercentile(outer, 10) > 0.3 else 0
    prof = _profile(outer, teeth)
    tip, root = float(prof.max()), float(prof.min())
    centre = c - (c @ n) * n + mid * n
    return dict(centre=centre.round(3).tolist(), normal=n.round(5).tolist(), thickness=round(gap, 3), bore=round(bore, 3),
                tip=round(tip, 3), root=round(root, 3), teeth=teeth, coverage=round(float(np.isfinite(edge).mean()), 2),
                _profile=prof)


def _profile(outer, teeth, n=64):
    """The outer edge over one tooth: the median across all teeth, so the teeth hidden or broken are the
    others' shape; a ring with no teeth is its median radius."""
    if not teeth:
        return np.full(n, np.nanmedian(outer))
    th = (np.arange(720) + 0.5) / 720
    phase = (th * teeth) % 1
    out = np.empty(n)
    for k in range(n):
        near = np.abs(((phase - (k + 0.5) / n) + 0.5) % 1 - 0.5) < 0.5 / n + 0.5 * teeth / 720
        out[k] = np.nanmedian(outer[near]) if np.isfinite(outer[near]).any() else np.nan
    return np.where(np.isfinite(out), out, np.nanmedian(outer))


def rod_knife(rod, grow=1.0, extend=(0.0, 0.0)):
    """A rod as followed (its path, bends and all), grown by grow mm round and at the ends, its ends moved
    out by extend (mm) along it (what a hand or hair hid): a manifold3d solid of cylinders along the path."""
    import manifold3d as m3d
    path = np.asarray(rod.get("path") or rod["ends"], float)
    a0 = path[1] - path[0] if len(path) > 1 else np.asarray(rod["axis"], float)
    a1 = path[-1] - path[-2] if len(path) > 1 else np.asarray(rod["axis"], float)
    path = np.vstack([path[0] - a0 / np.linalg.norm(a0) * (float(extend[0]) + grow), path[1:-1],
                      path[-1] + a1 / np.linalg.norm(a1) * (float(extend[1]) + grow)]) if len(path) > 1 else path
    r = rod["radius"] + grow
    keep = [0]                  # a cylinder per stretch of about a radius, a ball at each joint
    for i in range(1, len(path)):
        if np.linalg.norm(path[i] - path[keep[-1]]) >= r or i == len(path) - 1:
            keep.append(i)
    parts = []
    for i, j in zip(keep, keep[1:]):
        seg = path[j] - path[i]
        L = float(np.linalg.norm(seg))
        u, w = _frame(seg / L)
        parts.append(m3d.Manifold.cylinder(L, r, r, 48).transform(np.c_[u, w, seg / L, path[i]].tolist()))
        parts.append(m3d.Manifold.sphere(r, 48).translate(path[j].tolist()))
    return m3d.Manifold.batch_boolean(parts, m3d.OpType.Add)


def knife(ring, grow):
    """The ring as fitted, every side grown by grow mm: a manifold3d solid."""
    import manifold3d as m3d
    n = np.asarray(ring["normal"], float)
    u, w = _frame(n)
    prof, teeth = ring["_profile"], max(1, ring["teeth"])
    k = len(prof)
    ang = (np.arange(teeth * k) + 0.5) / (teeth * k) * 2 * np.pi - np.pi
    rad = np.tile(prof, teeth) + grow
    outer = np.c_[rad * np.cos(ang), rad * np.sin(ang)]
    bore = max(0.1, ring["bore"] - grow)
    a2 = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    hole = np.c_[bore * np.cos(a2), bore * np.sin(a2)][::-1]
    cs = m3d.CrossSection([outer, hole], m3d.FillRule.EvenOdd)
    h = ring["thickness"] + 2 * grow
    solid = m3d.Manifold.extrude(cs, h).translate([0, 0, -h / 2])
    c = np.asarray(ring["centre"], float)
    # the angle origin: _ring measured it from the in-plane centre with arctan2(w, u)
    M = np.c_[u, w, n, c]
    return solid.transform(M.tolist())


def follow_rod(rod, solid):
    """A rod followed out from both ends along the rod itself, bends and all: each step the cross-section a
    little ahead is read (inside the solid or not, on a small grid across the rod), the rod's own disc in it
    found, and the path re-centred on it. Where the disc runs into more material (a hand round a staff, the
    hair it passes) the path goes on straight; the rod ends where its core is no longer solid, or after 6
    radii (25 mm at least) hidden, at the last place it was seen. Sets path (centres, end to end), ends,
    length, and where it widened (a head, a knob: for the caller to look at)."""
    import igl
    from scipy import ndimage
    v, f = indexed(np.asarray(solid, float), 5)
    f = f.astype(np.int64)
    r = rod["radius"]
    R, h = r + 3.0, r / 5
    g = np.arange(-R, R + h / 2, h)
    gu, gw = np.meshgrid(g, g, indexing="ij")
    disc = gu ** 2 + gw ** 2 <= R ** 2
    halves = []
    for k, sgn in ((0, -1), (1, 1)):
        c = np.asarray(rod["ends"][k], float)
        d = sgn * np.asarray(rod["axis"], float)
        path, seen, s, widened = [c.copy()], 0, 0.0, None
        while s < 40 * max(r, 5):
            s += r / 2
            p = c + d * r / 2
            u, w = _frame(d)
            pts = p + gu[..., None] * u + gw[..., None] * w
            inside = (igl.fast_winding_number(v, f, pts.reshape(-1, 3)) > 0.5).reshape(gu.shape) & disc
            lab, _ = ndimage.label(inside)
            centre = np.unravel_index(np.argmin(np.where(inside, gu ** 2 + gw ** 2, np.inf)), gu.shape)
            if not inside[centre] or gu[centre] ** 2 + gw[centre] ** 2 > r ** 2:
                break                                       # no solid where the rod would go: it ended
            blob = lab == lab[centre]
            edge = (blob & ~ndimage.binary_erosion(disc, iterations=1)).any()
            req = np.sqrt(blob.sum() * h * h / np.pi)
            if not edge and abs(req - r) < 0.35 * r:        # its own disc, in the open: re-centre on it
                q = p + gu[blob].mean() * u + gw[blob].mean() * w
                step = q - c
                d = d * 0.6 + 0.4 * step / (np.linalg.norm(step) + 1e-12)
                d /= np.linalg.norm(d)
                c = q
                path.append(c.copy())
                seen = len(path) - 1
            else:                                           # in more material: straight on
                widened = widened or c.round(2).tolist()
                c = p
                path.append(c.copy())
                if (len(path) - 1 - seen) * r / 2 > max(6 * r, 25):
                    break
            if seen == len(path) - 1:
                widened = None
        halves.append((path[:seen + 1], widened))
    (a_, wa), (b_, wb) = halves
    # both halves start at their own end of the fitted run: b_ keeps its first point, or a far end followed no further
    # collapsed the rod to the near half and it was dropped as short (Frieren's staff at 112k faces)
    path = [*a_[::-1], *b_]
    path = np.array([x for x in path])
    rod["path"] = path.round(3).tolist()
    rod["ends"] = [path[0].round(2).tolist(), path[-1].round(2).tolist()]
    rod["length"] = round(float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()), 2)
    rod["centre"] = path[len(path) // 2].round(3).tolist()
    rod["widens"] = [x for x in (wa, wb) if x]
    rod["heads"] = []
    for k, (half, wid) in enumerate(halves):
        if wid:
            end = np.asarray(rod["ends"][k], float)
            d = ((path[0] - path[1]) if k == 0 else (path[-1] - path[-2])) if len(path) > 1 else \
                (-1 if k == 0 else 1) * np.asarray(rod["axis"], float)
            rod["heads"].append(_head(v, f, end, d / np.linalg.norm(d), r))
    return rod


def _head(v, f, end, d, r, reach=60.0, R=20.0):
    """What a rod runs into at an end (a staff's ornament, a pin's head): cross-sections walked on along
    the rod, the solid connected to the rod's line in each, until it ends - a box round it (along d, its
    width the widest it got), clean when it never ran past R of the line (nothing else joined it)."""
    import igl
    from scipy import ndimage
    h = 0.6
    g = np.arange(-R, R + h / 2, h)
    gu, gw = np.meshgrid(g, g, indexing="ij")
    disc = gu ** 2 + gw ** 2 <= R ** 2
    u, w = _frame(d)
    length, width, clean, lo_u, hi_u, lo_w, hi_w = 0.0, r, True, -r, r, -r, r
    s = 0.0
    while s < reach:
        s += r / 2
        p = end + d * s
        inside = (igl.fast_winding_number(v, f, (p + gu[..., None] * u + gw[..., None] * w).reshape(-1, 3)) > 0.5).reshape(gu.shape) & disc
        lab, _ = ndimage.label(inside)
        at = np.unravel_index(np.argmin(np.where(inside, gu ** 2 + gw ** 2, np.inf)), gu.shape)
        if not inside[at] or gu[at] ** 2 + gw[at] ** 2 > (2 * r) ** 2:
            break                       # nothing on the rod's line any more: past the head
        blob = lab == lab[at]
        if (blob & ~ndimage.binary_erosion(disc, iterations=1)).any():
            clean = False               # it runs on past R: something else joins it here
        lo_u, hi_u = min(lo_u, gu[blob].min()), max(hi_u, gu[blob].max())
        lo_w, hi_w = min(lo_w, gw[blob].min()), max(hi_w, gw[blob].max())
        length = s
    centre = end + d * length / 2 + u * (lo_u + hi_u) / 2 + w * (lo_w + hi_w) / 2
    return dict(at=end.round(2).tolist(), centre=centre.round(2).tolist(), size=[round(float(length) + 2, 1), round(float(hi_u - lo_u) + 2, 1), round(float(hi_w - lo_w) + 2, 1)],
                axes=[d.round(5).tolist(), u.round(5).tolist()], clean=bool(clean and length > r))


def find_rods(tris, of=None, samples=40000, tries=4000, seed=0, solid=None):
    """Long round cylinders on the surface, outside (a rod, staff, shaft, pin): RANSAC on oriented surface
    samples - two points' normals give the axis, their normal lines meeting give the centre - the
    cylinder with most samples on it (within 0.3 mm + 2 % of its radius, normals radial) taken, its own
    samples taken out, again. A rod is at least 4 radii long along one run (gaps under 4 radii or 12 mm
    bridged: a hand round it), seen round over half its turn, and over a third of its side on it.
    [{id, centre, axis, radius, length, ends, coverage, pieces}], longest first. Candidates, not a verdict."""
    from scipy.spatial import cKDTree
    t = np.asarray(tris, float)
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    a = np.linalg.norm(n, axis=1)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(t), samples, p=a / a.sum())
    r1, r2 = rng.random((2, samples))
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    P = t[pick, 0] + r1[:, None] * (t[pick, 1] - t[pick, 0]) + r2[:, None] * (t[pick, 2] - t[pick, 0])
    N = n[pick] / a[pick, None]
    tri_of = pick
    left = np.ones(samples, bool)
    tree = cKDTree(P)
    rods = []
    per = float(a.sum() / 2 / samples)          # surface each sample stands for
    for _ in range(24):
        idx = np.flatnonzero(left)
        if len(idx) < 200:
            break
        best = None
        sub = rng.choice(idx, min(len(idx), 4000), replace=False)     # tries scored on a subsample, the winner on all
        for i in rng.choice(idx, tries):
            near = [j for j in tree.query(P[i], 40)[1] if left[j] and j != i]       # a thin rod's own samples
            if not near:
                continue
            j = near[rng.integers(len(near))]
            ax = np.cross(N[i], N[j])
            if np.linalg.norm(ax) < 0.2:
                continue
            ax /= np.linalg.norm(ax)
            # the normal lines, in the plane across the axis, meet at the axis
            pi, pj = P[i] - (P[i] @ ax) * ax, P[j] - (P[j] @ ax) * ax
            ni, nj = N[i] - (N[i] @ ax) * ax, N[j] - (N[j] @ ax) * ax
            M = np.c_[ni, -nj]
            sol = np.linalg.lstsq(M, pj - pi, rcond=None)[0]
            c = pi + sol[0] * ni
            r = -sol[0]                        # outward normals point away from the axis: the axis lies behind
            if not 0.5 < r < 60 or abs(np.linalg.norm(pj - c) - r) > 0.3 + 0.02 * r:
                continue
            d = P[sub] - c
            d -= np.outer(d @ ax, ax)
            dist = np.linalg.norm(d, axis=1)
            score = int(((np.abs(dist - r) < 0.3 + 0.02 * r) & ((N[sub] * d).sum(1) > 0.85 * dist)).sum())
            if best is None or score > best[0]:
                best = (score, c, ax, r)
        if best is None:
            break
        _, c, ax, r = best
        d = P[idx] - c
        d -= np.outer(d @ ax, ax)
        dist = np.linalg.norm(d, axis=1)
        on = idx[(np.abs(dist - r) < 0.3 + 0.02 * r) & ((N[idx] * d).sum(1) > 0.85 * dist)]
        if len(on) < 100:       # a weak fit is set aside, not the end: Frieren's staff came after one (112k faces)
            left[on] = False
            continue
        first = on
        # refine on its samples: the centre and radius a circle fit across the axis, the axis their principal
        # direction where they run long along it (a short patch's principal direction is any)
        for _ in range(3):
            q = P[on]
            if len(q) < 50:
                break
            sv = np.linalg.svd(q - q.mean(0), full_matrices=False)
            if sv[1][0] > 3 * sv[1][1] and abs(sv[2][0] @ ax) > np.cos(np.radians(15)):
                ax = sv[2][0] * np.sign(sv[2][0] @ ax)
            u, w = _frame(ax)
            c2, r = _circle(np.c_[q @ u, q @ w])
            c = c2[0] * u + c2[1] * w
            dd = P[idx] - c
            dd -= np.outer(dd @ ax, ax)
            dist = np.linalg.norm(dd, axis=1)
            on = idx[(np.abs(dist - r) < 0.3 + 0.02 * r) & ((N[idx] * dd).sum(1) > 0.85 * dist)]
        left[first] = False
        left[on] = False
        if len(on) < 100:
            continue
        # the longest run along the axis without a gap over 2 radii
        h = np.sort((P[on] - c) @ ax)
        cut = np.flatnonzero(np.diff(h) > max(4 * r, 12.0))        # a hand round a staff hides that much
        runs = np.split(h, cut + 1)
        run = max(runs, key=lambda x: x[-1] - x[0])
        length = float(run[-1] - run[0])
        sel = on[(((P[on] - c) @ ax) >= run[0]) & (((P[on] - c) @ ax) <= run[-1])]
        u, w = _frame(ax)
        dd = P[sel] - c
        cov = len(np.unique((np.arctan2(dd @ w, dd @ u) / (2 * np.pi) * 36).astype(int) % 36)) / 36
        # most of its side must be on it: a torso fits a cylinder in places, not along its length
        if length < 4 * r or cov < 0.5 or len(sel) * per < 0.35 * 2 * np.pi * r * length:
            continue
        mid = c + ax * (run[0] + run[-1]) / 2
        rods.append(dict(centre=mid.round(3).tolist(), axis=ax.round(5).tolist(), radius=round(r, 3), length=round(length, 2),
                         ends=[(c + ax * run[0]).round(2).tolist(), (c + ax * run[-1]).round(2).tolist()], coverage=round(cov, 2),
                         pieces=sorted(set(np.asarray(of)[tri_of[sel]].tolist())) if of is not None else []))
    for r in rods:
        follow_rod(r, t if solid is None else solid)
    rods = [r for r in rods if r["length"] >= 4 * r["radius"]]        # followed, a rod still    rods.sort(key=lambda r: -r["length"])
    for k, r in enumerate(rods):
        r["id"] = k
    return rods


def draw_parts(t, rods, rings, png):
    """The solid with every rod and ring found in its own colour, labelled where it is."""
    from bodies import draw
    c = t.mean(1)
    lab = np.zeros(len(t), int)
    notes = []
    for k, r in enumerate(rods):
        ax, a = np.asarray(r["axis"]), np.asarray(r["ends"][0])
        h = (c - a) @ ax
        d = np.linalg.norm(c - a - np.outer(h, ax), axis=1)
        lab[(h > -1) & (h < r["length"] + 1) & (d < r["radius"] + 1)] = k + 1
        notes.append((f"rod {r['id']}", r["centre"]))
    for k, r in enumerate(rings):
        n, ctr = np.asarray(r["normal"]), np.asarray(r["centre"])
        u, w = _frame(n)
        h = (c - ctr) @ n
        rr = np.hypot((c - ctr) @ u, (c - ctr) @ w)
        lab[(np.abs(h) < r["thickness"] / 2 + 1) & (rr > r["bore"] - 1) & (rr < r["tip"] + 1)] = len(rods) + k + 1
        notes.append((f"ring {r['id']}", (ctr + u * (r["bore"] + r["tip"]) / 2).tolist()))
    draw(t, lab, notes, png, "parts found by shape: rods and rings/gears (grey: the rest)")


def find_stand(t):
    """A stand the solid stands on: its largest flat face at one end of it (the bottom), and a face parallel
    to it facing the other way within a quarter of the solid's height, a third of its size or more (the
    stand's top). {point, normal (down, into the stand), thickness, bottom_mm2, top_mm2} or None."""
    P = flat_patches(t)
    if not P:
        return None
    m, n, d, a = P[0]
    h = t.reshape(-1, 3) @ n
    height = float(h.max() - h.min())
    if d < h.max() - 0.05 * height:
        return None
    tops = [x for x in P if x[1] @ n < -0.99 and 0.3 < d + x[2] < 0.25 * height and x[3] >= 0.3 * a]
    if not tops:
        return None
    m2, n2, d2, a2 = max(tops, key=lambda x: x[3])
    c = t[m2].mean(1)
    return dict(point=np.average(c, axis=0, weights=_tri_area(t[m2])).round(3).tolist(), normal=n.round(5).tolist(),
                thickness=round(d + d2, 2), bottom_mm2=round(a, 1), top_mm2=round(a2, 1))


def _tri_area(t):
    return 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)


def describe_rods(rods):
    if not rods:
        return "no rods found (long round cylinders: 4 radii long at least, seen over half their turn)"
    return "\n".join([f"{len(rods)} rods/shafts:"] + [
        f"rod {r['id']}: R{r['radius']}, {r['length']} mm long from {r['ends'][0]} to {r['ends'][1]}, axis {r['axis']}, "
        f"seen over {100 * r['coverage']:.0f} % of the turn, in pieces {r['pieces']}"
        + "".join(f"; at {hd['at']} it runs into more material - a head? box {{centre {hd['centre']}, size {hd['size']}, axes {hd['axes']}}}"
                  + (" (clean: nothing else joins it)" if hd["clean"] else " (other material joins it: look before taking it)") for hd in r.get("heads", []))
        for r in rods])


def describe(rings):
    if not rings:
        return "no flat rings or gears found (two parallel flat annular faces round one axis)"
    return "\n".join([f"{len(rings)} rings/gears (flat faces both sides, round one axis):"] + [
        f"ring {r['id']}: centre {r['centre']}, axis {r['normal']}, thickness {r['thickness']}, bore R{r['bore']}, "
        f"tip R{r['tip']}, root R{r['root']}, {r['teeth'] or 'no'} teeth, seen over {100 * r['coverage']:.0f} % of the turn, "
        f"in pieces {r['pieces']}" for r in rings])


if __name__ == "__main__":
    from meshio_min import read_stl
    t = read_stl(sys.argv[1])
    print(describe_rods(find_rods(t)))
    if "--rings" in sys.argv:
        print(describe(find(t)))
