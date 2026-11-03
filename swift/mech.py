# SPDX-License-Identifier: AGPL-3.0-only
"""Engineered scans to analytic CAD (the mech route, .claude-work/RE_RESEARCH.md 5.2): the scan split at its
creases into smooth patches, each patch read as planes, cylinders, spheres and cones (one fit when it holds
within the noise, else RANSAC inside the patch - a tangent plane-fillet-plane chain), equal primitives merged,
then snapped to design intent: an orthogonal frame, coaxial bores, round diameters and plane spacings when the
snap moves the surface less than SNAP_MM. brep.py turns the primitives into one exact solid; the solid is read
back by OCCT and held to the validator's scan mode against the original mesh.

    python swift/mech.py part.stl [--out part_mech.step]
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meshio_min import read_stl, write_stl  # noqa: E402
from step_faceted import indexed  # noqa: E402
from validate import estimate_noise  # noqa: E402

CREASE_DEG = 25.0       # smooth patches end where the smoothed normals turn more than this
NORMAL_DEG = 20.0       # a face belongs to a primitive only when its normal agrees this well
AXIS_DEG = 2.0          # directions this close to a frame axis are taken as the axis
SNAP_MM = 0.015         # a round value is taken only when it moves the surface less than this
ORDER = ("plane", "cylinder", "cone", "sphere")      # first that holds within the noise wins (machined: cones over spheres)


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-300)


def perp(a):
    u = unit(np.cross(a, np.eye(3)[np.argmin(np.abs(a))]))
    return u, np.cross(a, u)


# ---- primitives: plane n.x = d; cylinder axis a through c, radius r; sphere c, r; cone apex c, axis a into the
# opening, half angle t. distance is signed (outward of the surface's own normal), normal is its gradient

def distance(p, P):
    k = p["kind"]
    if k == "plane":
        return P @ p["n"] - p["d"]
    if k == "sphere":
        return np.linalg.norm(P - p["c"], axis=1) - p["r"]
    v = P - p["c"]
    h = v @ p["a"]
    rho = np.linalg.norm(v - h[:, None] * p["a"], axis=1)
    return rho - p["r"] if k == "cylinder" else rho * np.cos(p["t"]) - h * np.sin(p["t"])


def normal(p, P):
    k = p["kind"]
    if k == "plane":
        return np.broadcast_to(p["n"], P.shape)
    if k == "sphere":
        return unit(P - p["c"])
    v = P - p["c"]
    rad = unit(v - (v @ p["a"])[:, None] * p["a"])
    return rad if k == "cylinder" else rad * np.cos(p["t"]) - p["a"] * np.sin(p["t"])


def fit_plane(P, N=None, scale=None, n=None):
    c = P.mean(0)
    if n is None:
        n = np.linalg.eigh(np.cov((P - c).T))[1][:, 0]
        if N is not None and (N @ n).sum() < 0:
            n = -n
    return dict(kind="plane", n=n, d=float(n @ c))


def refine(p, P, scale, fixed_axis=False):
    """Least squares on the surface distance, soft L1 past scale so stray points do not tilt it."""
    k = p["kind"]
    if k == "plane":
        q = fit_plane(P, n=p["n"] if fixed_axis else None)
        return q if q["n"] @ p["n"] >= 0 else dict(kind="plane", n=-q["n"], d=-q["d"])
    if k == "sphere":
        x = least_squares(lambda x: np.linalg.norm(P - x[:3], axis=1) - x[3], np.r_[p["c"], p["r"]],
                          loss="soft_l1", f_scale=scale, max_nfev=60).x
        return dict(kind="sphere", c=x[:3], r=abs(float(x[3])))
    a0, c0 = p["a"], p["c"]
    e1, e2 = perp(a0)
    na = 0 if fixed_axis else 2

    def make(x):
        a = a0 if fixed_axis else unit(a0 + x[0] * e1 + x[1] * e2)
        c = c0 + x[na] * e1 + x[na + 1] * e2 + (x[na + 2] * a0 if k == "cone" else 0)
        return dict(kind=k, a=a, c=c, **({"r": abs(float(x[-1]))} if k == "cylinder" else {"t": float(x[-1])}))
    x0 = np.r_[np.zeros(na + (3 if k == "cone" else 2)), p["r"] if k == "cylinder" else p["t"]]
    return make(least_squares(lambda x: distance(make(x), P), x0, loss="soft_l1", f_scale=scale, max_nfev=60).x)


def fit_cylinder(P, N, scale, a=None):
    a0 = np.linalg.eigh(N.T @ N)[1][:, 0] if a is None else a       # every normal is square to the axis
    u, v = perp(a0)
    x, y = P @ u, P @ v
    A, B, C = np.linalg.lstsq(np.c_[x, y, np.ones_like(x)], x * x + y * y, rcond=None)[0]   # Kasa circle
    cx, cy = A / 2, B / 2
    r = float(np.sqrt(max(C + cx * cx + cy * cy, 1e-12)))
    return refine(dict(kind="cylinder", c=cx * u + cy * v + (P.mean(0) @ a0) * a0, a=a0, r=r), P, scale, a is not None)


def fit_sphere(P, N, scale):
    x = np.linalg.lstsq(np.c_[2 * P, np.ones(len(P))], (P * P).sum(1), rcond=None)[0]
    return refine(dict(kind="sphere", c=x[:3], r=float(np.sqrt(max(x[3] + x[:3] @ x[:3], 1e-12)))), P, scale)


def fit_cone(P, N, scale):
    apex = np.linalg.lstsq(N, (N * P).sum(1), rcond=None)[0]       # every tangent plane holds the apex
    a = np.linalg.eigh(np.cov(N.T))[1][:, 0]                         # n.a is the same for every normal
    if (P - apex).mean(0) @ a < 0:
        a = -a
    v = P - apex
    h = v @ a
    rho = np.linalg.norm(v - h[:, None] * a, axis=1)
    return refine(dict(kind="cone", c=apex, a=a, t=float(np.arctan2(np.median(rho), np.median(h)))), P, scale)


FIT = dict(plane=fit_plane, cylinder=fit_cylinder, sphere=fit_sphere, cone=fit_cone)


def sane(p, size):
    k = p["kind"]
    if k in ("cylinder", "sphere"):
        return 0.05 < p["r"] < 5 * size and np.isfinite(p["c"]).all()
    if k == "cone":
        return np.radians(1) < p["t"] < np.radians(80) and np.linalg.norm(p["c"]) < 1e3 * size
    return True


def best(P, N, sigma, size, n=6000, seed=0):
    """The simplest primitive that holds within the noise on the region's surface samples (rms, p99): (prim, ok).
    Fitted on n of them, judged on all."""
    tried = []
    fit = np.random.default_rng(seed).choice(len(P), n, replace=False) if len(P) > n else slice(None)
    for k in ORDER:
        try:
            p = FIT[k](P[fit], N[fit], 2 * sigma)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if not sane(p, size):
            continue
        d = distance(p, P)
        rms = float(np.sqrt(np.mean(d * d)))
        if rms <= 1.5 * sigma + 0.003 and np.percentile(np.abs(d), 99) <= 5 * sigma + 0.01:
            return p, True
        tried.append((rms, k, p))
    return (min(tried, key=lambda x: x[0])[2] if tried else None), False


def inliers(p, P, N, eps, cosa):
    return (np.abs(distance(p, P)) < eps) & (np.abs((normal(p, P) * N).sum(1)) > cosa)


def two_point(p1, n1, p2, n2):
    """Cylinder and sphere through two oriented points (RANSAC's minimal sets)."""
    out = []
    a = np.cross(n1, n2)
    if np.linalg.norm(a) > 0.05:
        a = unit(a)
        u, v = perp(a)
        q1, q2 = np.array([p1 @ u, p1 @ v]), np.array([p2 @ u, p2 @ v])
        d1, d2 = np.array([n1 @ u, n1 @ v]), np.array([n2 @ u, n2 @ v])
        try:
            s1, _ = np.linalg.solve(np.c_[d1, -d2], q2 - q1)
            c2 = q1 + s1 * d1
            out.append(dict(kind="cylinder", c=c2[0] * u + c2[1] * v, a=a,
                            r=float((np.linalg.norm(q1 - c2) + np.linalg.norm(q2 - c2)) / 2)))
        except np.linalg.LinAlgError:
            pass
    w, b = p1 - p2, n1 @ n2
    if 1 - b * b > 1e-3:
        t1, t2 = (b * (n2 @ w) - n1 @ w) / (1 - b * b), (n2 @ w - b * (n1 @ w)) / (1 - b * b)
        c = (p1 + t1 * n1 + p2 + t2 * n2) / 2
        out.append(dict(kind="sphere", c=c, r=float((np.linalg.norm(p1 - c) + np.linalg.norm(p2 - c)) / 2)))
    return out


def largest_piece(mask, pairs, A):
    """The connected faces (edge-adjacent) of mask with the most area."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return idx
    m = np.full(len(mask), -1)
    m[idx] = np.arange(len(idx))
    e = m[pairs]
    e = e[(e >= 0).all(1)]
    _, lab = connected_components(csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(idx),) * 2), directed=False)
    return idx[lab == np.argmax(np.bincount(lab, weights=A[idx]))]


class Scan:
    """The mesh with what segmentation reads: face normals (raw and smoothed), areas, edge-adjacent pairs, and
    surface samples spread by area (a CAD mesh's 60 mm sliver triangle weighs what its area says, not three
    vertices) with their face; points(faces) gathers a region's samples."""

    def __init__(self, V, F, samples=200_000, seed=0):
        T = V[F]
        n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        self.V, self.F, self.A, self.N = V, F, np.linalg.norm(n, axis=1) / 2, unit(n)
        self.Ns, self.pairs = smooth_normals(F, self.N, self.A), edge_pairs(F)
        self.size = float(np.linalg.norm(np.ptp(V, axis=0)))
        k = np.clip(np.ceil(self.A / (self.A.sum() / samples)), 1, 400).astype(int)
        self.off = np.r_[0, np.cumsum(k)]
        self.Sf = np.repeat(np.arange(len(F)), k)
        r1, r2 = np.random.default_rng(seed).random((2, len(self.Sf)))
        s = np.sqrt(r1)
        t = T[self.Sf]
        self.S = t[:, 0] * (1 - s)[:, None] + t[:, 1] * (s * (1 - r2))[:, None] + t[:, 2] * (s * r2)[:, None]
        self.Sn = self.Ns[self.Sf]

    def index(self, faces):
        faces = np.asarray(faces)
        cnt = self.off[faces + 1] - self.off[faces]
        return np.repeat(self.off[faces] - np.r_[0, np.cumsum(cnt)[:-1]], cnt) + np.arange(cnt.sum())

    def points(self, faces):
        return self.S[self.index(faces)]


def ransac(sc, idx, sigma, min_area, rng, probe=20000):
    """Primitives of one smooth patch no single primitive explains (Schnabel et al. 2007, simplified: oriented
    minimal sets from samples, score = inlier samples, the winner refit; a face joins when most of its samples
    are inliers, and the region is the largest connected piece)."""
    eps, cosa = max(3 * sigma, 0.02), np.cos(np.radians(NORMAL_DEG))
    si = sc.index(idx)
    P, N, Fl = sc.S[si], sc.Sn[si], np.searchsorted(idx, sc.Sf[si])        # local face of each sample
    m = np.full(len(sc.F), -1)
    m[idx] = np.arange(len(idx))
    pl = m[sc.pairs]
    pl = pl[(pl >= 0).all(1)]
    Ai, cnt = sc.A[idx], np.bincount(Fl, minlength=len(idx))
    tree, R = cKDTree(P), max(2.0, 0.03 * sc.size)
    free, out = np.ones(len(idx), bool), []
    while Ai[free].sum() >= min_area and len(out) < 50:
        fs = np.flatnonzero(free[Fl])
        test = fs if len(fs) <= probe else rng.choice(fs, probe, replace=False)
        top, win = 0, None
        for _ in range(150):
            i = fs[rng.integers(len(fs))]
            cands = [dict(kind="plane", n=N[i], d=float(N[i] @ P[i]))]
            nb = tree.query_ball_point(P[i], R)
            j = nb[rng.integers(len(nb))]
            if free[Fl[j]] and j != i:
                cands += [p for p in two_point(P[i], N[i], P[j], N[j]) if sane(p, sc.size)]
            for p in cands:
                s = int(inliers(p, P[test], N[test], eps, cosa).sum())
                if s > top:
                    top, win = s, p
        if win is None:
            break
        for _ in range(2):
            ok = fs[inliers(win, P[fs], N[fs], eps, cosa)]
            if len(ok) < 30:
                break
            try:
                win = FIT[win["kind"]](P[ok], N[ok], 2 * sigma)
            except (np.linalg.LinAlgError, ValueError):
                break
        hit = np.bincount(Fl[fs[inliers(win, P[fs], N[fs], eps, cosa)]], minlength=len(idx))
        piece = largest_piece(free & (hit >= 0.7 * cnt), pl, Ai)
        if Ai[piece].sum() < min_area:
            break
        out.append(dict(prim=win, faces=idx[piece]))
        free[piece] = False
    return out


def smooth_normals(F, N, A, rounds=2, deg=40.0):
    """Face normals averaged over the faces sharing a vertex, only those within deg (creases stay sharp)."""
    nf = len(F)
    inc = csr_matrix((np.ones(3 * nf), (F.ravel(), np.repeat(np.arange(nf), 3))), shape=(F.max() + 1, nf))
    G = (inc.T @ inc).tocoo()
    w = A[G.col] * ((N[G.row] * N[G.col]).sum(1) > np.cos(np.radians(deg)))
    S = coo_matrix((w, (G.row, G.col)), shape=(nf, nf)).tocsr()
    for _ in range(rounds):
        N = unit(S @ N)
    return N


def edge_pairs(F):
    e = np.sort(F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    fid = np.repeat(np.arange(len(F)), 3)
    o = np.lexsort((e[:, 1], e[:, 0]))
    e, fid = e[o], fid[o]
    same = (e[1:] == e[:-1]).all(1)
    return np.c_[fid[:-1][same], fid[1:][same]]


def patches(sc):
    keep = sc.pairs[(sc.Ns[sc.pairs[:, 0]] * sc.Ns[sc.pairs[:, 1]]).sum(1) > np.cos(np.radians(CREASE_DEG))]
    return connected_components(csr_matrix((np.ones(len(keep)), (keep[:, 0], keep[:, 1])), shape=(len(sc.F),) * 2), directed=False)[1]


def patch_noise(sc, lab, n=6000, seed=0):
    """The scan's noise as its big patches (1 % of the area or more) show it: the area-weighted median rms of a
    plane fit, or of a cylinder fit where the normals spread past 5 deg. Both kinds count: a CAD mesh jittered
    into a 'scan' keeps its flat faces' vertices on their rims, so a turned part's planes hardly show the noise.
    None when no patch is big enough."""
    rows, total, rng = [], sc.A.sum(), np.random.default_rng(seed)
    area = np.bincount(lab, weights=sc.A)
    for pc in np.flatnonzero(area >= 0.01 * total):
        idx = np.flatnonzero(lab == pc)
        si = sc.index(idx)
        si = si if len(si) <= n else rng.choice(si, n, replace=False)
        P, N = sc.S[si], sc.Sn[si]
        p = fit_plane(P)
        if np.degrees(np.arccos(np.clip(np.percentile(N @ p["n"] * np.sign(N.sum(0) @ p["n"]), 5), -1, 1))) > 5:
            try:
                p = fit_cylinder(P, N, 0.1)
            except (np.linalg.LinAlgError, ValueError):
                continue
            if not sane(p, sc.size):
                continue
        d = distance(p, P)
        rows.append((float(np.sqrt((d * d).mean())), area[pc]))
    if not rows:
        return None
    rows.sort()
    w = np.cumsum([a for _, a in rows])
    return rows[int(np.searchsorted(w, w[-1] / 2))][0]


def segment(sc, sigma, lab, seed=0):
    """Regions {prim, faces, area} of the scan; lab = its crease patches."""
    min_area = max(1.0, 2e-4 * sc.A.sum())
    rng, regions = np.random.default_rng(seed), []
    for pc in np.flatnonzero(np.bincount(lab, weights=sc.A) >= min_area):
        idx = np.flatnonzero(lab == pc)
        si = sc.index(idx)
        p, ok = best(sc.S[si], sc.Sn[si], sigma, sc.size)
        regions += [dict(prim=p, faces=idx)] if ok else ransac(sc, idx, sigma, min_area, rng)
    for r in regions:
        r["area"] = float(sc.A[r["faces"]].sum())
    return regions, dict(faces=len(sc.F), area=float(sc.A.sum()), patches=int(lab.max() + 1), size=sc.size,
                         unexplained_mm2=round(float(sc.A.sum() - sum(r["area"] for r in regions)), 2))


def same(p, q, eps, deg=1.5):
    if p["kind"] != q["kind"]:
        return False
    cosd, k = np.cos(np.radians(deg)), p["kind"]
    if k == "plane":
        s = np.sign(p["n"] @ q["n"])
        return abs(p["n"] @ q["n"]) > cosd and abs(p["d"] - s * q["d"]) < eps
    if k == "sphere":
        return np.linalg.norm(p["c"] - q["c"]) < eps and abs(p["r"] - q["r"]) < eps
    if abs(p["a"] @ q["a"]) <= cosd:
        return False
    if k == "cylinder":
        w = p["c"] - q["c"]
        return abs(p["r"] - q["r"]) < eps and np.linalg.norm(w - (w @ p["a"]) * p["a"]) < eps
    return np.linalg.norm(p["c"] - q["c"]) < 3 * eps and abs(p["t"] - q["t"]) < np.radians(deg) and p["a"] @ q["a"] > 0


def coalesce(sc, regions, sigma, probe=6000, seed=0):
    """Neighbouring regions whose union one primitive explains within the noise become one, smallest first:
    noise tips the normals of thin triangles past the crease angle and splits a bore into strips."""
    rng, failed, ids = np.random.default_rng(seed), set(), itertools.count(len(regions))
    for i, r in enumerate(regions):
        r["id"] = i          # a pair that failed stays failed until one side changes (a new id)
    while True:
        lab = np.full(len(sc.F), -1)
        for i, r in enumerate(regions):
            lab[r["faces"]] = i
        e = lab[sc.pairs]
        e = np.unique(np.sort(e[(e >= 0).all(1) & (e[:, 0] != e[:, 1])], axis=1), axis=0)
        done = False
        for i, j in sorted(e.tolist(), key=lambda ij: min(regions[ij[0]]["area"], regions[ij[1]]["area"])):
            key = (regions[i]["id"], regions[j]["id"])
            if key in failed:
                continue
            faces = np.concatenate([regions[i]["faces"], regions[j]["faces"]])
            si = sc.index(faces)
            if len(si) > probe:
                si = rng.choice(si, probe, replace=False)
            p, ok = best(sc.S[si], sc.Sn[si], sigma, sc.size)
            if not ok:
                failed.add(key)
                continue
            regions[i] = dict(prim=p, faces=faces, area=regions[i]["area"] + regions[j]["area"], id=next(ids))
            del regions[j]
            done = True
            break
        if not done:
            return regions


def merge(regions, pts_of, sigma):
    """Regions on the same primitive (one plane split by a rib, a bore split by a cross hole) become one."""
    eps, root = max(3 * sigma, 0.03), list(range(len(regions)))

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i
    for i in range(len(regions)):
        for j in range(i):
            if same(regions[i]["prim"], regions[j]["prim"], eps):
                root[find(i)] = find(j)
    out = []
    for g in sorted({find(i) for i in range(len(regions))}):
        rs = [r for i, r in enumerate(regions) if find(i) == g]
        big = max(rs, key=lambda r: r["area"])
        faces = np.concatenate([r["faces"] for r in rs])
        out.append(dict(prim=refine(big["prim"], pts_of(faces), 2 * sigma), faces=faces, area=sum(r["area"] for r in rs)))
    # a small region lying on a bigger one's surface is part of it (a chamfer band piece a sphere also fits)
    out.sort(key=lambda r: -r["area"])
    i = len(out) - 1
    while i > 0:
        P = pts_of(out[i]["faces"])
        j = next((j for j in range(i) if np.mean(np.abs(distance(out[j]["prim"], P)) < eps) > 0.9), None)
        if j is not None:
            faces = np.concatenate([out[j]["faces"], out[i]["faces"]])
            out[j] = dict(prim=refine(out[j]["prim"], pts_of(faces), 2 * sigma), faces=faces, area=out[j]["area"] + out[i]["area"])
            del out[i]
        i -= 1
    return out


def snap(regions, pts_of, sigma):
    """Design intent: the frame of the biggest planes, directions on it, coaxial axes, round diameters and
    round spacings from each axis' biggest plane. Returns notes of what moved."""
    notes = []
    pts = [pts_of(r["faces"]) for r in regions]
    planes = sorted([r for r in regions if r["prim"]["kind"] == "plane"], key=lambda r: -r["area"])
    dirs = [r["prim"]["n"] for r in planes] + [r["prim"]["a"] for r in regions if r["prim"]["kind"] in ("cylinder", "cone")]
    if not dirs:
        return notes
    d1 = dirs[0]
    d2 = next((d for d in dirs[1:] if abs(d @ d1) < np.sin(np.radians(3))), perp(d1)[0])
    d2 = unit(d2 - (d2 @ d1) * d1)
    axes = np.array([d1, d2, np.cross(d1, d2)])
    cosa = np.cos(np.radians(AXIS_DEG))
    for r, P in zip(regions, pts):
        p, key = r["prim"], "n" if r["prim"]["kind"] == "plane" else "a"
        if key not in p:
            continue
        dots = axes @ p[key]
        k = int(np.argmax(np.abs(dots)))
        if abs(dots[k]) > cosa:
            p[key] = axes[k] * np.sign(dots[k])
            r["prim"] = fit_plane(P, n=p["n"]) if key == "n" else \
                (fit_cylinder(P, None, 2 * sigma, a=p["a"]) if p["kind"] == "cylinder" else refine(p, P, 2 * sigma, True))
    # coaxial: cylinders on one axis line share it
    eps = max(3 * sigma, 0.05)
    cyl = [i for i, r in enumerate(regions) if r["prim"]["kind"] == "cylinder"]
    done = set()
    for i in cyl:
        if i in done:
            continue
        a = regions[i]["prim"]["a"]
        grp = [j for j in cyl if j not in done and abs(regions[j]["prim"]["a"] @ a) > np.cos(np.radians(0.5))
               and np.linalg.norm((w := regions[j]["prim"]["c"] - regions[i]["prim"]["c"]) - (w @ a) * a) < eps]
        done.update(grp)
        if len(grp) > 1:
            w = np.array([regions[j]["area"] for j in grp])
            c = np.average([regions[j]["prim"]["c"] - (regions[j]["prim"]["c"] @ a) * a for j in grp], axis=0, weights=w)
            for j in grp:
                regions[j]["prim"].update(c=c, a=a)
                regions[j]["prim"]["r"] = float(np.mean(np.linalg.norm((v := pts[j] - c) - np.outer(v @ a, a), axis=1)))
            notes.append(f"coaxial: {len(grp)} cylinders")
    # round values: diameters to 0.1 mm, plane spacings and axis positions to 0.1 mm from each axis' biggest plane
    ref = {}
    for r in planes:
        k = int(np.argmax(np.abs(axes @ r["prim"]["n"])))
        if abs(axes[k] @ r["prim"]["n"]) > 0.999999:
            ref.setdefault(k, r["prim"]["d"] * np.sign(axes[k] @ r["prim"]["n"]))
    for r in regions:
        p = r["prim"]
        if p["kind"] == "cylinder":
            d = round(2 * p["r"], 1)
            if abs(d / 2 - p["r"]) <= SNAP_MM:
                p["r"] = d / 2
            for k in ref:
                if abs(axes[k] @ p["a"]) < 1e-9:
                    x = p["c"] @ axes[k] - ref[k]
                    if abs(round(x, 1) - x) <= SNAP_MM:
                        p["c"] = p["c"] + (round(x, 1) - x) * axes[k]
        elif p["kind"] == "plane":
            k = int(np.argmax(np.abs(axes @ p["n"])))
            s = np.sign(axes[k] @ p["n"])
            if k in ref and abs(axes[k] @ p["n"]) > 0.999999:
                x = p["d"] * s - ref[k]
                if abs(round(x, 1) - x) <= SNAP_MM:
                    p["d"] = float((ref[k] + round(x, 1)) * s)
    return notes


def blends(regions, pts_of, sigma, size):
    """Rounds: cylinder strips (under 200 deg around their axis) tangent to two planes square to the axis - the
    rolling-ball round of their edge (Kos, Martin, Varady 2000). They leave the cell complex, which cannot cut a
    tangent contact cleanly, and come back as a fillet of the sharp edge, tangent by construction. Pieces on one
    edge are one round; its radius is refit with the axis where the two planes put it (that radius from both),
    and taken as the nearest tenth when within 0.05 mm of it."""
    planes = [i for i, r in enumerate(regions) if r["prim"]["kind"] == "plane"]
    found = {}
    for i, r in enumerate(regions):
        p = r["prim"]
        if p["kind"] != "cylinder" or p["r"] > 0.2 * size:
            continue
        P = pts_of(r["faces"])
        u, v = perp(p["a"])
        ang = np.sort(np.arctan2((P - p["c"]) @ v, (P - p["c"]) @ u))
        if np.degrees(2 * np.pi - np.diff(np.r_[ang, ang[0] + 2 * np.pi]).max()) > 200:
            continue                    # a hole or a boss, not a strip
        near, tol = cKDTree(P), max(3 * sigma, 0.03, 0.15 * p["r"])
        tang = [j for j in planes if abs(regions[j]["prim"]["n"] @ p["a"]) < 0.02
                and abs(abs(distance(regions[j]["prim"], p["c"][None])[0]) - p["r"]) < tol
                and near.query(pts_of(regions[j]["faces"]), distance_upper_bound=0.5 * p["r"] + 1)[0].min() < np.inf]
        j2 = next((j for j in tang[1:] if abs(regions[j]["prim"]["n"] @ regions[tang[0]]["prim"]["n"]) < 0.99), None)
        if j2 is not None:
            found.setdefault(tuple(sorted((tang[0], j2))), []).append(i)
    out = []
    for (j1, j2), members in found.items():
        P = np.concatenate([pts_of(regions[i]["faces"]) for i in members])
        q1, q2 = regions[j1]["prim"], regions[j2]["prim"]
        a = unit(np.cross(q1["n"], q2["n"]))
        s1, s2 = np.sign(distance(q1, P).mean()), np.sign(distance(q2, P).mean())   # behind both: convex round
        M = np.array([q1["n"], q2["n"], a])

        def axis(rho):
            return np.linalg.solve(M, [q1["d"] + s1 * rho, q2["d"] + s2 * rho, P.mean(0) @ a])

        def cost(rho):
            w = P - axis(rho)
            return np.mean((np.linalg.norm(w - np.outer(w @ a, a), axis=1) - rho) ** 2)
        r0 = float(np.mean([regions[i]["prim"]["r"] for i in members]))
        rho = float(minimize_scalar(cost, bounds=(0.3 * r0, 2 * r0), method="bounded").x)
        if abs(round(rho, 1) - rho) <= 0.05:
            rho = round(rho, 1)
        h = P @ a
        # small regions on the round's own surface (a facet read as a plane, a stray strip) are the round too
        cyl, own = dict(kind="cylinder", a=a, c=axis(rho), r=rho), sum(regions[i]["area"] for i in members)
        for k, rk in enumerate(regions):
            if k in members or k in (j1, j2) or rk["area"] > own:
                continue
            Q = pts_of(rk["faces"])
            hq = Q @ a
            if np.mean(np.abs(distance(cyl, Q)) < max(3 * sigma, 0.05)) > 0.9 and hq.min() > h.min() - 1 and hq.max() < h.max() + 1:
                members.append(k)
        out.append(dict(regions=members, radius=rho, planes=[j1, j2], a=a, c=axis(rho), h=(float(h.min()), float(h.max()))))
    return out


def residuals(regions, pts_of):
    out = []
    for r in regions:
        d = distance(r["prim"], pts_of(r["faces"]))
        out.append(dict(kind=r["prim"]["kind"], area=round(r["area"], 1), mean=round(float(d.mean()), 4),
                        rms=round(float(np.sqrt((d * d).mean())), 4), p95=round(float(np.percentile(np.abs(d), 95)), 4),
                        **{k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else round(float(v), 5))
                           for k, v in r["prim"].items() if k != "kind"}))
    return out


def run(stl, out=None, log=print):
    """The mech route on one mesh: out (default <stl>_mech.step) written; the report dict."""
    import brep
    from step_faceted import check_import
    from validate import validate
    t0, stl = time.time(), Path(stl)
    out = Path(out) if out else stl.with_name(stl.stem + "_mech.step")
    tris = read_stl(stl)
    V, F = indexed(tris)
    size = float(np.linalg.norm(np.ptp(V, axis=0)))
    sc = Scan(V, F)
    lab = patches(sc)
    sigma = max(patch_noise(sc, lab) or estimate_noise(tris, max(0.05, size / 400)), 0.003)
    regions, seg = segment(sc, sigma, lab)
    log(f"[mech] noise {sigma:.4f} mm, {seg['patches']} patches -> {len(regions)} regions, "
        f"{seg['unexplained_mm2']} mm2 of {seg['area']:.0f} unexplained ({time.time() - t0:.0f} s)")
    regions = merge(coalesce(sc, regions, sigma), sc.points, sigma)
    snap(regions, sc.points, sigma)
    regions = merge(regions, sc.points, sigma)          # snapped, more of them read as one
    notes = snap(regions, sc.points, sigma)
    res = residuals(regions, sc.points)
    log(f"[mech] {len(regions)} primitives: " + ", ".join(f"{k} {sum(r['kind'] == k for r in res)}" for k in ORDER) + "; " + "; ".join(notes))
    rounds = blends(regions, sc.points, sigma, size)
    skip = {i for b in rounds for i in b["regions"]}
    for b in rounds:
        b["planes"] = [regions[j]["prim"] for j in b["planes"]]
    if rounds:
        log(f"[mech] {len(rounds)} rounds: " + ", ".join(f"R{b['radius']:.2f}" for b in rounds))
    shape, info = brep.solid([r["prim"] for i, r in enumerate(regions) if i not in skip], V, F, rounds=rounds, log=log)
    rep = dict(stl=str(stl), step=str(out), noise=sigma, segmentation=seg, primitives=res, notes=notes, solid=info)
    if shape is None:
        rep["verdict"] = "FAIL"
        return rep
    brep.write_step(shape, out)
    imp = check_import(out, meshes=True, deflection=0.01)
    if not imp["meshes"]:
        rep.update(verdict="FAIL", readback={k: imp.get(k) for k in ("ok", "solids", "problems")})
        return rep
    model = out.with_name(out.stem + "_readback.stl")
    write_stl(model, np.concatenate([v[f] for v, f in imp["meshes"]]))
    val = validate(stl, model, None, None, "scan", sigma)
    model.unlink(missing_ok=True)
    bambu = check_import(out, deflection=0.003)
    rep.update(validation=val, readback={k: imp[k] for k in ("ok", "solids", "faces", "valid", "problems")},
               bambu_like={k: bambu[k] for k in ("seconds", "faces", "triangles", "ok")},
               step_kb=round(out.stat().st_size / 1024, 1), seconds=round(time.time() - t0, 1))
    worst = max((abs(r["mean"]) for r in res), default=0)
    rep["verdict"] = "PASS" if val["verdict"] == "PASS" and imp["ok"] and info["valid"] else "FAIL"
    log(f"[mech] {rep['verdict']}: {info['faces']} faces ({info['kinds']}), STEP {rep['step_kb']} kB, scan->model rms "
        f"{val['input_to_model']['rms']:.4f} p95 {val['input_to_model']['p95']:.4f}, worst face bias {worst:.4f} mm, "
        f"Bambu-like {bambu['seconds']} s {bambu['triangles']} triangles ({rep['seconds']} s)")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stl")
    ap.add_argument("--out")
    a = ap.parse_args()
    rep = run(a.stl, a.out)
    Path(rep["step"]).with_suffix(".json").write_text(json.dumps(rep, indent=1, default=str))
    if rep.get("validation"):
        from validate import summary
        print(summary(rep["validation"]))
    sys.exit(0 if rep["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
