# SPDX-License-Identifier: AGPL-3.0-only
"""Independent accuracy gate: is the rebuilt CAD the part in the input mesh?

Shares no code with the scripts that built the model. Measures
  - two-way surface distance (input -> model and model -> input): RMS, P95, P99, max
  - per-face deviation of the input (every face of the scan must be reproduced)
  - volume difference
  - the analytic faces' dimensions (planes, cylinders, cones, spheres) matched input vs model
and returns PASS only when all are inside the tolerances for the input kind.

    python swift/validate.py input.stl model.stl [--faces data] [--model-faces data_model]
                             [--mode cad|scan] [--noise SIGMA] [--json out.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meshio_min import read_stl  # noqa: E402

# tolerances (mm, %). scan: distances are judged against the scan's own noise
TOL = {
    "cad": dict(rms=0.005, p95=0.01, p99=0.02, max=0.10, vol=0.25, face_p95=0.02, dim=0.005),
    "scan": dict(rms=None, p95=None, p99=None, max=None, vol=1.5, face_p95=None, dim=0.05),
}


def closest_on_tris(p, a, b, c):
    """Closest points on triangles (a, b, c) to points p, all (n, 3). Ericson, RTCD 5.1.5."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    v, w = vb / den, vc / den
    out = a + ab * v[:, None] + ac * w[:, None]                      # inside the face
    sel = lambda m, q: np.where(m[:, None], q, out)
    safe = lambda x: np.where(np.abs(x) < 1e-30, 1e-30, x)
    m = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)                 # edge bc
    out = sel(m, b + (c - b) * ((d4 - d3) / safe((d4 - d3) + (d5 - d6)))[:, None])
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)                           # edge ac
    out = sel(m, a + ac * (d2 / safe(d2 - d6))[:, None])
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)                           # edge ab
    out = sel(m, a + ab * (d1 / safe(d1 - d3))[:, None])
    out = sel((d6 >= 0) & (d5 <= d6), c)
    out = sel((d3 >= 0) & (d4 <= d3), b)
    out = sel((d1 <= 0) & (d2 <= 0), a)
    return out


class MeshDistance:
    """Exact point-to-mesh distance. Candidate triangles come from a dense surface sample
    (each sample remembers its triangle); the k nearest samples' triangles are tested exactly."""

    def __init__(self, tris, spacing):
        self.tris = tris
        pts, tid = sample(tris, spacing, return_tri=True)
        self.tid = tid
        self.tree = cKDTree(pts)

    def __call__(self, p, k=24, chunk=20000):
        """Unsigned distances, and their sign: +1 where p lies outside the mesh (along the nearest
        triangle's outward normal). shortcut: the sign is that one triangle's, so it can be wrong right
        at an edge or corner; good enough to say whether a cluster is extra or missing material."""
        d, sgn = np.empty(len(p)), np.empty(len(p))
        self.last_tri = np.empty(len(p), int)                       # the nearest triangle of each point
        nrm = np.cross(self.tris[:, 1] - self.tris[:, 0], self.tris[:, 2] - self.tris[:, 0])
        for s in range(0, len(p), chunk):
            q = p[s:s + chunk]
            _, idx = self.tree.query(q, k=k)
            cand = self.tid[idx]                                     # (m, k) triangle ids
            best = np.full(len(q), np.inf)
            side = np.ones(len(q))
            tri = cand[:, 0].copy()
            for j in range(k):
                t = self.tris[cand[:, j]]
                c = closest_on_tris(q, t[:, 0], t[:, 1], t[:, 2])
                dj = np.linalg.norm(q - c, axis=1)
                better = dj < best
                best = np.where(better, dj, best)
                tri = np.where(better, cand[:, j], tri)
                side = np.where(better, np.sign(((q - c) * nrm[cand[:, j]]).sum(1)) + (dj == 0), side)
            d[s:s + chunk], sgn[s:s + chunk] = best, np.where(side >= 0, 1.0, -1.0)
            self.last_tri[s:s + chunk] = tri
        return d, sgn


def shape_of(q):
    """What surface the points of a cluster lie on - a plane or a cylinder, with its numbers and rms -
    so the fix can be built from evidence, not from a bounding box. None when neither fits."""
    if len(q) < 12:
        return None
    c = q.mean(0)
    w, v = np.linalg.eigh(np.cov((q - c).T))
    if np.sqrt(max(w[0], 0)) < 0.03 + 0.002 * np.sqrt(w[2]):
        n = v[:, 0]
        return f"plane n={np.round(n, 3).tolist()} through {np.round(c, 2).tolist()} (rms {np.sqrt(max(w[0], 0)):.3f})"
    best = None
    for ax in [v[:, 2]] + list(np.eye(3)):        # the long direction, then the world axes
        e1 = np.cross(ax, [1, 0, 0] if abs(ax[0]) < 0.9 else [0, 1, 0]); e1 /= np.linalg.norm(e1)
        e2 = np.cross(ax, e1)
        x, y = (q - c) @ e1, (q - c) @ e2
        A = np.c_[2 * x, 2 * y, np.ones_like(x)]
        sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
        r = np.sqrt(max(sol[2] + sol[0] ** 2 + sol[1] ** 2, 0))
        res = float(np.sqrt(np.mean((np.hypot(x - sol[0], y - sol[1]) - r) ** 2)))
        if r > 0 and (best is None or res < best[0]):
            best = (res, r, ax, c + sol[0] * e1 + sol[1] * e2, np.ptp((q - c) @ ax))
    res, r, ax, pt, length = best
    if r > 1e3:
        return None
    # radius along the axis: constant is a cylinder, linear is a cone (a chamfer band, a drafted wall)
    t = (q - pt) @ ax
    rad = np.linalg.norm(q - pt - np.outer(t, ax), axis=1)
    k, a0 = np.polyfit(t, rad, 1) if np.ptp(t) > 1e-6 else (0.0, float(rad.mean()))
    cres = float(np.sqrt(np.mean((rad - (a0 + k * t)) ** 2)))
    tol = max(0.03, 0.01 * r)
    where = f"axis {np.round(ax, 3).tolist()} through {np.round(pt, 2).tolist()}"
    if res < tol and abs(k) < 0.02:
        return f"cylinder R{r:.3f} (dia {2 * r:.3f}) {where}, {length:.1f} mm long (rms {res:.3f})"
    if cres < tol:
        t0, t1 = float(t.min()), float(t.max())
        p0, p1 = np.round(pt + t0 * ax, 2).tolist(), np.round(pt + t1 * ax, 2).tolist()
        return (f"cone {where}: R{a0 + k * t0:.3f} at {p0} to R{a0 + k * t1:.3f} at {p1}, "
                f"half-angle {np.degrees(np.arctan(abs(k))):.1f} deg (rms {cres:.3f})")
    return None


def clusters(p, d, sgn, thresh, spacing, top=8):
    """Where the deviation exceeds thresh: connected groups of such points, largest first."""
    m = d > thresh
    if not m.any():
        return []
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    out = []
    for side in (1.0, -1.0):                    # outside and inside apart: one cluster is one kind of error
        k = m & (sgn == side)
        if k.sum() < 4:
            continue
        q, dd = p[k], d[k]
        pairs = cKDTree(q).query_pairs(2.5 * spacing, output_type="ndarray")
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(q), len(q))) if len(pairs) else coo_matrix((len(q), len(q)))
        _, lab = connected_components(g, directed=False)
        for c in np.unique(lab):
            i = lab == c
            if i.sum() < 4:
                continue
            out.append(dict(points=int(i.sum()), area_mm2=round(float(i.sum() * spacing ** 2), 2), shape=shape_of(q[i]),
                            mean=round(float(dd[i].mean()), 4), max=round(float(dd[i].max()), 4),
                            side="outside" if side > 0 else "inside",
                            center=np.round(q[i].mean(0), 2).tolist(),
                            box=[np.round(q[i].min(0), 2).tolist(), np.round(q[i].max(0), 2).tolist()]))
    return sorted(out, key=lambda c: -c["area_mm2"] * c["mean"])[:top]


def sample(tris, spacing, return_tri=False, seed=1):
    """Points spread evenly over the surface: about one per spacing^2 of area, at least one per triangle."""
    a = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    n = np.maximum(1, np.ceil(a / spacing ** 2)).astype(int)
    tid = np.repeat(np.arange(len(tris)), n)
    r = np.random.default_rng(seed).random((len(tid), 2))
    flip = r.sum(1) > 1
    r[flip] = 1 - r[flip]
    t = tris[tid]
    pts = t[:, 0] + (t[:, 1] - t[:, 0]) * r[:, :1] + (t[:, 2] - t[:, 0]) * r[:, 1:]
    return (pts, tid) if return_tri else pts


def sample_area(tris, n, seed=1):
    """n points spread over the surface in proportion to area - each stands for area/n of it, which
    sample()'s one-per-triangle floor breaks where triangles are small (a reduced mesh's details)."""
    a = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    tid = rng.choice(len(tris), n, p=a / a.sum())
    r = rng.random((n, 2))
    flip = r.sum(1) > 1
    r[flip] = 1 - r[flip]
    t = tris[tid]
    return t[:, 0] + (t[:, 1] - t[:, 0]) * r[:, :1] + (t[:, 2] - t[:, 0]) * r[:, 1:], tid


def volume(tris):
    return float(np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6)


def stats(d):
    return dict(rms=float(np.sqrt((d ** 2).mean())), p95=float(np.percentile(d, 95)),
                p99=float(np.percentile(d, 99)), max=float(d.max()), mean=float(d.mean()))


def load_faces(data):
    """analyze.py's data dir: faces.json (the faces), faces.npy (face of each triangle), V.npy/F.npy (the mesh)."""
    d = Path(data) if data else None
    if d is None or not (d / "faces.json").exists():
        return None
    fs = dict(prims=json.loads((d / "faces.json").read_text())["faces"])
    if (d / "faces.npy").exists() and (d / "V.npy").exists():
        fs["label"] = np.load(d / "faces.npy")
        fs["tris"] = np.load(d / "V.npy")[np.load(d / "F.npy")]
    return fs


def face_list(fs):
    return fs["prims"], fs.get("label")


def match_prims(src, dst, min_area):
    """For each analytic face of the input, the model face that is the same surface, and the dimension error."""
    out = []
    for f in src:
        k = f.get("kind")
        if k not in ("plane", "cylinder", "cone", "sphere") or f.get("area", 0) < min_area:
            continue
        best = None
        for g in dst:
            if g.get("kind") != k:
                continue
            if k == "plane":
                if abs(abs(np.dot(f["n"], g["n"])) - 1) > 1e-4:
                    continue
                err = abs(np.dot(f["n"], g["n"]) * g["d"] - f["d"])
                desc = f"plane n={np.round(f['n'], 4).tolist()} d={f['d']:.4f}"
            elif k == "sphere":
                err = max(abs(f["r"] - g["r"]), float(np.linalg.norm(np.subtract(f["c"], g["c"]))))
                desc = f"sphere R{f['r']:.4f}"
            else:
                ax_f, ax_g = np.asarray(f["axis"]), np.asarray(g["axis"])
                if abs(abs(ax_f @ ax_g) - 1) > 1e-3:
                    continue
                if k == "cylinder":
                    off = np.subtract(g["c"], f["c"])
                    line = float(np.linalg.norm(off - (off @ ax_f) * ax_f))
                    err = max(abs(f["r"] - g["r"]), line)
                    desc = f"cylinder R{f['r']:.4f} axis {np.round(ax_f, 3).tolist()} at {np.round(f['c'], 3).tolist()}"
                else:
                    off = np.subtract(g["apex"], f["apex"])
                    line = float(np.linalg.norm(off - (off @ ax_f) * ax_f))
                    err = max(abs(f["half"] - g["half"]) * 10, line)       # angle error weighted per 10 mm
                    desc = f"cone {np.degrees(f['half']):.3f}deg"
            if best is None or err < best[0]:
                best = (err, g)
        out.append(dict(face=f.get("id"), kind=k, desc=desc if best else f"{k} (no {k} in model)",
                        area=round(float(f.get("area", 0)), 2), err=None if best is None else float(best[0])))
    return out


def estimate_noise(tris, spacing):
    """The scan's noise sigma: the median local-plane residual (PCA over ~1.5 mm patches) of its surface.
    Curvature inflates it on rounds, so the median over the whole part is used. A patch is fitted on its own
    connected piece only: shells stacked a fraction of a mm apart (a game model's hair cards and outline shells)
    are no noise, and read across them Kurisu, a clean mesh, came out at 0.23 mm and passed a smeared face."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    pts, tid = sample(tris, spacing, return_tri=True)
    _, inv = np.unique(np.round(tris.reshape(-1, 3), 6), axis=0, return_inverse=True)
    f = inv.reshape(-1, 3)
    nv = int(f.max()) + 1
    piece = connected_components(coo_matrix((np.ones(3 * len(f)), (f.ravel(), np.roll(f, 1, axis=1).ravel())), shape=(nv, nv)),
                                 directed=False)[1][f[:, 0]][tid]
    tree = cKDTree(pts)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(pts), min(4000, len(pts)), replace=False)
    sig = []
    for k, nb in zip(pick, tree.query_ball_point(pts[pick], r=max(1.5, 4 * spacing))):
        nb = np.asarray(nb)
        nb = nb[piece[nb] == piece[k]]
        if len(nb) < 12:
            continue
        x = pts[nb] - pts[nb].mean(0)
        sig.append(np.sqrt(max(np.linalg.eigvalsh(x.T @ x / len(nb))[0], 0)))
    return float(np.median(sig)) if sig else 0.05


def validate(inp, model, faces=None, model_faces=None, mode="cad", noise=None, spacing=None):
    fa, fb = load_faces(faces), load_faces(model_faces)
    # label the input's own triangles when analyze.py kept them (its order is the faces.npy order)
    A = fa["tris"] if fa and "tris" in fa else read_stl(inp)
    B = read_stl(model)
    size = float(np.linalg.norm(np.ptp(A.reshape(-1, 3), axis=0)))
    spacing = spacing or max(0.05, size / 400)
    if mode == "scan" and noise is None:
        noise = estimate_noise(A, spacing)
    tol = dict(TOL[mode])
    if mode == "scan":
        # a fit to a noisy scan cannot beat its noise: allow ~1 sigma RMS and the tails of N(0, sigma)
        tol.update(rms=1.25 * noise + 0.01, p95=2.5 * noise + 0.02, p99=3.5 * noise + 0.03, max=8 * noise + 0.2,
                   face_p95=3 * noise + 0.03)

    pa, ta = sample(A, spacing, return_tri=True)
    pb = sample(B, spacing)
    d_ab, sg_ab = MeshDistance(B, spacing / 2)(pa)                  # input -> model: missing/misplaced features
    d_ba, sg_ba = MeshDistance(A, spacing / 2)(pb)                  # model -> input: extra material
    # input outside the model = material the model lacks; model outside the input = material it has too much of
    thresh = max(tol["p99"], 3 * (noise or 0) + 0.02)
    devs = [dict(c, of="scan", kind="missing material" if c["side"] == "outside" else "extra material") for c in clusters(pa, d_ab, sg_ab, thresh, spacing)]
    devs += [dict(c, of="model", kind="extra material" if c["side"] == "outside" else "missing material") for c in clusters(pb, d_ba, sg_ba, thresh, spacing)]
    s_ab, s_ba = stats(d_ab), stats(d_ba)
    area = float(0.5 * np.linalg.norm(np.cross(A[:, 1] - A[:, 0], A[:, 2] - A[:, 0]), axis=1).sum())
    min_band = max(2.0, 5e-4 * area)
    systematic = [c for c in devs if c["area_mm2"] >= min_band]
    va, vb = volume(A), volume(B)
    vol_pct = 100 * (vb - va) / va

    worst_faces = []
    if fa is not None:
        prims, labels = face_list(fa)
        if labels is not None and len(labels) == len(A):
            lab = labels[ta]
            for f in prims:
                m = lab == f.get("id")
                if m.sum() < 5:
                    continue
                p95 = float(np.percentile(d_ab[m], 95))
                if p95 > tol["face_p95"]:
                    b0, b1 = f.get("bbox", [[0] * 3, [0] * 3])
                    worst_faces.append(dict(face=f.get("id"), kind=f.get("kind"), area=round(float(f.get("area", 0)), 2),
                                            p95=round(p95, 4), box=[np.round(b0, 3).tolist(), np.round(b1, 3).tolist()]))
            worst_faces.sort(key=lambda x: -x["p95"] * x["area"])

    dims = []
    if fa is not None and fb is not None:
        total = sum(p.get("area", 0) for p in face_list(fa)[0])
        dims = match_prims(face_list(fa)[0], face_list(fb)[0], min_area=max(0.5, 1e-3 * total))
    bad_dims = [d for d in dims if d["err"] is None or d["err"] > tol["dim"]]

    checks = {
        "rms": max(s_ab["rms"], s_ba["rms"]) <= tol["rms"],
        "p95": max(s_ab["p95"], s_ba["p95"]) <= tol["p95"],
        "p99": max(s_ab["p99"], s_ba["p99"]) <= tol["p99"],
        "max": max(s_ab["max"], s_ba["max"]) <= tol["max"],
        "volume": abs(vol_pct) <= tol["vol"],
        "faces": not worst_faces,
        # noise beyond the threshold is scattered; a connected band of it is a wrong surface (a fillet for a chamfer)
        "systematic": not systematic,
        "dimensions": not bad_dims,
    }
    return dict(
        verdict="PASS" if all(checks.values()) else "FAIL", mode=mode, noise=noise, spacing=spacing,
        checks=checks, tolerances=tol,
        input_to_model=s_ab, model_to_input=s_ba,
        volume=dict(input=va, model=vb, diff_pct=vol_pct),
        worst_faces=worst_faces[:10], bad_dimensions=bad_dims[:15], deviation_clusters=sorted(devs, key=lambda c: -c["area_mm2"] * c["mean"])[:10],
        dimensions_checked=len(dims), dimensions_ok=len(dims) - len(bad_dims),
    )


def edge_counts(tris):
    """Open (one face) and non-manifold (three or more) edges of a triangle soup, by shared vertex."""
    _, inv = np.unique(np.round(tris.reshape(-1, 3), 6), axis=0, return_inverse=True)
    f = inv.reshape(-1, 3)
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, n = np.unique(e, axis=0, return_counts=True)
    return int((n == 1).sum()), int((n > 2).sum())


def enclosing(tris, pts, eps=0.05):
    """Which sample points lie on surface that bounds the scan's volume (generalized winding number):
    inside just behind them, outside just in front. The rest is single sheet - a flap outside the body,
    a stray layer inside it, a patch facing in - that no closed solid can lie on. "Behind" is along the
    normal of the surface around the point (its 24 nearest triangles), not of its own triangle, which a
    noisy scan tips every way; eps should clear the noise."""
    import igl
    v, inv = np.unique(tris.reshape(-1, 3), axis=0, return_inverse=True)
    f = inv.reshape(-1, 3).astype(np.int64)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])       # area-weighted
    _, near = cKDTree(tris.mean(1)).query(pts, k=min(24, len(tris)))
    nrm = n[near].sum(1) if near.ndim == 2 else n[near]
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-20
    return (igl.fast_winding_number(v, f, pts - eps * nrm) > 0.75) & (igl.fast_winding_number(v, f, pts + eps * nrm) < 0.25)


def tunnels(tris, closed_only=False):
    """Handles through a closed mesh (its genus, summed over its shells): a figurine has a few, a thin surface
    closed as lace has hundreds. closed_only: summed over the shells with no open edge only (an open scan's
    holes are no handles) - the topology a mesh brings of its own (a lattice, a chain, a pierced relief)."""
    _, inv = np.unique(np.round(tris.reshape(-1, 3), 6), axis=0, return_inverse=True)
    f = inv.reshape(-1, 3)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    nv = int(f.max()) + 1
    comp = connected_components(coo_matrix((np.ones(3 * len(f)), (f.ravel(), np.roll(f, 1, axis=1).ravel())),
                                           shape=(nv, nv)), directed=False)[1]
    e, n = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1), axis=0, return_counts=True)
    keep = ~np.isin(comp, comp[e[n == 1, 0]]) if closed_only else np.ones(nv, bool)
    fs, es = keep[f[:, 0]], keep[e[:, 0]]
    shells = len(np.unique(comp[f[fs, 0]]))
    chi = len(np.unique(f[fs])) - int(es.sum()) + int(fs.sum())
    return max(0, (2 * shells - chi) // 2)


def solid_gate(scan, bodies, grid, noise, points=3_000_000, seed=0, sheet=None):
    """What a slicer fills, against what the scan holds - the bodies exactly as written for it.
    - every body a closed shell facing out (a slicer drops an open or inside-out part: nothing printed there);
    - the scan's volume is solid: points clearly inside the scan (winding number) and away from its surface
      must be inside a body, and so must every point of thin scanned surface (a hand, a face, hair: it is a
      plate there) - lace or a void fails;
    - the solid stays in the scan: points clearly outside it, past a plate's half thickness from its
      surface, must be outside every body;
    - tunnels: a thin surface closed as lace has hundreds."""
    import igl
    from organic_solid import indexed as ix, plate_radius, sheet_faces
    V, F = ix(scan)
    per, inside = [], None
    every = np.concatenate([V] + [t.reshape(-1, 3) for _, t in bodies])     # solid grown past the scan is in the box too
    lo, hi = every.min(0), every.max(0)
    rng = np.random.default_rng(seed)
    P = lo + rng.random((points, 3)) * (hi - lo)
    cell = float(np.prod(hi - lo)) / points
    thin = V[F][sheet_faces(V, F, grid) if sheet is None else sheet]
    thin_mm2 = float(0.5 * np.linalg.norm(np.cross(thin[:, 1] - thin[:, 0], thin[:, 2] - thin[:, 0]), axis=1).sum())
    S = sample_area(thin, int(min(300_000, max(1000, thin_mm2 / 0.04))), seed=seed)[0] if len(thin) else np.zeros((0, 3))   # 5 per mm2
    in_p, in_s, holes = np.zeros(points, bool), np.zeros(len(S), bool), 0
    for name, t in bodies:
        oe, nm = edge_counts(t)
        vol = volume(t)
        k = tunnels(t)
        holes += k
        per.append(dict(name=name, open_edges=oe, nonmanifold_edges=nm, volume=round(vol, 1), tunnels=k, ok=oe == 0 and nm == 0 and vol > 0))
        bv, bf = ix(t)
        in_p |= igl.fast_winding_number(bv, bf, P) > 0.5
        if len(S):
            in_s |= igl.fast_winding_number(bv, bf, S) > 0.5
    own = tunnels(scan, closed_only=True)       # the mesh's own handles (a lattice): no lace the closing made
    w = igl.fast_winding_number(V, F, P)
    d = np.sqrt(igl.point_mesh_squared_distance(P, V, F)[0])
    band = max(2 * noise + 0.05, 1.5 * grid)
    held = (w > 0.8) & (d > band)
    missing = held & ~in_p
    extra = (w < 0.2) & (d > plate_radius(grid) + band) & in_p
    scan_vol = max(float(held.sum()) * cell, 1e-9)
    r = dict(bodies=per, tunnels=int(holes), tunnels_allowed=150 + own, scan_volume_mm3=round(scan_vol, 1),
             missing_mm3=round(float(missing.sum()) * cell, 1), extra_mm3=round(float(extra.sum()) * cell, 1),
             thin_surface_mm2=round(thin_mm2, 1),
             thin_not_solid_share=round(float((~in_s).mean()) if len(S) else 0.0, 4))
    r["checks"] = dict(bodies_closed=all(b["ok"] for b in per), volume_solid=r["missing_mm3"] <= 1e-3 * scan_vol,
                       stays_in_scan=r["extra_mm3"] <= 1e-3 * scan_vol, thin_surface_solid=r["thin_not_solid_share"] <= 0.01,
                       no_lace=holes <= 150 + own)      # Kurisu: 79 with its gear rings fused on, 1522 when closed as lace
    return r


def validate_organic(inp, repaired, reduced, tol=0.1, max_points=300_000, closed=None, grid=None, bodies=None):
    """The organic route's gate: the delivered solid (reduced) is the scan, closed, within the scan's own
    noise (the scan-mode rule). The max is judged by area, not by its one worst point: at most a thousandth
    of the measured surface past the max tolerance, and none of it in a band (a connected region) larger
    than scan mode's systematic band - a crevice sealed shut or an unscanned mouth is a few mm2. Distances are measured only where the scan has surface that bounds a
    volume: islands the repair removed, holes it filled (no scan under them) and single sheets (flaps,
    stray inner layers) are reported as areas, not counted as error. So is surface the closing bridged
    (closed: the unreduced closed mesh, grid: its step): marching cubes on a distance field lays surface
    within a fraction of a cell of the scan wherever there is scan, so closed surface over a cell from it
    spans a gap with no scan in it - a crevice's mouth, the gap between two scan layers - like a filled hole."""
    A, R0, R = read_stl(inp), read_stl(repaired), read_stl(reduced)
    size = float(np.linalg.norm(np.ptp(A.reshape(-1, 3), axis=0)))
    spacing = max(0.1, size / 600)
    # a closed manifold mesh is a model as given (a print file, a lattice), not raw scan: it has no noise to allow -
    # its plane residual is the curvature of thin struts (a Voronoi car: 0.33 mm, loose enough to smear them)
    noise = 0.0 if edge_counts(A) == (0, 0) else estimate_noise(A, 0.3)     # 1.5 mm patches: wider ones measure a figurine's curvature, not its noise
    area = lambda T: float(0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1).sum())
    count = lambda T: int(min(max_points, max(1000, area(T) / spacing ** 2)))
    # a repaired triangle is the scan's own when the scan has that very triangle (filling adds new ones)
    q = np.round(np.concatenate([A.reshape(-1, 3), R0.reshape(-1, 3)]) * 1e4).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    _, key = np.unique(np.sort(inv.ravel().reshape(-1, 3), axis=1), axis=0, return_inverse=True)
    key = key.ravel()
    own, kept = np.isin(key[len(A):], key[:len(A)]), np.isin(key[:len(A)], key[len(A):])

    pr, tr = sample_area(R0, count(R0))
    scanned = own[tr]
    bounding = enclosing(R0, pr, eps=max(0.05, 2 * noise))
    pa = pr[scanned & bounding]
    d_ar, sg_ar = MeshDistance(R, spacing)(pa)
    pb, _ = sample_area(R, count(R), seed=2)
    d_ra, sg = MeshDistance(A, spacing)(pb)
    near = MeshDistance(R0, spacing)
    near(pb)
    supported = own[near.last_tri]                          # over scan surface, not over a filled hole
    bridged_mm2 = 0.0
    if closed is not None:
        import igl
        C = read_stl(closed)
        v, inv = np.unique(R0.reshape(-1, 3), axis=0, return_inverse=True)
        gap = igl.signed_distance(C.mean(1), v, inv.reshape(-1, 3).astype(np.int64), igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_UNSIGNED)[0] > grid
        bridged_mm2 = area(C[gap])
        on_gap = gap[cKDTree(C.mean(1)).query(pb)[1]]
        supported &= ~on_gap
    open_e, nonman = edge_counts(R)
    tolr = dict(rms=max(tol, 1.25 * noise + 0.01), p95=2.5 * noise + 0.02, p99=3.5 * noise + 0.03, max=8 * noise + 0.2,
                vol=None, dim=None, face_p95=None)
    # scan surface buried in the solid past the max: an inner layer the solid fills (the skull under a hair shell,
    # a fold's inner sheet) - reported, not measured; material the solid adds over the scan is judged from the
    # solid's side (detail check)
    per_a, per_b = area(R0) / len(pr), area(R) / len(pb)
    buried = (sg_ar < 0) & (d_ar > tolr["max"])
    buried_mm2 = float(buried.sum() * per_a)
    # thin surface the closing made a plate (organic_solid.sheet_faces: a wall thinner than 1.5 grid steps, a
    # flap) lies within it by design, up to the plate's half thickness under the solid's face, as the solid
    # gate allows plates outside the scan: reported, not measured. Off the plate (outside the solid, deeper
    # in) it counts
    from organic_solid import already_solid, indexed as ix, plate_radius, sheet_faces
    g = grid or spacing
    # a scan that is a closed solid already was delivered as it is: a thin wall of it bounds volume, no open
    # sheet to plate (judged as one, its own surface points sit at winding number 1/2 and flicker)
    sheet = np.zeros(len(R0), bool) if already_solid(R0) else sheet_faces(*ix(R0), g)
    plated = sheet[tr][scanned & bounding] & (sg_ar < 0) & (d_ar <= plate_radius(g) + tolr["p99"]) & ~buried
    plated_mm2 = float(plated.sum() * per_a)
    pa, d_ar, sg_ar = pa[~buried & ~plated], d_ar[~buried & ~plated], sg_ar[~buried & ~plated]
    s_ar, s_ba = stats(d_ar), stats(d_ra[supported])
    # each sample stands for its share of its surface: past-max area and bands in mm2
    measured_a, measured_b = len(pa) * per_a, int(supported.sum()) * per_b
    past = dict(scan=float((d_ar > tolr["max"]).sum() * per_a), solid=float((d_ra[supported] > tolr["max"]).sum() * per_b))
    min_band = max(2.0, 5e-4 * measured_a)
    bands = [round(c["points"] * per_a, 2) for c in clusters(pa, d_ar, sg_ar, tolr["max"], spacing, top=10_000)]
    bands += [round(c["points"] * per_b, 2) for c in clusters(pb[supported], d_ra[supported], sg[supported], tolr["max"], spacing, top=10_000)]
    # detail lost against the original mesh: noise averages out over a point's 24 nearest, a feature the solid
    # misses (a tip cut off, a groove filled, an edge rounded) does not - a connected area of that local mean
    # signed deviation past the noise is detail lost, however small the overall rms. On the scan's side only
    # scan outside the solid counts (material missing): scan inside it is mostly a fold's inner layer under
    # scanned surface the solid follows; material added is judged from the solid's side, off any scan
    detail_tol, lost = max(0.05, noise), []
    for p, d, s, per, sides, what in ((pa, d_ar, sg_ar, per_a, ("outside",), "scan surface the solid misses"),
                                      (pb[supported], d_ra[supported], sg[supported], per_b, ("outside", "inside"), "solid surface off the scan")):
        if len(p) >= 24:
            b = (d * s)[cKDTree(p).query(p, 24)[1]].mean(1)
            lost += [dict(c, area_mm2=round(c["points"] * per, 2), kind=f"detail lost: {what}", of="model")
                     for c in clusters(p, np.abs(b), np.sign(b), detail_tol, spacing, top=10_000) if c["side"] in sides]
    lost.sort(key=lambda c: -c["area_mm2"])
    detail_band = max(1.0, 1e-4 * measured_a)
    # the bodies as written for the slicer, against the scan's volume (solid_gate)
    gate = solid_gate(R0, bodies or [("solid", R)], g, noise, sheet=sheet)
    checks = {
        "closed": open_e == 0 and nonman == 0,
        "rms": max(s_ar["rms"], s_ba["rms"]) <= tolr["rms"],
        "p99": max(s_ar["p99"], s_ba["p99"]) <= tolr["p99"],
        "max_area": past["scan"] <= 1e-3 * measured_a and past["solid"] <= 1e-3 * measured_b,
        "max_bands": max(bands, default=0) < min_band,
        "detail": max((c["area_mm2"] for c in lost), default=0) < detail_band,
        **{f"solid: {k}": v for k, v in gate["checks"].items()},
        # a verdict on a sliver of the scan is none: at least a quarter of its own surface must be measured
        "coverage": bool((scanned & bounding).sum() >= 0.25 * scanned.sum()),
    }
    devs = [dict(c, of="model", kind="off the scan") for c in clusters(pb[supported], d_ra[supported], sg[supported], tolr["p99"], spacing)]
    return dict(
        verdict="PASS" if all(checks.values()) else "FAIL", mode="organic", noise=noise, spacing=spacing,
        checks=checks, tolerances=tolr, input_to_model=s_ar, model_to_input=s_ba,
        volume=dict(input=None, model=abs(volume(R)), diff_pct=0.0),
        worst_faces=[], bad_dimensions=[], deviation_clusters=(lost[:5] + devs)[:10], dimensions_checked=0, dimensions_ok=0,
        solid=dict(triangles=len(R), open_edges=open_e, nonmanifold_edges=nonman),
        removed_area_mm2=round(area(A[~kept]), 1),
        filled_area_mm2=round(float((~supported).mean()) * area(R), 1),
        sheet_area_mm2=round(float((scanned & ~bounding).mean()) * area(R0), 1),
        buried_mm2=round(buried_mm2, 1), plated_mm2=round(plated_mm2, 1),
        sheet_share=round(float((scanned & ~bounding).sum() / max(1, scanned.sum())), 4),
        past_max_mm2={k: round(v, 1) for k, v in past.items()},
        past_max_share=dict(scan=round(past["scan"] / max(measured_a, 1e-9), 5), solid=round(past["solid"] / max(measured_b, 1e-9), 5)),
        largest_band_mm2=max(bands, default=0.0), min_band_mm2=round(min_band, 1), bridged_mm2=round(bridged_mm2, 1),
        solid_gate=gate,
        detail=dict(tol=round(detail_tol, 4), largest_lost_mm2=max((c["area_mm2"] for c in lost), default=0.0),
                    lost_mm2=round(sum(c["area_mm2"] for c in lost), 1), allowed_mm2=round(detail_band, 1)),
    )


def summary(r):
    if r.get("mode") == "organic":
        f = lambda s: f"rms {s['rms']:.4f} p95 {s['p95']:.4f} p99 {s['p99']:.4f} max {s['max']:.4f}"
        return "\n".join([
            f"VERDICT {r['verdict']} (organic: the repaired scan as a solid)",
            f"scan->solid {f(r['input_to_model'])}  (over surface that bounds the scan's volume)", f"solid->scan {f(r['model_to_input'])}  (over scanned surface)",
            f"solid: {r['solid']['triangles']} faces, open edges {r['solid']['open_edges']}, non-manifold {r['solid']['nonmanifold_edges']}, volume {r['volume']['model']:.1f} mm3",
            f"noise sigma {r['noise']:.4f}: tolerances rms {r['tolerances']['rms']:.3f} p99 {r['tolerances']['p99']:.3f} max {r['tolerances']['max']:.3f}",
            f"filled holes {r['filled_area_mm2']} mm2, removed islands {r['removed_area_mm2']} mm2, "
            f"single sheets {r['sheet_area_mm2']} mm2 ({100 * r['sheet_share']:.1f} % of the scan: no closed solid lies on them, not measured), "
            f"inner scan layers buried in the solid {r.get('buried_mm2', 0)} mm2 (filled, not measured), "
            f"thin surface made a plate {r.get('plated_mm2', 0)} mm2 (within it, not measured), "
            f"gaps bridged by the closing {r.get('bridged_mm2', 0)} mm2 (no scan there, not measured)",
            f"past max {r['tolerances']['max']:.3f}: scan {r['past_max_mm2']['scan']} mm2 ({100 * r['past_max_share']['scan']:.3f} %), "
            f"solid {r['past_max_mm2']['solid']} mm2 ({100 * r['past_max_share']['solid']:.3f} %) of 0.1 % allowed; "
            f"largest band {r['largest_band_mm2']} mm2 of {r['min_band_mm2']} allowed",
            *([f"detail vs the original mesh (local mean deviation past {r['detail']['tol']:.3f}): largest area lost "
               f"{r['detail']['largest_lost_mm2']} mm2 of {r['detail']['allowed_mm2']} allowed, {r['detail']['lost_mm2']} mm2 in all"] if r.get("detail") else []),
            *([f"solid as the slicer gets it: {len(r['solid_gate']['bodies'])} bodies, open/inside-out "
               f"{sum(not b['ok'] for b in r['solid_gate']['bodies'])}, tunnels {r['solid_gate']['tunnels']} of {r['solid_gate'].get('tunnels_allowed', 150)}; scan volume not solid "
               f"{r['solid_gate']['missing_mm3']} mm3, solid outside the scan {r['solid_gate']['extra_mm3']} mm3 (of "
               f"{r['solid_gate']['scan_volume_mm3']}); thin surface ({r['solid_gate']['thin_surface_mm2']} mm2) not solid "
               f"{100 * r['solid_gate']['thin_not_solid_share']:.1f} %"] if r.get("solid_gate") else []),
            "failed: " + (", ".join(k for k, v in r["checks"].items() if not v) or "none")])
    f = lambda s: f"rms {s['rms']:.4f} p95 {s['p95']:.4f} p99 {s['p99']:.4f} max {s['max']:.4f}"
    lines = [f"VERDICT {r['verdict']} ({r['mode']}" + (f", noise sigma {r['noise']:.4f}" if r["noise"] else "") + ")",
             f"input->model {f(r['input_to_model'])}", f"model->input {f(r['model_to_input'])}",
             f"volume {r['volume']['diff_pct']:+.3f} %  dimensions {r['dimensions_ok']}/{r['dimensions_checked']} within {r['tolerances']['dim']} mm",
             "failed: " + (", ".join(k for k, v in r["checks"].items() if not v) or "none")]
    for w in r["worst_faces"]:
        lines.append(f"  face {w['face']} {w['kind']} area {w['area']} not reproduced: p95 {w['p95']} mm, box {w['box']}")
    for c in r["deviation_clusters"]:
        lines.append(f"  {c['kind']}: {c['area_mm2']} mm2, mean {c['mean']} max {c['max']} mm, center {c['center']}, box {c['box']}"
                     + (f"\n      the {'scan' if c.get('of') == 'scan' else 'model'} surface there: {c['shape']}" if c.get("shape") else ""))
    for d in r["bad_dimensions"]:
        lines.append(f"  dim {d['desc']} area {d['area']}: " + ("missing in model" if d["err"] is None else f"off by {d['err']:.4f} mm"))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("model")
    ap.add_argument("--faces"); ap.add_argument("--model-faces")
    ap.add_argument("--mode", choices=["cad", "scan"], default="cad")
    ap.add_argument("--noise", type=float)
    ap.add_argument("--json")
    a = ap.parse_args()
    r = validate(a.input, a.model, a.faces, a.model_faces, a.mode, a.noise)
    print(summary(r))
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1))
    sys.exit(0 if r["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
