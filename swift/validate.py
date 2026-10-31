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
        nrm = np.cross(self.tris[:, 1] - self.tris[:, 0], self.tris[:, 2] - self.tris[:, 0])
        for s in range(0, len(p), chunk):
            q = p[s:s + chunk]
            _, idx = self.tree.query(q, k=k)
            cand = self.tid[idx]                                     # (m, k) triangle ids
            best = np.full(len(q), np.inf)
            side = np.ones(len(q))
            for j in range(k):
                t = self.tris[cand[:, j]]
                c = closest_on_tris(q, t[:, 0], t[:, 1], t[:, 2])
                dj = np.linalg.norm(q - c, axis=1)
                better = dj < best
                best = np.where(better, dj, best)
                side = np.where(better, np.sign(((q - c) * nrm[cand[:, j]]).sum(1)) + (dj == 0), side)
            d[s:s + chunk], sgn[s:s + chunk] = best, np.where(side >= 0, 1.0, -1.0)
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
    Curvature inflates it on rounds, so the median over the whole part is used."""
    pts = sample(tris, spacing)
    tree = cKDTree(pts)
    rng = np.random.default_rng(0)
    probe = pts[rng.choice(len(pts), min(4000, len(pts)), replace=False)]
    sig = []
    for nb in tree.query_ball_point(probe, r=max(1.5, 4 * spacing)):
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


def summary(r):
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
