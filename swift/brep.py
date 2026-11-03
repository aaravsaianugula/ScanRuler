# SPDX-License-Identifier: AGPL-3.0-only
"""Analytic primitives -> one exact solid with OCCT (cadquery-ocp). Every primitive becomes a face reaching past
the part, BOPAlgo_MakerVolume splits the space they bound into cells, a cell is kept when the scan's generalized
winding number says most of it is inside, the faces between kept and dropped cells are sewn into the solid and
pieces of one surface merged (ShapeUpgrade_UnifySameDomain). PolyFit's hypothesize-and-select (Nan & Wonka 2017)
with quadrics and a winding-number vote instead of an optimizer; .claude-work/RE_RESEARCH.md 5.2."""
from collections import Counter

import numpy as np

from OCP.Bnd import Bnd_Box
from OCP.BOPAlgo import BOPAlgo_MakerVolume
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepGProp import BRepGProp
from OCP.BRepLib import BRepLib
from OCP.GeomAbs import GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Sphere, GeomAbs_Torus
from OCP.GProp import GProp_GProps
from OCP.gp import gp_Ax3, gp_Cone, gp_Cylinder, gp_Dir, gp_Pln, gp_Pnt, gp_Sphere
from OCP.IFSelect import IFSelect_RetDone
from OCP.ShapeFix import ShapeFix_Solid
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_FORWARD, TopAbs_IN, TopAbs_SHELL, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS

KINDS = {GeomAbs_Plane: "plane", GeomAbs_Cylinder: "cylinder", GeomAbs_Cone: "cone", GeomAbs_Sphere: "sphere",
         GeomAbs_Torus: "torus"}
# OCP 7.x names static methods with _s, OCP 8 drops it for the TopoDS casts
CAST = {n: getattr(TopoDS, n + "_s", None) or getattr(TopoDS, n) for n in ("Edge", "Face", "Shell", "Solid")}


def shapes(s, kind):
    e, out = TopExp_Explorer(s, kind), []
    while e.More():
        out.append(e.Current())
        e.Next()
    return out


def pnt(x):
    return gp_Pnt(*map(float, x))


def face(p, mid, L):
    """The primitive as a face covering the part's box (centre mid, half size L): planes and cylinders cut
    square across it, cones kept clear of their apex."""
    k = p["kind"]
    if k == "plane":
        n = np.asarray(p["n"])
        o = mid - (n @ mid - p["d"]) * n
        return BRepBuilderAPI_MakeFace(gp_Pln(gp_Ax3(pnt(o), gp_Dir(*n))), -L, L, -L, L).Face()
    if k == "sphere":
        return BRepBuilderAPI_MakeFace(gp_Sphere(gp_Ax3(pnt(p["c"]), gp_Dir(0, 0, 1)), p["r"])).Face()
    a, c = np.asarray(p["a"]), np.asarray(p["c"])
    if k == "cylinder":
        o = c + ((mid - c) @ a) * a
        return BRepBuilderAPI_MakeFace(gp_Cylinder(gp_Ax3(pnt(o), gp_Dir(*a)), p["r"]), 0, 2 * np.pi, -L, L).Face()
    # cone: v runs along the generatrix from the apex (reference radius 0)
    h = (mid - c) @ a
    v0, v1 = max(1e-3 * L, (h - L) / np.cos(p["t"])), (h + L) / np.cos(p["t"])
    return BRepBuilderAPI_MakeFace(gp_Cone(gp_Ax3(pnt(c), gp_Dir(*a)), p["t"], 0.0), 0, 2 * np.pi, v0, v1).Face()


def inside_share(cells, V, F, rng, want=12, tries=200):
    """Per cell, the share of points inside it that the scan's winding number puts inside the scan."""
    import igl
    pts, owner = [], []
    for i, c in enumerate(cells):
        b = Bnd_Box()
        BRepBndLib.Add_s(c, b)
        lo, hi = np.array(b.CornerMin().Coord()), np.array(b.CornerMax().Coord())
        cls, got = BRepClass3d_SolidClassifier(c), []
        for q in rng.uniform(lo, hi, (tries, 3)):
            cls.Perform(pnt(q), 1e-7)
            if cls.State() == TopAbs_IN:
                got.append(q)
                if len(got) >= want:
                    break
        if not got:            # a sliver no random point hits: its centre of mass decides
            g = GProp_GProps()
            BRepGProp.VolumeProperties_s(c, g)
            got = [np.array(g.CentreOfMass().Coord())]
        pts += got
        owner += [i] * len(got)
    w = igl.fast_winding_number(V.astype(float), F.astype(np.int64), np.array(pts))
    owner = np.array(owner)
    return np.bincount(owner, weights=w > 0.5, minlength=len(cells)) / np.bincount(owner, minlength=len(cells))


def kinds(shape):
    return dict(Counter(KINDS.get(BRepAdaptor_Surface(CAST["Face"](f)).GetType(), "other") for f in shapes(shape, TopAbs_FACE)))


def solid(prims, V, F, rounds=(), margin=0.08, log=print, seed=0):
    """(shape or None, info) from primitives and the scan mesh (V, F) that judges inside; rounds (mech.blends)
    fillet the sharp edges they stood for."""
    lo, hi = V.min(0), V.max(0)
    mid, L = (lo + hi) / 2, float(np.linalg.norm(hi - lo))
    pad = margin * (hi - lo).max() + 1.0
    box = [dict(kind="plane", n=np.eye(3)[k] * s, d=float(s * (hi[k] + pad if s > 0 else lo[k] - pad)))
           for k in range(3) for s in (1, -1)]
    mv = BOPAlgo_MakerVolume()
    for p in list(prims) + box:
        mv.AddArgument(face(p, mid, L))
    mv.SetRunParallel(True)
    mv.Perform()
    if mv.HasErrors():
        return None, dict(error="MakerVolume failed", valid=False, faces=0, kinds={})
    cells = shapes(mv.Shape(), TopAbs_SOLID)
    share = inside_share(cells, V, F, np.random.default_rng(seed))
    kept = [c for c, s in zip(cells, share) if s > 0.5]
    # the solid's faces: those of kept cells that no other kept cell shares
    seen = {}
    for c in kept:
        for f in shapes(c, TopAbs_FACE):
            k = f.Oriented(TopAbs_FORWARD)
            hit = next((e for e in seen.get(hash(k), []) if e[0].IsSame(f)), None)
            if hit:
                hit[1] += 1
            else:
                seen.setdefault(hash(k), []).append([f, 1])
    sew = BRepBuilderAPI_Sewing(1e-6)
    for f, n in (e for es in seen.values() for e in es):
        if n == 1:
            sew.Add(f)
    sew.Perform()
    solids = []
    for sh in shapes(sew.SewedShape(), TopAbs_SHELL):
        ms = BRepBuilderAPI_MakeSolid(CAST["Shell"](sh))
        if ms.IsDone():
            fix = ShapeFix_Solid(ms.Solid())
            fix.Perform()
            s = CAST["Solid"](fix.Solid())
            BRepLib.OrientClosedSolid_s(s)
            solids.append(s)
    if not solids:
        return None, dict(error="no closed shell", cells=len(cells), kept=len(kept), valid=False, faces=0, kinds={})
    # shortcut: one solid (the biggest shell); several bodies or inner voids need a compound / shells per solid
    s = max(solids, key=lambda x: abs(volume(x)))
    u = ShapeUpgrade_UnifySameDomain(s, True, True, False)
    u.Build()
    out, filleted = round_edges(u.Shape(), rounds) if rounds else (u.Shape(), 0)
    if filleted > 0:
        u = ShapeUpgrade_UnifySameDomain(out, True, True, False)
        u.Build()
        out = u.Shape()
    info = dict(cells=len(cells), kept=len(kept), shells=len(solids), faces=len(shapes(out, TopAbs_FACE)), filleted_edges=filleted,
                kinds=kinds(out), valid=bool(BRepCheck_Analyzer(out).IsValid()), volume=round(volume(out), 2),
                undecided=int(((share > 0.2) & (share < 0.8)).sum()))
    log(f"[brep] {len(cells)} cells, {len(kept)} inside ({info['undecided']} undecided), {len(solids)} shells -> "
        f"{info['faces']} faces {info['kinds']}, valid {info['valid']}")
    return out, info


def round_edges(shape, rounds, tol=1e-6):
    """Fillets of the sharp edges the rounds stood for: a straight edge on both support planes, along the
    round's axis, within two radii of it and overlapping its extent. (shape, edges filleted; negative = failed)"""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCP.GeomAbs import GeomAbs_Line
    mf, n = BRepFilletAPI_MakeFillet(shape), 0
    for e in shapes(shape, TopAbs_EDGE):
        c = BRepAdaptor_Curve(CAST["Edge"](e))
        if c.GetType() != GeomAbs_Line:
            continue
        p0, p1 = (np.array(c.Value(t).Coord()) for t in (c.FirstParameter(), c.LastParameter()))
        m, d = (p0 + p1) / 2, (p1 - p0) / np.linalg.norm(p1 - p0)
        for b in rounds:
            a, w = np.asarray(b["a"]), m - b["c"]
            if abs(d @ a) < 0.9999 or any(abs(q["n"] @ m - q["d"]) > tol for q in b["planes"])                     or np.linalg.norm(w - (w @ a) * a) > 2 * b["radius"]:
                continue
            lo, hi = sorted((p0 @ a, p1 @ a))
            if hi > b["h"][0] and lo < b["h"][1]:
                mf.Add(b["radius"], CAST["Edge"](e))
                n += 1
                break
    if not n:
        return shape, 0
    mf.Build()
    return (mf.Shape(), n) if mf.IsDone() else (shape, -n)


def volume(s):
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(s, g)
    return g.Mass()


def write_step(shape, path):
    import os
    import sys
    # OCCT prints its transfer statistics on the process stdout, which carries run.py's JSON events
    sys.stdout.flush()
    keep, null = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    try:
        w = STEPControl_Writer()
        w.Transfer(shape, STEPControl_AsIs)
        ok = w.Write(str(path)) == IFSelect_RetDone
    finally:
        os.dup2(keep, 1)
        os.close(keep)
        os.close(null)
    if not ok:
        raise RuntimeError(f"STEP write failed: {path}")
