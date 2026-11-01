# SPDX-License-Identifier: AGPL-3.0-only
"""A closed manifold triangle mesh as a STEP solid (AP214 MANIFOLD_SOLID_BREP: a planar face per triangle,
edges and vertices shared between faces).

For the organic route: a figurine has no analytic faces to rebuild, so the deliverable solid is the
repaired scan itself, in a form every CAD system imports as a body. ScanRuler cannot write one
(meshstep only reads STEP), hence this writer.

    python swift/step_faceted.py mesh.stl out.step [--name Figure]
"""
import argparse
import datetime
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meshio_min import read_stl  # noqa: E402


def indexed(tris, decimals=6):
    """Shared vertices: a closed shell must reference the same point from every face that meets it."""
    v, inv = np.unique(np.round(tris.reshape(-1, 3), decimals), axis=0, return_inverse=True)
    f = inv.reshape(-1, 3)
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])    # drop collapsed triangles
    return v, f[keep]


def num(x, decimals):
    """A STEP real in its shortest exact form at that many decimals: 12.5, not 12.500000 (a third of the file)."""
    s = f"{x:.{decimals}f}".rstrip("0")
    return "0." if s in ("-0.", "0.") else s


def solid_pieces(v, f):
    """The closed shells of an indexed mesh: a body in several pieces is a solid per piece, and a piece
    facing in (a sealed cavity) is a void of the smallest solid around it. Returns f with cavities turned to
    face out of the cavity, each face's shell, the shells' volumes (x6), the solids (shell ids, largest
    first), each solid's voids and the faces of cavities no solid holds (they cross a wall)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    adj = coo_matrix((np.ones(3 * len(f)), (f.ravel(), np.roll(f, 1, axis=1).ravel())), shape=(len(v), len(v)))
    comp = connected_components(adj, directed=False)[1][f[:, 0]]
    t = v[f]
    cvol = np.bincount(comp, weights=np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])))
    f = np.where((cvol[comp] < 0)[:, None], f[:, ::-1], f)
    solids = sorted((c for c in range(len(cvol)) if cvol[c] > 0), key=lambda c: -cvol[c])
    voids, lost = {c: [] for c in solids}, 0
    lo = np.full((len(cvol), 3), np.inf)
    hi = -lo
    np.minimum.at(lo, comp, t.min(1))
    np.maximum.at(hi, comp, t.max(1))
    for c in np.flatnonzero(cvol < 0):
        import igl     # in the smallest solid around it: all its corners inside that, none of that's inside it
        p = v[np.unique(f[comp == c])]
        cav = f[comp == c].astype(np.int64)
        boxed = [s for s in solids if (lo[s] <= lo[c]).all() and (hi[c] <= hi[s]).all()]
        around = [s for s in boxed if (igl.fast_winding_number(v, f[comp == s].astype(np.int64), p) > 0.5).all()
                  and not (igl.fast_winding_number(v, cav, v[np.unique(f[comp == s])]) > 0.5).any()]
        if around:
            voids[min(around, key=lambda s: cvol[s])].append(int(c))
        else:
            lost += int((comp == c).sum())
    return f, comp, cvol, solids, voids, lost


def write_step(path, tris, name="Body", bodies=None):
    """bodies: [(name, triangles or (vertices, faces))], each written as a product of its own so named (a
    solid per closed piece, its cavities as voids); default the one body (name, tris). Bodies are indexed
    apart: two that touch (halves of a cut) share no vertex or edge; an indexed body keeps its indexing."""
    bodies = bodies or [(name, tris)]
    lines, n = [], [0]

    def ent(text):
        n[0] += 1
        lines.append(f"#{n[0]}={text};")
        return n[0]

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
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
    o = ent("CARTESIAN_POINT('',(0.,0.,0.))")
    z = ent("DIRECTION('',(0.,0.,1.))")
    x = ent("DIRECTION('',(1.,0.,0.))")
    axis = ent(f"AXIS2_PLACEMENT_3D('',#{o},#{z},#{x})")
    info = dict(vertices=0, faces=0, solids=0, voids=0, faces_left_out=0, products=len(bodies))
    for bname, btris in bodies:
        bname = str(bname).replace("'", "")
        v, f = (np.asarray(btris[0], float), np.asarray(btris[1], np.int64)) if isinstance(btris, tuple) else indexed(np.asarray(btris, float))
        f, comp, cvol, solids, voids, lost = solid_pieces(v, f)
        # real topology, not POLY_LOOPs: a face per triangle, but one edge per mesh edge shared by its two
        # faces, so a reader gets one connected closed shell (OCCT splits a POLY_LOOP shell into many solids)
        pt = [ent(f"CARTESIAN_POINT('',({num(p[0], 6)},{num(p[1], 6)},{num(p[2], 6)}))") for p in v]
        dirs = {}

        def direction(d):       # one entity per distinct direction: a flat face's triangles share theirs
            key = f"({num(d[0], 9)},{num(d[1], 9)},{num(d[2], 9)})"
            if key not in dirs:
                dirs[key] = ent(f"DIRECTION('',{key})")
            return dirs[key]
        vx = [ent(f"VERTEX_POINT('',#{i})") for i in pt]
        he = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])            # half-edges, face-major by corner
        und, e_of = np.unique(np.sort(he, axis=1), axis=0, return_inverse=True)
        e_of = e_of.ravel()
        edge = []
        # an edge's curve is a two-point POLYLINE on the vertices' own points: one entity, where a LINE takes
        # three (its VECTOR and DIRECTION), and exact by construction
        edge = [ent(f"EDGE_CURVE('',#{vx[a]},#{vx[b]},#{ent(f'POLYLINE({chr(39)}{chr(39)},(#{pt[a]},#{pt[b]}))')},.T.)") for a, b in und]
        nrm = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-300
        nf, faces = len(f), []
        for i, ((a, b, c), nn) in enumerate(zip(f, nrm)):
            oe = []
            for k, (s, t) in enumerate(((a, b), (b, c), (c, a))):
                oe.append(ent(f"ORIENTED_EDGE('',*,*,#{edge[e_of[k * nf + i]]},{'.T.' if s < t else '.F.'})"))
            loop = ent("EDGE_LOOP('',(" + ",".join(f"#{j}" for j in oe) + "))")
            bound = ent(f"FACE_OUTER_BOUND('',#{loop},.T.)")
            # no ref_direction ($, optional): a plane's in-plane axes are the reader's to choose
            pl = ent(f"PLANE('',#{ent(f'AXIS2_PLACEMENT_3D({chr(39)}{chr(39)},#{pt[a]},#{direction(nn)},$)')})")
            faces.append(ent(f"ADVANCED_FACE('',(#{bound}),#{pl},.T.)"))
        shell = {c: ent("CLOSED_SHELL('',(" + ",".join(f"#{faces[i]}" for i in np.flatnonzero(comp == c)) + "))") for c in np.unique(comp)}
        breps = []
        for k, c in enumerate(solids):
            label = f"{bname}{'' if k == 0 else f' {k + 1}'}"
            vs = [ent(f"ORIENTED_CLOSED_SHELL('',*,#{shell[o]},.F.)") for o in voids[c]]
            breps.append(ent(f"BREP_WITH_VOIDS('{label}',#{shell[c]},(" + ",".join(f"#{o}" for o in vs) + "))") if vs
                         else ent(f"MANIFOLD_SOLID_BREP('{label}',#{shell[c]})"))
        prod = ent(f"PRODUCT('{bname}','{bname}','',(#{pctx}))")
        pd = ent(f"PRODUCT_DEFINITION('design','',#{ent(f'PRODUCT_DEFINITION_FORMATION({chr(39)}{chr(39)},{chr(39)}{chr(39)},#{prod})')},#{dctx})")
        rep = ent(f"ADVANCED_BREP_SHAPE_REPRESENTATION('{bname}',(" + ",".join(f"#{b}" for b in breps) + f",#{axis}),#{ctx})")
        ent(f"SHAPE_DEFINITION_REPRESENTATION(#{ent(f'PRODUCT_DEFINITION_SHAPE({chr(39)}{chr(39)},{chr(39)}{chr(39)},#{pd})')},#{rep})")
        for k, val in dict(vertices=len(v), faces=len(f), solids=len(solids), voids=sum(map(len, voids.values())), faces_left_out=lost).items():
            info[k] += val
    head = ["ISO-10303-21;", "HEADER;", "FILE_DESCRIPTION(('faceted solid from a repaired scan'),'2;1');",
            f"FILE_NAME('{Path(path).name}','{now}',(''),(''),'scanruler-swift','scanruler-swift','');",
            "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));", "ENDSEC;", "DATA;"]
    Path(path).write_text("\n".join(head + lines + ["ENDSEC;", "END-ISO-10303-21;", ""]), encoding="ascii")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh"); ap.add_argument("out"); ap.add_argument("--name", default="Body")
    a = ap.parse_args()
    print(write_step(a.out, read_stl(a.mesh), a.name))


if __name__ == "__main__":
    main()
