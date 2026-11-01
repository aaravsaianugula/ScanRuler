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


def write_step(path, tris, name="Body"):
    v, f = indexed(np.asarray(tris, float))
    lines, n = [], [0]

    def ent(text):
        n[0] += 1
        lines.append(f"#{n[0]}={text};")
        return n[0]

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    app = ent("APPLICATION_CONTEXT('automotive design')")
    ent(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{app})")
    pctx = ent(f"PRODUCT_CONTEXT('',#{app},'mechanical')")
    prod = ent(f"PRODUCT('{name}','{name}','',(#{pctx}))")
    pdf = ent(f"PRODUCT_DEFINITION_FORMATION('','',#{prod})")
    dctx = ent(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{app},'design')")
    pd = ent(f"PRODUCT_DEFINITION('design','',#{pdf},#{dctx})")
    pds = ent(f"PRODUCT_DEFINITION_SHAPE('','',#{pd})")
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
    # a closed shell is one connected piece: a body in several pieces is a solid per piece, and a piece
    # facing in (a sealed cavity) is a void of the piece around it - written as a shell facing out, reversed
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    adj = coo_matrix((np.ones(3 * len(f)), (f.ravel(), np.roll(f, 1, axis=1).ravel())), shape=(len(v), len(v)))
    comp = connected_components(adj, directed=False)[1][f[:, 0]]
    t = v[f]
    cvol = np.bincount(comp, weights=np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])))
    hollow = cvol[comp] < 0
    f = np.where(hollow[:, None], f[:, ::-1], f)
    # real topology, not POLY_LOOPs: a face per triangle, but one edge per mesh edge shared by its two
    # faces, so a reader gets one connected closed shell (OCCT splits a POLY_LOOP shell into many solids)
    pt = [ent(f"CARTESIAN_POINT('',({p[0]:.6f},{p[1]:.6f},{p[2]:.6f}))") for p in v]
    vx = [ent(f"VERTEX_POINT('',#{i})") for i in pt]
    he = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])            # half-edges, face-major by corner
    und, e_of = np.unique(np.sort(he, axis=1), axis=0, return_inverse=True)
    e_of = e_of.ravel()
    edge = []
    for a, b in und:
        d = v[b] - v[a]
        ln = float(np.linalg.norm(d))
        d = d / ln
        line = ent(f"LINE('',#{pt[a]},#{ent(f'VECTOR({chr(39)}{chr(39)},#{ent(f'DIRECTION({chr(39)}{chr(39)},({d[0]:.9f},{d[1]:.9f},{d[2]:.9f}))')},{ln:.6f})')})")
        edge.append(ent(f"EDGE_CURVE('',#{vx[a]},#{vx[b]},#{line},.T.)"))
    nrm = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-300
    ref = v[f[:, 1]] - v[f[:, 0]]
    ref /= np.linalg.norm(ref, axis=1, keepdims=True) + 1e-300
    nf, faces = len(f), []
    for i, ((a, b, c), nn, rr) in enumerate(zip(f, nrm, ref)):
        oe = []
        for k, (s, t) in enumerate(((a, b), (b, c), (c, a))):
            oe.append(ent(f"ORIENTED_EDGE('',*,*,#{edge[e_of[k * nf + i]]},{'.T.' if s < t else '.F.'})"))
        loop = ent("EDGE_LOOP('',(" + ",".join(f"#{j}" for j in oe) + "))")
        bound = ent(f"FACE_OUTER_BOUND('',#{loop},.T.)")
        dn = ent(f"DIRECTION('',({nn[0]:.9f},{nn[1]:.9f},{nn[2]:.9f}))")
        dr = ent(f"DIRECTION('',({rr[0]:.9f},{rr[1]:.9f},{rr[2]:.9f}))")
        pl = ent(f"PLANE('',#{ent(f'AXIS2_PLACEMENT_3D({chr(39)}{chr(39)},#{pt[a]},#{dn},#{dr})')})")
        faces.append(ent(f"ADVANCED_FACE('',(#{bound}),#{pl},.T.)"))
    shell = {c: ent("CLOSED_SHELL('',(" + ",".join(f"#{faces[i]}" for i in np.flatnonzero(comp == c)) + "))") for c in np.unique(comp)}
    solids = [c for c in shell if cvol[c] > 0]
    voids, lost = {c: [] for c in solids}, 0
    for c in shell:
        if cvol[c] < 0:     # in the smallest solid around it: all its corners inside that, none of that's inside it
            import igl
            p = v[np.unique(f[comp == c])]
            cav = f[comp == c].astype(np.int64)                 # reversed above: faces out of the cavity, a solid of its own
            around = [s for s in solids if (igl.fast_winding_number(v, f[comp == s].astype(np.int64), p) > 0.5).all()
                      and not (igl.fast_winding_number(v, cav, v[np.unique(f[comp == s])]) > 0.5).any()]
            if around:
                voids[min(around, key=lambda s: cvol[s])].append(ent(f"ORIENTED_CLOSED_SHELL('',*,#{shell[c]},.F.)"))
            else:       # a cavity no solid holds (it crosses one's wall): left out, and said
                lost += int((comp == c).sum())
    breps = []
    for k, c in enumerate(solids):
        label = f"{name}{'' if k == 0 else f' {k + 1}'}"
        breps.append(ent(f"BREP_WITH_VOIDS('{label}',#{shell[c]},(" + ",".join(f"#{o}" for o in voids[c]) + "))") if voids[c]
                     else ent(f"MANIFOLD_SOLID_BREP('{label}',#{shell[c]})"))
    rep = ent("ADVANCED_BREP_SHAPE_REPRESENTATION('',(" + ",".join(f"#{b}" for b in breps) + f",#{axis}),#{ctx})")
    ent(f"SHAPE_DEFINITION_REPRESENTATION(#{pds},#{rep})")
    head = ["ISO-10303-21;", "HEADER;", "FILE_DESCRIPTION(('faceted solid from a repaired scan'),'2;1');",
            f"FILE_NAME('{Path(path).name}','{now}',(''),(''),'scanruler-swift','scanruler-swift','');",
            "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));", "ENDSEC;", "DATA;"]
    Path(path).write_text("\n".join(head + lines + ["ENDSEC;", "END-ISO-10303-21;", ""]), encoding="ascii")
    return dict(vertices=len(v), faces=len(f), solids=len(breps), voids=sum(map(len, voids.values())), faces_left_out=lost)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh"); ap.add_argument("out"); ap.add_argument("--name", default="Body")
    a = ap.parse_args()
    print(write_step(a.out, read_stl(a.mesh), a.name))


if __name__ == "__main__":
    main()
