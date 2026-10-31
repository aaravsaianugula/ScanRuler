# SPDX-License-Identifier: AGPL-3.0-only
"""A closed triangle mesh as a STEP solid (AP214 FACETED_BREP: one planar face per triangle).

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
    first = n[0] + 1
    for p in v:
        ent(f"CARTESIAN_POINT('',({p[0]:.6f},{p[1]:.6f},{p[2]:.6f}))")
    faces = []
    # each face its own plane: OCCT's reader takes a FACE_SURFACE (bound + geometry), not a bare FACE
    nrm = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-300
    ref = v[f[:, 1]] - v[f[:, 0]]
    ref /= np.linalg.norm(ref, axis=1, keepdims=True) + 1e-300
    for (a, b, c), nn, rr in zip(f, nrm, ref):
        dn = ent(f"DIRECTION('',({nn[0]:.9f},{nn[1]:.9f},{nn[2]:.9f}))")
        dr = ent(f"DIRECTION('',({rr[0]:.9f},{rr[1]:.9f},{rr[2]:.9f}))")
        pl = ent(f"PLANE('',#{ent(f'AXIS2_PLACEMENT_3D({chr(39)}{chr(39)},#{first + a},#{dn},#{dr})')})")
        loop = ent(f"POLY_LOOP('',(#{first + a},#{first + b},#{first + c}))")
        bound = ent(f"FACE_OUTER_BOUND('',#{loop},.T.)")
        faces.append(ent(f"FACE_SURFACE('',(#{bound}),#{pl},.T.)"))
    shell = ent("CLOSED_SHELL('',(" + ",".join(f"#{i}" for i in faces) + "))")
    brep = ent(f"FACETED_BREP('{name}',#{shell})")
    rep = ent(f"FACETED_BREP_SHAPE_REPRESENTATION('',(#{brep},#{axis}),#{ctx})")
    ent(f"SHAPE_DEFINITION_REPRESENTATION(#{pds},#{rep})")
    head = ["ISO-10303-21;", "HEADER;", "FILE_DESCRIPTION(('faceted solid from a repaired scan'),'2;1');",
            f"FILE_NAME('{Path(path).name}','{now}',(''),(''),'scanruler-swift','scanruler-swift','');",
            "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));", "ENDSEC;", "DATA;"]
    Path(path).write_text("\n".join(head + lines + ["ENDSEC;", "END-ISO-10303-21;", ""]), encoding="ascii")
    return dict(vertices=len(v), faces=len(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh"); ap.add_argument("out"); ap.add_argument("--name", default="Body")
    a = ap.parse_args()
    print(write_step(a.out, read_stl(a.mesh), a.name))


if __name__ == "__main__":
    main()
