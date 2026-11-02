# SPDX-License-Identifier: AGPL-3.0-only
"""Binary/ASCII STL read, binary STL and 3MF write, numpy only."""
import struct
from pathlib import Path

import numpy as np


def read_stl(path):
    """Triangles as an (n, 3, 3) float64 array."""
    data = open(path, "rb").read()
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if 84 + 50 * n == len(data):
            rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n, offset=84)
            return rec["v"].astype(np.float64)
    vs = [list(map(float, l.split()[1:4])) for l in data.decode("ascii", "replace").splitlines() if l.strip().startswith("vertex")]
    return np.array(vs, dtype=np.float64).reshape(-1, 3, 3)


def write_stl(path, tris):
    tris = np.asarray(tris, dtype=np.float32)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-20
    rec = np.zeros(len(tris), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"], rec["v"] = n, tris
    with open(path, "wb") as fh:
        fh.write(b"scanruler-swift".ljust(80, b"\0"))
        fh.write(struct.pack("<I", len(tris)))
        fh.write(rec.tobytes())


def write_3mf(path, bodies):
    """bodies: [(name, triangles or (vertices, faces))] as a 3MF in millimetres, one object with a part per body.
    Triangles are welded at 1e-6 mm; (vertices, faces) stay as given, so shells touching at an edge stay apart
    (welded, they would share it four ways: non-manifold to a slicer). The print copy of the faceted STEP: the
    same surface at a few percent of the size, which slicers open directly."""
    import zipfile
    from xml.sax.saxutils import quoteattr
    f6 = lambda x: f"{x:.6f}".rstrip("0").rstrip(".")
    objs, items = [], []
    for i, (name, mesh) in enumerate(bodies, 1):
        if isinstance(mesh, tuple):     # (vertices, faces) as bodies._mesh made them: its shells stay apart, never welded
            v, f = np.asarray(mesh[0], float), np.asarray(mesh[1], np.int64)
        else:
            v, inv = np.unique(np.round(np.asarray(mesh, float).reshape(-1, 3), 6), axis=0, return_inverse=True)
            f = inv.reshape(-1, 3)
        f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
        vs = "".join(f'<vertex x="{f6(a)}" y="{f6(b)}" z="{f6(c)}"/>' for a, b, c in v)
        ts = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in f)
        objs.append(f'<object id="{i}" name={quoteattr(str(name))} type="model"><mesh><vertices>{vs}</vertices>'
                    f'<triangles>{ts}</triangles></mesh></object>')
        items.append(f'<component objectid="{i}"/>')
    # one object of parts: a slicer (Bambu Studio, PrusaSlicer) puts it on the plate assembled, as scanned
    top = len(objs) + 1
    objs.append(f'<object id="{top}" name={quoteattr(Path(path).stem)} type="model"><components>{"".join(items)}</components></object>')
    model = ('<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" '
             'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
             # without a producer Bambu Studio calls the file's config invalid; named, it loads the geometry quietly
             '<metadata name="Application">ScanRuler-Swift</metadata>'
             f'<resources>{"".join(objs)}</resources><build><item objectid="{top}"/></build></model>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
        z.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
        z.writestr("3D/3dmodel.model", model)
