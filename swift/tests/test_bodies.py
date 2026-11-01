# SPDX-License-Identifier: AGPL-3.0-only
"""Bodies from pieces: a big cube with a sliver beside it and a speck far off, grouped, the cube split by a
plane, written as named products and read back by native OCCT (XDE) with their names and volumes.
    python swift/tests/test_bodies.py"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bodies import group, survey  # noqa: E402
from step_faceted import write_step  # noqa: E402

V = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
F = [(0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3), (0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6)]
box = lambda lo, size: V[np.array(F)] * size + lo
tris = np.concatenate([box([0, 0, 0], [10, 10, 10]), box([10.3, 0, 0], [0.4, 10, 10]), box([40, 0, 0], [1, 1, 1])])

pieces = survey(tris, 0.3)
assert [p["volume"] for p in pieces] == [1000.0, 40.0, 1.0], pieces
assert pieces[1]["touching"] == [0] and pieces[2]["touching"] == [], pieces
bodies, kept, dropped, cuts = group(tris, pieces, [{"name": "Figure", "pieces": [0]}],
                                   splits=[{"piece": 0, "point": [0, 0, 3], "normal": [0, 0, -1], "name": "Base"}])
assert [n for n, _ in bodies] == ["Figure", "Base"] and [d["id"] for d in dropped] == [2], (bodies, dropped)
assert abs(cuts[0]["volume"] - 300) < 1e-6 and cuts[0]["pieces"] == [0], cuts
assert len(kept) == 24                  # the validator judges the uncut cube and the sliver, not the speck
for bad in ([{"name": "A", "pieces": [0, 0]}], [{"name": "", "pieces": [0]}], [{"name": "A", "pieces": [7]}]):
    try:
        group(tris, pieces, bad)
        raise AssertionError(f"accepted {bad}")
    except ValueError:
        pass

# a gear wheel fused to a post through its bore and touching it: found by find(), cut out by its ring knife,
# the two bodies adding up to the fused solid, the wheel whole (its volume at least the wheel's)
import manifold3d as m3d  # noqa: E402
from bodies import _tris  # noqa: E402
from rings import find  # noqa: E402
teeth = 12
ang = np.linspace(-np.pi, np.pi, teeth * 8, endpoint=False)
rad = 30 + 3 * (np.sin(ang * teeth) > 0)
wheel = m3d.Manifold.extrude(m3d.CrossSection([np.c_[rad * np.cos(ang), rad * np.sin(ang)],
                                               np.c_[22 * np.cos(ang), 22 * np.sin(ang)][::-1]], m3d.FillRule.EvenOdd), 4)
post = m3d.Manifold.cube([8, 8, 60], True).translate([18, 0, 2])        # through the bore, pressing on its wall
solid = wheel + post
fused = _tris(solid)
gp = survey(fused, 0.3)
assert len(gp) == 1, gp
found = find(fused, np.zeros(len(fused), int))
assert len(found) == 1 and abs(found[0]["bore"] - 22) < 0.3 and abs(found[0]["thickness"] - 4) < 0.1 and found[0]["teeth"] == teeth, found
gb, *_ = group(fused, gp, [{"name": "Post", "pieces": [0]}], splits=[{"name": "Wheel", "kind": "ring", "ring": 0}], rings=found)
vol = {n: abs(sum(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2]))) / 6) for n, t in gb}
assert abs(vol["Post"] + vol["Wheel"] - solid.volume()) < 1e-3 * solid.volume() and vol["Wheel"] >= wheel.volume() - 1, (vol, wheel.volume())
print("gear wheel ok", {k: round(x, 1) for k, x in vol.items()})

try:
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDF import TDF_ChildIterator
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
except ImportError:
    print("ok (no native OCCT to read the STEP back)")
    sys.exit(0)
with tempfile.TemporaryDirectory() as d:
    out = Path(d) / "bodies.step"
    info = write_step(out, None, bodies=bodies)
    assert info["products"] == 2 and info["solids"] == 3, info
    doc, r = TDocStd_Document(TCollection_ExtendedString("XmlOcaf")), STEPCAFControl_Reader()
    r.ReadFile(str(out)); r.Transfer(doc)
    tool, got = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main()), {}
    it = TDF_ChildIterator(tool.Label())
    while it.More():
        lab, nm, props = it.Value(), TDataStd_Name(), GProp_GProps()
        if tool.IsFree_s(lab) and lab.FindAttribute(TDataStd_Name.GetID_s(), nm):
            s = tool.GetShape_s(lab)
            BRepGProp.VolumeProperties_s(s, props)
            assert BRepCheck_Analyzer(s).IsValid(), nm.Get().ToExtString()
            got[nm.Get().ToExtString()] = round(props.Mass(), 3)
        it.Next()
    assert got == {"Figure": 740.0, "Base": 300.0}, got
print("ok")
