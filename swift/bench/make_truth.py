# SPDX-License-Identifier: AGPL-3.0-only
"""Build ground-truth parts of known design in headless ScanRuler and export each as
STL (the input) + STEP (the truth), so the Swift pipeline can be scored against reality.

    python swift/bench/make_truth.py <out_dir> [--noise 0.03]   # noise>0 also writes a fake 'scan'
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mcp_client import ScanRulerMCP  # noqa: E402
from meshio_min import read_stl, write_stl  # noqa: E402


def F(**f):
    return {"tool": "reverse_feature_add", "arguments": {"feature": f, "detail": "none"}}


def sketch(name, axis, offset, items):
    sk = {"shapes": [s for s in items if s["kind"] != "circle"], "entities": [s for s in items if s["kind"] == "circle"]}
    return F(kind="sketch", name=name, plane={"kind": "world", "axis": axis, "offset": offset}, sketch={k: v for k, v in sk.items() if v})


def circle(i, c, r):
    return {"kind": "circle", "id": i, "c": c, "r": r, "fixed": True}


PARTS = {
    # plate 60x40x10, two Ã˜8 holes, Ã˜16 boss 12 high with Ã˜6 bore, Ã˜4 cross hole, R3 corners, 1 mm boss chamfer
    "bracket": {
        "truth": {"plate": [60, 40, 10], "holes_d": 8, "boss_d": 16, "boss_h": 12, "bore_d": 6, "cross_d": 4,
                  "corner_r": 3, "chamfer": 1},
        "calls": [
            sketch("Base", "z", 0, [{"kind": "rectangle", "id": 1, "from": [0, 0], "to": [60, 40]},
                                    circle(5, [15, 20], 4), circle(6, [45, 20], 4)]),
            F(kind="extrude", name="Plate", sketchId="Base", regions=[{"at": [30, 5]}],
              extent={"kind": "distance", "from": 0, "to": 10}, op="new"),
            sketch("Boss sk", "z", 10, [circle(1, [30, 20], 8)]),
            F(kind="extrude", name="Boss", sketchId="Boss sk", extent={"kind": "distance", "from": 0, "to": 12},
              op="join", target="Plate"),
            sketch("Bore sk", "z", 0, [circle(1, [30, 20], 3)]),
            F(kind="extrude", name="Bore", sketchId="Bore sk", extent={"kind": "distance", "from": 0, "to": 22},
              op="cut", target="Plate"),
            sketch("Cross sk", "x", 0, [circle(1, [10, 5], 2)]),
            F(kind="extrude", name="Cross", sketchId="Cross sk", extent={"kind": "distance", "from": -1, "to": 61},
              op="cut", target="Plate"),
            F(kind="fillet", name="Corners", body="Plate", radius=3,
              edges=[{"through": {"at": p, "within": 0.01}} for p in ([0, 0, 5], [60, 0, 5], [60, 40, 5], [0, 40, 5])]),
            F(kind="chamfer", name="Boss chamfer", body="Plate", distance=1,
              edges=[{"through": {"at": [38, 20, 22], "within": 0.01}}]),
        ],
    },
    # block 50x30x20 with a side pocket cut along X (walls along another axis than the main build) and a
    # through slot along Y: the deterministic slabs-along-z pass misses both, so a fix along X/Y is needed
    "block": {
        "truth": {"block": [50, 30, 20], "side_pocket": "Y 10..20, Z 6..14, X 0..18", "y_slot": "X 30..40, Z 14..20, through Y"},
        "calls": [
            sketch("Block sk", "z", 0, [{"kind": "rectangle", "id": 1, "from": [0, 0], "to": [50, 30]}]),
            F(kind="extrude", name="Block", sketchId="Block sk", extent={"kind": "distance", "from": 0, "to": 20}, op="new"),
            sketch("Pocket sk", "x", 0, [{"kind": "rectangle", "id": 1, "from": [10, 6], "to": [20, 14]}]),
            F(kind="extrude", name="Pocket", sketchId="Pocket sk", extent={"kind": "distance", "from": -1, "to": 18},
              op="cut", target="Block"),
            sketch("Slot sk", "y", 0, [{"kind": "rectangle", "id": 1, "from": [14, 30], "to": [21, 40]}]),
            F(kind="extrude", name="Slot", sketchId="Slot sk", extent={"kind": "distance", "from": -1, "to": 31},
              op="cut", target="Block"),
        ],
    },
    # turned spacer: Ã˜30 x 8 flange, Ã˜20 x 20 shank, Ã˜10 bore, 1 mm chamfers on the flange
    "spacer": {
        "truth": {"flange_d": 30, "flange_h": 8, "shank_d": 20, "shank_h": 20, "bore_d": 10},
        "calls": [
            sketch("Flange sk", "z", 0, [circle(1, [0, 0], 15)]),
            F(kind="extrude", name="Flange", sketchId="Flange sk", extent={"kind": "distance", "from": 0, "to": 8}, op="new"),
            sketch("Shank sk", "z", 8, [circle(1, [0, 0], 10)]),
            F(kind="extrude", name="Shank", sketchId="Shank sk", extent={"kind": "distance", "from": 0, "to": 20},
              op="join", target="Flange"),
            sketch("Bore sk", "z", 0, [circle(1, [0, 0], 5)]),
            F(kind="extrude", name="Bore", sketchId="Bore sk", extent={"kind": "distance", "from": -1, "to": 29},
              op="cut", target="Flange"),
            F(kind="chamfer", name="Flange chamfer", body="Flange", distance=1,
              edges=[{"through": {"at": [15, 0, 0], "within": 0.01}}, {"through": {"at": [15, 0, 8], "within": 0.01}}]),
        ],
    },
}


# the bracket left at an angle, as a part exported in its assembly position: frame.py's case
PARTS["bracket_tilted"] = {
    "truth": dict(PARTS["bracket"]["truth"], tilt_deg=[12, -7, 20]),
    "calls": PARTS["bracket"]["calls"] + [F(kind="moveBody", name="Tilt", bodies=["Plate"], rotate=[12, -7, 20],
                                           pivot=[0, 0, 0], translate=[5, -3, 2])],
}


def add_noise(stl, out, sigma, seed=0):
    """A crude 'scan': vertices jittered along their normals by N(0, sigma)."""
    tris = read_stl(stl)
    v, inv = np.unique(tris.reshape(-1, 3).round(6), axis=0, return_inverse=True)
    f = inv.reshape(-1, 3)
    n = np.zeros_like(v)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    v = v + n * np.random.default_rng(seed).normal(0, sigma, (len(v), 1))
    write_stl(out, v[f])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--only")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    m = ScanRulerMCP(out)
    try:
        for name, part in PARTS.items():
            if a.only and name != a.only:
                continue
            m.call("scanruler_batch", {"calls": [{"tool": "reverse_feature_clear", "arguments": {}}]})
            txt, _, err = m.call("scanruler_batch", {"calls": part["calls"]})
            res = json.loads(txt)
            if res.get("failed") or err:
                sys.exit(f"{name}: {json.dumps(res.get('failed') or txt)[:1500]}")
            txt, _, _ = m.call("scanruler_batch", {"calls": [
                {"tool": "reverse_export_stl", "arguments": {"chord": 0.005, "path": str(out / f"{name}.stl")}},
                {"tool": "reverse_export_step", "arguments": {"path": str(out / f"{name}_truth.step")}}]})
            if json.loads(txt).get("failed"):
                sys.exit(f"{name} export: {txt[:1500]}")
            (out / f"{name}_truth.json").write_text(json.dumps(part["truth"], indent=1))
            if a.noise > 0:
                add_noise(out / f"{name}.stl", out / f"{name}_scan.stl", a.noise)
            print(f"{name}: ok")
    finally:
        m.close()


if __name__ == "__main__":
    main()
