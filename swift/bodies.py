# SPDX-License-Identifier: AGPL-3.0-only
"""Which closed pieces of the organic solid make one body. The closing leaves a figurine in many pieces:
the figure, a base it stands on, a staff it holds but does not touch, a sliver of double scan layer
hugging the body, specks. survey() lists them (volume, box, centre, gap to the nearest other piece) and
draws them numbered; group() turns a decision - these pieces are the Figure, this one the Base, those
junk - into named products for the STEP writer and the mesh the validator judges.

A piece the closing fused (a figure on its base, a gear touching a leg) is cut apart with a knife - a plane,
a box, a rod or tube, a fitted ring or gear (rings.py) - by exact booleans (manifold3d: pip install
manifold3d): every scanned detail stays in one body or the other, each body is closed, and their volumes add
up to the solid's.

    python swift/bodies.py reduced.stl [--grid 0.3] [--png pieces.png]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meshio_min import read_stl  # noqa: E402
from step_faceted import indexed, solid_pieces  # noqa: E402

LISTED = 30         # pieces described one by one; the rest summed up


def _pieces(tris):
    v, f = indexed(np.asarray(tris, float))
    f, comp, cvol, solids, voids, lost = solid_pieces(v, f)
    # a solid's faces with its cavities': what it keeps or drops
    member = np.full(len(cvol), -1)
    for i, c in enumerate(solids):
        member[[c, *voids[c]]] = i
    f = np.where((cvol[comp] < 0)[:, None], f[:, ::-1], f)      # cavities facing in again, as in the mesh
    return v, f, member[comp], cvol, solids, voids


def survey(tris, grid, png=None):
    """The solids, largest first: [{id, volume, share, lo, hi, centre, cavities, nearest, gap, touching}].
    gap is the closest approach (mm, vertex to vertex) to any other piece; touching lists the pieces within
    two grid cells - the closing keeps pieces apart by about a cell even where the scan touches."""
    v, f, of, cvol, solids, voids = _pieces(tris)
    total = sum(cvol[c] for c in solids)
    vert_of = np.full(len(v), -1)
    vert_of[f.ravel()] = np.repeat(of, 3)
    used = vert_of >= 0
    tree, owner = cKDTree(v[used]), vert_of[used]
    out = []
    for i, c in enumerate(solids):
        p = v[np.unique(f[of == i])]
        lo, hi = p.min(0), p.max(0)
        t = v[f[(of == i)]]
        area = float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())
        others = owner != i
        d, j = cKDTree(v[used][others]).query(p) if others.any() else (np.full(len(p), np.inf), None)
        near = int(owner[others][j[d.argmin()]]) if j is not None else None
        touch = {int(owner[k]) for ks in tree.query_ball_point(p, 2 * grid) for k in ks} - {i}
        out.append(dict(id=i, volume=round(float(cvol[c] / 6), 1), share=round(float(cvol[c] / total), 5), area=round(area, 1),
                        thickness=round(float(2 * cvol[c] / 6 / area), 2), sheet=bool(2 * cvol[c] / 6 / area < 2 * grid),
                        lo=lo.round(1).tolist(), hi=hi.round(1).tolist(), centre=((lo + hi) / 2).round(1).tolist(),
                        cavities=len(voids[c]), nearest=near, gap=round(float(d.min()), 2) if near is not None else None,
                        touching=sorted(touch)))
    if png:
        render(v, f, of, out, png)
    return out


def describe(pieces, grid):
    """The survey as text for Swift: the big pieces one by one, the rest summed up."""
    rows = [f"{len(pieces)} closed pieces (grid {grid:.3g} mm; touching = within 2 cells). id: volume mm3 (share), "
            "surface mm2, thickness mm (2 volume / area: under 2 cells is a sheet of scanned surface, not a speck), box lo..hi, cavities, "
            "nearest piece and gap mm, touching"]
    for p in pieces[:LISTED]:
        rows.append(f"{p['id']}: {p['volume']} ({100 * p['share']:.2f} %), {p['area']} mm2, {p['thickness']} thick, {p['lo']}..{p['hi']}, {p['cavities']} cav, "
                    f"nearest {p['nearest']} gap {p['gap']}, touching {p['touching'][:8]}{'...' if len(p['touching']) > 8 else ''}")
    rest = pieces[LISTED:]
    if rest:
        near = {}
        for p in rest:
            near[p["nearest"]] = near.get(p["nearest"], 0) + 1
        rows.append(f"{len(rest)} more (ids {LISTED}-{len(pieces) - 1}): {sum(p['volume'] for p in rest):.1f} mm3, {sum(p['area'] for p in rest):.0f} mm2 of surface "
                    f"({100 * sum(p['share'] for p in rest):.2f} %), largest {rest[0]['volume']} mm3, max gap "
                    f"{max(p['gap'] or 0 for p in rest)} mm; nearest piece: " + ", ".join(f"{k}: {n}" for k, n in sorted(near.items(), key=lambda x: -x[1])[:8]))
    return "\n".join(rows)


def group(tris, pieces, groups, drop=(), rest="touching", splits=(), rings=(), rods=()):
    """groups: [{name, pieces: [ids]}]; drop: ids that are junk. Pieces in neither follow rest: 'touching'
    (join the group of a grouped piece they touch, else dropped), 'nearest' (join the nearest grouped
    piece's group), 'drop'. splits: [{name, kind, ..., pieces?}] - what a knife (see knife()) takes of the kept
    pieces it reaches (or of pieces) becomes the body name, the rest stays where it was; in order, so a later
    cut takes from what earlier ones left. Bits a cut tears off the rest that are under 5 % of what it took go
    with the body. Every cut is checked to keep the volume, and the bodies to add up to the kept solid.
    Returns the bodies [(name, (vertices, faces))] for the STEP writer, the kept triangles (uncut: what the validator
    judges), the dropped pieces and the cuts made. Refuses ids out of range, a piece in two places, an
    unnamed or empty group, a cut that reaches nothing."""
    n = len(pieces)
    bad = lambda i: not isinstance(i, int) or not 0 <= i < n
    where = {}
    for gi, g in enumerate(groups):
        if not str(g.get("name", "")).strip() or not g.get("pieces"):
            raise ValueError(f"group {gi} needs a name and pieces")
        for i in g["pieces"]:
            if bad(i) or i in where:
                raise ValueError(f"piece {i}: " + ("no such piece" if bad(i) else "in two groups"))
            where[i] = gi
    for i in drop:
        if bad(i) or i in where:
            raise ValueError(f"piece {i}: " + ("no such piece" if bad(i) else "both grouped and dropped"))
    if rest not in ("touching", "nearest", "drop"):
        raise ValueError("rest is touching, nearest or drop")
    grouped = dict(where)
    dropped = {i: "dropped as junk" for i in drop}
    for p in pieces:
        i = p["id"]
        if i in grouped or i in dropped:
            continue
        hit = [grouped[j] for j in p["touching"] if j in grouped]
        if rest == "touching" and hit:
            where[i] = hit[0]
        elif rest == "nearest" and grouped:
            j = p["nearest"] if p["nearest"] in grouped else \
                min(grouped, key=lambda j: np.linalg.norm(np.subtract(pieces[j]["centre"], p["centre"])))
            where[i] = grouped[j]
        else:
            dropped[i] = "in no group, touching none" if rest == "touching" else "in no group"
    names = [str(g["name"]).strip().replace("'", "") for g in groups]
    cut_names = {str(c.get("name", "")).strip().replace("'", "") for c in splits}
    if len(set(names)) != len(names) or cut_names & set(names):
        raise ValueError("two bodies with one name (cuts may share one: they make one body, a rod and its head)")
    v, f, of, *_ = _pieces(tris)
    # a body is a list of parts: a piece as scanned (its own triangles, no boolean touched it) or what a cut
    # left of one (manifold3d). Booleans only where a knife goes through a solid piece: a sheet of scanned
    # surface (a sliver a fraction of a cell thick) is never cut - it goes whole with the knife holding its
    # centre - since booleans on next to no volume make junk shells
    orig = {i: v[f[of == i]] for i in where}
    part = {i: orig[i] for i in where}           # triangles, or a manifold once cut
    vol = lambda x: x.volume() if not isinstance(x, np.ndarray) else _volume(x)
    total = sum(vol(x) for x in part.values())
    made, extra = [], []
    for k, c in enumerate(splits):
        nm = str(c.get("name", "")).strip().replace("'", "")
        if not nm:
            raise ValueError(f"cut {k}: needs a name")
        K = knife(c, rings, rods, 3 * float(np.linalg.norm(np.ptp(v, axis=0))) + 10)
        targets = c.get("pieces", None)
        if targets is None and c.get("piece") is not None:
            targets = [c["piece"]]
        if targets is None:     # every kept piece the knife reaches
            kb = K.bounding_box()
            targets = [i for i, x in part.items() if x is not None and _overlap(_box(x), kb)]
        took_parts, torn, took = [], 0.0, []
        for i in targets:
            if bad(i) or i not in part:
                raise ValueError(f"cut {k}: piece {i} is not kept")
            if part[i] is None:
                continue
            if pieces[i].get("sheet") and isinstance(part[i], np.ndarray):
                if _holds(K, part[i].reshape(-1, 3).mean(0)):
                    took_parts.append(part[i]); part[i] = None; took.append(i)
                continue
            m = part[i] if not isinstance(part[i], np.ndarray) else _manifold(part[i])
            inside, outside = m ^ K, m - K
            if inside.is_empty():
                continue
            if abs(inside.volume() + outside.volume() - m.volume()) > 1e-6 * abs(m.volume()) + 1e-3:
                raise ValueError(f"cut {k}: cutting piece {i} lost volume")
            # what the knife tore off the rest (a tooth tip past the knife's edge: small, and touching what it took)
            # goes with the body; a bit of the rest that only came apart elsewhere stays with the rest. The rest
            # stays as the boolean left it (its cavities with it), less what moved
            comps = sorted((x for x in outside.decompose() if x.volume() > 0), key=lambda x: -x.volume()) if not outside.is_empty() else []
            big = comps[0].volume() if comps else 0.0
            moved = [x for x in comps if (x.volume() < big or big <= 0.05 * inside.volume()) and x.volume() < 0.05 * inside.volume()
                     and inside.min_gap(x, 0.05) < 0.01]
            # joined to what it took: two solids touching face to face weld non-manifold (Kurisu's leg gear); the union's
            # zero-volume junk shells are left out by _mesh; a bit whose union pinches an edge stays with the rest
            u = _sum([inside, *moved]) if moved else inside
            if moved and not _clean(u):
                u = inside
                for x in moved:
                    if _clean(u + x):
                        u = u + x
                    else:
                        moved = [y for y in moved if y is not x]
            for x in moved:
                torn += x.volume()
                outside = outside - x
            inside = u
            took_parts.append(inside)
            part[i] = None if outside.is_empty() else outside
            took.append(i)
        if not took_parts:
            on = [r.get("pieces") for r in rings if c.get("kind") == "ring" and r["id"] == c.get("ring")]
            raise ValueError(f"cut {k} ({nm}): the knife reaches no kept solid"
                             + (f" - ring {c.get('ring')} is in pieces {on[0]}, not {targets}: where that piece is a body of its own "
                                "(a group), the ring is a body already and needs no cut" if on and on[0] else ""))
        hit = [x for x in extra if x[0] == nm]
        if hit:
            hit[0][1].extend(took_parts)
        else:
            extra.append((nm, took_parts))
        made.append(dict(name=nm, kind=c.get("kind", "plane"), pieces=took, volume=round(sum(vol(x) for x in took_parts), 1),
                         torn_off_joined=round(torn, 2)))
    # a piece no group named (a sliver hugging the solid) goes with the body nearest it once the cuts are made:
    # the sliver under a stand with the stand, not with the figure its piece was
    named = {i for g in groups for i in g["pieces"]}
    loose = [i for i in where if i not in named and part.get(i) is not None]
    bodies = [(nm, [part[i] for i in where if where[i] == gi and i not in loose and part[i] is not None]) for gi, nm in enumerate(names)] + extra
    bodies = [(nm, ps_) for nm, ps_ in bodies if ps_]
    if loose:
        from scipy.spatial import cKDTree
        trees = [cKDTree(np.concatenate([_pts(x) for x in ps_])) for _, ps_ in bodies]
        for i in loose:
            q = _pts(part[i])
            gaps = [float(tr.query(q)[0].min()) for tr in trees]
            k = int(np.argmin(gaps)) if gaps and min(gaps) < 5.0 else None
            if k is None:       # nothing within 5 mm: as rest said (the group it joined)
                nm = names[where[i]]
                hit = [j for j, (n_, _) in enumerate(bodies) if n_ == nm]
                k = hit[0] if hit else len(bodies)
                if not hit:
                    bodies.append((nm, []))
            bodies[k][1].append(part[i])
    # every bit of the kept solid in exactly one body: the bodies together are what the validator judges
    got = sum(vol(x) for _, ps_ in bodies for x in ps_)
    if abs(got - total) > 1e-6 * abs(total) + 1e-2:
        raise ValueError(f"the bodies do not add up to the solid: {got:.3f} mm3 of {total:.3f} "
                         f"({', '.join(f'{n} {sum(vol(x) for x in ps_):.1f}' for n, ps_ in bodies)})")
    kept = v[f[np.isin(of, list(where))]]
    out = [(nm, _mesh(ps_)) for nm, ps_ in bodies]
    return out, kept, [dict(id=i, volume=pieces[i]["volume"], area=pieces[i]["area"], why=w) for i, w in sorted(dropped.items())], made


def _box(x):
    if isinstance(x, np.ndarray):
        p = x.reshape(-1, 3)
        return (*p.min(0), *p.max(0))
    return x.bounding_box()


def _pts(x):
    return x.reshape(-1, 3) if isinstance(x, np.ndarray) else np.asarray(x.to_mesh64().vert_properties)[:, :3]


def _holds(K, p):
    import igl
    t = _tris(K)
    V, F = indexed(t)
    return bool(igl.fast_winding_number(V, F.astype(np.int64), np.asarray([p], float))[0] > 0.5)


def _volume(t):
    return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6)


def _manifold(t):
    import manifold3d as m3d
    v, f = indexed(t)
    m = m3d.Manifold(m3d.Mesh64(vert_properties=np.ascontiguousarray(v, np.float64), tri_verts=np.ascontiguousarray(f, np.uint32)))
    if m.status() != m3d.Error.NoError:
        raise ValueError(f"a piece is not a manifold solid to cut: {m.status()}")
    return m


def _mesh(parts):
    """A body's parts as one (vertices, faces): a piece as scanned welded as the scan was, a cut one shell by
    shell (to a micron: a sliver of a triangle collapses away), no vertex shared between parts or
    shells - two solids touching at an edge would share it four ways (non-manifold)."""
    vs, fs, base = [], [], 0
    for x in parts:
        for t, dec in ([(x, 6)] if isinstance(x, np.ndarray) else [(_tris(c), 6) for c in x.decompose() if abs(c.volume()) > 1e-6]):
            v, f = indexed(t, dec)
            vs.append(v); fs.append(f + base); base += len(v)
    return np.concatenate(vs), np.concatenate(fs).astype(np.int64)


def _clean(m):
    """m as _mesh writes it is closed and manifold (every edge in two triangles)."""
    f = _mesh([m])[1]
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1)
    return bool((np.unique(e, axis=0, return_counts=True)[1] == 2).all())


def _tris(m):
    g = m.to_mesh64()
    return np.asarray(g.vert_properties)[:, :3][np.asarray(g.tri_verts, np.int64)]


def _sum(ms):
    ms = [m for m in ms if m is not None and not m.is_empty()]
    if not ms:
        return None
    out = ms[0]
    for m in ms[1:]:
        out = out + m
    return out


def _overlap(a, b):
    return all(a[k] <= b[k + 3] and b[k] <= a[k + 3] for k in range(3))


def knife(c, rings=(), rods=(), span=1e4):
    """The region a cut takes (manifold3d), from {kind, ...}:
    plane {point, normal}: the normal's side; box {centre, size, axes?: [x, y] directions};
    cylinder {centre, axis, radius, height, inner?}: a rod or a tube; ring {ring, grow?}: a ring or gear
    rings.find() fitted, grown by grow mm (default 1) every side; rod {rod, grow?, extend?: [mm, mm]}: a rod
    rings.find_rods() fitted, grown, its ends moved out along its axis."""
    import manifold3d as m3d
    kind = c.get("kind", "plane")
    if kind not in ("plane", "box", "cylinder", "ring", "rod"):
        raise ValueError(f"no knife kind {kind}: plane, box, cylinder, ring or rod")
    big = float(span)   # past the solid, no further: a boolean's tolerance grows with its operands' size
    if kind == "plane":
        n = np.asarray(c["normal"], float)
        n /= np.linalg.norm(n)
        box = m3d.Manifold.cube([2 * big, 2 * big, big], True).translate([0, 0, big / 2])
        return box.transform(_place(n, np.asarray(c["point"], float)))
    if kind == "box":
        size = np.asarray(c["size"], float)
        x = np.asarray((c.get("axes") or [[1, 0, 0], [0, 1, 0]])[0], float)
        y = np.asarray((c.get("axes") or [[1, 0, 0], [0, 1, 0]])[1], float)
        x /= np.linalg.norm(x)
        y -= (y @ x) * x
        y /= np.linalg.norm(y)
        M = np.c_[x, y, np.cross(x, y), np.asarray(c["centre"], float)]
        return m3d.Manifold.cube(size.tolist(), True).transform(M.tolist())
    if kind == "cylinder":
        r, h, ri = float(c["radius"]), float(c["height"]), float(c.get("inner") or 0)
        tube = m3d.Manifold.cylinder(h, r, r, 256, True)
        if ri > 0:
            tube = tube - m3d.Manifold.cylinder(h + 1, ri, ri, 256, True)
        a = np.asarray(c["axis"], float)
        return tube.transform(_place(a / np.linalg.norm(a), np.asarray(c["centre"], float)))
    if kind == "rod":
        from rings import rod_knife
        hit = [r for r in (rods or ()) if r["id"] == c.get("rod")]
        if not hit:
            raise ValueError(f"no rod {c.get('rod')}: call find_parts first")
        return rod_knife(hit[0], float(c.get("grow", 1.0)), c.get("extend") or (0, 0))
    if kind == "ring":
        from rings import knife as ring_knife
        hit = [r for r in rings if r["id"] == c.get("ring")]
        if not hit:
            raise ValueError(f"no ring {c.get('ring')}: call find_parts first")
        return ring_knife(hit[0], float(c.get("grow", 1.0)))
    raise AssertionError(kind)


def _place(z, origin):
    """A 3x4 transform taking +z to z, the origin to origin."""
    x = np.cross(z, [1, 0, 0] if abs(z[0]) < 0.9 else [0, 1, 0])
    x /= np.linalg.norm(x)
    return np.c_[x, np.cross(z, x), z, origin].tolist()


def render(v, f, of, pieces, png):
    """The pieces, each its own colour (the largest grey), the listed ones numbered."""
    keep = of >= 0
    draw(v[f[keep]], of[keep], [(str(p["id"]), p["centre"]) for p in pieces[:LISTED]], png)


def draw(t, lab, notes, png, title=""):
    """Three views (up = the longest axis; front, side, back), triangles coloured by label (0 grey, the rest
    distinct colours), notes [(text, point)] written at their points - for Swift to look at."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    pts = t.reshape(-1, 3)
    up = int(np.ptp(pts, axis=0).argmax())
    rng = np.random.default_rng(1)
    base = np.array([[0.9, 0.3, 0.25], [0.25, 0.55, 0.9], [0.95, 0.75, 0.2], [0.35, 0.75, 0.35], [0.7, 0.4, 0.85], [0.2, 0.8, 0.8]])
    colours = np.vstack([[0.82, 0.82, 0.8], base, rng.uniform(0.15, 0.95, (max(1, int(lab.max()) - len(base)), 3))])
    nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-20
    looks = [(a, s) for a in range(3) if a != up for s in (-1,)] + [([a for a in range(3) if a != up][0], 1)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 7.5), dpi=80)
    for ax, (look, sgn) in zip(axes, looks):
        right = 3 - up - look
        front = sgn * nrm[:, look] > 0                  # faces towards the viewer on that side
        order = np.flatnonzero(front)[np.argsort(sgn * t[front][:, :, look].mean(1))]     # far first
        shade = 0.35 + 0.65 * np.clip(sgn * nrm[order, look] * 0.8 + nrm[order, up] * 0.2, 0, 1)
        xy = t[order][:, :, [right, up]] * [-sgn, 1]    # seen from behind, left and right swap
        ax.add_collection(PolyCollection(xy, facecolors=colours[lab[order]] * shade[:, None], edgecolors="none",
                                         antialiased=False, rasterized=True))
        for text, at in notes:
            ax.annotate(text, (-sgn * at[right], at[up]), fontsize=9, weight="bold", ha="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.75, lw=0))
        xs = -sgn * pts[:, right]
        ax.set_xlim(xs.min(), xs.max()); ax.set_ylim(pts[:, up].min(), pts[:, up].max())
        # a mm grid in world coordinates: where a part is can be read off the picture
        from matplotlib.ticker import FuncFormatter, MaxNLocator
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(MaxNLocator(8)); ax.yaxis.set_major_locator(MaxNLocator(10))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _, s=sgn: f"{-s * x:g}"))
        ax.grid(True, lw=0.4, alpha=0.5); ax.set_axisbelow(False)
        ax.tick_params(labelsize=7)
        ax.set_xlabel(f"{'xyz'[right]} mm", fontsize=8); ax.set_ylabel(f"{'xyz'[up]} mm", fontsize=8)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(f"{'back' if sgn > 0 else 'looking along'} {'xyz'[look]}{' (from +' + 'xyz'[look] + ')' if sgn > 0 else ''}")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(png)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh"); ap.add_argument("--grid", type=float, default=0.3); ap.add_argument("--png")
    a = ap.parse_args()
    print(describe(survey(read_stl(a.mesh), a.grid, a.png), a.grid))
