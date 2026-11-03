"""Small tunnels and handles off a voxel solid, everything else kept voxel for voxel (Zhou, Ju, Hu 2007; Wood et
al. 2004 for why: a closed scan's hundreds of handles block any disk patch layout). The solid is closed (tunnels
fill) or opened (thin handles go), the change is thinned back by simple points - what cannot go without changing
the topology is a plug across a tunnel or a cut through a handle - and each such piece is kept only when the
Euler characteristic says it takes a handle away. RE_RESEARCH.md 8, step 4a.
Topology is that of closed unit cubes: a 26-connected solid with a 6-connected outside."""
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

OFF = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)])
_L1 = np.abs(OFF).sum(1)
_bits = lambda sel: np.uint32(sum(1 << int(b) for b in np.flatnonzero(sel)))
FACE, N18 = _bits(_L1 == 1), _bits(_L1 <= 2)
ADJ26 = [_bits((np.abs(OFF - o).max(1) == 1)) for o in OFF]
ADJ6 = [_bits((np.abs(OFF - o).sum(1) == 1) & (_L1 <= 2)) for o in OFF]


def _grow(comp, cfg, adj):
    """The pieces of cfg's bits that hold comp's bits (bit sets, one per element)."""
    todo = np.arange(len(comp))
    comp = comp.copy()
    while len(todo):
        c = comp[todo]
        new = c.copy()
        for i in range(26):
            new |= ((c >> i) & 1) * adj[i]
        new &= cfg[todo]
        comp[todo] = new
        todo = todo[new != c]
    return comp


def simple(cfg):
    """Whether a voxel whose 26 neighbours are cfg's bits (bit i: OFF[i] solid) is simple: setting or clearing
    it changes no piece, tunnel or cavity - its solid neighbours are one 26-connected piece and its outside
    neighbours in the 18-neighbourhood that touch it by a face are one 6-connected piece."""
    ok = ((~cfg & FACE) != 0) & (cfg != 0)     # no outside face neighbour or no solid one: never simple
    c = cfg[ok]
    out = ~c & N18
    face = out & FACE
    one26 = _grow(c & (~c + np.uint32(1)), c, ADJ26) == c
    one6 = (_grow(face & (~face + np.uint32(1)), out, ADJ6) & face) == face
    ok[ok] = one26 & one6
    return ok


def _cfg(flat, idx, off):
    cfg = np.zeros(len(idx), np.uint32)
    for b, o in enumerate(off):
        cfg |= flat[idx + o].astype(np.uint32) << b
    return cfg


def settle(Y, cand, value):
    """Set the voxels cand (flat indices) of Y to value, each only while that changes no topology, until none
    can go. Eight subfields by coordinate parity: no two voxels of one are neighbours, so a sweep of one is a
    sequence of single flips. Returns the voxels left unchanged."""
    flat, off = Y.reshape(-1), OFF @ (np.array(Y.strides) // Y.itemsize)
    zyx = np.unravel_index(cand, Y.shape)
    par = (zyx[0] % 2) * 4 + (zyx[1] % 2) * 2 + zyx[2] % 2
    left = [cand[par == k] for k in range(8)]
    while True:
        moved = 0
        for k in range(8):
            c = left[k]
            if len(c):
                cfg = _cfg(flat, c, off)
                ok = simple(cfg)
                flat[c[ok]] = value
                left[k], moved = c[~ok], moved + ok.sum()
        if not moved:
            return np.concatenate(left)


def euler(B):
    """Euler characteristic of the union of the closed unit cubes at B's set voxels."""
    P = np.pad(B, 1)

    def cells(axes):
        A = P
        for ax in axes:
            lo, hi = [slice(None)] * 3, [slice(None)] * 3
            lo[ax], hi[ax] = slice(None, -1), slice(1, None)
            A = A[tuple(lo)] | A[tuple(hi)]
        return int(A.sum())
    return (cells((0, 1, 2)) - cells((0, 1)) - cells((0, 2)) - cells((1, 2))
            + cells((0,)) + cells((1,)) + cells((2,)) - int(B.sum()))


def pieces(idx, shape):
    """26-connected pieces of a sparse voxel set (flat indices, sorted, off the array's border): labels."""
    st = np.array([shape[1] * shape[2], shape[2], 1])
    i, j = [], []
    for o in OFF[:13] @ st:
        pos = np.searchsorted(idx, idx + o)
        hit = pos < len(idx)
        hit[hit] = idx[pos[hit]] == idx[hit] + o
        i.append(np.flatnonzero(hit)); j.append(pos[hit])
    i, j = np.concatenate(i), np.concatenate(j)
    return connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(len(idx),) * 2), directed=False)[1]


def _keep(X, base, idx, adding, most):
    """Which pieces of idx (voxels a closing adds or an opening takes, thinned) take a handle off X: the genus
    change of flipping one piece alone is the change in pieces less the change in Euler characteristic
    (cavities taken as unchanged: simplify checks). base: the labelled pieces the flipped piece touches - X's
    when adding, the solid with every such voxel cleared when taking away. A piece over most voxels is no small
    handle: kept out. Returns (keep, piece) per voxel."""
    lab = pieces(idx, X.shape)
    order = np.argsort(lab, kind="stable")
    cuts = np.flatnonzero(np.diff(lab[order])) + 1
    keep = np.zeros(lab.max() + 1, bool)
    zyx = np.array(np.unravel_index(idx, X.shape)).T
    for p, grp in enumerate(np.split(order, cuts)):
        q = zyx[grp]
        if len(q) > most:
            continue
        lo, hi = np.maximum(q.min(0) - 2, 0), q.max(0) + 3
        win = tuple(slice(a, b) for a, b in zip(lo, hi))
        before = X[win]
        after = before.copy()
        after[tuple((q - lo).T)] = adding
        w, touch = base[win], set()
        for o in OFF:
            v = w[tuple((q - lo + o).T)]
            touch.update(v[v > 0].tolist())
        m = len(touch)
        d_pieces = (1 - m) if adding else (m - 1)
        keep[p] = d_pieces - (euler(after) - euler(before)) < 0
    return keep[lab], lab


def _cube(A, r, grow):
    """A dilated (grow) or eroded by a (2r+1)^3 cube, outside the array empty: three 1-D max or min filters, each
    run on threads in slabs across another axis (ndimage lets go of the GIL)."""
    f = ndimage.maximum_filter1d if grow else ndimage.minimum_filter1d
    A = A.view(np.uint8)
    with ThreadPoolExecutor(16) as pool:
        for ax in range(3):
            out, b = np.empty_like(A), (ax + 1) % 3
            cut = np.linspace(0, A.shape[b], 17).astype(int)

            def run(i):
                sl = tuple(slice(cut[i], cut[i + 1]) if k == b else slice(None) for k in range(3))
                f(A[sl], 2 * r + 1, axis=ax, output=out[sl], mode="constant", cval=0)
            list(pool.map(run, range(16)))
            A = out
    return A.view(bool)


def simplify(X, r=3, most=None, log=lambda s: print(s, file=sys.stderr)):
    """X (a boolean volume, the solid) with tunnels and handles under about 2r voxels across taken off: the
    smallest first (radius 1, 2, .. r: a wide closing joins neighbouring pinholes into one membrane, which is no
    small fix). A plug or cut over most voxels (default 16 rr^2 at radius rr: a slit's, not a pinhole's) is no
    small handle and stays out. Returns (voxels to set solid, voxels to clear) as flat indices into X."""
    box = ndimage.generate_binary_structure(3, 3)
    objs = ndimage.find_objects(X.astype(np.uint8))
    if not objs:            # no solid: nothing to simplify
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    sl, m = objs[0], r + 3
    sub0 = np.pad(X[sl], m)
    sub = sub0.copy()
    for rr in range(1, r + 1):
        sub = _stage(sub, rr, most or 16 * rr * rr, box, log)

    def full(i):
        q = np.array(np.unravel_index(i, sub.shape)).T - m + [s.start for s in sl]
        q = q[((q >= 0) & (q < X.shape)).all(1)]
        return np.ravel_multi_index(tuple(q.T), X.shape)
    return full(np.flatnonzero(sub & ~sub0)), full(np.flatnonzero(sub0 & ~sub))


def _stage(sub, r, most, box, log):
    """One radius of simplify: sub with its kept plugs and cuts."""
    for adding in (True, False):
        # a closing fills tunnels and is thinned back; an opening cuts handles and is grown back
        Y = (_cube(_cube(sub, r, True), r, False) | sub) if adding else (_cube(_cube(sub, r, False), r, True) & sub)
        if not adding:      # a piece the opening took whole grows back from one voxel: it keeps a cut, not nothing
            lab, n = ndimage.label(sub, box)
            have = np.zeros(n + 1, bool)
            have[lab[Y]] = True
            for c in np.flatnonzero(~have[1:]) + 1:
                w = ndimage.find_objects(lab, c)[c - 1]
                k = np.flatnonzero(lab[w] == c)[0]
                Y[tuple(a + sl_.start for a, sl_ in zip(np.unravel_index(k, lab[w].shape), w))] = True
            del lab
        cand = np.flatnonzero(Y != sub)
        left = np.sort(settle(Y, cand, not adding))
        keep, piece = np.zeros(0, bool), np.zeros(0, int)
        if len(left):
            keep, piece = _keep(sub, ndimage.label(sub if adding else Y, box)[0], left, adding, most)
        del Y
        if adding and keep.any():
            # a plug across a crevice's mouth seals it: a cavity, not a tunnel less - those plugs stay out
            six = ndimage.generate_binary_structure(3, 1)
            T = sub.copy()
            T.reshape(-1)[left[keep]] = True
            was = ndimage.label(~sub, six)[0]
            now = ndimage.label(~T, six)[0]
            del T
            q = np.array(np.unravel_index(left[keep], sub.shape)).T
            bad = np.zeros(len(q), bool)
            for o in OFF[_L1 == 1]:     # outside beside a plug that was open to the outer air and now is not
                at = tuple((q + o).T)
                bad |= (now[at] > 0) & (now[at] != now.flat[0]) & (was[at] == was.flat[0])
            del was, now
            keep &= ~np.isin(piece, np.unique(piece[keep][bad]))
        log(f"[topology] {'closing' if adding else 'opening'} r={r}: {len(cand)} voxels changed, {len(left)} left after "
            f"thinning, {keep.sum()} kept in {len(np.unique(piece[keep]))} pieces"
            + (f", largest {np.bincount(piece[keep]).max()}; {(np.bincount(piece) > most).sum()} pieces over {most} left out" if keep.any() else ""))
        sub = sub.copy()
        sub.reshape(-1)[left[keep]] = adding
    return sub
