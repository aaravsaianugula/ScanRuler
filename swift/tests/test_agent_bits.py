# SPDX-License-Identifier: AGPL-3.0-only
"""Self-checks for the agent's context helpers and the validator's geometry (no model, no ScanRuler).
    python swift/tests/test_agent_bits.py"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import Agent, slim  # noqa: E402
from validate import MeshDistance, closest_on_tris, volume  # noqa: E402

# slim: big numeric arrays go, structure and small values stay
s = json.loads(slim(json.dumps({"a": {"grid": [[1.5] * 50] * 20, "n": 3}, "b": [1, 2], "c": "x"})))
assert s["a"]["n"] == 3 and s["b"] == [1, 2] and s["c"] == "x" and isinstance(s["a"]["grid"], str), s
assert slim("not json") == "not json"

# closest point on a triangle: inside, edge, vertex regions
a, b, c = (np.array([[0.0, 0, 0]]), np.array([[1.0, 0, 0]]), np.array([[0.0, 1, 0]]))
for p, want in (([0.2, 0.2, 1], [0.2, 0.2, 0]), ([0.5, -1, 0], [0.5, 0, 0]), ([-1, -1, 0], [0, 0, 0]), ([2, 2, 0], [0.5, 0.5, 0])):
    got = closest_on_tris(np.array([p], float), a, b, c)[0]
    assert np.allclose(got, want), (p, got)

# a unit cube: volume 1, distance from a point 0.3 above its top is 0.3, signed outside
V = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
F = [(0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3), (0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6)]
T = V[np.array(F)]
assert abs(abs(volume(T)) - 1) < 1e-12
d, sg = MeshDistance(T, 0.1)(np.array([[0.5, 0.5, 1.3], [0.5, 0.5, 0.8]]))
assert np.allclose(d, [0.3, 0.2]) and sg[0] == -sg[1], (d, sg)


# compaction: old images go, history comes under budget, the last messages stay whole
class WB:  # the agent only needs .work for these helpers
    work = Path(".")


ag = Agent(WB(), "cad", log=lambda *_: None)
msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
for i in range(40):
    msgs.append({"role": "tool", "tool_call_id": str(i), "content": "x" * 12000})
    msgs.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:"}}]})
ag.compact(msgs)
from agent import BUDGET, est  # noqa: E402
assert sum(est(m) for m in msgs) <= BUDGET * 1.1, sum(est(m) for m in msgs)
assert isinstance(msgs[-1]["content"], list) and sum(isinstance(m["content"], list) for m in msgs) == 1
print("ok")

# give_up on a scan with no closed solid and most turns left: refused once, then allowed; late or with a solid, allowed
from types import SimpleNamespace  # noqa: E402
ag = Agent(SimpleNamespace(), "scan", log=lambda *a: None, max_turns=60)
ag.turn = 9
assert str(ag.t_finish("stuck", give_up=True)).startswith("refused") and ag.t_finish("stuck", give_up=True) is None
ag = Agent(SimpleNamespace(), "scan", log=lambda *a: None, max_turns=60)
ag.turn = 45
assert ag.t_finish("stuck", give_up=True) is None
ag = Agent(SimpleNamespace(whole="x.stl"), "scan", log=lambda *a: None, max_turns=60)
assert ag.t_finish("stuck", give_up=True) is None
print("ok give-up guard")

# the solid gate: a 10 mm cube judged against itself passes; inside out, half missing, or grown, it fails
from validate import solid_gate, tunnels  # noqa: E402
C = T * 10
assert tunnels(C) == 0
ok = solid_gate(C, [("cube", C)], 0.2, 0.01, points=200_000)["checks"]
assert all(ok.values()), ok
assert not solid_gate(C, [("cube", C[:, ::-1])], 0.2, 0.01, points=200_000)["checks"]["bodies_closed"]
half = C * np.array([1, 1, 0.5])
assert not solid_gate(C, [("half", half)], 0.2, 0.01, points=200_000)["checks"]["volume_solid"]
grown = (C - 5) * 1.3 + 5
assert not solid_gate(C, [("grown", grown)], 0.2, 0.01, points=200_000)["checks"]["stays_in_scan"]
print("solid gate ok")
