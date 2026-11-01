# SPDX-License-Identifier: AGPL-3.0-only
"""The organic route's ScanRuler repair is best-effort, with ScanRuler stubbed: a step that finds nothing,
refuses or gets no answer is reported and the next runs; the scan carried forward is the edited export when
there is one, else the scan as it is. A hiccup (no answer) is tried once more.
    python swift/tests/test_organic_repair.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from workbench import Workbench  # noqa: E402


def bench(d, answers):
    """A Workbench with no ScanRuler: batch answers from answers(tool, arguments), raising for a hiccup."""
    wb = Workbench.__new__(Workbench)
    wb.input, wb.repaired = Path(d) / "input.stl", Path(d) / "x_repaired.stl"
    wb.input.write_bytes(b"the scan")
    wb.calls = []

    def batch(calls):
        c = calls[0]
        wb.calls.append(c["tool"])
        out = answers(c["tool"], c["arguments"])
        if isinstance(out, Exception):
            raise out
        return json.dumps({"results": [out]}), ""
    wb.batch = batch
    return wb


nothing = {"mesh_remove_islands": {"ok": False, "error": {"message": "No piece smaller than that"}},
           "mesh_fill_holes": {"ok": False, "error": {"message": "No hole smaller than that"}},
           "mesh_export": {"ok": False, "error": {"message": "There is no edited copy - the first edit makes one."}}}

with tempfile.TemporaryDirectory() as d:
    # nothing to repair: no edited copy, the scan itself goes on (the kurisu_repaired.stl case)
    wb, report = bench(d, lambda t, a: nothing.get(t, {"ok": True, "result": {}})), []
    wb.repaired.write_bytes(b"stale copy of an earlier run")
    wb._repair(100.0, report)
    assert wb.repaired.read_bytes() == b"the scan", wb.repaired.read_bytes()
    assert wb.calls == ["mesh_discard_edited", "mesh_remove_islands", "mesh_fill_holes", "mesh_fill_holes", "mesh_export"], wb.calls
    assert "carried on from the scan as it is" in report[-1], report

    # a hiccup on the first fill is tried once more; a refusal is not, and the next step still runs
    seen = {"fill": 0}

    def flaky(t, a):
        if t == "mesh_fill_holes" and a.get("patch") == "curved":
            seen["fill"] += 1
            return RuntimeError("page closed") if seen["fill"] == 1 else {"ok": True, "result": {"status": "Filled 3 holes", "edit": {"maxSize": 4.2}}}
        if t == "mesh_fill_holes":
            return {"ok": False, "error": {"message": "the rim crosses itself"}}
        if t == "mesh_export":
            Path(a["path"]).write_bytes(b"edited")
            return {"ok": True, "result": {"status": "exported"}}
        return {"ok": True, "result": {}}
    wb, report = bench(d, flaky), []
    wb._repair(100.0, report)
    assert seen["fill"] == 2 and wb.hole_size == 4.2, (seen, getattr(wb, "hole_size", None))
    assert wb.repaired.read_bytes() == b"edited", wb.repaired.read_bytes()
    assert any("did not go through: the rim crosses itself" in r for r in report), report

    # ScanRuler gone altogether: every step reported, the scan itself goes on
    wb, report = bench(d, lambda t, a: RuntimeError("bridge down")), []
    wb._repair(100.0, report)
    assert wb.repaired.read_bytes() == b"the scan" and sum("no answer from ScanRuler" in r for r in report) == 4, report
print("ok organic repair fallbacks")
