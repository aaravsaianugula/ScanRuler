# SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic first pass, no model: follow analyze.py's recommendation (frame, slabs or
revolve, holes, blends), build, validate. Swift starts from here and fixes what fails."""
import json
import re

from workbench import Workbench


def first_pass(wb: Workbench, log=print):
    out = wb.script("analyze")
    log(out)
    move = None
    if "recommend: frame.py" in out:
        # tilted part (the skill's frame step): build in its own frame, turn the body back as the last feature
        log(wb.script("frame"))
        fr = wb.work / "data" / "frame.json"
        if fr.exists():
            move = json.loads(fr.read_text())
            out = wb.script("analyze", [str(wb.work / "data" / "local.stl")])
            log(out)
    rec = re.search(r"recommend: (\S+\.py)\s*([^\n(]*)", out)
    files = []
    if rec and rec.group(1) == "revolve.py":
        args = rec.group(2).split()
        log(wb.script("revolve", args + ["--plot", "revolve.png"]))
        wb.main_axis = args[args.index("--axis") + 1] if "--axis" in args else "z"
        wb.body = "Turned body"
        files = ["data/revolve.json"]
    else:
        m = re.search(r"slabs\.py --axis (\w)", out)
        wb.main_axis = m.group(1) if m else "z"
        wb.body = f"{wb.main_axis.upper()} slab 0"
        log(wb.script("slabs", ["--axis", wb.main_axis, "--plot", f"slabs_{wb.main_axis}.png"]))
        files = [f"data/slabs_{wb.main_axis}.json"]
    log(wb.script("holes", ["--main", wb.main_axis, "--body", wb.body]))
    files.append("data/holes.json")
    log(wb.rebuild(files))
    log(wb.script("blends", ["--main", wb.main_axis, "--body", wb.body]))
    files.append("data/blends.json")
    if move:
        (wb.work / "data" / "frame_move.json").write_text(json.dumps({"steps": [{"name": "reverse.feature.add", "input": {
            "feature": {"kind": "moveBody", "name": "Back to the STL frame", "bodies": [wb.body],
                        "rotate": move["rotate"], "pivot": move["pivot"]}, "detail": "none"}}]}))
        files.append("data/frame_move.json")
    log(wb.rebuild(files))
    return files, out
