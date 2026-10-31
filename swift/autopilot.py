# SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic first pass, no model: follow analyze.py's recommendation (frame, slabs or
revolve, holes, blends), build, validate. Swift starts from here and fixes what fails."""
import re

from workbench import Workbench


def first_pass(wb: Workbench, log=print):
    out = wb.script("analyze")
    log(out)
    if "build in the STL frame" not in out and "frame.py" in out:
        log(wb.script("frame"))     # tilted part: the skill's frame step; Swift handles the moveBody
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
    log(wb.rebuild(files))
    return files, out
