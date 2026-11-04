# SPDX-License-Identifier: AGPL-3.0-only
"""Swift 1.5 (through the AI Gateway, OpenAI-compatible) driving the workbench with a small tool set.
The validator, not the model, decides when the part is done."""
import base64
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path

from workbench import Workbench, clip

GATEWAY = os.environ.get("SWIFT_URL", "http://127.0.0.1:11434/v1")
MODEL = os.environ.get("SWIFT_MODEL", "swift-1.5-qwen3.8-27b")
PROMPTS = Path(__file__).resolve().parent / "prompts"
CTX_TOKENS = 49152
BUDGET = 30000          # history kept under this (est. tokens), leaving room for the reply and thinking
PIN_BUDGET = 10000      # guide text kept verbatim
READ_ONLY = {"guide_read", "scanruler_tools", "reverse_feature_list", "reverse_feature_schema", "reverse_body_describe",
             "reverse_body_check", "reverse_parameter_list", "reverse_note_list", "session_state", "scan_query", "scan_nearest",
             "scan_qa", "scan_outline", "scan_draft", "scan_heightmap", "scan_pattern", "view_render", "section_render",
             "reverse_body_review", "reverse_body_compare", "reverse_body_volume_compare", "reverse_section_fit",
             "reverse_section_outline", "reverse_section_stack", "reverse_sketch_candidates", "reverse_body_fillet_read",
             "reverse_plane_describe", "reverse_thread_read", "report_get", "mesh_inspect", "mesh_parts", "deviation_hotspots"}


# tools that change the scan itself: the validator must see the scan as scanned
NO_EDIT = {"mesh_smooth", "mesh_delete_marked", "mesh_reduce", "mesh_fill_holes", "mesh_remove_islands", "mesh_cut",
           "mesh_offset", "mesh_shell", "mesh_boolean", "mesh_use_edited", "mesh_flip_normals"}


def tool(name, desc, props=None, req=()):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props or {}, "required": list(req)}}}


COMMON = [
    tool("check", "Export the current model and run the independent validator (and, for CAD meshes, compare.py's clusters and suggested fixes). Call after every change."),
    tool("view_image", "Look at a PNG in the work folder (a script plot, a ScanRuler render).", {"path": {"type": "string"}}, ["path"]),
    tool("read_file", "Read a text/JSON file in the work folder.", {"path": {"type": "string"}, "max_chars": {"type": "integer"}}, ["path"]),
    tool("scanruler", "Run ScanRuler tools in order: calls = [{tool, arguments}]. A refused change rolls the batch back.",
         {"calls": {"type": "array", "items": {"type": "object", "properties": {"tool": {"type": "string"}, "arguments": {"type": "object"}}, "required": ["tool"]}},
          "stopOnError": {"type": "boolean"}}, ["calls"]),
    tool("scanruler_help", "ScanRuler tools' definitions by name (no names: the list of all tools).", {"names": {"type": "array", "items": {"type": "string"}}}),
    tool("guide", "Read a ScanRuler guide: no topic = the index with token costs; topic (+ section) = that text.",
         {"topic": {"type": "string"}, "section": {"type": "string"}}),
    tool("finish", "Deliver. Refused unless the last check PASSED on the current model, or give_up is true.",
         {"report": {"type": "string"}, "give_up": {"type": "boolean"}}, ["report"]),
]
SCAN_TOOLS = [
    tool("organic_route", "For an organic part (figurine, sculpture, anatomy, a grip with no analytic faces): repair the scan "
         "(islands off, holes filled, closed), reduce it to max_triangles and write it as a closed STEP solid. The validator then "
         "judges that solid against the scan. Call it once: it takes minutes on a big scan, and what it cannot follow (sealed crevices, "
         "unscanned mouths) does not change with max_triangles.", {"reason": {"type": "string"}, "max_triangles": {"type": "integer"}}, ["reason"]),
    tool("body_pieces", "After organic_route: look at the solid - three views with a mm grid, its closed pieces coloured and numbered - "
         "and the pieces listed largest first (volume, box, cavities, nearest piece and gap, touching). Decide from the picture which "
         "parts a person would call separate bodies, and where they are."),
    tool("group_bodies", "After organic_route: name the bodies and write each as a named solid in the STEP, then the validator judges what is kept. "
         "groups: [{name, pieces: [ids]}] (one body each: a sliver hugging the figure goes in the figure's group); drop: junk ids (specks, "
         "stray layers; reported); rest: what unlisted pieces do - 'touching' (join the group of a grouped piece they touch, else dropped), "
         "'nearest', 'drop'. cuts: where the scan fused separate parts (a figure on its base, a gear touching a leg, a bolt in its hole), "
         "[{name, kind, ...}] carves the body name out of the kept pieces the knife reaches, by exact booleans - every scanned detail "
         "stays in one body or the other: plane {point, normal} (the normal's side), box {centre, size, axes?}, cylinder {centre, axis, "
         "radius, height, inner?} (a rod, a tube), ring {ring, grow?} (a ring/gear from find_parts, grown grow mm, default 1), rod {rod, "
         "grow?, extend?: [mm, mm]} (a rod from find_parts, its ends moved out along its axis over what hid it). Cuts sharing a name make "
         "one body (a staff's rod and a box round its head). "
         "Call again to regroup: it starts from the whole solid.",
         {"groups": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "pieces": {"type": "array", "items": {"type": "integer"}}}, "required": ["name", "pieces"]}},
          "drop": {"type": "array", "items": {"type": "integer"}}, "rest": {"type": "string", "enum": ["touching", "nearest", "drop"]},
          "cuts": {"type": "array", "items": {"type": "object", "properties": {
              "name": {"type": "string"}, "kind": {"type": "string", "enum": ["plane", "box", "cylinder", "ring", "rod"]},
              "point": {"type": "array", "items": {"type": "number"}}, "normal": {"type": "array", "items": {"type": "number"}},
              "centre": {"type": "array", "items": {"type": "number"}}, "size": {"type": "array", "items": {"type": "number"}},
              "axes": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
              "axis": {"type": "array", "items": {"type": "number"}}, "radius": {"type": "number"}, "height": {"type": "number"},
              "inner": {"type": "number"}, "ring": {"type": "integer"}, "rod": {"type": "integer"}, "grow": {"type": "number"},
              "extend": {"type": "array", "items": {"type": "number"}},
              "pieces": {"type": "array", "items": {"type": "integer"}}}, "required": ["name", "kind"]}}}, ["groups"]),
    tool("propose_bodies", "After organic_route: the finders' bodies, applied and drawn - separate pieces, rings/gears and long "
         "straight rods cut out where fused, a stand cut at its top face - with the group_bodies call they made. Look at the "
         "picture: keep it, or change it with group_bodies (edit that call: rename, regroup, cut what they missed, undo a wrong cut)."),
    tool("find_parts", "After organic_route: parts found by their shape, fused to others or not - rods (staffs, shafts, pins: radius, "
         "ends, axis) and flat rings and gears (centre, axis, thickness, bore, tip, teeth) - with how much of each was seen and its pieces. "
         "Candidates, not verdicts: a leg or an arm can read as a rod. near + within (mm): look only there - for a part you saw in "
         "a picture, its position read off the mm grid (much faster, and finds what the whole-solid search missed). A few minutes on "
         "the whole of a big solid.", {"near": {"type": "array", "items": {"type": "number"}}, "within": {"type": "number"}}),
]
CAD_TOOLS = [
    tool("run_script", "Run a reverse-engineering script (analyze, frame, slabs, revolve, holes, blends, compare, helix, profile, rz, centers, two_way, slice, vslice) with CLI args.",
         {"name": {"type": "string"}, "args": {"type": "array", "items": {"type": "string"}}}, ["name"]),
    tool("rebuild", "Clear the part and run these step files in order (main body, fills, trims, holes, blends).",
         {"files": {"type": "array", "items": {"type": "string"}}}, ["files"]),
]


def chat(messages, tools, thinking=True, max_tokens=6000, retries=3, on_delta=None):
    """One completion, streamed: on_delta(kind, text) sees the reasoning and the reply as they come
    (kind "reasoning" or "content"). Returns the assembled message and the usage."""
    body = {"model": MODEL, "messages": messages, "tools": tools, "tool_choice": "auto", "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": thinking}, "stream": True, "stream_options": {"include_usage": True}}
    # the model card's sampling, per mode
    body.update(dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0, presence_penalty=0, reasoning_effort="medium") if thinking
                else dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0, presence_penalty=1.5))
    req = urllib.request.Request(f"{GATEWAY}/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    for k in range(retries):
        try:
            return _read_stream(urllib.request.urlopen(req, timeout=1800), on_delta or (lambda *_: None))
        except Exception:  # the Gateway queues behind other clients' turns; a dropped call is retried
            if k == retries - 1:
                raise
            time.sleep(5 * (k + 1))


def _read_stream(resp, on_delta):
    content, calls, usage = [], {}, {}
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            usage = chunk.get("usage") or usage
            for ch in chunk.get("choices", []):
                d = ch.get("delta", {})
                if d.get("reasoning_content"):
                    on_delta("reasoning", d["reasoning_content"])
                if d.get("content"):
                    content.append(d["content"]); on_delta("content", d["content"])
                for tc in d.get("tool_calls") or []:
                    c = calls.setdefault(tc.get("index", 0), {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    c["id"] = tc.get("id") or c["id"]
                    f = tc.get("function") or {}
                    c["function"]["name"] += f.get("name") or ""
                    c["function"]["arguments"] += f.get("arguments") or ""
    msg = {"role": "assistant", "content": "".join(content)}
    if calls:
        msg["tool_calls"] = [calls[i] for i in sorted(calls)]
    return msg, usage


def slim(txt):
    """Long numeric arrays (heightmap grids, point lists) out of a tool answer: a picture or a
    narrower query reads them better than the model's context does."""
    def walk(v):
        if isinstance(v, list):
            flat = json.dumps(v)
            if len(flat) > 400 and all(isinstance(x, (int, float, list, type(None))) for x in v):
                return f"[{len(v)} values elided - ask for a picture or a smaller box]"
            return [walk(x) for x in v]
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        return v
    try:
        return json.dumps(walk(json.loads(txt)), ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return txt


def est(m):
    c = m.get("content")
    n = len(c) if isinstance(c, str) else sum(len(p.get("text", "")) + (1200 * 4 if p.get("type") == "image_url" else 0) for p in (c or []))
    return (n + len(json.dumps(m.get("tool_calls", "")))) // 3.5


def available(timeout=4):
    """Swift is there to ask: the Gateway answers and serves its model (a run without it goes on deterministic)."""
    try:
        with urllib.request.urlopen(f"{GATEWAY}/models", timeout=timeout) as r:
            return MODEL in [m.get("id") for m in json.loads(r.read()).get("data", [])]
    except Exception:
        return False


class Agent:
    def __init__(self, wb: Workbench, mode, log=print, max_turns=60, emit=None, deadline=None, review=False):
        self.wb, self.mode, self.log, self.max_turns = wb, mode, log, max_turns
        self.deadline = deadline    # time.time() past which no new turn starts
        self.review = review        # the solid is built: bodies and names only, no organic_route
        self.emit = emit or (lambda *a, **k: None)     # the desktop app's event stream
        self.tools = COMMON + (CAD_TOOLS if mode == "cad" else SCAN_TOOLS)
        self.changes = 0            # bumped by anything that may change the model
        self.checked_at = -1        # self.changes when the last check ran
        self.passed = False
        self.best = None            # (score, validation) of the best model seen
        self.pinned = set()         # message indices kept verbatim (guide text)
        self.turn, self.give_ups = 0, 0

    # -- tools -----------------------------------------------------------
    def t_check(self):
        out = self.wb.check(self.mode)
        r = self.wb.last_validation
        self.checked_at = self.changes
        self.passed = bool(r and r["verdict"] == "PASS")
        self.emit("validation", result=r, summary=out)
        if r:
            score = (sum(not v for v in r["checks"].values()), max(r["input_to_model"]["p95"], r["model_to_input"]["p95"]))
            if self.mode != "organic" and (self.best is None or score < self.best[0]):   # organic: one solid, kept as built
                self.best = (score, r)
                shutil.copyfile(self.wb.work / "model.stl", self.wb.work / "best_model.stl")
                self.wb.batch([{"tool": "reverse_export_step", "arguments": {"path": str(self.wb.work / "best_attempt_CAD.step")}}])
        if self.mode == "cad" and not self.passed and (self.wb.work / "model.stl").exists():
            out += "\n\n" + self.wb.compare()
        return out

    def t_view_image(self, path):
        p = (self.wb.work / path).resolve()
        if not p.exists() or self.wb.work not in p.parents and p.parent != self.wb.work:
            return f"no image at {path}", None
        return f"{path} attached", base64.b64encode(p.read_bytes()).decode(), str(p)

    def t_read_file(self, path, max_chars=6000):
        p = (self.wb.work / path).resolve()
        if self.wb.work not in p.parents or not p.is_file():
            return f"no file {path} in the work folder"
        return clip(p.read_text(encoding="utf-8", errors="replace"), max_chars)

    def t_scanruler(self, calls, stopOnError=True):
        banned = [c.get("tool") for c in calls if c.get("tool") in NO_EDIT]
        if banned:
            return f"refused: {banned} edit the scan, and the model must be judged against the scan as scanned."
        if any(c.get("tool") not in READ_ONLY for c in calls):
            self.changes += 1
        for c in calls:
            c.setdefault("arguments", {})
        txt, err = self.wb.batch(calls, stopOnError=stopOnError)
        out = clip(slim(txt), 6000)
        # a call refused for its arguments: hand back that tool's schema, so the next try is right
        try:
            bad = {r["tool"] for r in json.loads(txt).get("results", []) if (r.get("error") or {}).get("code") == "invalid_input"}
        except Exception:
            bad = set()
        if bad:
            out += "\n\nSchemas of the refused tools:\n" + self.schemas(sorted(bad))
        return out

    def schemas(self, names):
        txt, _, _ = self.wb.mcp.call("scanruler_tools", {"names": names})
        try:
            return clip(json.dumps([{"name": t["name"], "inputSchema": t["inputSchema"]} for t in json.loads(txt)["tools"]],
                                   separators=(",", ":")), 5000)
        except Exception:
            return clip(txt, 5000)

    def t_scanruler_help(self, names=None):
        txt, _, _ = self.wb.mcp.call("scanruler_tools", {"names": names} if names else {})
        return clip(txt, 9000)

    def t_guide(self, topic=None, section=None):
        args = {k: v for k, v in (("topic", topic), ("section", section)) if v}
        txt, _ = self.wb.batch([{"tool": "guide_read", "arguments": args}])
        try:
            res = json.loads(txt)["results"][0]["result"]
            txt = res.get("text") or json.dumps(res)
            if res.get("sectionsLeft"):
                txt += "\n[sections left, read each by section: " + ", ".join(res["sectionsLeft"]) + "]"
        except Exception:
            pass
        return clip(txt, 24000)

    def t_run_script(self, name, args=()):
        return self.wb.script(name, args)       # scripts write step files; the model changes on rebuild

    def t_organic_route(self, reason, max_triangles=150000):
        if self.review:
            return "refused: the solid is built and validated already; judge the bodies (group_bodies), check, finish"
        self.log(f"[swift] organic route: {reason}")
        self.emit("phase", phase="organic", label="Repairing the scan into a solid")
        self.emit("mode", mode="organic")
        out = self.wb.organic(max_triangles)
        self.mode = "organic"            # the gate is now the organic solid against the scan
        self.changes += 1
        return out + "\n\n" + self.t_check()

    def _with(self, out, png):
        """A tool answer with its picture: Swift decides bodies by looking, not from numbers alone."""
        p = self.wb.work / png
        return (out, base64.b64encode(p.read_bytes()).decode(), str(p)) if p.exists() and not out.startswith(("no ", "refused")) else out

    def t_body_pieces(self):
        return self._with(self.wb.body_pieces(), "pieces.png")

    def t_propose_bodies(self):
        out = self.wb.propose_bodies()
        if out.startswith(("refused", "no organic")):
            return out
        self.changes += 1
        return self._with(out + "\n\n" + self.t_check(), "bodies.png")

    def t_find_parts(self, near=None, within=40.0):
        return self._with(self.wb.find_parts(near, within), "parts.png")

    def t_group_bodies(self, groups, drop=(), rest="touching", cuts=()):
        out = self.wb.group_bodies(groups, drop, rest, cuts)
        if out.startswith(("refused", "no organic")):
            return out
        self.changes += 1
        return self._with(out + "\n\n" + self.t_check(), "bodies.png")

    def t_rebuild(self, files):
        self.changes += 1
        return self.wb.rebuild(files)

    def t_finish(self, report, give_up=False):
        if not give_up and not (self.passed and self.checked_at == self.changes):
            why = "the model changed since the last check" if self.checked_at != self.changes else "the last check did not PASS"
            return f"refused: {why}. Run check and fix what fails, or finish with give_up=true explaining every deviation."
        if give_up and self.mode != "cad":
            # a scan always has the organic route: giving up early with no closed solid leaves the user nothing.
            # Refused once, and only while over a third of the turns are left, so it cannot loop
            self.give_ups += 1
            left = self.max_turns - self.turn
            if self.give_ups == 1 and left > self.max_turns / 3 and not getattr(self.wb, "whole", None):
                return (f"refused: {left} of {self.max_turns} turns left and no closed solid yet. Next: call organic_route "
                        "(when ScanRuler has nothing to repair it carries on from the scan as it is, and closes what is open), "
                        "then propose_bodies or group_bodies on what it made, then check. If organic_route itself fails, its "
                        "answer says what to do next. Ask for give_up again only when that cannot work.")
        return None

    # -- loop ------------------------------------------------------------
    def compact(self, msgs):
        """Keep the history under BUDGET: images only from the latest turn, then old tool results elided
        oldest first (the newest guide text kept within PIN_BUDGET), the last 3 messages always whole."""
        last_img = max((i for i, m in enumerate(msgs) if isinstance(m.get("content"), list)), default=-1)
        for i, m in enumerate(msgs):
            if isinstance(m.get("content"), list) and i != last_img:
                m["content"] = "[older image elided]"
        total = sum(est(m) for m in msgs)
        keep, pin = set(), 0
        for i in sorted(self.pinned, reverse=True):       # newest guide reads first
            if i < len(msgs) and pin + est(msgs[i]) <= PIN_BUDGET:
                keep.add(i); pin += est(msgs[i])
        for i, m in enumerate(msgs[2:-3], start=2):
            if total <= BUDGET:
                break
            c = m.get("content")
            if i in keep or not isinstance(c, (str, list)):
                continue
            if m["role"] == "tool" and isinstance(c, str) and len(c) > 300:
                total -= est(m); m["content"] = c[:240] + " ...[older result elided]"; total += est(m)
            elif m["role"] == "user" and isinstance(c, list):
                total -= est(m); m["content"] = "[older image elided]"; total += est(m)

    def _stream(self):
        """on_delta for chat(): batches tokens into events of ~60 chars or 0.1 s, so the app sees the
        reasoning live without an event per token."""
        buf = {"reasoning": [], "content": []}
        last = [time.time()]

        def flush(kind):
            if buf[kind]:
                self.emit(kind, delta="".join(buf[kind])); buf[kind].clear()

        def on_delta(kind, text):
            buf[kind].append(text)
            if sum(map(len, buf[kind])) > 60 or time.time() - last[0] > 0.1:
                flush("reasoning"); flush("content"); last[0] = time.time()
        on_delta.flush = lambda: (flush("reasoning"), flush("content"))
        return on_delta

    def run(self, first_message):
        sys_prompt = (PROMPTS / f"{'cad' if self.mode == 'cad' else 'scan'}.md").read_text(encoding="utf-8")   # organic is a scan's route
        msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": first_message}]
        for turn in range(self.max_turns):
            self.turn = turn
            if self.deadline and time.time() > self.deadline:
                self.log(f"[swift] time budget spent after {turn} turns")
                return "time budget spent", True
            self.compact(msgs)
            self.emit("turn", n=turn, max=self.max_turns, context=int(sum(est(m) for m in msgs)), limit=CTX_TOKENS)
            stream = self._stream()
            msg, usage = chat(msgs, self.tools, on_delta=stream)
            stream.flush()
            self.emit("turn_end", n=turn, prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"))
            calls = msg.get("tool_calls") or []
            msgs.append({"role": "assistant", "content": msg.get("content") or "", **({"tool_calls": calls} if calls else {})})
            self.log(f"[swift {turn}] prompt {usage.get('prompt_tokens')} tok; " + (msg.get("content") or "")[:400].replace("\n", " ")
                     + "".join(f"\n   -> {c['function']['name']} {c['function'].get('arguments', '')[:300]}" for c in calls))
            if not calls:
                msgs.append({"role": "user", "content": "Use the tools. When the validator passes, call finish with the report."})
                continue
            images = []
            for c in calls:
                name = c["function"]["name"]
                self.emit("tool", name=name, args=c["function"].get("arguments", "")[:2000])
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError as e:
                    out = f"arguments are not valid JSON: {e}"
                    args = None
                if args is not None:
                    fn = getattr(self, f"t_{name}", None)
                    if fn is None or not any(t["function"]["name"] == name for t in self.tools):
                        out = f"no tool {name}"
                    else:
                        try:
                            out = fn(**args)
                        except TypeError as e:
                            out = f"bad arguments for {name}: {e}"
                        except Exception as e:
                            out = f"{name} failed: {type(e).__name__}: {e}"
                    if name == "finish" and out is None:
                        return args.get("report", ""), args.get("give_up", False) or not self.passed
                    if isinstance(out, tuple):          # a picture with it (view_image, pieces, parts, bodies)
                        out, img, *where = out
                        if img:
                            images.append(img)
                            self.emit("image", path=where[0] if where else str((self.wb.work / args.get("path", "")).resolve()))
                if name == "guide":
                    self.pinned.add(len(msgs))
                self.log(f"   <- {name}: {str(out)[:600]}")
                self.emit("tool_result", name=name, text=str(out)[:1500])
                msgs.append({"role": "tool", "tool_call_id": c.get("id", name), "content": str(out)})
            for img in images:      # vision goes in a user turn: tool messages carry text only
                msgs.append({"role": "user", "content": [{"type": "text", "text": "The picture with that answer - look at it before deciding:"},
                                                         {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}]})
        return f"turn budget ({self.max_turns}) spent", True
