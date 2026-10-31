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
BUDGET = 34000          # history kept under this (est. tokens), leaving room for the reply and thinking
PIN_BUDGET = 12000      # guide text kept verbatim
READ_ONLY = {"guide_read", "scanruler_tools", "reverse_feature_list", "reverse_feature_schema", "reverse_body_describe",
             "reverse_body_check", "reverse_parameter_list", "reverse_note_list", "session_state", "scan_query", "scan_nearest",
             "scan_qa", "scan_outline", "scan_draft", "scan_heightmap", "scan_pattern", "view_render", "section_render",
             "reverse_body_review", "reverse_body_compare", "reverse_body_volume_compare", "reverse_section_fit",
             "reverse_section_outline", "reverse_section_stack", "reverse_sketch_candidates", "reverse_body_fillet_read",
             "reverse_plane_describe", "reverse_thread_read", "report_get", "mesh_inspect", "mesh_parts", "deviation_hotspots"}


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
CAD_TOOLS = [
    tool("run_script", "Run a reverse-engineering script (analyze, frame, slabs, revolve, holes, blends, compare, helix, profile, rz, centers, two_way, slice, vslice) with CLI args.",
         {"name": {"type": "string"}, "args": {"type": "array", "items": {"type": "string"}}}, ["name"]),
    tool("rebuild", "Clear the part and run these step files in order (main body, fills, trims, holes, blends).",
         {"files": {"type": "array", "items": {"type": "string"}}}, ["files"]),
]


def chat(messages, tools, thinking=True, max_tokens=6000, retries=3):
    body = {"model": MODEL, "messages": messages, "tools": tools, "tool_choice": "auto", "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    # the model card's sampling, per mode
    body.update(dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0, presence_penalty=0, reasoning_effort="medium") if thinking
                else dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0, presence_penalty=1.5))
    req = urllib.request.Request(f"{GATEWAY}/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    for k in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                res = json.loads(r.read())
            return res["choices"][0]["message"], res.get("usage", {})
        except Exception as e:  # the Gateway queues behind Sena's live turns; a dropped call is retried
            if k == retries - 1:
                raise
            time.sleep(5 * (k + 1))


def est(m):
    c = m.get("content")
    n = len(c) if isinstance(c, str) else sum(len(p.get("text", "")) + (1200 * 4 if p.get("type") == "image_url" else 0) for p in (c or []))
    return (n + len(json.dumps(m.get("tool_calls", "")))) // 3.5


class Agent:
    def __init__(self, wb: Workbench, mode, log=print, max_turns=60):
        self.wb, self.mode, self.log, self.max_turns = wb, mode, log, max_turns
        self.tools = COMMON + (CAD_TOOLS if mode == "cad" else [])
        self.changes = 0            # bumped by anything that may change the model
        self.checked_at = -1        # self.changes when the last check ran
        self.passed = False
        self.best = None            # (score, validation) of the best model seen
        self.pinned = set()         # message indices kept verbatim (guide text)

    # -- tools -----------------------------------------------------------
    def t_check(self):
        out = self.wb.check(self.mode)
        r = self.wb.last_validation
        self.checked_at = self.changes
        self.passed = bool(r and r["verdict"] == "PASS")
        if r:
            score = (sum(not v for v in r["checks"].values()), max(r["input_to_model"]["p95"], r["model_to_input"]["p95"]))
            if self.best is None or score < self.best[0]:
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
        return f"{path} attached", base64.b64encode(p.read_bytes()).decode()

    def t_read_file(self, path, max_chars=6000):
        p = (self.wb.work / path).resolve()
        if self.wb.work not in p.parents or not p.is_file():
            return f"no file {path} in the work folder"
        return clip(p.read_text(encoding="utf-8", errors="replace"), max_chars)

    def t_scanruler(self, calls, stopOnError=True):
        if any(c.get("tool") not in READ_ONLY for c in calls):
            self.changes += 1
        for c in calls:
            c.setdefault("arguments", {})
        txt, err = self.wb.batch(calls, stopOnError=stopOnError)
        return clip(txt, 8000)

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

    def t_rebuild(self, files):
        self.changes += 1
        return self.wb.rebuild(files)

    def t_finish(self, report, give_up=False):
        if not give_up and not (self.passed and self.checked_at == self.changes):
            why = "the model changed since the last check" if self.checked_at != self.changes else "the last check did not PASS"
            return f"refused: {why}. Run check and fix what fails, or finish with give_up=true explaining every deviation."
        return None

    # -- loop ------------------------------------------------------------
    def compact(self, msgs):
        """Elide old tool results (keep the newest, the pinned guide text within its budget, and the task)."""
        total = sum(est(m) for m in msgs)
        pinned_tok = 0
        for i, m in enumerate(msgs[2:-6], start=2):
            if total <= BUDGET:
                break
            if i in self.pinned:
                pinned_tok += est(m)
                if pinned_tok <= PIN_BUDGET:
                    continue
            c = m.get("content")
            if m["role"] == "tool" and isinstance(c, str) and len(c) > 300:
                total -= est(m); m["content"] = c[:240] + " ...[older result elided]"; total += est(m)
            elif m["role"] == "user" and isinstance(c, list):
                total -= est(m); m["content"] = "[older image elided]"; total += est(m)

    def run(self, first_message):
        sys_prompt = (PROMPTS / f"{self.mode}.md").read_text(encoding="utf-8")
        msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": first_message}]
        for turn in range(self.max_turns):
            self.compact(msgs)
            msg, usage = chat(msgs, self.tools)
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
                    if name == "view_image":
                        out, img = out
                        if img:
                            images.append(img)
                if name == "guide":
                    self.pinned.add(len(msgs))
                self.log(f"   <- {name}: {str(out)[:600]}")
                msgs.append({"role": "tool", "tool_call_id": c.get("id", name), "content": str(out)})
            for img in images:      # vision goes in a user turn: tool messages carry text only
                msgs.append({"role": "user", "content": [{"type": "text", "text": "The image you asked for:"},
                                                         {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}]})
        return f"turn budget ({self.max_turns}) spent", True
