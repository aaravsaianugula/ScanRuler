# SPDX-License-Identifier: AGPL-3.0-only
"""Minimal MCP stdio client for scanruler-mcp (newline-delimited JSON-RPC)."""
import json
import os
import subprocess
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


class ScanRulerMCP:
    def __init__(self, out_dir, headless=True, tools_mode="compact", extra=(), env=None):
        args = ["node", str(REPO / "mcp" / "bin" / "scanruler-mcp.js"), "--no-update",
                "--tools", tools_mode, "--out", str(out_dir), "--answer-within", "0", *extra]
        if headless:
            args.append("--headless")
        env = {**os.environ, "SCANRULER_MCP_NO_UPDATE": "1", **(env or {})}
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, cwd=str(out_dir), env=env,
                                     text=True, encoding="utf-8", bufsize=1)
        self.stderr = []
        threading.Thread(target=lambda: [self.stderr.append(l) for l in self.proc.stderr], daemon=True).start()
        self._id = 0
        self.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                "clientInfo": {"name": "scanruler-swift", "version": "1.0"}})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _send(self, msg):
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def rpc(self, method, params=None):
        self._id += 1
        self._send({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}})
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("scanruler-mcp exited: " + "".join(self.stderr[-20:]))
            msg = json.loads(line)
            if msg.get("id") == self._id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg["result"]

    def tools(self):
        return self.rpc("tools/list")["tools"]

    def call(self, name, arguments=None):
        """Returns (text, images, is_error). Images are (mime, base64) pairs."""
        res = self.rpc("tools/call", {"name": name, "arguments": arguments or {}})
        text, images = [], []
        for c in res.get("content", []):
            if c.get("type") == "text":
                text.append(c["text"])
            elif c.get("type") == "image":
                images.append((c.get("mimeType", "image/png"), c["data"]))
        return "\n".join(text), images, bool(res.get("isError"))

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
