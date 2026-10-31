# ScanRuler Swift Studio

Desktop app for this fork: drop in a scan or CAD-exported STL. You get a STEP file that an independent validator has checked against the mesh.

- **Live view:** the real ScanRuler page runs inside the app, so you watch the part being measured and built.
- **Swift 1.5 panel:** streams the model's reasoning, each tool call, and the current phase.
- **Validator panel:** shows each check. A part is delivered only on PASS.
- **Autopilot:** the exact-reading pipeline with no AI, for CAD-exported meshes.

## Requirements
- This checkout, built: `npm ci && npm --prefix mcp ci && npm run build && npm run build:engine` in the repo root.
- Python 3 with numpy, scipy and matplotlib (the app looks for `../.venv` next to the checkout; you can set the path in Settings).
- An OpenAI-compatible endpoint serving Swift 1.5. The default is the AI Gateway at `http://127.0.0.1:11434/v1`, model `swift-1.5-qwen3.8-27b`.

## Run
```bash
cd desktop
npm ci
npm run dev        # build the UI and open the app
npm run dist       # Windows installer + portable exe in desktop/release/
```

The CLI behind the app is `python swift/run.py part.stl` (see its docstring for the flags).
