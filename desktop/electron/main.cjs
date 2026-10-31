// SPDX-License-Identifier: AGPL-3.0-only
// ScanRuler Swift Studio — the desktop shell. Runs swift/run.py (--events --live) as a child process,
// relays its JSON event lines to the window, and owns the few things a page may not do: pick a file,
// open a folder, read a run's report and images, and stop a run.
const { app, BrowserWindow, dialog, ipcMain, session, shell } = require('electron')
const { spawn, execFile } = require('node:child_process')
const fs = require('node:fs')
const path = require('node:path')
const readline = require('node:readline')

const DEFAULTS = (() => {
  try { return JSON.parse(fs.readFileSync(path.join(__dirname, 'defaults.json'), 'utf8')) } catch { return {} }
})()
const SETTINGS_FILE = () => path.join(app.getPath('userData'), 'settings.json')

function loadSettings() {
  const repoDir = DEFAULTS.repoDir || path.resolve(__dirname, '..', '..')
  const venv = path.resolve(repoDir, '..', '.venv', 'Scripts', 'python.exe')
  const base = {
    repoDir,
    python: fs.existsSync(venv) ? venv : 'python',
    gatewayUrl: 'http://127.0.0.1:11434/v1',
    model: 'swift-1.5-qwen3.8-27b',
    maxTurns: 70,
  }
  try { return { ...base, ...JSON.parse(fs.readFileSync(SETTINGS_FILE(), 'utf8')) } } catch { return base }
}

function saveSettings(s) {
  fs.mkdirSync(path.dirname(SETTINGS_FILE()), { recursive: true })
  fs.writeFileSync(SETTINGS_FILE(), JSON.stringify(s, null, 1))
}

let win = null
let run = null          // { child, work }

function send(channel, payload) {
  if (win && !win.isDestroyed()) win.webContents.send(channel, payload)
}

function createWindow() {
  win = new BrowserWindow({
    width: 1600, height: 980, minWidth: 1180, minHeight: 720,
    // STUDIO_HIDDEN=1: never shown or focused (checks with STUDIO_SHOTS); it still paints for capturePage
    show: !process.env.STUDIO_HIDDEN, paintWhenInitiallyHidden: true,
    backgroundColor: '#0d0d0f',
    title: 'ScanRuler Swift Studio',
    icon: path.join(__dirname, '..', 'build', 'icon.png'),
    titleBarStyle: 'hidden',
    titleBarOverlay: { color: '#0d0d0f', symbolColor: '#f2ede4', height: 38 },
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webviewTag: true,
    },
  })
  win.loadFile(path.join(__dirname, '..', 'dist', 'index.html'))
  debugHooks()
  // links never open inside the app
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//.test(url)) shell.openExternal(url)
    return { action: 'deny' }
  })
}

// opt-in, for checking the app without hands: STUDIO_AUTORUN=<stl> [STUDIO_AUTORUN_SWIFT=0] [STUDIO_AUTORUN_MODE=organic] starts a
// run once the window is up; STUDIO_SHOTS=<dir> saves a screenshot of the window every 15 s
function debugHooks() {
  const { STUDIO_AUTORUN: stl, STUDIO_SHOTS: dir } = process.env
  if (stl) win.webContents.once('did-finish-load', () => setTimeout(() => startRun(null, { stl, mode: process.env.STUDIO_AUTORUN_MODE || 'auto', swift: process.env.STUDIO_AUTORUN_SWIFT !== '0' }), 2500))
  if (dir) {
    fs.mkdirSync(dir, { recursive: true })
    const logFile = path.join(dir, 'console.log')
    win.webContents.on('console-message', (e) => fs.appendFileSync(logFile, `[${e.level}] ${e.message}\n`))
    let n = 0
    setInterval(async () => {
      if (!win || win.isDestroyed()) return
      const k = String(n++).padStart(3, '0')
      fs.writeFileSync(path.join(dir, `shot-${k}.png`), (await win.webContents.capturePage()).toPNG())
      // the live view is a guest page of its own: a window capture leaves it out
      const { webContents } = require('electron')
      for (const wc of webContents.getAllWebContents().filter((w) => w.getType() === 'webview')) {
        fs.writeFileSync(path.join(dir, `live-${k}.png`), (await wc.capturePage()).toPNG())
      }
    }, 15000)
  }
}

// the live view is a ScanRuler page on 127.0.0.1 only; its preload answers the first-run questions
app.on('web-contents-created', (_e, contents) => {
  contents.on('will-attach-webview', (_ev, prefs, params) => {
    if (!/^http:\/\/127\.0\.0\.1:\d+\//.test(params.src || '')) { _ev.preventDefault(); return }
    prefs.nodeIntegration = false
    prefs.contextIsolation = true
  })
})

function startRun(_e, opts) {
  if (run) return { error: 'A run is already going.' }
  const s = loadSettings()
  const script = path.join(s.repoDir, 'swift', 'run.py')
  if (!fs.existsSync(script)) return { error: `swift/run.py not found under ${s.repoDir} — set the ScanRuler folder in Settings.` }
  if (!fs.existsSync(opts.stl)) return { error: `No file at ${opts.stl}` }
  const args = ['-u', script, opts.stl, '--events', '--live', '--max-turns', String(opts.maxTurns || s.maxTurns)]
  if (opts.mode && opts.mode !== 'auto') args.push('--mode', opts.mode)
  if (!opts.swift) args.push('--no-swift')
  const child = spawn(s.python, args, {
    cwd: s.repoDir,
    env: { ...process.env, SWIFT_URL: s.gatewayUrl, SWIFT_MODEL: s.model, PYTHONIOENCODING: 'utf-8' },
    windowsHide: true,
  })
  run = { child }
  readline.createInterface({ input: child.stdout }).on('line', (line) => {
    try { send('run:event', JSON.parse(line)) } catch { send('run:event', { type: 'log', text: line }) }
  })
  readline.createInterface({ input: child.stderr }).on('line', (line) => send('run:event', { type: 'stderr', text: line }))
  child.on('error', (err) => send('run:event', { type: 'error', text: `Could not start Python: ${err.message}` }))
  child.on('exit', (code) => { run = null; send('run:exit', { code }) })
  return { ok: true }
}

function stopRun() {
  if (!run) return { ok: true }
  // the run has children of its own (scanruler-mcp, node): end the whole tree
  if (process.platform === 'win32') execFile('taskkill', ['/pid', String(run.child.pid), '/T', '/F'], () => {})
  else run.child.kill('SIGTERM')
  return { ok: true }
}

// files a run wrote: only inside its work folder (the event stream names it)
function insideWork(file, work) {
  const rel = path.relative(path.resolve(work), path.resolve(file))
  return rel && !rel.startsWith('..') && !path.isAbsolute(rel)
}

ipcMain.handle('run:start', startRun)
ipcMain.handle('run:stop', stopRun)
ipcMain.handle('settings:get', () => loadSettings())
ipcMain.handle('settings:set', (_e, s) => { saveSettings({ ...loadSettings(), ...s }); return loadSettings() })
ipcMain.handle('dialog:open-stl', async () => {
  const r = await dialog.showOpenDialog(win, { title: 'Open a scan or mesh', properties: ['openFile'], filters: [{ name: 'Meshes', extensions: ['stl'] }] })
  return r.canceled ? null : r.filePaths[0]
})
ipcMain.handle('dialog:pick', async (_e, kind) => {
  const r = await dialog.showOpenDialog(win, kind === 'folder' ? { properties: ['openDirectory'] } : { properties: ['openFile'] })
  return r.canceled ? null : r.filePaths[0]
})
ipcMain.handle('shell:show', (_e, file) => { if (file && fs.existsSync(file)) shell.showItemInFolder(file) })
ipcMain.handle('shell:open', (_e, file) => (file && fs.existsSync(file) ? shell.openPath(file) : 'missing'))
ipcMain.handle('fs:text', (_e, file, work) => (insideWork(file, work) && fs.existsSync(file) ? fs.readFileSync(file, 'utf8') : null))
ipcMain.handle('fs:image', (_e, file, work) => {
  if (!insideWork(file, work) || !/\.png$/i.test(file) || !fs.existsSync(file)) return null
  return `data:image/png;base64,${fs.readFileSync(file).toString('base64')}`
})
ipcMain.handle('health', async () => {
  const s = loadSettings()
  try {
    const r = await fetch(`${s.gatewayUrl}/models`, { signal: AbortSignal.timeout(4000) })
    const ids = (await r.json()).data.map((m) => m.id)
    return { gateway: true, model: ids.includes(s.model), models: ids, repo: fs.existsSync(path.join(s.repoDir, 'swift', 'run.py')), python: s.python }
  } catch (err) {
    return { gateway: false, model: false, error: err.message, repo: fs.existsSync(path.join(s.repoDir, 'swift', 'run.py')), python: s.python }
  }
})

app.whenReady().then(() => {
  // the live view's session runs the first-run answers in every frame it loads
  const ses = session.fromPartition('persist:scanruler')
  const filePath = path.join(__dirname, 'webview-preload.cjs')
  if (ses.registerPreloadScript) ses.registerPreloadScript({ type: 'frame', filePath })
  else ses.setPreloads([filePath])
  createWindow()
})
app.on('window-all-closed', () => { stopRun(); app.quit() })
