import { useCallback, useEffect, useReducer, useRef, useState } from 'react'
import { ThinkingOrb } from 'thinking-orbs'
import AsciiHero from './components/AsciiHero.jsx'
import LiveView from './components/LiveView.jsx'
import { Bodies, OrbCard, Reasoning, Timeline, Validation } from './components/SwiftPanels.jsx'
import { estimate, record } from './eta.js'
import { initial, PHASES, reduce } from './run.js'

// "ANSI Shadow" block letters, joined at run time so every row lines up
const GLYPHS = {
  S: ['███████╗', '██╔════╝', '███████╗', '╚════██║', '███████║', '╚══════╝'],
  C: [' ██████╗', '██╔════╝', '██║     ', '██║     ', '╚██████╗', ' ╚═════╝'],
  A: [' █████╗ ', '██╔══██╗', '███████║', '██╔══██║', '██║  ██║', '╚═╝  ╚═╝'],
  N: ['███╗   ██╗', '████╗  ██║', '██╔██╗ ██║', '██║╚██╗██║', '██║ ╚████║', '╚═╝  ╚═══╝'],
  R: ['██████╗ ', '██╔══██╗', '██████╔╝', '██╔══██╗', '██║  ██║', '╚═╝  ╚═╝'],
  U: ['██╗   ██╗', '██║   ██║', '██║   ██║', '██║   ██║', '╚██████╔╝', ' ╚═════╝ '],
  L: ['██╗     ', '██║     ', '██║     ', '██║     ', '███████╗', '╚══════╝'],
  E: ['███████╗', '██╔════╝', '█████╗  ', '██╔══╝  ', '███████╗', '╚══════╝'],
}
const BANNER = [0, 1, 2, 3, 4, 5].map((r) => [...'SCANRULER'].map((ch) => GLYPHS[ch][r]).join('')).join('\n')

const RECENT_KEY = 'studio.recent'
const readRecent = () => { try { return JSON.parse(localStorage.getItem(RECENT_KEY)) || [] } catch { return [] } }

function useElapsed(run) {
  const [, tick] = useState(0)
  useEffect(() => {
    if (run.status !== 'running') return
    const id = setInterval(() => tick((n) => n + 1), 1000)
    return () => clearInterval(id)
  }, [run.status])
  if (!run.startedAt) return '00:00'
  const s = Math.floor(((run.endedAt || Date.now()) - run.startedAt) / 1000)
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`
}

export default function App() {
  const [run, dispatch] = useReducer(reduce, initial)
  const [screen, setScreen] = useState('home')
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [report, setReport] = useState(null)
  const [card, setCard] = useState(null)
  const elapsed = useElapsed(run)
  const eta = estimate(run, Date.now() / 1000)

  useEffect(() => {
    const offE = window.studio.onEvent((e) => {
      if (e.type === 'start') setScreen('run')     // a run the app did not start from Home (autorun)
      dispatch(e)
    })
    const offX = window.studio.onExit((e) => dispatch({ type: 'exit', code: e.code }))
    return () => { offE(); offX() }
  }, [])

  // the anime title card on each new phase
  const lastPhase = useRef(null)
  useEffect(() => {
    if (!run.phase || run.phase === lastPhase.current) return
    lastPhase.current = run.phase
    const p = PHASES.find((x) => x.id === run.phase)
    if (!p || matchMedia('(prefers-reduced-motion: reduce)').matches) return
    setCard({ ...p, label: run.phaseLabel })
    const id = setTimeout(() => setCard(null), 1500)
    return () => clearTimeout(id)
  }, [run.phase, run.phaseLabel])

  // remember finished runs, and how long their phases took (the ETA learns from them)
  useEffect(() => {
    if (!run.done || run.replay) return
    record(run, run.done.t || Date.now() / 1000)
    const entry = { stl: run.stl, work: run.done.work, status: run.done.status, step: run.done.step, report: run.done.report, t: Date.now() }
    const list = [entry, ...readRecent().filter((r) => r.work !== entry.work)].slice(0, 8)
    try { localStorage.setItem(RECENT_KEY, JSON.stringify(list)) } catch { /* storage blocked: no history */ }
  }, [run.done])

  const start = useCallback(async (opts) => {
    lastPhase.current = null
    dispatch({ type: 'reset', stl: opts.stl, swift: opts.swift })
    setScreen('run')
    let r
    try { r = await window.studio.start(opts) } catch (err) { r = { error: `The run could not start: ${err.message}` } }
    if (r?.error) dispatch({ type: 'error', text: r.error })
  }, [])

  // a finished run, read back from the events it recorded: its reasoning, actions and checks as they were
  const replay = async (work) => {
    const text = await window.studio.readText(`${work}\\events.jsonl`, work)
    if (!text) { setReport('This run has no recorded events: runs are recorded from this version of the app on.'); return }
    lastPhase.current = 'replay'
    dispatch({ type: 'reset', replay: true })
    for (const line of text.split('\n')) { try { if (line) dispatch(JSON.parse(line)) } catch { /* a torn last line */ } }
    dispatch({ type: 'exit', code: 0 })
    setScreen('run')
  }

  const openReport = async (file, work) => setReport(await window.studio.readText(file, work) || 'The report could not be read.')

  return (
    <div className="app">
      <div className="crt" aria-hidden />
      <TopBar run={run} screen={screen} onHome={() => setScreen('home')} onSettings={() => setSettingsOpen(true)} />
      {screen === 'home'
        ? <Home onStart={start} onOpenReport={openReport} onReplay={replay} running={run.status === 'running'} onResume={() => setScreen('run')} />
        : <RunView run={run} elapsed={elapsed} eta={eta} onStop={() => window.studio.stop()} onNew={() => setScreen('home')} onReport={() => openReport(run.done.report, run.done.work)} />}
      {card && <TitleCard card={card} />}
      {settingsOpen && <Settings onClose={() => setSettingsOpen(false)} />}
      {report && <ReportModal text={report} onClose={() => setReport(null)} />}
    </div>
  )
}

function TopBar({ run, screen, onHome, onSettings }) {
  return (
    <header className="topbar">
      <button className="brand" onClick={onHome} aria-label="Home">
        <span className="mark"><i className="c" /><i className="t" /><i className="s" /></span>
        <span className="brand-name">SCANRULER<em>×</em>SWIFT</span>
        <span className="brand-sub">STUDIO</span>
      </button>
      <nav className="top-nav">
        {screen === 'run' && run.stl && <span className="top-file">{run.stl.split(/[\\/]/).pop()}</span>}
        <button className="ghost" onClick={onSettings}>SETTINGS</button>
      </nav>
    </header>
  )
}

function Health() {
  const [h, setH] = useState(null)
  useEffect(() => {
    let live = true
    const poll = () => window.studio.health().then((x) => live && setH(x))
    poll()
    const id = setInterval(poll, 15000)
    return () => { live = false; clearInterval(id) }
  }, [])
  const row = (ok, label, detail) => (
    <li className={ok ? 'ok' : ok === false ? 'no' : ''}><i />{label}<small>{detail}</small></li>
  )
  return (
    <ul className="health" aria-label="System status">
      {row(h?.gateway, 'AI GATEWAY', h ? (h.gateway ? 'online' : 'offline') : '…')}
      {row(h?.model, 'SWIFT 1.5', h ? (h.model ? 'ready on the edge PC' : 'not listed') : '…')}
      {row(h?.repo, 'SCANRULER', h ? (h.repo ? 'checkout found' : 'set the folder in Settings') : '…')}
    </ul>
  )
}

function Home({ onStart, onOpenReport, onReplay, running, onResume }) {
  const [stl, setStl] = useState(null)
  const [mode, setMode] = useState('auto')
  const [swift, setSwift] = useState(true)
  const [turns, setTurns] = useState(70)
  const [drag, setDrag] = useState(false)
  const recent = readRecent()

  const onDrop = (e) => {
    e.preventDefault(); setDrag(false)
    const f = e.dataTransfer.files[0]
    if (f && /\.stl$/i.test(f.name)) setStl(window.studio.pathForFile(f))
  }

  return (
    <main className="home">
      <section className="hero">
        <div className="bauhaus" aria-hidden><i className="b-circle" /><i className="b-tri" /><i className="b-square" /><i className="b-bar" /></div>
        <pre className="banner" aria-label="ScanRuler">{BANNER}</pre>
        <AsciiHero />
        <div className="hero-copy">
          <h1>Scan in.<br /><span>Exact CAD</span> out.</h1>
          <p>Swift 1.5 measures your scan in ScanRuler, builds the part feature by feature, and an independent validator signs it off — or sends it back.</p>
        </div>
        <div className="speedlines" aria-hidden />
      </section>

      <section className="launch">
        <div className="panel launch-card">
          <header className="panel-head red"><span className="glyph">●</span> NEW RUN</header>
          <div
            className={`drop ${drag ? 'over' : ''} ${stl ? 'has' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDrag(true) }} onDragLeave={() => setDrag(false)} onDrop={onDrop}
            onClick={async () => { const p = await window.studio.openStl(); if (p) setStl(p) }}
            role="button" tabIndex={0} onKeyDown={async (e) => { if (e.key === 'Enter') { const p = await window.studio.openStl(); if (p) setStl(p) } }}
          >
            {stl ? (<><b>{stl.split(/[\\/]/).pop()}</b><small>{stl}</small></>) : (<><b>DROP AN .STL</b><small>scan or CAD export — or click to browse</small></>)}
          </div>

          <fieldset className="seg">
            <legend>INPUT</legend>
            {[['auto', 'AUTO', 'detects the kind'], ['cad', 'CAD EXACT', 'clean export'], ['scan', 'SCAN', '3D scan'], ['organic', 'ORGANIC', 'figurine → solid']].map(([v, l, s]) => (
              <label key={v} className={mode === v ? 'on' : ''}><input type="radio" name="mode" value={v} checked={mode === v} onChange={() => setMode(v)} /><b>{l}</b><small>{s}</small></label>
            ))}
          </fieldset>

          <fieldset className="seg">
            <legend>ENGINE</legend>
            <label className={swift ? 'on' : ''}>
              <input type="radio" name="engine" checked={swift} onChange={() => setSwift(true)} />
              <span className="seg-orb"><ThinkingOrb state="weaving" size={20} theme="dark" color="#f5c518" /></span><b>SWIFT 1.5</b><small>agent + autopilot</small>
            </label>
            <label className={!swift ? 'on' : ''}>
              <input type="radio" name="engine" checked={!swift} onChange={() => setSwift(false)} />
              <span className="seg-orb"><ThinkingOrb state="working" size={20} theme="dark" color="#3a6fd8" /></span><b>AUTOPILOT</b><small>exact reading, no AI</small>
            </label>
          </fieldset>

          {swift && (
            <label className="slider">
              <span>TURN BUDGET <b>{turns}</b></span>
              <input type="range" min="20" max="120" step="5" value={turns} onChange={(e) => setTurns(+e.target.value)} />
            </label>
          )}

          <div className="launch-actions">
            {running && <button className="ghost" onClick={onResume}>VIEW CURRENT RUN</button>}
            <button className="go" disabled={!stl || running} onClick={() => onStart({ stl, mode, swift, maxTurns: turns })}>
              <span>LAUNCH</span><small>発進</small>
            </button>
          </div>
        </div>

        <div className="panel status-card">
          <header className="panel-head blue"><span className="glyph">■</span> SYSTEM</header>
          <Health />
        </div>

        {recent.length > 0 && (
          <div className="panel recent">
            <header className="panel-head yellow"><span className="glyph">▲</span> RECENT</header>
            <ul>
              {recent.map((r) => (
                <li key={r.work}>
                  <span className={`dot ${r.status === 'PASS' ? 'ok' : 'no'}`} />
                  <span className="recent-name">{(r.stl || '').split(/[\\/]/).pop()}</span>
                  <button className="link" onClick={() => onReplay(r.work)}>replay</button>
                  <button className="link" onClick={() => onOpenReport(r.report, r.work)}>report</button>
                  <button className="link" onClick={() => window.studio.show(r.step || r.report)}>folder</button>
                </li>
              ))}
            </ul>
          </div>
        )}
      </section>
    </main>
  )
}

function RunView({ run, elapsed, eta, onStop, onNew, onReport }) {
  const active = run.status === 'running'
  return (
    <main className="runview">
      <div className="run-left">
        <LiveView run={run} elapsed={elapsed} eta={eta} />
        <Timeline items={run.timeline} />
      </div>
      <div className="run-right">
        <OrbCard run={run} elapsed={elapsed} eta={eta} />
        <Reasoning turns={run.turns} logs={run.logs} running={active && (run.activity === 'thinking' || run.activity === 'writing')} />
        <Validation validations={run.validations} />
        <Bodies bodies={run.bodies} />
      </div>
      <footer className="runbar">
        {active && <button className="stop" onClick={onStop}>■ STOP</button>}
        {run.status === 'done' && run.done && (
          <>
            <span className={`verdict ${run.done.status === 'PASS' ? 'ok' : 'no'}`}>{run.done.status === 'PASS' ? '✓ VALIDATED' : '✗ NOT VALIDATED'}</span>
            {/* the slicer's copy: the same triangles and bodies in a few MB, where a faceted STEP takes a slicer minutes */}
            {run.done.print && <button title="Opens the 3MF in your slicer (Bambu Studio, PrusaSlicer...)" onClick={() => window.studio.open(run.done.print)}>OPEN IN SLICER</button>}
            {run.done.step && <button onClick={() => window.studio.open(run.done.step)}>OPEN STEP</button>}
            <button onClick={() => window.studio.show(run.done.step || run.done.report)}>SHOW FOLDER</button>
            <button onClick={onReport}>REPORT</button>
          </>
        )}
        {run.status === 'error' && <span className="verdict no">✗ {run.error}</span>}
        <span className="runbar-log">{run.logs.findLast((l) => /^\[run\]/.test(l))?.slice(0, 160) || run.phaseLabel}</span>
        {!active && <button className="go small" onClick={onNew}>NEW RUN</button>}
      </footer>
    </main>
  )
}

function TitleCard({ card }) {
  return (
    <div className="titlecard" role="status">
      <div className="tc-lines" />
      <div className="tc-text">
        <small>PHASE {card.n}</small>
        <b>{card.label}</b>
        <span>{card.kana}</span>
      </div>
    </div>
  )
}

function Settings({ onClose }) {
  const [s, setS] = useState(null)
  useEffect(() => { window.studio.settings().then(setS) }, [])
  if (!s) return null
  const field = (k, label, pick) => (
    <label className="field">
      <span>{label}</span>
      <div><input value={s[k]} onChange={(e) => setS({ ...s, [k]: e.target.value })} />{pick && <button onClick={async () => { const p = await window.studio.pick(pick); if (p) setS({ ...s, [k]: p }) }}>…</button>}</div>
    </label>
  )
  return (
    <div className="modal" role="dialog" aria-modal="true" aria-label="Settings" onClick={onClose}>
      <div className="panel modal-box" onClick={(e) => e.stopPropagation()}>
        <header className="panel-head yellow"><span className="glyph">▲</span> SETTINGS</header>
        <div className="pad">
          {field('repoDir', 'ScanRuler checkout (has swift/run.py)', 'folder')}
          {field('python', 'Python (with numpy, scipy, matplotlib)', 'file')}
          {field('gatewayUrl', 'AI Gateway URL')}
          {field('model', 'Model id')}
          <div className="modal-actions">
            <button className="ghost" onClick={onClose}>CANCEL</button>
            <button className="go small" onClick={async () => { await window.studio.saveSettings(s); onClose() }}>SAVE</button>
          </div>
        </div>
      </div>
    </div>
  )
}

function ReportModal({ text, onClose }) {
  useEffect(() => {
    const k = (e) => e.key === 'Escape' && onClose()
    addEventListener('keydown', k)
    return () => removeEventListener('keydown', k)
  }, [onClose])
  return (
    <div className="modal" role="dialog" aria-modal="true" aria-label="Report" onClick={onClose}>
      <div className="panel modal-box wide" onClick={(e) => e.stopPropagation()}>
        <header className="panel-head blue"><span className="glyph">■</span> REPORT<button className="x" onClick={onClose} aria-label="Close">✕</button></header>
        <pre className="report">{text}</pre>
      </div>
    </div>
  )
}
