import { useEffect, useRef, useState } from 'react'
import { ThinkingOrb } from 'thinking-orbs'
import { clock } from '../eta.js'
import { outcomeOf, PHASES } from '../run.js'

// orb state, label and Bauhaus colour per activity
export const ACTIVITY = {
  idle: ['breathing', 'Standing by', '#f2ede4'],
  linking: ['connecting', 'Linking', '#f2ede4'],
  prefill: ['breathing', 'Reading context', '#f5c518'],
  thinking: ['weaving', 'Thinking', '#f5c518'],
  writing: ['composing', 'Writing', '#f5c518'],
  measuring: ['searching', 'Measuring', '#3a6fd8'],
  looking: ['searching', 'Looking', '#3a6fd8'],
  reading: ['listening', 'Reading', '#f2ede4'],
  building: ['shaping', 'Building', '#e63b2e'],
  validating: ['solving', 'Validating', '#3a6fd8'],
  working: ['working', 'Working', '#f2ede4'],
}

export function OrbCard({ run, elapsed, eta }) {
  const [state, label, color] = ACTIVITY[run.activity] || ACTIVITY.idle
  const ctx = run.contextLimit ? Math.min(1, run.context / run.contextLimit) : 0
  const end = outcomeOf(run)
  return (
    <section className="panel orb-card" aria-live="polite">
      <div className="orb-row">
        <div className="orb-disc" style={{ '--tint': color }}>
          <ThinkingOrb state={state} size={64} theme="dark" color={color} paused={run.status !== 'running'} />
        </div>
        <div className="orb-text">
          <div className="orb-label">{run.status === 'running' ? `${label}…` : end ? (end.ok ? 'Run complete' : 'Run ended') : 'Standing by'}</div>
          <div className={`orb-sub ${end && !end.ok ? 'bad' : ''}`}>{end ? end.title.toLowerCase().replace(/^./, (c) => c.toUpperCase()) : run.phaseLabel || '—'}</div>
          <div className="chips">
            <span className="chip">{run.swift ? 'SWIFT 1.5' : 'AUTOPILOT'}</span>
            {run.mode && <span className="chip yellow">{{ cad: 'CAD EXACT', organic: 'ORGANIC' }[run.mode] || 'SCAN'}</span>}
            {run.turn != null && <span className="chip">TURN {run.turn + 1}/{run.maxTurns}</span>}
            <span className="chip">{elapsed}</span>
            {eta && <span className="chip yellow" title={eta.learning ? 'No finished runs yet to learn from: a first guess' : 'From the runs before this one and this run’s pace'}>ETA {clock(eta.left)}</span>}
          </div>
        </div>
      </div>
      <PhaseRail phase={run.phase} done={run.status === 'done'} />
      <div className="sync" title="Context window used">
        <span>CTX</span>
        <div className="sync-bar"><i style={{ width: `${ctx * 100}%` }} className={ctx > 0.8 ? 'hot' : ''} /></div>
        <span>{Math.round(run.context / 1000)}k/{Math.round(run.contextLimit / 1000)}k</span>
      </div>
    </section>
  )
}

function PhaseRail({ phase, done }) {
  const at = PHASES.findIndex((p) => p.id === phase)
  return (
    <ol className="rail">
      {PHASES.map((p, i) => (
        <li key={p.id} className={done || i < at ? 'past' : i === at ? 'now' : ''}>
          <b>{p.n}</b><span>{p.label}</span><small>{p.kana}</small>
        </li>
      ))}
    </ol>
  )
}

const firstLine = (t) => (t.content || t.reasoning || '').replace(/\s+/g, ' ').trim().slice(0, 110)

// every turn of the run: the last few open, the earlier ones folded to a line (click to read them); the box
// follows the newest text until the reader scrolls up, and a button brings it back to live
export function Reasoning({ turns, running, logs }) {
  const box = useRef(null)
  const follow = useRef(true)
  const [behind, setBehind] = useState(false)
  const [open, setOpen] = useState(() => new Set())
  const [all, setAll] = useState(false)
  const lastTurn = turns[turns.length - 1]
  const log = turns.length === 0       // before Swift speaks (or with the autopilot alone) the run's own log
  const toLive = () => { follow.current = true; setBehind(false); if (box.current) box.current.scrollTop = box.current.scrollHeight }
  useEffect(() => {
    if (follow.current && box.current) box.current.scrollTop = box.current.scrollHeight
  }, [lastTurn?.reasoning, lastTurn?.content, turns.length, logs.length])
  const onScroll = (e) => {
    const el = e.currentTarget
    const atEnd = el.scrollHeight - el.scrollTop - el.clientHeight < 40
    follow.current = atEnd
    setBehind(!atEnd)
  }
  const toggle = (n) => setOpen((s) => { const x = new Set(s); x.has(n) ? x.delete(n) : x.add(n); return x })
  return (
    <section className="panel reasoning">
      <header className="panel-head yellow">
        <span className="glyph">▲</span> {log ? 'RUN // LOG' : 'SWIFT // REASONING'}
        <span className="head-right">
          {!log && turns.length > 3 && <button className="head-btn" onClick={() => setAll((a) => !a)}>{all ? 'FOLD' : 'OPEN ALL'}</button>}
          {log ? `${logs.length} lines` : `${turns.length} turns`}
        </span>
      </header>
      <div className="term-wrap">
        <div className="term" ref={box} onScroll={onScroll}>
          {log && logs.length === 0 && <p className="term-dim">{'> waiting…'}</p>}
          {log && <pre className="log">{logs.slice(-400).join('\n')}</pre>}
          {turns.map((t, i) => {
            const recent = i >= turns.length - 3
            const shown = all || recent || open.has(t.n)
            const head = `TURN ${String(t.n + 1).padStart(2, '0')}${t.tokens ? ` · ${t.tokens} tok` : ''}`
            return (
              <div key={t.n} className={`turn ${shown ? '' : 'folded'}`}>
                {recent && !all
                  ? <div className="turn-head">┌─ {head}</div>
                  : <button className="turn-head" onClick={() => toggle(t.n)} aria-expanded={shown}>{shown ? '▾' : '▸'} {head}{!shown && <span className="turn-peek"> {firstLine(t)}</span>}</button>}
                {shown && t.reasoning && <p className="think">{t.reasoning}</p>}
                {shown && t.content && <p className="say">{t.content}</p>}
                {running && i === turns.length - 1 && <span className="cursor">█</span>}
              </div>
            )
          })}
        </div>
        {behind && <button className="to-live" onClick={toLive}>▼ LIVE</button>}
      </div>
    </section>
  )
}

export function summarize(name, args) {
  try {
    const a = JSON.parse(args || '{}')
    if (name === 'scanruler') return (a.calls || []).map((c) => c.tool).join(' → ')
    if (name === 'run_script') return `${a.name} ${(a.args || []).join(' ')}`
    if (name === 'rebuild') return (a.files || []).join(', ')
    if (name === 'guide') return [a.topic, a.section].filter(Boolean).join(' § ') || 'index'
    if (name === 'view_image' || name === 'read_file') return (a.path || '').split(/[\\/]/).pop()
    if (name === 'scanruler_help') return (a.names || []).join(', ') || 'all tools'
    if (name === 'finish') return a.give_up ? 'give up' : 'deliver'
    if (name === 'group_bodies') return [...new Set([...(a.groups || []).map((g) => g.name), ...(a.cuts || []).map((c) => c.name)])].join(', ')
    return ''
  } catch { return (args || '').slice(0, 80) }
}

const ICON = { check: '◆', scanruler: '▶', rebuild: '■', run_script: '▣', guide: '◉', view_image: '◎', finish: '★', read_file: '▤', scanruler_help: '?', body_pieces: '◫', find_parts: '◎', group_bodies: '▦' }

export function Timeline({ items }) {
  const box = useRef(null)
  const [open, setOpen] = useState(null)
  useEffect(() => { if (box.current) box.current.scrollTop = box.current.scrollHeight }, [items.length])
  return (
    <section className="panel timeline">
      <header className="panel-head red"><span className="glyph">●</span> ACTIONS<span className="head-right">{items.length}</span></header>
      <ol className="tl" ref={box}>
        {items.length === 0 && <li className="term-dim">no actions yet</li>}
        {items.map((it) => (
          <li key={it.id} className={`tl-item ${it.result === null && it.name !== 'finish' ? 'pending' : ''} ${/FAIL|refused|failed|error/i.test(it.result || '') || (it.name === 'finish' && /"give_up":\s*true/.test(it.args || '')) ? 'bad' : ''}`}>
            <button onClick={() => setOpen(open === it.id ? null : it.id)} aria-expanded={open === it.id}>
              <span className="tl-icon">{ICON[it.name] || '▸'}</span>
              <span className="tl-name">{it.name}</span>
              <span className="tl-args">{summarize(it.name, it.args)}</span>
              <span className="tl-time">{new Date(it.t * 1000).toLocaleTimeString([], { hour12: false })}</span>
            </button>
            {open === it.id && <pre className="tl-detail">{`ARGS  ${it.args}\n\nRESULT\n${it.result ?? '…running'}`}</pre>}
          </li>
        ))}
      </ol>
    </section>
  )
}

function bar(v, tol, width = 18) {
  if (tol == null) return ''.padEnd(width, '·')
  const f = Math.min(1, v / (tol * 1.5))
  const n = Math.round(f * width)
  return '█'.repeat(n) + '░'.repeat(width - n)
}

export function Validation({ validations }) {
  const v = validations[validations.length - 1]
  if (!v) {
    return (
      <section className="panel validation">
        <header className="panel-head blue"><span className="glyph">◆</span> VALIDATOR</header>
        <p className="term-dim pad">The independent check runs after every change. Only a PASS is delivered.</p>
      </section>
    )
  }
  const r = v.result
  const t = r.tolerances
  const worst = (k) => Math.max(r.input_to_model[k], r.model_to_input[k])
  const metrics = [['RMS', worst('rms'), t.rms], ['P95', worst('p95'), t.p95], ['P99', worst('p99'), t.p99], ['MAX', worst('max'), r.mode === 'organic' ? null : t.max], ...(r.mode === 'organic' ? [] : [['VOL%', Math.abs(r.volume.diff_pct), t.vol]])]
  const history = validations.map((x) => Math.max(x.result.input_to_model.p95, x.result.model_to_input.p95))
  const pass = r.verdict === 'PASS'
  return (
    <section className="panel validation">
      <header className="panel-head blue"><span className="glyph">◆</span> VALIDATOR<span className="head-right">check #{validations.length}{r.noise ? ` · σ ${r.noise.toFixed(4)}` : ''}</span></header>
      <div className="val-body">
        <div key={validations.length} className={`stamp ${pass ? 'pass' : 'fail'}`}>
          <span>{pass ? 'PASS' : 'FAIL'}</span><small>{pass ? '承認' : '警告'}</small>
        </div>
        <pre className="meters">
          {metrics.map(([k, val, tol]) => `${k.padEnd(5)}${bar(val, tol)} ${val.toFixed(3).padStart(7)} ${tol != null ? `/ ${tol.toFixed(3)}` : ''}${tol == null ? '  ·' : val > tol ? '  ✗' : '  ✓'}`).join('\n')}
        </pre>
        <Spark values={history} />
        <ul className="checks">
          {Object.entries(r.checks).map(([k, ok]) => <li key={k} className={ok ? 'ok' : 'no'}>{ok ? '✓' : '✗'} {k}</li>)}
        </ul>
        {(r.deviation_clusters || []).slice(0, 4).map((c, i) => (
          <div key={i} className="cluster">
            <b>{c.kind}</b> {c.area_mm2} mm² · max {c.max} mm @ [{c.center.join(', ')}]
            {c.shape && <div className="cluster-shape">↳ {c.of === 'scan' ? 'scan' : 'model'}: {c.shape}</div>}
          </div>
        ))}
      </div>
    </section>
  )
}

function Spark({ values }) {
  if (values.length < 2) return null
  const max = Math.max(...values, 1e-6)
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * 100},${28 - (v / max) * 26}`).join(' ')
  return (
    <svg className="spark" viewBox="0 0 100 30" preserveAspectRatio="none" aria-label="p95 deviation over checks">
      <polyline points={pts} />
    </svg>
  )
}

// the named solids the STEP holds (organic route): Swift's grouping of the closed pieces, or the one body
export function Bodies({ bodies }) {
  if (!bodies) return null
  const size = (b) => b.hi.map((h, i) => Math.round(h - b.lo[i])).join(' × ')
  const lost = bodies.dropped.reduce((a, d) => a + d.volume, 0)
  return (
    <section className="panel bodies">
      <header className="panel-head yellow"><span className="glyph">■</span> BODIES<span className="head-right">{bodies.bodies.length} in STEP{bodies.dropped.length ? ` · ${bodies.dropped.length} dropped` : ''}</span></header>
      <ul className="body-list">
        {bodies.bodies.map((b, i) => (
          <li key={b.name}><i style={{ '--hue': (i * 67) % 360 }} /><b>{b.name}</b><span>{Math.round(b.volume).toLocaleString()} mm³</span><small>{size(b)} mm</small></li>
        ))}
      </ul>
      {bodies.cuts.map((c, i) => <div key={i} className="cluster">{c.kind} cut → <b>{c.name}</b> {Math.round(c.volume).toLocaleString()} mm³</div>)}
      {bodies.dropped.length > 0 && <div className="cluster dropped">dropped {bodies.dropped.length} pieces · {lost.toFixed(1)} mm³ (junk, reported)</div>}
    </section>
  )
}
