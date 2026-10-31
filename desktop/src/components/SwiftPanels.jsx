import { useEffect, useRef, useState } from 'react'
import { ThinkingOrb } from 'thinking-orbs'
import { PHASES } from '../run.js'

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

export function OrbCard({ run, elapsed }) {
  const [state, label, color] = ACTIVITY[run.activity] || ACTIVITY.idle
  const ctx = run.contextLimit ? Math.min(1, run.context / run.contextLimit) : 0
  return (
    <section className="panel orb-card" aria-live="polite">
      <div className="orb-row">
        <div className="orb-disc" style={{ '--tint': color }}>
          <ThinkingOrb state={state} size={64} theme="dark" color={color} paused={run.status !== 'running'} />
        </div>
        <div className="orb-text">
          <div className="orb-label">{run.status === 'running' ? `${label}…` : run.status === 'done' ? 'Run complete' : run.status === 'error' ? 'Stopped' : 'Standing by'}</div>
          <div className="orb-sub">{run.phaseLabel || '—'}</div>
          <div className="chips">
            <span className="chip">{run.swift ? 'SWIFT 1.5' : 'AUTOPILOT'}</span>
            {run.mode && <span className="chip yellow">{{ cad: 'CAD EXACT', organic: 'ORGANIC' }[run.mode] || 'SCAN'}</span>}
            {run.turn != null && <span className="chip">TURN {run.turn + 1}/{run.maxTurns}</span>}
            <span className="chip">{elapsed}</span>
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

export function Reasoning({ turns, running, logs }) {
  const box = useRef(null)
  const [follow, setFollow] = useState(true)
  const lastTurn = turns[turns.length - 1]
  const log = turns.length === 0       // before Swift speaks (or with the autopilot alone) the run's own log
  useEffect(() => {
    if (follow && box.current) box.current.scrollTop = box.current.scrollHeight
  }, [lastTurn?.reasoning, lastTurn?.content, turns.length, logs.length, follow])
  return (
    <section className="panel reasoning">
      <header className="panel-head yellow"><span className="glyph">▲</span> {log ? 'RUN // LOG' : 'SWIFT // REASONING'}<span className="head-right">{log ? `${logs.length} lines` : `${turns.length} turns`}</span></header>
      <div className="term" ref={box} onScroll={(e) => { const el = e.currentTarget; setFollow(el.scrollHeight - el.scrollTop - el.clientHeight < 40) }}>
        {log && logs.length === 0 && <p className="term-dim">{'> waiting…'}</p>}
        {log && <pre className="log">{logs.slice(-250).join('\n')}</pre>}
        {turns.slice(-12).map((t, i, arr) => (
          <div key={t.n} className="turn">
            <div className="turn-head">┌─ TURN {String(t.n + 1).padStart(2, '0')} {t.tokens ? `· ${t.tokens} tok` : ''}</div>
            {t.reasoning && <p className="think">{t.reasoning}</p>}
            {t.content && <p className="say">{t.content}</p>}
            {running && i === arr.length - 1 && <span className="cursor">█</span>}
          </div>
        ))}
      </div>
    </section>
  )
}

function summarize(name, args) {
  try {
    const a = JSON.parse(args || '{}')
    if (name === 'scanruler') return (a.calls || []).map((c) => c.tool).join(' → ')
    if (name === 'run_script') return `${a.name} ${(a.args || []).join(' ')}`
    if (name === 'rebuild') return (a.files || []).join(', ')
    if (name === 'guide') return [a.topic, a.section].filter(Boolean).join(' § ') || 'index'
    if (name === 'view_image' || name === 'read_file') return (a.path || '').split(/[\\/]/).pop()
    if (name === 'scanruler_help') return (a.names || []).join(', ') || 'all tools'
    if (name === 'finish') return a.give_up ? 'give up' : 'deliver'
    return ''
  } catch { return (args || '').slice(0, 80) }
}

const ICON = { check: '◆', scanruler: '▶', rebuild: '■', run_script: '▣', guide: '◉', view_image: '◎', finish: '★', read_file: '▤', scanruler_help: '?' }

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
          <li key={it.id} className={`tl-item ${it.result === null ? 'pending' : ''} ${/FAIL|refused|failed|error/i.test(it.result || '') ? 'bad' : ''}`}>
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
  const metrics = [['RMS', worst('rms'), t.rms], ['P95', worst('p95'), t.p95], ['P99', worst('p99'), t.p99], ['MAX', worst('max'), t.max], ...(r.mode === 'organic' ? [] : [['VOL%', Math.abs(r.volume.diff_pct), t.vol]])]
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
          {metrics.map(([k, val, tol]) => `${k.padEnd(5)}${bar(val, tol)} ${val.toFixed(3).padStart(7)} ${tol != null ? `/ ${tol.toFixed(3)}` : ''}${tol != null && val > tol ? '  ✗' : '  ✓'}`).join('\n')}
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
