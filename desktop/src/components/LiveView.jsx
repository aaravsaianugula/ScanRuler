import { BorderBeam } from 'border-beam'
import { Component, useCallback, useEffect, useRef, useState } from 'react'
import { clock } from '../eta.js'
import { outcomeOf, STAGES } from '../run.js'
import Director, { FILMABLE_BYTES, SHOTS, shotFor } from './Director.jsx'

// a crash in the 3D view (a driver, a mesh it cannot draw) costs the view, never the app or the run
class Guard extends Component {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  componentDidCatch(err) { console.error('director failed:', err) }
  render() { return this.state.failed ? <div className="director-wait">3D VIEW UNAVAILABLE · SEE SCANRULER</div> : this.props.children }
}

// what a quiet run is waiting on, so a long step never looks like a hang
function quietNote(run, now) {
  if (run.status !== 'running' || !run.lastEventAt) return null
  const s = (now - run.lastEventAt) / 1000
  if (s < 45) return null
  const pending = run.timeline.findLast((x) => x.result === null)
  if (pending) return `STILL ON ${pending.name.toUpperCase()} · ${clock(s)} · long steps take minutes`
  if (run.phase === 'swift') return `WAITING ON SWIFT (THE GATEWAY) · ${clock(s)}`
  return `${(run.phaseLabel || 'WORKING').toUpperCase()} · ${clock(s)} without news · long steps take minutes`
}
import { ACTIVITY, summarize } from './SwiftPanels.jsx'

const NOISE = '░▒▓█▀▄■□▪▫◘◙'
const VIEW_KEY = 'studio.view'

function Static() {
  // the "no signal" screen: a field of block characters that keeps changing
  const [rows, setRows] = useState([])
  useEffect(() => {
    const make = () => Array.from({ length: 14 }, () => Array.from({ length: 56 }, () => (Math.random() < 0.55 ? ' ' : NOISE[(Math.random() * NOISE.length) | 0])).join(''))
    setRows(make())
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const id = setInterval(() => setRows(make()), 120)
    return () => clearInterval(id)
  }, [])
  return <pre className="static" aria-hidden>{rows.join('\n')}</pre>
}

// the shot holds at least a few seconds, so quick tool calls do not make the camera jitter
function useShot(run) {
  const [shot, setShot] = useState({ id: 'wide', n: 1, since: 0 })
  const want = shotFor(run.activity, run.status, run.focus, Date.now() / 1000)
  const focusT = run.focus?.t
  useEffect(() => {
    const go = () => setShot((s) => (s.id === want && !(want === 'dolly' && s.focusT !== focusT) ? s : { id: want, n: s.n + 1, since: Date.now(), focusT }))
    const id = setTimeout(go, Math.max(0, 3500 - (Date.now() - shot.since)))
    return () => clearTimeout(id)
  }, [want, focusT, shot.since])
  return shot
}

// Swift's plan as it stands: the last sentence of what it is thinking or saying
function planOf(turns) {
  const t = turns[turns.length - 1]
  const text = (t?.content || t?.reasoning || '').replace(/\s+/g, ' ').trim()
  if (!text) return ''
  const parts = text.split(/(?<=[.!?])\s+/)
  const tail = parts[parts.length - 1]
  return (tail.length > 40 || parts.length < 2 ? tail : parts[parts.length - 2]).slice(-170)
}

// the names over the bodies and the validator's places, moved every frame without re-rendering the view
function Labels({ register }) {
  const [items, setItems] = useState([])
  useEffect(() => register(setItems), [register])
  return items.map((l) => (
    <span key={l.id} className={`dlabel ${l.kind}`} style={{ transform: `translate(${l.x}px, ${l.y}px)` }}>{l.label}</span>
  ))
}

export default function LiveView({ run, elapsed, eta }) {
  const { pageUrl: url, image, work, phaseLabel, status } = run
  const active = status === 'running' && run.activity !== 'idle'
  const [pic, setPic] = useState(null)
  const [view, setView] = useState(() => { try { return localStorage.getItem(VIEW_KEY) || 'director' } catch { return 'director' } })
  const shot = useShot(run)
  const labelSink = useRef(null)
  const register = useCallback((fn) => { labelSink.current = fn }, [])
  const [flash, setFlash] = useState(0)

  useEffect(() => {
    if (!image || !work) return
    window.studio.readImage(image, work).then(setPic)
  }, [image, work])
  useEffect(() => { try { localStorage.setItem(VIEW_KEY, view) } catch { /* remembered for this session only */ } }, [view])
  useEffect(() => { if (shot.n > 1) setFlash((n) => n + 1) }, [shot.n])

  const [, label, color] = ACTIVITY[run.activity] || ACTIVITY.idle
  const lastTool = run.timeline.findLast((x) => x.result === null) || run.timeline[run.timeline.length - 1]
  const what = lastTool && summarize(lastTool.name, lastTool.args)
  const detail = lastTool ? `${lastTool.name}${what ? ` · ${what}` : ''}` : phaseLabel
  const plan = planOf(run.turns)
  const lastVal = run.validations[run.validations.length - 1]
  const director = view === 'director' && !!run.stl
  const thinking = status === 'running' && ['thinking', 'prefill', 'writing'].includes(run.activity)
  const end = outcomeOf(run)
  const quiet = quietNote(run, Date.now())
  const reached = new Set(['scan', ...run.stages.map((x) => x.kind)])
  const latest = STAGES.map(([k]) => k).filter((k) => reached.has(k)).pop()
  const stage = run.stages.filter((x) => x.bytes <= FILMABLE_BYTES).sort((a, b) => a.t - b.t).pop()
  const showStages = run.mode === 'organic' || run.stages.length > 0
  const [hideEnd, setHideEnd] = useState(false)
  useEffect(() => setHideEnd(false), [run.startedAt])

  return (
    <section className="panel live" aria-label="Live view">
      <header className="panel-head blue">
        <span className="glyph">■</span> LIVE //
        <span className="view-tabs" role="tablist">
          <button role="tab" aria-selected={view === 'director'} className={view === 'director' ? 'on' : ''} onClick={() => setView('director')}>DIRECTOR</button>
          <button role="tab" aria-selected={view === 'page'} className={view === 'page' ? 'on' : ''} onClick={() => setView('page')}>SCANRULER</button>
        </span>
        <span className="head-right">{url || run.stl ? <><i className="rec" /> REC</> : 'NO SIGNAL'}</span>
      </header>
      <BorderBeam className="beam" size="md" colorVariant="ocean" theme="dark" active={!!active} duration={2.4} strength={0.9}>
        <div className="live-frame" style={{ '--tint': color }}>
          {/* the page Swift drives stays mounted and full size under the director: it must keep rendering */}
          {url ? (
            <webview className="live-webview" src={url} partition="persist:scanruler" />
          ) : !director && (
            <div className="awaiting">
              <Static />
              <div className="awaiting-label"><span>AWAITING SIGNAL</span><small>{phaseLabel || 'start a run to link ScanRuler'}</small></div>
            </div>
          )}
          {run.stl && (
            <div className={`director-layer ${director ? '' : 'off'} ${thinking ? 'letterbox' : ''}`} aria-hidden={!director}>
              <Guard>
                <Director stl={run.stl} shot={shot.id} focus={run.focus} bodies={run.bodies} clusters={lastVal?.result?.deviation_clusters}
                  visible={director} onLabels={(items) => labelSink.current?.(items)} stage={stage} />
              </Guard>
              <div className="dlabels"><Labels register={register} /></div>
              <div className="hud">
                <div className="hud-top">
                  <span className="hud-cam"><i className="rec" /> CAM {String(shot.n).padStart(2, '0')} · {SHOTS[shot.id]}</span>
                  <span className="hud-tc">{elapsed}</span>
                  {eta && <span className="hud-eta">ETA {clock(eta.left)}{eta.learning ? ' · first guess' : ''}<i style={{ width: `${eta.frac * 100}%` }} /></span>}
                </div>
                {showStages && (
                  <div className="hud-stages" aria-label="From scan to solid">
                    {STAGES.map(([k, l]) => <span key={k} className={`${reached.has(k) ? 'on' : ''} ${k === latest && status === 'running' ? 'now' : ''}`}>{l}</span>)}
                  </div>
                )}
                {quiet && <div className="hud-quiet">{quiet}</div>}
                {run.activity === 'measuring' && <div className="hud-scan" />}
                <div className="hud-third" key={`${run.activity}-${lastTool?.id}`}>
                  {status === 'running' && <><b style={{ background: color }}>{label.toUpperCase()}</b><span>{detail}</span></>}
                </div>
                {plan && status === 'running' && <div className="hud-plan" key={plan.slice(0, 30)}><small>SWIFT ▸ PLAN</small>{plan}</div>}
                {flash > 0 && <div className="hud-flash" key={flash} />}
              </div>
            </div>
          )}
          {/* how the run ended and why, over either view: a run never just stops without saying so */}
          {end && !hideEnd && (
            <div className={`endcard ${end.ok ? 'ok' : 'bad'}`} role="status">
              <button className="x" onClick={() => setHideEnd(true)} aria-label="Hide">✕</button>
              <small>RUN ENDED · {elapsed}{run.turns.length ? ` · ${run.turns.length} turns` : ''}</small>
              <b>{end.title}</b>
              <p>{end.why}</p>
              {!end.ok && <em>The full story is in REPORT; the reasoning panel keeps every turn.</em>}
            </div>
          )}
          <i className="corner tl" /><i className="corner tr" /><i className="corner bl" /><i className="corner br" />
          {pic && (
            <figure className="swift-eye">
              <img src={pic} alt="The last picture Swift looked at" />
              <figcaption>SWIFT'S EYE</figcaption>
            </figure>
          )}
        </div>
      </BorderBeam>
    </section>
  )
}
