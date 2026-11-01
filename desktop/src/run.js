// The run, as the events of swift/run.py --events make it. One reducer: the UI only reads this state.

export const PHASES = [
  { id: 'open', n: '01', label: 'Link', kana: 'リンク' },
  { id: 'analyze', n: '02', label: 'Read', kana: '解析' },
  { id: 'autopilot', n: '03', label: 'Autopilot', kana: '自動' },
  { id: 'swift', n: '04', label: 'Swift', kana: '思考' },
  { id: 'organic', n: '05', label: 'Solid', kana: '造形' },
  { id: 'bodies', n: '06', label: 'Bodies', kana: '部品' },
  { id: 'check', n: '07', label: 'Validate', kana: '検証' },
  { id: 'deliver', n: '08', label: 'Deliver', kana: '納品' },
]

export const initial = {
  status: 'idle',          // idle | running | done | error
  stl: null, work: null, model: null, swift: true,
  mode: null, phase: null, phaseLabel: '', pageUrl: null,
  turn: null, maxTurns: null, context: 0, contextLimit: 49152,
  turns: [],               // [{ n, reasoning, content, tokens }]
  activity: 'idle',        // what the orb shows: thinking | writing | measuring | building | validating | reading | looking | prefill | working | linking
  timeline: [],            // [{ id, name, args, result, t }]
  validations: [],         // [{ result, summary, t }]
  image: null,             // last picture Swift looked at
  bodies: null,            // { bodies: [{ name, volume, triangles, lo, hi }], dropped, cuts, image }: the named solids of the STEP
  logs: [],
  done: null, error: null, startedAt: null, endedAt: null,
  tris: null,              // triangles in the input (for the ETA)
  phaseTimes: [],          // [{ id, t }]: when each phase began (run clock, s)
  turnTimes: [],           // when each Swift turn began
  focus: null,             // { p: [x, y, z], r, t }: where the last tool call looked, for the director's camera
  replay: false,           // a finished run read back from its events.jsonl
  finish: null,            // { giveUp, report }: what Swift said when it ended the run
  stages: [],              // [{ kind, file, bytes, t }]: the scan's way to a solid, as the work folder shows it
  lastEventAt: null,       // when the run last said anything (ms): how long it has been quiet
}

export const STAGES = [['scan', 'SCAN'], ['repaired', 'REPAIRED'], ['closed', 'CLOSED'], ['reduced', 'SOLID'], ['bodies', 'BODIES']]

// how a finished run ended, in words: the verdict and why
export function outcomeOf(s) {
  if (s.status === 'error') return { ok: false, title: 'STOPPED BY AN ERROR', why: s.error || 'The run stopped.' }
  if (s.status !== 'done') return null
  if (s.done?.status === 'PASS') return { ok: true, title: 'VALIDATED', why: `${s.done.step?.split(/[\\/]/).pop() || 'The STEP'} passed the independent check.` }
  const said = (s.finish?.report || s.turns[s.turns.length - 1]?.content || '').trim()
  const title = s.finish?.giveUp ? 'SWIFT GAVE UP' : s.validations.length ? 'NOT VALIDATED' : 'ENDED WITHOUT A SOLID'
  return { ok: false, title, why: said || (s.validations.length ? 'The last check did not pass.' : 'No closed solid reached the validator.') }
}

const vec = (v) => (Array.isArray(v) && v.length === 3 && v.every((x) => typeof x === 'number') ? v : null)

// where a tool call looks: the first box, point or point set in its arguments
export function focusOf(args) {
  let found = null
  const walk = (o, depth) => {
    if (found || !o || typeof o !== 'object' || depth > 7) return
    if (Array.isArray(o)) {
      const pts = o.map(vec)
      if (pts.length > 1 && pts.every(Boolean)) {
        const c = [0, 1, 2].map((i) => pts.reduce((a, p) => a + p[i], 0) / pts.length)
        found = { p: c, r: Math.max(...pts.map((p) => Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]))) }
        return
      }
      for (const x of o) walk(x, depth + 1)
      return
    }
    const mn = vec(o.min) || vec(o.lo), mx = vec(o.max) || vec(o.hi)
    if (mn && mx) { found = { p: mn.map((m, i) => (m + mx[i]) / 2), r: Math.hypot(...mx.map((m, i) => m - mn[i])) / 2 }; return }
    for (const k of ['at', 'center', 'centre', 'centroid', 'point', 'origin', 'near']) {
      const p = vec(o[k])
      if (p) { found = { p, r: [o.size / 2, o.radius * 3, o.within].find((x) => x > 0) || null }; return }
    }
    for (const v of Object.values(o)) walk(v, depth + 1)
  }
  try { walk(JSON.parse(args || '{}'), 0) } catch { /* not JSON: no place */ }
  return found
}

const MEASURE = /scan_|element_|section|heightmap|outline|fillet_read|thread_read|draft|qa|nearest|query|describe|view_render|body_compare|body_review/
const BUILD = /feature_|sketch_|rebuild|parameter_|build|import_step|body_set/

// what a tool call means for the orb
export function activityOf(name, args) {
  if (name === 'check') return 'validating'
  if (name === 'rebuild') return 'building'
  if (name === 'view_image') return 'looking'
  if (name === 'guide' || name === 'scanruler_help' || name === 'read_file') return 'reading'
  if (name === 'finish') return 'writing'
  if (name === 'body_pieces' || name === 'find_parts') return 'looking'
  if (name === 'group_bodies') return 'building'
  if (name === 'run_script') return /slabs|revolve|holes|blends/.test(args || '') ? 'building' : 'measuring'
  if (name === 'scanruler') return BUILD.test(args || '') ? 'building' : MEASURE.test(args || '') ? 'measuring' : 'working'
  return 'working'
}

const last = (a) => a[a.length - 1]

export function reduce(s, e) {
  const n = step(s, e)
  return n === s || e.type === 'reset' ? n : { ...n, lastEventAt: Date.now() }
}

function step(s, e) {
  switch (e.type) {
    case 'stage':
      return { ...s, stages: [...s.stages.filter((x) => x.kind !== e.kind), { kind: e.kind, file: e.file, bytes: e.bytes, t: e.t }] }
    case 'reset':
      return { ...initial, status: 'running', stl: e.stl, swift: e.swift, startedAt: Date.now(), replay: !!e.replay }
    case 'start':
      return { ...s, status: 'running', stl: s.stl || e.stl, startedAt: s.replay ? e.t * 1000 : s.startedAt || Date.now(), work: e.work, model: e.model, swift: e.swift }
    case 'meta':
      return { ...s, tris: e.tris }
    case 'phase':
      return { ...s, phase: e.phase, phaseLabel: e.label, phaseTimes: [...s.phaseTimes, { id: e.phase, t: e.t }], activity: e.phase === 'open' ? 'linking' : e.phase === 'check' ? 'validating' : e.phase === 'swift' ? 'prefill' : 'working' }
    case 'mode':
      return { ...s, mode: e.mode }
    case 'page':
      return { ...s, pageUrl: e.url }
    case 'turn':
      return { ...s, turn: e.n, maxTurns: e.max, context: e.context, contextLimit: e.limit, activity: 'prefill', turnTimes: [...s.turnTimes, e.t], turns: [...s.turns, { n: e.n, reasoning: '', content: '', tokens: null }] }
    case 'turn_end': {
      const turns = s.turns.slice()
      if (turns.length) turns[turns.length - 1] = { ...last(turns), tokens: e.completion_tokens, prompt: e.prompt_tokens }
      return { ...s, turns, context: e.prompt_tokens || s.context }
    }
    case 'reasoning':
    case 'content': {
      const turns = s.turns.length ? s.turns.slice() : [{ n: 0, reasoning: '', content: '' }]
      const t = last(turns)
      turns[turns.length - 1] = { ...t, [e.type]: t[e.type] + e.delta }
      return { ...s, turns, activity: e.type === 'reasoning' ? 'thinking' : 'writing' }
    }
    case 'tool': {
      const f = focusOf(e.args)
      let finish = s.finish
      if (e.name === 'finish') { try { const a = JSON.parse(e.args || '{}'); finish = { giveUp: !!a.give_up, report: a.report || '' } } catch { finish = { giveUp: false, report: '' } } }
      return { ...s, finish, activity: activityOf(e.name, e.args), focus: f ? { ...f, t: e.t } : s.focus, timeline: [...s.timeline, { id: s.timeline.length, name: e.name, args: e.args, result: null, t: e.t }].slice(-300) }
    }
    case 'tool_result': {
      const timeline = s.timeline.slice()
      for (let i = timeline.length - 1; i >= 0; i--) if (timeline[i].name === e.name && timeline[i].result === null) { timeline[i] = { ...timeline[i], result: e.text }; break }
      return { ...s, timeline }
    }
    case 'image':
      return { ...s, image: e.path }
    case 'bodies':
      return { ...s, bodies: e, image: e.image || s.image }
    case 'validation':
      return e.result ? { ...s, validations: [...s.validations, { result: e.result, summary: e.summary, t: e.t }] } : s
    case 'log':
    case 'stderr':
      return { ...s, logs: [...s.logs, e.text].slice(-400) }
    case 'done':
      return { ...s, done: e, status: 'done', activity: 'idle', endedAt: s.replay ? e.t * 1000 : Date.now() }
    case 'error':
      return { ...s, error: e.text, status: 'error', activity: 'idle', endedAt: s.replay && e.t ? e.t * 1000 : Date.now() }
    case 'exit':
      return s.status === 'running' ? { ...s, status: e.code === 0 ? 'done' : 'error', error: s.error || (e.code ? `The run stopped (exit ${e.code}).` : null), activity: 'idle', endedAt: s.replay ? s.endedAt || Date.now() : Date.now() } : s
    default:
      return s
  }
}
