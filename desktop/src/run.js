// The run, as the events of swift/run.py --events make it. One reducer: the UI only reads this state.

export const PHASES = [
  { id: 'open', n: '01', label: 'Link', kana: 'リンク' },
  { id: 'analyze', n: '02', label: 'Read', kana: '解析' },
  { id: 'autopilot', n: '03', label: 'Autopilot', kana: '自動' },
  { id: 'swift', n: '04', label: 'Swift', kana: '思考' },
  { id: 'check', n: '05', label: 'Validate', kana: '検証' },
  { id: 'deliver', n: '06', label: 'Deliver', kana: '納品' },
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
  logs: [],
  done: null, error: null, startedAt: null, endedAt: null,
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
  if (name === 'run_script') return /slabs|revolve|holes|blends/.test(args || '') ? 'building' : 'measuring'
  if (name === 'scanruler') return BUILD.test(args || '') ? 'building' : MEASURE.test(args || '') ? 'measuring' : 'working'
  return 'working'
}

const last = (a) => a[a.length - 1]

export function reduce(s, e) {
  switch (e.type) {
    case 'reset':
      return { ...initial, status: 'running', stl: e.stl, swift: e.swift, startedAt: Date.now() }
    case 'start':
      return { ...s, status: 'running', stl: s.stl || e.stl, startedAt: s.startedAt || Date.now(), work: e.work, model: e.model, swift: e.swift }
    case 'phase':
      return { ...s, phase: e.phase, phaseLabel: e.label, activity: e.phase === 'open' ? 'linking' : e.phase === 'check' ? 'validating' : e.phase === 'swift' ? 'prefill' : 'working' }
    case 'mode':
      return { ...s, mode: e.mode }
    case 'page':
      return { ...s, pageUrl: e.url }
    case 'turn':
      return { ...s, turn: e.n, maxTurns: e.max, context: e.context, contextLimit: e.limit, activity: 'prefill', turns: [...s.turns, { n: e.n, reasoning: '', content: '', tokens: null }] }
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
    case 'tool':
      return { ...s, activity: activityOf(e.name, e.args), timeline: [...s.timeline, { id: s.timeline.length, name: e.name, args: e.args, result: null, t: e.t }].slice(-300) }
    case 'tool_result': {
      const timeline = s.timeline.slice()
      for (let i = timeline.length - 1; i >= 0; i--) if (timeline[i].name === e.name && timeline[i].result === null) { timeline[i] = { ...timeline[i], result: e.text }; break }
      return { ...s, timeline }
    }
    case 'image':
      return { ...s, image: e.path }
    case 'validation':
      return e.result ? { ...s, validations: [...s.validations, { result: e.result, summary: e.summary, t: e.t }] } : s
    case 'log':
    case 'stderr':
      return { ...s, logs: [...s.logs, e.text].slice(-400) }
    case 'done':
      return { ...s, done: e, status: 'done', activity: 'idle', endedAt: Date.now() }
    case 'error':
      return { ...s, error: e.text, status: 'error', activity: 'idle', endedAt: Date.now() }
    case 'exit':
      return s.status === 'running' ? { ...s, status: e.code === 0 ? 'done' : 'error', error: s.error || (e.code ? `The run stopped (exit ${e.code}).` : null), activity: 'idle', endedAt: Date.now() } : s
    default:
      return s
  }
}
