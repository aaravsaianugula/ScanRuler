// Time left in a run, learned from the runs before it: each finished run leaves how long each phase took
// (per triangle for the mesh-bound phases) and Swift's turns and seconds per turn; the estimate is the
// median of those for the same route, the current phase's elapsed time and the current run's own pace.

const KEY = 'studio.timing'
// first guesses, from a Kurisu run (59k triangles); every finished run replaces them with what it measured
const DEFAULTS = { open: 8, analyze: 120, autopilot: 90, organic: 300, bodies: 100, check: 35, deliver: 5 }
// the exact reading scales with the input; the solid, the bodies and the check work on a remesh of fixed size
const MESH = new Set(['analyze'])
const BASE_TRIS = 800_000
const DEFAULT_TURNS = 25
const DEFAULT_SPT = 25

const history = () => { try { return JSON.parse(localStorage.getItem(KEY)) || [] } catch { return [] } }
const median = (a) => { const s = a.filter((x) => x > 0).sort((x, y) => x - y); return s.length ? s[s.length >> 1] : null }
const scale = (tris) => Math.min(5, Math.max(0.2, (tris || BASE_TRIS) / BASE_TRIS))

// the phases still to come after `phase`, by the route the run takes
function ahead(run) {
  const seen = new Set(run.phaseTimes.map((p) => p.id))
  const organicRoute = seen.has('organic') || (run.mode === 'organic' && !run.swift)
  const tail = run.mode === 'cad' ? ['autopilot', 'check', 'deliver']
    : organicRoute || !run.swift ? ['organic', 'bodies', 'check', 'deliver'] : ['swift', 'deliver']
  const route = ['open', 'analyze', ...tail]
  const at = route.indexOf(run.phase)
  return at < 0 ? (run.phase ? ['deliver'] : route) : route.slice(at + 1)
}

export function estimate(run, nowS) {
  if (run.status !== 'running' || !run.startedAt) return null
  const past = history()
  const same = past.filter((h) => h.mode === run.mode && h.swift === run.swift)
  const pool = same.length ? same : past
  const k = scale(run.tris)
  const expect = (id) => {
    const m = median(pool.map((h) => h.phases?.[id] / (MESH.has(id) ? scale(h.tris) : 1)))
    return (m ?? DEFAULTS[id] ?? 30) * (MESH.has(id) ? k : 1)
  }
  // Swift: turns still expected at the pace of this run (or the runs before it)
  const done = run.turnTimes.length
  const own = done >= 3 ? (nowS - run.turnTimes[0]) / done : null
  const spt = own ?? median(pool.map((h) => h.spt)) ?? DEFAULT_SPT
  const turns = Math.min(run.maxTurns || 70, Math.max(done + 2, median(pool.filter((h) => h.turns).map((h) => h.turns)) ?? DEFAULT_TURNS))
  const swiftLeft = () => Math.max(0, turns - done) * spt + (done ? Math.max(0, spt - (nowS - run.turnTimes[done - 1])) : 0)

  const cur = run.phaseTimes[run.phaseTimes.length - 1]
  let left = 0
  // once Swift has the run, its turns are the clock: the phases its tools start (a solid, the bodies) run inside them
  if (run.phaseTimes.some((p) => p.id === 'swift') && cur?.id !== 'deliver') {
    const spent = nowS - run.startedAt / 1000
    left = swiftLeft() + expect('deliver')
    return { left, frac: Math.min(0.99, spent / (spent + left)), learning: pool.length === 0 }
  }
  if (cur) {
    const inPhase = nowS - cur.t
    left = cur.id === 'swift' ? swiftLeft() : Math.max(expect(cur.id) - inPhase, expect(cur.id) * 0.1)
  }
  for (const id of ahead(run)) left += id === 'swift' ? turns * spt : expect(id)
  const spent = nowS - run.startedAt / 1000
  return { left, frac: Math.min(0.99, spent / (spent + left)), learning: pool.length === 0 }
}

// remember how long each phase of a finished run took
export function record(run, endS) {
  if (!run.phaseTimes.length) return
  const phases = {}
  run.phaseTimes.forEach((p, i) => { phases[p.id] = (phases[p.id] || 0) + ((run.phaseTimes[i + 1]?.t ?? endS) - p.t) })
  const n = run.turnTimes.length
  const entry = { mode: run.mode, swift: run.swift, tris: run.tris, phases, turns: n || null, spt: n >= 2 ? (endS - run.turnTimes[0]) / n : null, t: Date.now() }
  try { localStorage.setItem(KEY, JSON.stringify([entry, ...history()].slice(0, 30))) } catch { /* storage blocked: no learning */ }
}

export const clock = (s) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`
