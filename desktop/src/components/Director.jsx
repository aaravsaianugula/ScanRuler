import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { STLLoader } from 'three/addons/loaders/STLLoader.js'

// The director's camera: the run's own mesh in a scene of its own (the ScanRuler page Swift drives is never
// touched), filmed in shots chosen by what the run is doing - a wide orbit while Swift thinks, a dolly in on the
// place a tool call measures, a low push while it builds, a sweep while the validator checks, a turntable at the end.

export const SHOTS = {
  wide: 'WIDE · ORBIT',
  dolly: 'DOLLY IN',
  hero: 'LOW ANGLE · PUSH',
  sweep: 'TOP SWEEP',
  final: 'TURNTABLE',
}

export function shotFor(activity, status, focus, nowS) {
  if (status === 'done') return 'final'
  if (activity === 'validating') return 'sweep'
  if (activity === 'building') return 'hero'
  const fresh = focus && nowS - focus.t < 45
  if (fresh && (activity === 'measuring' || activity === 'looking' || activity === 'working')) return 'dolly'
  return 'wide'
}

const BAND = { sweep: [0.23, 0.44, 0.85], hero: [0.9, 0.23, 0.18] }

export default function Director({ stl, shot, focus, bodies, clusters, visible, onLabels }) {
  const host = useRef(null)
  const live = useRef({})            // what the render loop reads each frame
  const [state, setState] = useState('loading')
  live.current.shot = shot
  live.current.focus = focus
  live.current.visible = visible

  useEffect(() => {
    if (!stl) return
    let alive = true
    const el = host.current
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'low-power' })
    renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5))
    renderer.outputColorSpace = THREE.SRGBColorSpace
    el.appendChild(renderer.domElement)
    const scene = new THREE.Scene()
    scene.background = new THREE.Color('#07080b')
    const camera = new THREE.PerspectiveCamera(32, 1, 1, 10000)
    camera.up.set(0, 0, 1)
    scene.add(new THREE.HemisphereLight(0xf2ede4, 0x101018, 1.1))
    const key = new THREE.DirectionalLight(0xfff1dc, 2.2)
    const rim = new THREE.DirectionalLight(0x3a6fd8, 2.6)
    scene.add(key, rim)

    const u = { uBand: { value: 0 }, uBandOn: { value: 0 }, uBandW: { value: 1 }, uBandColor: { value: new THREE.Vector3(...BAND.sweep) }, uFocus: { value: new THREE.Vector3() }, uFocusR: { value: 1 }, uFocusOn: { value: 0 } }
    const material = new THREE.MeshStandardMaterial({ color: 0x8e9bab, roughness: 0.5, metalness: 0.12 })
    // a scan band that sweeps the surface and a yellow ring where the run looks, drawn on the mesh itself
    material.onBeforeCompile = (sh) => {
      Object.assign(sh.uniforms, u)
      sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vW;')
        .replace('#include <project_vertex>', '#include <project_vertex>\nvW = (modelMatrix * vec4(transformed, 1.0)).xyz;')
      sh.fragmentShader = sh.fragmentShader.replace('#include <common>', `#include <common>
varying vec3 vW; uniform float uBand, uBandOn, uBandW, uFocusR, uFocusOn; uniform vec3 uBandColor, uFocus;`)
        .replace('#include <dithering_fragment>', `#include <dithering_fragment>
float band = uBandOn * exp(-pow((vW.z - uBand) / uBandW, 2.0));
gl_FragColor.rgb = mix(gl_FragColor.rgb, uBandColor * 1.7, clamp(band, 0.0, 1.0) * 0.85);
float d = length(vW - uFocus);
float ring = uFocusOn * (exp(-pow((d - uFocusR) / (uFocusR * 0.07), 2.0)) + 0.22 * step(d, uFocusR));
gl_FragColor.rgb = mix(gl_FragColor.rgb, vec3(0.96, 0.77, 0.09), clamp(ring, 0.0, 1.0) * 0.85);`)
    }
    const extras = new THREE.Group()
    scene.add(extras)

    let box = null, centre = new THREE.Vector3(), R = 100
    const look = new THREE.Vector3()
    let az = -Math.PI / 3, lastShot = null, shotStart = 0, last = performance.now(), acc = 0, frame = 0
    const still = matchMedia('(prefers-reduced-motion: reduce)').matches

    window.studio.readMesh(stl).then((bytes) => {
      if (!alive) return
      if (!bytes) { setState('missing'); return }
      const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
      const geom = new STLLoader().parse(buf)
      geom.computeBoundingBox()
      box = geom.boundingBox
      box.getCenter(centre)
      R = box.getSize(new THREE.Vector3()).length() / 2
      camera.near = R / 200; camera.far = R * 40; camera.updateProjectionMatrix()
      scene.add(new THREE.Mesh(geom, material))
      const grid = new THREE.GridHelper(R * 4, 40, 0x3a6fd8, 0x1b2233)
      grid.rotation.x = Math.PI / 2
      grid.position.set(centre.x, centre.y, box.min.z - R * 0.002)
      grid.material.transparent = true; grid.material.opacity = 0.45
      scene.add(grid)
      look.copy(centre)
      camera.position.copy(centre).add(new THREE.Vector3(R * 2.2, -R * 2.2, R))
      setState('ready')
    }).catch(() => alive && setState('missing'))

    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight
      if (!w || !h) return
      renderer.setSize(w, h, false)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
    }
    const ro = new ResizeObserver(resize)
    ro.observe(el)
    resize()

    const aim = new THREE.Vector3(), pos = new THREE.Vector3(), tmp = new THREE.Vector3()
    let raf = 0
    const tick = (now) => {
      raf = requestAnimationFrame(tick)
      const dt = Math.min(0.1, (now - last) / 1000)
      last = now
      acc += dt
      if (!box || !live.current.visible || acc < 1 / 30) return          // 30 fps is plenty for slow camera moves
      const step = acc
      acc = 0
      const { shot: s, focus: f } = live.current
      const cut = s !== lastShot
      if (cut) { lastShot = s; shotStart = now }
      const t = (now - shotStart) / 1000
      if (!still) az += step * ({ wide: 0.1, dolly: 0.05, hero: 0.06, sweep: 0.08, final: 0.28 }[s] || 0.1)

      let dist = R * 2.6, elev = 0.32
      aim.copy(centre)
      if (s === 'dolly' && f) {
        aim.set(...f.p)
        const r = Math.min(R * 0.6, Math.max(R * 0.08, f.r || R * 0.12))
        dist = r * 4.2 * (1.35 - 0.35 * Math.min(1, t / 8))
        elev = 0.38
      } else if (s === 'hero') {
        aim.z = box.min.z + (box.max.z - box.min.z) * 0.55
        dist = R * (2.1 - 0.4 * Math.min(1, t / 12)); elev = -0.05
      } else if (s === 'sweep') {
        dist = R * 2.5; elev = 0.95
      } else if (s === 'final') {
        dist = R * 2.3; elev = 0.22
      }
      dist *= Math.max(1, 1.3 / camera.aspect)          // a tall frame needs the camera further back to keep the part in
      pos.set(Math.cos(az) * Math.cos(elev), Math.sin(az) * Math.cos(elev), Math.sin(elev)).multiplyScalar(dist).add(aim)
      if (cut && !still && s !== 'dolly') { camera.position.copy(pos); look.copy(aim) }       // a cut, not a move
      else {
        const k = 1 - Math.exp(-step * (still ? 6 : 1.8))
        camera.position.lerp(pos, k)
        look.lerp(aim, k)
      }
      camera.lookAt(look)
      key.position.copy(camera.position).add(tmp.set(-R, -R * 0.2, R * 1.5))
      rim.position.copy(look).sub(tmp.copy(camera.position).sub(look)).add(new THREE.Vector3(0, 0, R))

      // the band: up the part while it builds, through it while the validator checks
      const span = box.max.z - box.min.z
      u.uBandOn.value = s === 'sweep' || s === 'hero' ? 1 : 0
      u.uBandW.value = span * 0.025
      u.uBandColor.value.set(...(BAND[s] || BAND.sweep))
      u.uBand.value = s === 'hero' ? box.min.z + span * ((t / 6) % 1) : box.min.z + span * (0.5 + 0.5 * Math.sin(t * 1.3))
      u.uFocusOn.value = s === 'dolly' && f ? 1 : 0
      if (f) { u.uFocus.value.set(...f.p); u.uFocusR.value = Math.min(R * 0.6, Math.max(R * 0.04, f.r || R * 0.1)) * (1 + 0.08 * Math.sin(now / 180)) }

      renderer.render(scene, camera)
      // where the overlays sit on screen this frame
      if (++frame % 2 === 0 && live.current.onLabels) {
        const w = el.clientWidth, h = el.clientHeight
        const out = []
        for (const o of extras.children) {
          if (!o.userData.label) continue
          tmp.copy(o.userData.anchor).project(camera)
          if (tmp.z < 1) out.push({ id: o.userData.id, label: o.userData.label, kind: o.userData.kind, x: (tmp.x + 1) / 2 * w, y: (1 - tmp.y) / 2 * h })
        }
        live.current.onLabels(out)
      }
    }
    raf = requestAnimationFrame(tick)
    live.current.extras = extras
    live.current.R = () => R

    return () => {
      alive = false
      cancelAnimationFrame(raf)
      ro.disconnect()
      scene.traverse((o) => { o.geometry?.dispose(); if (o.material && o.material !== material) o.material.dispose?.() })
      material.dispose()
      renderer.dispose()
      renderer.domElement.remove()
    }
  }, [stl])

  live.current.onLabels = onLabels

  // the named bodies as coloured boxes, the validator's worst places as red markers
  useEffect(() => {
    const g = live.current.extras
    if (!g || state !== 'ready') return
    for (const o of [...g.children]) { g.remove(o); o.geometry?.dispose(); o.material?.dispose() }
    ;(bodies?.bodies || []).forEach((b, i) => {
      const bx = new THREE.Box3(new THREE.Vector3(...b.lo), new THREE.Vector3(...b.hi))
      const helper = new THREE.Box3Helper(bx, new THREE.Color().setHSL(((i * 67) % 360) / 360, 0.7, 0.55))
      helper.userData = { id: `b${i}`, kind: 'body', label: b.name, anchor: new THREE.Vector3(b.lo[0], b.lo[1], b.hi[2]) }
      g.add(helper)
    })
    const R = live.current.R()
    ;(clusters || []).slice(0, 6).forEach((c, i) => {
      if (!vecOk(c.center)) return
      const m = new THREE.Mesh(new THREE.SphereGeometry(R * 0.018, 16, 12), new THREE.MeshBasicMaterial({ color: 0xe63b2e }))
      m.position.set(...c.center)
      m.userData = { id: `c${i}`, kind: 'cluster', label: `${c.kind} · ${c.max} mm`, anchor: m.position.clone() }
      g.add(m)
    })
  }, [bodies, clusters, state])

  return (
    <div className="director" ref={host}>
      {state !== 'ready' && <div className="director-wait">{state === 'loading' ? 'LOADING THE SCAN…' : 'NO MESH TO FILM'}</div>}
    </div>
  )
}

const vecOk = (v) => Array.isArray(v) && v.length === 3 && v.every(Number.isFinite)
