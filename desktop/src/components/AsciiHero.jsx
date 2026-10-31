import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { AsciiEffect } from 'three/examples/jsm/effects/AsciiEffect.js'

// A part on a turntable, rendered in ASCII: a plate with a boss, a bore and two holes — the bench
// bracket — built from primitives, lit hard so the characters carry the shading.
function bracket() {
  const g = new THREE.Group()
  const mat = new THREE.MeshPhongMaterial({ color: 0xffffff, flatShading: true, shininess: 40 })
  const plate = new THREE.Shape()
  const r = 3, w = 60, h = 40
  plate.moveTo(-w / 2 + r, -h / 2)
  plate.lineTo(w / 2 - r, -h / 2); plate.absarc(w / 2 - r, -h / 2 + r, r, -Math.PI / 2, 0)
  plate.lineTo(w / 2, h / 2 - r); plate.absarc(w / 2 - r, h / 2 - r, r, 0, Math.PI / 2)
  plate.lineTo(-w / 2 + r, h / 2); plate.absarc(-w / 2 + r, h / 2 - r, r, Math.PI / 2, Math.PI)
  plate.lineTo(-w / 2, -h / 2 + r); plate.absarc(-w / 2 + r, -h / 2 + r, r, Math.PI, Math.PI * 1.5)
  for (const x of [-15, 15]) { const p = new THREE.Path(); p.absarc(x, 0, 4, 0, Math.PI * 2, true); plate.holes.push(p) }
  const bore = new THREE.Path(); bore.absarc(0, 0, 3, 0, Math.PI * 2, true); plate.holes.push(bore)
  const base = new THREE.Mesh(new THREE.ExtrudeGeometry(plate, { depth: 10, bevelEnabled: false, curveSegments: 24 }), mat)
  g.add(base)
  const ring = new THREE.Shape(); ring.absarc(0, 0, 8, 0, Math.PI * 2)
  const hole = new THREE.Path(); hole.absarc(0, 0, 3, 0, Math.PI * 2, true); ring.holes.push(hole)
  const boss = new THREE.Mesh(new THREE.ExtrudeGeometry(ring, { depth: 11, bevelEnabled: true, bevelThickness: 1, bevelSize: -1, bevelSegments: 1, curveSegments: 32 }), mat)
  boss.position.z = 10
  g.add(boss)
  g.rotation.x = -Math.PI / 2.6
  return g
}

export default function AsciiHero({ chars = ' .:-=+*#%@', resolution = 0.16 }) {
  const host = useRef(null)
  useEffect(() => {
    const el = host.current
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(40, 1, 1, 1000)
    camera.position.set(0, 18, 95)
    camera.lookAt(0, 4, 0)
    scene.add(new THREE.AmbientLight(0xffffff, 0.08))
    const key = new THREE.DirectionalLight(0xffffff, 1.5); key.position.set(60, 80, 60); scene.add(key)
    const rim = new THREE.DirectionalLight(0xffffff, 0.8); rim.position.set(-80, -20, -40); scene.add(rim)
    const part = bracket()
    const spin = new THREE.Group(); spin.add(part); scene.add(spin)
    const renderer = new THREE.WebGLRenderer({ antialias: false })
    const effect = new AsciiEffect(renderer, chars, { invert: true, resolution })
    effect.domElement.className = 'ascii-hero-out'
    el.appendChild(effect.domElement)
    const fit = () => {
      const { clientWidth: w, clientHeight: h } = el
      if (!w || !h) return
      camera.aspect = w / Math.max(h, 1); camera.updateProjectionMatrix()
      effect.setSize(w, h)
    }
    fit()
    const ro = new ResizeObserver(fit); ro.observe(el)
    const still = matchMedia('(prefers-reduced-motion: reduce)').matches
    let raf = 0
    const t0 = performance.now()
    const tick = (now) => {
      raf = requestAnimationFrame(tick)
      if (!el.clientWidth || !el.clientHeight) return      // not laid out yet: the effect cannot draw at 0 px
      const t = (now - t0) / 1000
      spin.rotation.y = still ? 0.6 : t * 0.45
      spin.position.y = still ? 0 : Math.sin(t * 1.3) * 1.5
      effect.render(scene, camera)
    }
    raf = requestAnimationFrame(tick)
    return () => { cancelAnimationFrame(raf); ro.disconnect(); renderer.dispose(); el.replaceChildren() }
  }, [chars, resolution])
  return <div className="ascii-hero" ref={host} aria-label="A turning bracket drawn in ASCII characters" role="img" />
}
