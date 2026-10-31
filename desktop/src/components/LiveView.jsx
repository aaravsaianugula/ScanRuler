import { BorderBeam } from 'border-beam'
import { useEffect, useState } from 'react'

const NOISE = '░▒▓█▀▄■□▪▫◘◙'

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

export default function LiveView({ url, active, image, work, phaseLabel }) {
  const [pic, setPic] = useState(null)
  useEffect(() => {
    if (!image || !work) return
    window.studio.readImage(image, work).then(setPic)
  }, [image, work])

  return (
    <section className="panel live" aria-label="Live view of ScanRuler">
      <header className="panel-head blue">
        <span className="glyph">■</span> LIVE // SCANRULER
        <span className="head-right">{url ? <><i className="rec" /> REC</> : 'NO SIGNAL'}</span>
      </header>
      <BorderBeam className="beam" size="md" colorVariant="ocean" theme="dark" active={!!active} duration={2.4} strength={0.9}>
        <div className="live-frame">
          {url ? (
            <webview className="live-webview" src={url} partition="persist:scanruler" />
          ) : (
            <div className="awaiting">
              <Static />
              <div className="awaiting-label"><span>AWAITING SIGNAL</span><small>{phaseLabel || 'start a run to link ScanRuler'}</small></div>
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
