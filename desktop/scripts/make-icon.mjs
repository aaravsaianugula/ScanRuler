// SPDX-License-Identifier: AGPL-3.0-only
// The app icon: the Bauhaus mark (circle, triangle, square) on a black tile, drawn from SVG and
// rasterised headless with the puppeteer-core the ScanRuler checkout already has.
//   node scripts/make-icon.mjs   ->  build/icon.png (512 px)
import { existsSync, mkdirSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'

const require = createRequire(new URL('../../package.json', import.meta.url))
const puppeteer = require('puppeteer-core')
const browser = ['C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', 'C:/Program Files/Google/Chrome/Application/chrome.exe', process.env.CHROME].find((p) => p && existsSync(p))

const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  <rect x="16" y="16" width="480" height="480" rx="72" fill="#0d0d0f" stroke="#000" stroke-width="12"/>
  <rect x="40" y="40" width="432" height="432" rx="54" fill="none" stroke="#f2ede4" stroke-width="6" stroke-dasharray="2 14"/>
  <circle cx="186" cy="196" r="104" fill="#e63b2e" stroke="#000" stroke-width="12"/>
  <polygon points="300,120 440,370 160,370" fill="#f5c518" stroke="#000" stroke-width="12" stroke-linejoin="round"/>
  <rect x="96" y="300" width="132" height="132" fill="#1f4fa8" stroke="#000" stroke-width="12"/>
  <rect x="300" y="404" width="140" height="22" fill="#f2ede4"/>
</svg>`

const b = await puppeteer.launch({ executablePath: browser, headless: 'new' })
const page = await b.newPage()
await page.setViewport({ width: 512, height: 512 })
await page.setContent(`<html><body style="margin:0;background:transparent">${svg}</body></html>`)
const out = fileURLToPath(new URL('../build/', import.meta.url))
mkdirSync(out, { recursive: true })
writeFileSync(out + 'icon.png', await page.screenshot({ omitBackground: true, clip: { x: 0, y: 0, width: 512, height: 512 } }))
writeFileSync(out + 'icon.svg', svg)
await b.close()
console.log('wrote build/icon.png')
