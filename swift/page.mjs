// SPDX-License-Identifier: AGPL-3.0-only
// Open the ScanRuler page for the Swift agent: the local build the MCP server
// serves, paired through its link, in Edge or Chrome. Stays open until stdin
// closes (the Python side holds it). The same first-run answers as the e2e
// harness (scripts/e2e-lib.mjs), so nothing stalls on a dialog.
//   node swift/page.mjs <pairing-url> [--headless]
import puppeteer from 'puppeteer-core'
import { existsSync } from 'node:fs'

const url = process.argv[2]
const headless = process.argv.includes('--headless') ? 'new' : false
const browsers = [process.env.CHROME,
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', 'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome', '/usr/bin/chromium']
const found = browsers.filter((p) => p && existsSync(p))
if (!found.length) { console.error('no Edge or Chrome found: set CHROME'); process.exit(2) }

// the next one when a browser will not start (an Edge build that exits at once, headless, with code 0)
let browser
for (const executablePath of found) {
  try {
    browser = await puppeteer.launch({ executablePath, headless, defaultViewport: null, protocolTimeout: 0,
      args: ['--window-size=1500,950', '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows'] })
    break
  } catch (e) { console.error(`${executablePath} did not start: ${e.message.split('\n')[0]}`) }
}
if (!browser) { console.error('no browser started: set CHROME to a working Chrome or Edge'); process.exit(2) }
const [page] = await browser.pages()
await page.evaluateOnNewDocument(() => {
  for (const [k, v] of [['scanruler.askstlunits', '0'], ['scanruler.beta', 'all'], ['scanruler.supportask', '0'], ['scanruler.agentdataok', '1']]) {
    try { localStorage.setItem(k, v) } catch { /* private mode */ }
  }
})
page.on('pageerror', (e) => console.error('pageerror:', e.message))
await page.goto(url, { waitUntil: 'networkidle0' })
console.log('page open')
process.stdin.resume()
process.stdin.on('end', async () => { await browser.close(); process.exit(0) })
