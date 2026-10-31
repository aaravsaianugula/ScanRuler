// SPDX-License-Identifier: AGPL-3.0-only
// Bake the ScanRuler checkout this app was built from into the app, as the default for Settings.
const fs = require('node:fs')
const path = require('node:path')
const repoDir = path.resolve(__dirname, '..', '..')
fs.writeFileSync(path.join(__dirname, '..', 'electron', 'defaults.json'), JSON.stringify({ repoDir }, null, 1))
console.log('defaults: repoDir =', repoDir)
