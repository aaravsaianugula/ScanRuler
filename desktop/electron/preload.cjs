// SPDX-License-Identifier: AGPL-3.0-only
const { contextBridge, ipcRenderer, webUtils } = require('electron')

const on = (channel) => (cb) => {
  const h = (_e, payload) => cb(payload)
  ipcRenderer.on(channel, h)
  return () => ipcRenderer.removeListener(channel, h)
}

contextBridge.exposeInMainWorld('studio', {
  start: (opts) => ipcRenderer.invoke('run:start', opts),
  stop: () => ipcRenderer.invoke('run:stop'),
  onEvent: on('run:event'),
  onExit: on('run:exit'),
  openStl: () => ipcRenderer.invoke('dialog:open-stl'),
  pick: (kind) => ipcRenderer.invoke('dialog:pick', kind),
  pathForFile: (file) => webUtils.getPathForFile(file),
  show: (file) => ipcRenderer.invoke('shell:show', file),
  open: (file) => ipcRenderer.invoke('shell:open', file),
  readText: (file, work) => ipcRenderer.invoke('fs:text', file, work),
  readImage: (file, work) => ipcRenderer.invoke('fs:image', file, work),
  readMesh: (file) => ipcRenderer.invoke('fs:mesh', file),
  settings: () => ipcRenderer.invoke('settings:get'),
  saveSettings: (s) => ipcRenderer.invoke('settings:set', s),
  health: () => ipcRenderer.invoke('health'),
})
