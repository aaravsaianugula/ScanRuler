import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// built into dist/ and loaded from file:// by electron/main.cjs
export default defineConfig({ base: './', plugins: [react()], build: { outDir: 'dist', emptyOutDir: true, chunkSizeWarningLimit: 1500 } })
