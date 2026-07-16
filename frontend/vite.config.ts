import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

// Builds the SPA directly into ../app/static so FastAPI serves it.
// base:'./' => relative asset URLs, so it works when served from "/".
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': path.resolve(__dirname, './src') } },
  base: './',
  build: { outDir: '../app/static', emptyOutDir: true },
})
