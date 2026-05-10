import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const BACKEND = 'http://165.245.137.57:7860'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // strip /gradio prefix so local dev talks to plain Gradio server
      '/gradio': {
        target: BACKEND,
        changeOrigin: true,
        ws: true,
        rewrite: (path) => path.replace(/^\/gradio/, ''),
      },
    },
  },
})
