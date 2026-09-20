import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// In development the UI runs on :5173 and proxies /api to the FastAPI server,
// so the browser sees a single origin and SSE streaming works without CORS.
// In production the built bundle is either served by FastAPI itself or by a
// static host, with VITE_API_BASE pointing at the API.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
