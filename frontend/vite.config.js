import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Em dev, a API roda em localhost:8000 (uvicorn); em produção o build é
// servido pela própria FastAPI (mesma origem), então só caminhos relativos.
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        // Vendors em chunks próprios: mudam muito menos que o código do app,
        // então o navegador reaproveita o cache entre deploys. O Recharts já sai
        // do chunk inicial via lazy import dos painéis (ver App.jsx).
        manualChunks: {
          maplibre: ["maplibre-gl"],
          react: ["react", "react-dom"],
        },
      },
    },
  },
  server: {
    proxy: {
      "/predict": "http://localhost:8000",
      "/listings": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
});
