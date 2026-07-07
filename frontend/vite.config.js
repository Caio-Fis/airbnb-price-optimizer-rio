import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Em dev, a API roda em localhost:8000 (uvicorn); em produção o build é
// servido pela própria FastAPI (mesma origem), então só caminhos relativos.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/predict": "http://localhost:8000",
      "/listings": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
});
