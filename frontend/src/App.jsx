import { Suspense, lazy, useCallback, useState } from "react";
import MapView from "./components/MapView.jsx";

// Os dois painéis só aparecem depois de uma interação (clique no mapa ou aba
// "Novo imóvel") e ambos puxam o Recharts, que sozinho é ~40% do bundle.
// Carregar sob demanda tira o Recharts do first paint — o mapa é o que importa
// na abertura. MapView fica estático: é a tela inicial.
const ListingPanel = lazy(() => import("./components/ListingPanel.jsx"));
const NewListingForm = lazy(() => import("./components/NewListingForm.jsx"));

export default function App() {
  const [mode, setMode] = useState("map"); // "map" | "novo"
  const [selected, setSelected] = useState(null); // properties do listing clicado
  const [customPoint, setCustomPoint] = useState(null); // {lat, lon} do form

  const handleSelect = useCallback((props) => {
    setMode("map");
    setSelected(props);
  }, []);

  return (
    <div className="app">
      <MapView
        onSelect={handleSelect}
        selectedId={selected?.id ?? null}
        customPoint={mode === "novo" ? customPoint : null}
      />

      <header className="header">
        <div>
          <h1>Otimizador de Preço — Airbnb Rio</h1>
          <div className="subtitle">Clique numa acomodação para calcular o preço ótimo</div>
        </div>
        <nav className="mode-toggle" aria-label="Modo">
          <button className={mode === "map" ? "active" : ""} onClick={() => setMode("map")}>
            Mapa
          </button>
          <button className={mode === "novo" ? "active" : ""} onClick={() => setMode("novo")}>
            Novo imóvel
          </button>
        </nav>
      </header>

      <Suspense fallback={<div className="panel-loading">Carregando…</div>}>
        {mode === "map" && selected && (
          <ListingPanel key={selected.id} listing={selected} onClose={() => setSelected(null)} />
        )}
        {mode === "novo" && (
          <NewListingForm onLocate={setCustomPoint} onClose={() => setMode("map")} />
        )}
      </Suspense>
    </div>
  );
}
