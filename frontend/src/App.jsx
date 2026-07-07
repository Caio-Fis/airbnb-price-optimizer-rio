import { useCallback, useState } from "react";
import MapView from "./components/MapView.jsx";
import ListingPanel from "./components/ListingPanel.jsx";
import NewListingForm from "./components/NewListingForm.jsx";

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

      {mode === "map" && selected && (
        <ListingPanel key={selected.id} listing={selected} onClose={() => setSelected(null)} />
      )}
      {mode === "novo" && (
        <NewListingForm onLocate={setCustomPoint} onClose={() => setMode("map")} />
      )}
    </div>
  );
}
