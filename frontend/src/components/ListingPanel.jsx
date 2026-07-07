import { useState } from "react";
import { predictListing, brl } from "../api.js";
import Results from "./Results.jsx";

const ROOM_TYPE_PT = {
  "Entire home/apt": "Imóvel inteiro",
  "Private room": "Quarto privativo",
  "Shared room": "Quarto compartilhado",
  "Hotel room": "Quarto de hotel",
};

export default function ListingPanel({ listing, onClose }) {
  const [targetDate, setTargetDate] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  async function calculate() {
    setLoading(true);
    setError(null);
    try {
      setResult(await predictListing(listing.id, targetDate || undefined));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <aside className="panel">
      <button className="close" onClick={onClose} aria-label="Fechar">✕</button>
      <h2>{listing.name}</h2>
      <div className="meta-row">
        <span className="chip"><strong>{listing.nb}</strong></span>
        <span className="chip">{ROOM_TYPE_PT[listing.rt] || listing.rt}</span>
        <span className="chip">{listing.acc} hóspedes</span>
        {listing.price != null && (
          <span className="chip" title="Preço anunciado no snapshot do Inside Airbnb (Set/2025)">
            Set/2025: <strong>{brl.format(listing.price)}</strong>/noite
          </span>
        )}
      </div>

      <label className="field">
        <span>Data da estadia (ajusta a sazonalidade)</span>
        <input
          type="date"
          value={targetDate}
          onChange={(e) => setTargetDate(e.target.value)}
        />
      </label>

      <button className="btn-primary" onClick={calculate} disabled={loading}>
        {loading ? "Calculando…" : "Calcular preço otimizado"}
      </button>

      {error && <div className="error-box">{error}</div>}
      {result && <Results result={result} />}
    </aside>
  );
}
