import { useState } from "react";
import { predictNew } from "../api.js";
import { geocode, snapNeighbourhood } from "../geo.js";
import Results from "./Results.jsx";

const ROOM_TYPES = [
  ["Entire home/apt", "Imóvel inteiro"],
  ["Private room", "Quarto privativo"],
  ["Shared room", "Quarto compartilhado"],
  ["Hotel room", "Quarto de hotel"],
];

const AMENITIES = [
  ["wifi", "Wi-Fi"],
  ["kitchen", "Cozinha"],
  ["air_conditioning", "Ar-condicionado"],
  ["heating", "Aquecimento"],
  ["washer", "Máquina de lavar"],
  ["dryer", "Secadora"],
  ["pool", "Piscina"],
  ["gym", "Academia"],
  ["elevator", "Elevador"],
  ["parking", "Estacionamento"],
  ["breakfast", "Café da manhã"],
  ["workspace", "Espaço de trabalho"],
  ["tv", "TV"],
];

function Stepper({ label, value, onChange, min = 0, max = 16 }) {
  return (
    <label className="field">
      <span>{label}</span>
      <div className="stepper">
        <button type="button" onClick={() => onChange(Math.max(min, value - 1))}>−</button>
        <output>{value}</output>
        <button type="button" onClick={() => onChange(Math.min(max, value + 1))}>+</button>
      </div>
    </label>
  );
}

export default function NewListingForm({ onLocate, onClose }) {
  const [address, setAddress] = useState("");
  const [roomType, setRoomType] = useState("Entire home/apt");
  const [accommodates, setAccommodates] = useState(4);
  const [bedrooms, setBedrooms] = useState(2);
  const [bathrooms, setBathrooms] = useState(1);
  const [hasLavabo, setHasLavabo] = useState(false);
  const [beds, setBeds] = useState(2);
  const [superhost, setSuperhost] = useState(false);
  const [instant, setInstant] = useState(false);
  const [amenities, setAmenities] = useState(["wifi", "kitchen", "air_conditioning"]);
  const [targetDate, setTargetDate] = useState("");

  const [neighbourhood, setNeighbourhood] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  function toggleAmenity(key) {
    setAmenities((prev) =>
      prev.includes(key) ? prev.filter((a) => a !== key) : [...prev, key]
    );
  }

  async function submit(e) {
    e.preventDefault();
    if (!address.trim()) {
      setError("Preencha o endereço do imóvel.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const { lat, lon } = await geocode(address);
      const nb = snapNeighbourhood(lat, lon);
      setNeighbourhood(nb);
      onLocate({ lat, lon });

      const payload = {
        neighbourhood: nb,
        room_type: roomType,
        accommodates,
        bathrooms: bathrooms + (hasLavabo ? 0.5 : 0),
        bedrooms,
        beds,
        host_is_superhost: superhost,
        instant_bookable: instant,
        amenities,
        latitude: lat,
        longitude: lon,
      };
      if (targetDate) payload.target_date = targetDate;

      setResult(await predictNew(payload));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <aside className="panel">
      <button className="close" onClick={onClose} aria-label="Fechar">✕</button>
      <h2>Novo imóvel</h2>
      <p className="info-note" style={{ marginTop: 0 }}>
        Estime o preço ideal de um imóvel que ainda não está no Airbnb.
      </p>

      <form onSubmit={submit}>
        <label className="field">
          <span>Endereço no Rio de Janeiro</span>
          <input
            type="text"
            value={address}
            onChange={(e) => setAddress(e.target.value)}
            placeholder="Av. Atlântica, 1500, Copacabana"
          />
        </label>

        <label className="field">
          <span>Tipo de imóvel</span>
          <select value={roomType} onChange={(e) => setRoomType(e.target.value)}>
            {ROOM_TYPES.map(([value, label]) => (
              <option key={value} value={value}>{label}</option>
            ))}
          </select>
        </label>

        <div className="field-grid">
          <Stepper label="Hóspedes" value={accommodates} onChange={setAccommodates} min={1} max={16} />
          <Stepper label="Quartos" value={bedrooms} onChange={setBedrooms} min={0} max={10} />
          <Stepper label="Banheiros" value={bathrooms} onChange={setBathrooms} min={1} max={8} />
          <Stepper label="Camas" value={beds} onChange={setBeds} min={1} max={16} />
        </div>

        <label className="check-row">
          <input type="checkbox" checked={hasLavabo} onChange={(e) => setHasLavabo(e.target.checked)} />
          Tem lavabo (banheiro parcial)
        </label>
        <label className="check-row">
          <input type="checkbox" checked={superhost} onChange={(e) => setSuperhost(e.target.checked)} />
          Superanfitrião
        </label>
        <label className="check-row">
          <input type="checkbox" checked={instant} onChange={(e) => setInstant(e.target.checked)} />
          Reserva instantânea
        </label>

        <label className="field" style={{ marginTop: 10 }}>
          <span>Comodidades</span>
          <div className="amenities-grid">
            {AMENITIES.map(([key, label]) => (
              <label className="check-row" key={key} style={{ margin: 0 }}>
                <input
                  type="checkbox"
                  checked={amenities.includes(key)}
                  onChange={() => toggleAmenity(key)}
                />
                {label}
              </label>
            ))}
          </div>
        </label>

        <label className="field">
          <span>Data da estadia (opcional)</span>
          <input type="date" value={targetDate} onChange={(e) => setTargetDate(e.target.value)} />
        </label>

        <button className="btn-primary" type="submit" disabled={loading}>
          {loading ? "Calculando…" : "Calcular preço otimizado"}
        </button>
      </form>

      {neighbourhood && !error && (
        <p className="info-note">Bairro detectado: <strong>{neighbourhood}</strong></p>
      )}
      {error && <div className="error-box">{error}</div>}
      {result && <Results result={result} />}
    </aside>
  );
}
