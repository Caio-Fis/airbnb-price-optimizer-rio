// Geocoding (Nominatim/OSM) e snap de bairro — mesma lógica do app antigo.

export const NEIGHBOURHOOD_COORDS = {
  "Barra da Tijuca": [-23.0058, -43.3498],
  "Barra de Guaratiba": [-23.0536, -43.5602],
  Botafogo: [-22.9504, -43.1841],
  Camorim: [-22.9798, -43.4196],
  Catete: [-22.9263, -43.1789],
  Centro: [-22.9092, -43.1824],
  Copacabana: [-22.9724, -43.1861],
  Flamengo: [-22.9324, -43.1765],
  Glória: [-22.9219, -43.1757],
  Guaratiba: [-23.0307, -43.5785],
  Gávea: [-22.9759, -43.2272],
  Humaitá: [-22.9565, -43.1941],
  Ipanema: [-22.9836, -43.2034],
  Itanhangá: [-22.9966, -43.3445],
  Jacarepaguá: [-22.9523, -43.3734],
  "Jardim Botânico": [-22.9654, -43.2234],
  Lagoa: [-22.9722, -43.2153],
  Laranjeiras: [-22.9353, -43.1875],
  Leblon: [-22.9847, -43.2228],
  Leme: [-22.9618, -43.1684],
  "Recreio dos Bandeirantes": [-23.0222, -43.4733],
  "Santa Teresa": [-22.9205, -43.1869],
  "Santo Cristo": [-22.9025, -43.2062],
  "São Conrado": [-22.9966, -43.2599],
  Taquara: [-22.9242, -43.3797],
  Tijuca: [-22.9268, -43.2318],
  Urca: [-22.9494, -43.1656],
  "Vargem Grande": [-22.988, -43.498],
  "Vargem Pequena": [-22.9877, -43.4584],
  Vidigal: [-22.9947, -43.2376],
};

function haversineKm(lat1, lon1, lat2, lon2) {
  const R = 6371;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.asin(Math.sqrt(a));
}

export function snapNeighbourhood(lat, lon) {
  let best = null;
  let bestDist = Infinity;
  for (const [name, [nlat, nlon]] of Object.entries(NEIGHBOURHOOD_COORDS)) {
    const d = haversineKm(lat, lon, nlat, nlon);
    if (d < bestDist) {
      bestDist = d;
      best = name;
    }
  }
  return best;
}

// bbox do Rio: viewbox=lon1,lat1,lon2,lat2 (esq, cima, dir, baixo)
const RJ_VIEWBOX = "-43.8,-22.7,-42.9,-23.1";

export async function geocode(address) {
  const url =
    "https://nominatim.openstreetmap.org/search?" +
    new URLSearchParams({
      q: `${address}, Rio de Janeiro, Brasil`,
      format: "json",
      limit: "1",
      viewbox: RJ_VIEWBOX,
      bounded: "1",
    });
  const resp = await fetch(url, { headers: { Accept: "application/json" } });
  if (!resp.ok) throw new Error("Serviço de geocoding indisponível. Tente novamente.");
  const results = await resp.json();
  if (!results.length) {
    throw new Error(
      "Endereço não encontrado no Rio de Janeiro. Inclua rua e bairro (ex: Av. Atlântica, 1500, Copacabana)."
    );
  }
  return { lat: parseFloat(results[0].lat), lon: parseFloat(results[0].lon) };
}
