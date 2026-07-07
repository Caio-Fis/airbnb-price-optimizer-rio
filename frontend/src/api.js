// Cliente da API — caminhos relativos (mesma origem em produção, proxy do Vite em dev).

async function post(path, body) {
  const resp = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
  });
  if (!resp.ok) {
    const detail = await resp.json().then((d) => d.detail).catch(() => null);
    throw new Error(detail || `Erro ${resp.status} ao consultar a API`);
  }
  return resp.json();
}

// listing_id é serializado manualmente: ids do Airbnb passam de 2^53 e
// Number(id) corromperia o valor. A string é injetada crua no JSON — o
// backend (Python) lê o inteiro sem perda.
export function predictListing(idStr, targetDate) {
  const payload = { listing_id: "__ID__" };
  if (targetDate) payload.target_date = targetDate;
  const body = JSON.stringify(payload).replace('"__ID__"', idStr);
  return post("/predict/listing", body);
}

export function predictNew(payload) {
  return post("/predict", JSON.stringify(payload));
}

export async function fetchListings() {
  const resp = await fetch("/listings/map");
  if (!resp.ok) throw new Error("Não foi possível carregar os listings do mapa");
  return resp.json();
}

export const brl = new Intl.NumberFormat("pt-BR", {
  style: "currency",
  currency: "BRL",
  maximumFractionDigits: 0,
});
