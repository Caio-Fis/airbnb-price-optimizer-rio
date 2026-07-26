import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import { brl } from "../api.js";

const STRATEGY_LABELS = {
  revenue_optimal: "Ótimo de receita",
  premium_positioning: "Posicionamento premium",
  fallback: "Mediana local",
};

function revenueCurve(result) {
  const predicted = result.predicted_price;
  const optimal = result.revenue_optimal_price || predicted;
  const occ = result.expected_occupancy_pct || 0;
  const low = (result.price_range_low || predicted * 0.85) * 0.5;
  // o eixo precisa cobrir o preço ótimo, que pode ficar bem acima do intervalo de mercado
  const high = Math.max((result.price_range_high || predicted * 1.15) * 1.5, optimal * 1.35);

  const points = [];
  for (let i = 0; i < 120; i++) {
    const price = low + ((high - low) * i) / 119;
    let occupancy = 0.4;
    if (occ > 0 && optimal > 0 && predicted > 0) {
      const b = -2.0;
      const a = -b * Math.log(optimal / predicted);
      occupancy = 1 / (1 + Math.exp(-(a + b * Math.log(price / predicted))));
    }
    points.push({ price: Math.round(price), revenue: Math.round(price * occupancy) });
  }
  return points;
}

function PriceBars({ result }) {
  const rows = [
    { name: "Mediana local", value: result.local_median_price, color: "#898781" },
    { name: "Mercado", value: result.predicted_price, color: "#3987e5" },
    { name: "Ótimo", value: result.revenue_optimal_price, color: "#199e70" },
  ].filter((r) => r.value != null);
  const max = Math.max(...rows.map((r) => r.value));

  return (
    <div className="price-bars">
      {rows.map((r) => (
        <div className="price-bar" key={r.name}>
          <span className="name">{r.name}</span>
          <div className="track">
            <div
              className="fill"
              style={{ width: `${(100 * r.value) / max}%`, background: r.color }}
            />
          </div>
          <span className="val">{brl.format(r.value)}</span>
        </div>
      ))}
    </div>
  );
}

export default function Results({ result }) {
  const predicted = result.predicted_price;
  // Quando a estratégia é "fallback", a API não conseguiu estimar a curva de
  // demanda do segmento e devolve revenue_optimal_price/expected_occupancy nulos.
  // Nesse caso não existe preço ótimo — anunciar o preço de mercado como "ótimo"
  // (o antigo `|| predicted`) mostrava um delta de R$ 0,00 e uma curva inventada.
  const hasOptimal = result.revenue_optimal_price != null;
  const optimal = hasOptimal ? result.revenue_optimal_price : predicted;
  const delta = optimal - predicted;
  const curve = hasOptimal ? revenueCurve(result) : null;

  return (
    <div className="results">
      <div className="kpi-grid">
        {hasOptimal ? (
          <div className="kpi hero">
            <div className="label">Preço ótimo de receita</div>
            <div className="value" style={{ color: "#199e70" }}>{brl.format(optimal)}</div>
            <div className={`delta ${delta >= 0 ? "up" : ""}`}>
              {delta >= 0 ? "+" : "−"}{brl.format(Math.abs(delta))} vs mercado
              {result.seasonal_note ? ` · ${result.seasonal_note}` : ""}
            </div>
          </div>
        ) : (
          <div className="kpi hero">
            <div className="label">Preço de mercado</div>
            <div className="value">{brl.format(predicted)}</div>
            <div className="delta">
              sem dados de demanda suficientes neste segmento para calcular o ótimo
              {result.seasonal_note ? ` · ${result.seasonal_note}` : ""}
            </div>
          </div>
        )}

        {hasOptimal && (
          <div className="kpi">
            <div className="label">Preço de mercado</div>
            <div className="value">{brl.format(predicted)}</div>
            <div className="sub">o que imóveis similares cobram</div>
          </div>
        )}

        <div className="kpi">
          <div className="label">Ocupação esperada</div>
          <div className="value">
            {result.expected_occupancy_pct ? `${Math.round(result.expected_occupancy_pct)}%` : "—"}
          </div>
          <div className="sub">no preço ótimo</div>
        </div>

        <div className="kpi">
          <div className="label">Estratégia</div>
          <div className="value" style={{ fontSize: 15 }}>
            {STRATEGY_LABELS[result.pricing_strategy] || "—"}
          </div>
        </div>

        <div className="kpi">
          <div className="label">Intervalo de mercado</div>
          <div className="value" style={{ fontSize: 15 }}>
            {brl.format(result.price_range_low)} – {brl.format(result.price_range_high)}
          </div>
          <div className="sub">P10–P90 dos resíduos do modelo</div>
        </div>

        {result.seasonal_multiplier != null && (
          <div className="kpi">
            <div className="label">Ajuste sazonal</div>
            <div className="value" style={{ fontSize: 15 }}>
              {result.seasonal_multiplier.toFixed(2)}×
            </div>
            <div className="sub">{result.seasonal_note || "dia da semana × evento"}</div>
          </div>
        )}
      </div>

      <div className="chart-block">
        <h3>Comparativo de preços (R$/noite)</h3>
        <PriceBars result={result} />
      </div>

      {curve && (
      <div className="chart-block">
        <h3>Curva de receita estimada</h3>
        <ResponsiveContainer width="100%" height={190}>
          <AreaChart data={curve} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#2c2c2a" vertical={false} />
            <XAxis
              dataKey="price"
              tickFormatter={(v) => `R$${v}`}
              stroke="#383835"
              tick={{ fill: "#898781", fontSize: 11 }}
              tickLine={false}
            />
            <YAxis
              tickFormatter={(v) => `R$${v}`}
              stroke="#383835"
              tick={{ fill: "#898781", fontSize: 11 }}
              tickLine={false}
              width={54}
            />
            <Tooltip
              formatter={(value) => [brl.format(value), "Receita esperada"]}
              labelFormatter={(label) => `Preço: ${brl.format(label)}`}
              contentStyle={{
                background: "#1a1a19",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: 8,
                color: "#fff",
                fontSize: 12,
              }}
            />
            <Area
              type="monotone"
              dataKey="revenue"
              stroke="#3987e5"
              strokeWidth={2}
              fill="#3987e5"
              fillOpacity={0.12}
              dot={false}
              activeDot={{ r: 4 }}
            />
            <ReferenceLine
              x={curve.reduce((best, p) => (p.revenue > best.revenue ? p : best)).price}
              stroke="#199e70"
              strokeDasharray="5 4"
              label={{
                value: `Ótimo ${brl.format(optimal)}`,
                fill: "#199e70",
                fontSize: 11,
                position: "insideTopRight",
              }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      )}
    </div>
  );
}
