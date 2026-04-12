import math
import re
import requests
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from geopy.point import Point

API_URL = "https://airbnb-price-api-966533570956.us-central1.run.app"

_geolocator = Nominatim(user_agent="airbnb-rj-optimizer", timeout=8)
_RJ_VIEWBOX = [Point(-23.1, -43.8), Point(-22.7, -42.9)]  # bbox do Rio de Janeiro

NEIGHBOURHOOD_COORDS = {
    "Barra da Tijuca": (-23.0058, -43.3498),
    "Barra de Guaratiba": (-23.0536, -43.5602),
    "Botafogo": (-22.9504, -43.1841),
    "Camorim": (-22.9798, -43.4196),
    "Catete": (-22.9263, -43.1789),
    "Centro": (-22.9092, -43.1824),
    "Copacabana": (-22.9724, -43.1861),
    "Flamengo": (-22.9324, -43.1765),
    "Glória": (-22.9219, -43.1757),
    "Guaratiba": (-23.0307, -43.5785),
    "Gávea": (-22.9759, -43.2272),
    "Humaitá": (-22.9565, -43.1941),
    "Ipanema": (-22.9836, -43.2034),
    "Itanhangá": (-22.9966, -43.3445),
    "Jacarepaguá": (-22.9523, -43.3734),
    "Jardim Botânico": (-22.9654, -43.2234),
    "Lagoa": (-22.9722, -43.2153),
    "Laranjeiras": (-22.9353, -43.1875),
    "Leblon": (-22.9847, -43.2228),
    "Leme": (-22.9618, -43.1684),
    "Recreio dos Bandeirantes": (-23.0222, -43.4733),
    "Santa Teresa": (-22.9205, -43.1869),
    "Santo Cristo": (-22.9025, -43.2062),
    "São Conrado": (-22.9966, -43.2599),
    "Taquara": (-22.9242, -43.3797),
    "Tijuca": (-22.9268, -43.2318),
    "Urca": (-22.9494, -43.1656),
    "Vargem Grande": (-22.988, -43.498),
    "Vargem Pequena": (-22.9877, -43.4584),
    "Vidigal": (-22.9947, -43.2376),
}


def _snap_neighbourhood(lat: float, lon: float) -> str:
    """Retorna o bairro cujo centroide é mais próximo de (lat, lon)."""
    def _hav(la1, lo1, la2, lo2):
        R = 6371
        dlat = math.radians(la2 - la1)
        dlon = math.radians(lo2 - lo1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(la1)) * math.cos(math.radians(la2)) * math.sin(dlon / 2) ** 2
        return R * 2 * math.asin(math.sqrt(a))
    return min(NEIGHBOURHOOD_COORDS, key=lambda b: _hav(lat, lon, *NEIGHBOURHOOD_COORDS[b]))


def _extract_listing_id(text: str) -> int | None:
    """Extrai listing_id de URL Airbnb ou inteiro direto."""
    text = text.strip()
    # URL: https://www.airbnb.com/rooms/12345678[?...]
    match = re.search(r"/rooms/(\d+)", text)
    if match:
        return int(match.group(1))
    # Numérico direto
    if re.fullmatch(r"\d+", text):
        return int(text)
    return None


ROOM_TYPE_LABELS = {
    "Imóvel inteiro": "Entire home/apt",
    "Quarto privativo": "Private room",
    "Quarto compartilhado": "Shared room",
    "Quarto de hotel": "Hotel room",
}

AMENITIES_LABELS = {
    "wifi": "Wi-Fi",
    "kitchen": "Cozinha",
    "air_conditioning": "Ar-condicionado",
    "heating": "Aquecimento",
    "washer": "Máquina de lavar",
    "dryer": "Secadora",
    "pool": "Piscina",
    "gym": "Academia",
    "elevator": "Elevador",
    "parking": "Estacionamento",
    "breakfast": "Café da manhã",
    "workspace": "Espaço de trabalho",
    "tv": "TV",
}

st.set_page_config(
    page_title="Otimizador de Preço — Airbnb Rio de Janeiro",
    page_icon="🏖️",
    layout="wide",
)

st.title("🏖️ Otimizador de Preço — Airbnb Rio de Janeiro")
st.caption("Descubra o preço que maximiza sua receita no Airbnb Rio de Janeiro.")


def _render_results(result: dict, lat: float | None, lon: float | None):
    """Renderiza KPIs, gráficos e mapa a partir de um resultado da API."""
    predicted = result["predicted_price"]
    optimal = result.get("revenue_optimal_price") or predicted
    occupancy = result.get("expected_occupancy_pct") or 0.0
    strategy_map = {
        "revenue_optimal": "Ótimo de receita",
        "premium_positioning": "Premium",
        "fallback": "Mediana local",
    }
    strategy = strategy_map.get(result.get("pricing_strategy", ""), "—")
    price_low = result.get("price_range_low", predicted * 0.85)
    price_high = result.get("price_range_high", predicted * 1.15)
    local_median = result.get("local_median_price")

    st.divider()

    # KPIs — linha 1
    k1, k2, k3 = st.columns(3)
    k1.metric(
        "Preço de mercado",
        f"R$ {predicted:,.0f}",
        help="Benchmark: o que imóveis similares cobram no mesmo bairro e tipo de imóvel",
    )
    k2.metric(
        "Preço ótimo",
        f"R$ {optimal:,.0f}",
        delta=f"R$ {optimal - predicted:+,.0f} vs mercado",
        help="Preço que maximiza a receita esperada (preço × taxa de ocupação), estimado via curva de demanda",
    )
    k3.metric(
        "Ocupação esperada",
        f"{occupancy:.0f}%" if occupancy else "—",
        help="Taxa de ocupação estimada no preço ótimo, baseada na elasticidade do segmento",
    )

    # KPIs — linha 2
    s1, s2, _, _ = st.columns(4)
    s1.metric(
        "Estratégia de precificação",
        strategy,
        help=(
            "Otimização de receita: demanda elástica — o modelo encontra via grid search o preço "
            "que maximiza preço × ocupação.\n\n"
            "Posicionamento premium: demanda inelástica — sobe o preço até P75 local, "
            "perdendo pouca ocupação e ganhando margem.\n\n"
            "Mediana local: dados insuficientes para estimar demanda."
        ),
    )
    s2.metric(
        "Intervalo de mercado (P10–P90)",
        f"R$ {int(price_low)} – R$ {int(price_high)}",
        help="Intervalo de confiança data-driven calculado sobre os resíduos out-of-fold do modelo",
    )

    col_a, col_b, col_c = st.columns(3)

    # Gráfico de preços
    with col_a:
        st.subheader("Comparativo de preços")
        labels = ["Mediana local", "Preço de mercado", "Preço ótimo"]
        values = [local_median or predicted, predicted, optimal]
        colors = ["#94a3b8", "#3b82f6", "#10b981"]

        fig = go.Figure(go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"R$ {v:,.0f}" for v in values],
            textposition="outside",
        ))
        fig.update_layout(
            yaxis_title="R$/noite",
            showlegend=False,
            height=320,
            margin=dict(t=50, b=20),
            yaxis=dict(range=[0, max(values) * 1.2]),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Curva de receita
    with col_b:
        st.subheader("Curva de receita estimada")
        prices = np.linspace(price_low * 0.5, price_high * 1.5, 200)

        if occupancy > 0 and optimal > 0:
            b = -2.0
            a = -b * np.log(optimal / predicted) if predicted > 0 else 0
            occ_curve = 1 / (1 + np.exp(-(a + b * np.log(prices / predicted + 1e-9))))
        else:
            occ_curve = np.ones_like(prices) * 0.4

        revenue_curve = prices * occ_curve

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=prices, y=revenue_curve,
            mode="lines",
            line=dict(color="#3b82f6", width=2),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.1)",
        ))
        fig2.add_vline(
            x=optimal, line_dash="dash", line_color="#10b981",
            annotation_text=f"Ótimo: R${optimal:,.0f}",
            annotation_position="top right",
        )
        fig2.update_layout(
            xaxis_title="Preço (R$/noite)",
            yaxis_title="Receita estimada (R$)",
            height=320,
            margin=dict(t=30, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Mapa interativo
    with col_c:
        st.subheader("Localização no Rio")
        if lat is not None and lon is not None:
            st.map(
                pd.DataFrame({"lat": [lat], "lon": [lon]}),
                zoom=15,
                use_container_width=True,
            )
        else:
            st.info("Coordenadas não disponíveis para este listing.")


# ── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Listing existente", "Novo listing"])


# ── Tab 1: Listing existente ────────────────────────────────────────────────
with tab1:
    st.markdown(
        "Cole a URL do seu anúncio no Airbnb ou informe o ID do listing para obter a previsão de preço ótimo."
    )

    with st.form("listing_form"):
        listing_input = st.text_input(
            "URL ou ID do listing",
            placeholder="https://www.airbnb.com/rooms/12345678  ou  12345678",
        )
        target_date_1 = st.date_input(
            "Data alvo (opcional — ajusta sazonalidade)",
            value=None,
            help="Deixe em branco para usar a data de hoje.",
        )
        submitted_listing = st.form_submit_button(
            "Analisar listing", use_container_width=True, type="primary"
        )

    if submitted_listing:
        listing_id = _extract_listing_id(listing_input)
        if listing_id is None:
            st.error("Informe uma URL válida do Airbnb (ex: https://www.airbnb.com/rooms/12345678) ou um ID numérico.")
            st.stop()

        payload = {"listing_id": listing_id}
        if target_date_1:
            payload["target_date"] = str(target_date_1)

        with st.spinner("Consultando modelo..."):
            try:
                resp = requests.post(f"{API_URL}/predict/listing", json=payload, timeout=15)
                if resp.status_code == 404:
                    st.error(
                        f"Listing **{listing_id}** não encontrado no dataset (Set/2025). "
                        "Verifique o ID ou use a aba **Novo listing** para inserir as características manualmente."
                    )
                    st.stop()
                resp.raise_for_status()
                result = resp.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Erro ao consultar a API: {e}")
                st.stop()

        # Metadados do listing
        current_price = result.get("listing_current_price")
        st.success(f"Listing encontrado: **{result.get('listing_name', '—')}**")

        meta1, meta2, meta3, meta4 = st.columns(4)
        meta1.metric("Bairro", result.get("listing_neighbourhood", "—"))
        meta2.metric("Tipo", result.get("listing_room_type", "—"))
        meta3.metric("Hóspedes", result.get("listing_accommodates", "—"))
        meta4.metric(
            "Preço atual (dataset)",
            f"R$ {current_price:,.0f}" if current_price else "—",
            help="Preço cadastrado no snapshot de Set/2025 do Inside Airbnb",
        )

        # Lat/lon podem não estar no response — usamos None para o mapa
        _render_results(result, lat=None, lon=None)


# ── Tab 2: Novo listing ─────────────────────────────────────────────────────
with tab2:
    st.markdown("Preencha as características do imóvel para obter uma estimativa de preço.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Localização")
            address_input = st.text_input(
                "Endereço do imóvel",
                placeholder="Ex: Av. Atlântica, 1500, Copacabana",
            )
            room_type_label = st.selectbox("Tipo de imóvel", list(ROOM_TYPE_LABELS.keys()))

        with col2:
            st.subheader("Capacidade")
            accommodates = st.slider("Hóspedes", 1, 16, 4)
            bedrooms = st.slider("Quartos", 0, 10, 2)
            n_bathrooms = st.slider("Banheiros completos", 1, 8, 1)
            has_lavabo = st.checkbox("Tem lavabo (banheiro social parcial)")
            bathrooms = float(n_bathrooms) + (0.5 if has_lavabo else 0.0)
            beds = st.slider("Camas", 1, 16, 2)

        with col3:
            st.subheader("Perfil do anfitrião")
            host_is_superhost = st.checkbox("Superanfitrião")
            instant_bookable = st.checkbox("Reserva instantânea")

        st.subheader("Comodidades")
        amenities_selected = st.multiselect(
            "Selecione as comodidades disponíveis",
            list(AMENITIES_LABELS.keys()),
            default=["wifi", "kitchen", "air_conditioning"],
            format_func=lambda x: AMENITIES_LABELS[x],
        )

        submitted = st.form_submit_button(
            "Calcular preço ótimo", use_container_width=True, type="primary"
        )

    if submitted:
        if not address_input.strip():
            st.error("Preencha o endereço do imóvel.")
            st.stop()

        try:
            geo_result = _geolocator.geocode(
                address_input + ", Rio de Janeiro, Brasil",
                viewbox=_RJ_VIEWBOX,
                bounded=True,
            )
        except (GeocoderTimedOut, GeocoderServiceError):
            st.error("Serviço de geocoding indisponível. Tente novamente em alguns segundos.")
            st.stop()

        if geo_result is None:
            st.error(
                "Endereço não encontrado no Rio de Janeiro. "
                "Tente ser mais específico (ex: inclua o nome da rua e o bairro)."
            )
            st.stop()

        lat = geo_result.latitude
        lon = geo_result.longitude
        neighbourhood = _snap_neighbourhood(lat, lon)

        payload = {
            "neighbourhood": neighbourhood,
            "room_type": ROOM_TYPE_LABELS[room_type_label],
            "accommodates": accommodates,
            "bathrooms": bathrooms,
            "bedrooms": bedrooms,
            "beds": beds,
            "host_is_superhost": host_is_superhost,
            "instant_bookable": instant_bookable,
            "amenities": amenities_selected,
            "latitude": lat,
            "longitude": lon,
        }

        st.info(f"Bairro detectado: **{neighbourhood}** ({lat:.4f}, {lon:.4f})")

        with st.spinner("Consultando modelo..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Erro ao consultar a API: {e}")
                st.stop()

        _render_results(result, lat=lat, lon=lon)
