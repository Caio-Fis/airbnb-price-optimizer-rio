"""
Demo Streamlit — Airbnb Price Optimizer Rio de Janeiro.

Uso:
    streamlit run demo/app.py
"""
import sys
from pathlib import Path

import requests
import streamlit as st

# Permite importar src.features.constants mesmo rodando de fora do pacote
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.constants import TOP_AMENITIES

API_URL = "https://airbnb-price-api-966533570956.us-central1.run.app"

NEIGHBOURHOODS = [
    "Barra da Tijuca", "Botafogo", "Centro", "Copacabana", "Flamengo",
    "Glória", "Humaitá", "Ipanema", "Itanhangá", "Jacarepaguá",
    "Lagoa", "Laranjeiras", "Leme", "Maracanã", "Méier",
    "Penha", "Recreio dos Bandeirantes", "Santa Teresa", "São Conrado",
    "Urca", "Vidigal", "Vila Isabel", "Zona Norte", "Zona Oeste",
    "Alto da Boa Vista", "Andaraí", "Catete", "Cosme Velho",
    "Jardim Botânico", "Tijuca",
]

ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]

STRATEGY_CONFIG = {
    "revenue_optimal": ("Revenue Optimal", "success",
                        "Preço que maximiza receita (elástica — reduzir preço aumenta ocupação)"),
    "premium_positioning": ("Premium Positioning", "info",
                            "Demanda inelástica — subir até P75 local aumenta margem sem perder ocupação"),
    "fallback": ("Fallback", "warning",
                 "Dados insuficientes para estimar elasticidade — usando mediana do segmento"),
}

CONFIDENCE_LABELS = {
    "high": "Alta (20+ reviews)",
    "medium": "Média (5–19 reviews)",
    "low": "Baixa (< 5 reviews)",
}


def call_api(payload: dict) -> dict | None:
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        st.error("Timeout ao chamar a API. Tente novamente.")
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        st.error(f"Erro da API ({e.response.status_code}): {detail or str(e)}")
    except Exception as e:
        st.error(f"Erro inesperado: {e}")
    return None


def render_strategy_badge(strategy: str):
    label, kind, description = STRATEGY_CONFIG.get(
        strategy, ("Desconhecido", "warning", "")
    )
    getattr(st, kind)(f"**Estratégia: {label}** — {description}")


# ── Layout ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Airbnb Price Optimizer — RJ",
    page_icon="🏠",
    layout="wide",
)

st.title("Airbnb Price Optimizer — Rio de Janeiro")
st.caption(
    "Encontra o preço que maximiza sua receita usando curva de demanda estimada "
    "por dois snapshots do Inside Airbnb (Jun/Set 2025)."
)

col_input, col_result = st.columns([1, 1], gap="large")

# ── Inputs ────────────────────────────────────────────────────────────────────
with col_input:
    st.subheader("Sobre o listing")

    neighbourhood = st.selectbox("Bairro", NEIGHBOURHOODS, index=3)  # Copacabana
    room_type = st.selectbox("Tipo de acomodação", ROOM_TYPES)

    c1, c2, c3 = st.columns(3)
    accommodates = c1.slider("Hóspedes", 1, 16, 2)
    bedrooms = c2.slider("Quartos", 0, 8, 1)
    beds = c3.slider("Camas", 1, 12, 1)
    bathrooms = st.slider("Banheiros", 0.5, 6.0, 1.0, step=0.5)

    st.subheader("Avaliações e host")
    c4, c5 = st.columns(2)
    number_of_reviews = c4.number_input("Nº de reviews", 0, 5000, 0, step=1)
    review_scores_rating = c5.slider("Nota média", 1.0, 5.0, 4.5, step=0.1)

    c6, c7 = st.columns(2)
    host_is_superhost = c6.checkbox("Superhost")
    instant_bookable = c7.checkbox("Reserva instantânea")

    st.subheader("Comodidades")
    amenities = st.multiselect(
        "Selecione as comodidades disponíveis",
        options=TOP_AMENITIES,
        default=["wifi", "kitchen", "air conditioning"],
    )

    st.subheader("Localização e data (opcional)")
    c8, c9 = st.columns(2)
    latitude = c8.number_input("Latitude", value=-22.9711, format="%.4f")
    longitude = c9.number_input("Longitude", value=-43.1822, format="%.4f")
    target_date = st.date_input("Data alvo", value=None)

    predict_btn = st.button("Calcular preço ótimo", type="primary", use_container_width=True)

# ── Resultado ─────────────────────────────────────────────────────────────────
with col_result:
    st.subheader("Resultado")

    if predict_btn:
        payload = {
            "neighbourhood": neighbourhood,
            "room_type": room_type,
            "accommodates": accommodates,
            "bedrooms": bedrooms,
            "beds": beds,
            "bathrooms": bathrooms,
            "number_of_reviews": number_of_reviews,
            "review_scores_rating": float(review_scores_rating),
            "host_is_superhost": host_is_superhost,
            "instant_bookable": instant_bookable,
            "amenities": amenities,
            "latitude": float(latitude) if latitude else None,
            "longitude": float(longitude) if longitude else None,
            "target_date": str(target_date) if target_date else None,
        }

        with st.spinner("Consultando API..."):
            data = call_api(payload)

        if data:
            predicted = data["predicted_price"]
            local_median = data.get("local_median_price")
            revenue_optimal = data.get("revenue_optimal_price")
            occ_pct = data.get("expected_occupancy_pct")
            strategy = data.get("pricing_strategy", "fallback")
            confidence = data.get("confidence", "low")
            low = data.get("price_range_low")
            high = data.get("price_range_high")
            seasonal_note = data.get("seasonal_note")

            # Métricas principais
            m1, m2, m3 = st.columns(3)
            m1.metric("Benchmark de mercado", f"R$ {predicted:,.0f}",
                      help="O que listings similares cobram")
            if revenue_optimal:
                delta_str = f"{((revenue_optimal / predicted) - 1) * 100:+.1f}% vs mercado"
                m2.metric("Preço ótimo (receita)", f"R$ {revenue_optimal:,.0f}",
                          delta=delta_str,
                          help="Preço que maximiza receita = preço × ocupação")
            else:
                m2.metric("Preço ótimo (receita)", "—")

            if occ_pct:
                m3.metric("Ocupação esperada", f"{occ_pct:.1f}%",
                          help="Ocupação estimada no preço ótimo")
            else:
                m3.metric("Ocupação esperada", "—")

            # Intervalo de confiança
            if low and high:
                st.caption(
                    f"Intervalo de confiança (P10–P90): "
                    f"R$ {low:,.0f} — R$ {high:,.0f}"
                )

            # Badge de estratégia
            render_strategy_badge(strategy)

            # Posicionamento vs mediana local
            if local_median and local_median > 0:
                st.markdown(f"**Mediana local:** R$ {local_median:,.0f}")
                progress_val = min(predicted / (local_median * 2), 1.0)
                st.caption("Posicionamento de preço vs mediana local (0 = metade da mediana, 100% = 2× a mediana)")
                st.progress(progress_val)

            # Sazonalidade e confiança
            info_parts = [f"Confiança: **{CONFIDENCE_LABELS.get(confidence, confidence)}**"]
            if seasonal_note:
                info_parts.append(f"Sazonalidade: {seasonal_note}")
            st.caption(" | ".join(info_parts))

    else:
        st.info("Preencha os campos à esquerda e clique em **Calcular preço ótimo**.")

# ── Mapa ──────────────────────────────────────────────────────────────────────
if latitude and longitude:
    st.divider()
    st.subheader("Localização")

    try:
        import folium
        from streamlit_folium import st_folium

        m = folium.Map(location=[latitude, longitude], zoom_start=14)
        folium.Marker(
            [latitude, longitude],
            tooltip="Listing",
            icon=folium.Icon(color="red", icon="home", prefix="fa"),
        ).add_to(m)
        st_folium(m, width=None, height=380, returned_objects=[])
    except ImportError:
        st.map(data={"lat": [latitude], "lon": [longitude]})
