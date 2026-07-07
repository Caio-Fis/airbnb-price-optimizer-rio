import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import { fetchListings, brl } from "../api.js";

const RIO_CENTER = [-43.33, -22.95];
const STYLE_URL = "https://tiles.openfreemap.org/styles/dark";

// Rampa sequencial de preço (azul claro = mais caro, sobre mapa escuro)
const PRICE_RAMP = [
  [100, "#184f95"],
  [300, "#256abf"],
  [600, "#3987e5"],
  [1200, "#86b6ef"],
  [2500, "#cde2fb"],
];

export default function MapView({ onSelect, selectedId, customPoint }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const customMarkerRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState(null);

  useEffect(() => {
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: STYLE_URL,
      center: RIO_CENTER,
      zoom: 10.3,
      attributionControl: { compact: true },
    });
    mapRef.current = map;
    window.__map = map; // handle para testes e2e/debug
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");

    const hoverTip = new maplibregl.Popup({
      closeButton: false,
      closeOnClick: false,
      className: "hover-tip",
      offset: 10,
    });

    map.on("load", async () => {
      let geojson;
      try {
        geojson = await fetchListings();
      } catch (e) {
        setLoadError(e.message);
        setLoading(false);
        return;
      }
      map.addSource("listings", { type: "geojson", data: geojson });

      map.addLayer({
        id: "listings-dots",
        type: "circle",
        source: "listings",
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 9, 1.6, 12, 3.5, 15, 7],
          "circle-color": [
            "case",
            ["==", ["get", "price"], null],
            "#898781",
            ["interpolate", ["linear"], ["get", "price"], ...PRICE_RAMP.flat()],
          ],
          "circle-opacity": 0.85,
        },
      });

      // Destaque do listing selecionado: anel branco por cima
      map.addLayer({
        id: "listings-selected",
        type: "circle",
        source: "listings",
        filter: ["==", ["get", "id"], ""],
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 9, 5, 15, 11],
          "circle-color": "rgba(0,0,0,0)",
          "circle-stroke-color": "#ffffff",
          "circle-stroke-width": 2.5,
        },
      });

      map.on("click", "listings-dots", (e) => {
        const feature = e.features?.[0];
        if (!feature) return;
        onSelect({ ...feature.properties, lat: e.lngLat.lat, lon: e.lngLat.lng });
        map.easeTo({ center: feature.geometry.coordinates, duration: 500 });
      });

      map.on("mouseenter", "listings-dots", () => {
        map.getCanvas().style.cursor = "pointer";
      });
      map.on("mousemove", "listings-dots", (e) => {
        const p = e.features?.[0]?.properties;
        if (!p) return;
        const price = p.price != null ? brl.format(p.price) : "sem preço";
        hoverTip
          .setLngLat(e.lngLat)
          .setHTML(`<strong>${p.name}</strong><br/>${p.nb} · ${price}/noite`)
          .addTo(map);
      });
      map.on("mouseleave", "listings-dots", () => {
        map.getCanvas().style.cursor = "";
        hoverTip.remove();
      });

      setLoading(false);
    });

    return () => map.remove();
  }, [onSelect]);

  // Atualiza o anel de seleção sem recriar o mapa
  useEffect(() => {
    const map = mapRef.current;
    if (map?.getLayer("listings-selected")) {
      map.setFilter("listings-selected", ["==", ["get", "id"], selectedId ?? ""]);
    }
  }, [selectedId]);

  // Marcador do endereço geocodificado (modo "Novo imóvel")
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    customMarkerRef.current?.remove();
    customMarkerRef.current = null;
    if (customPoint) {
      customMarkerRef.current = new maplibregl.Marker({ color: "#199e70" })
        .setLngLat([customPoint.lon, customPoint.lat])
        .addTo(map);
      map.flyTo({ center: [customPoint.lon, customPoint.lat], zoom: 14.5, duration: 900 });
    }
  }, [customPoint]);

  return (
    <>
      <div ref={containerRef} className="map-container" />
      {loading && <div className="loading-overlay">Carregando 43 mil acomodações…</div>}
      {loadError && (
        <div className="loading-overlay">
          Falha ao carregar as acomodações ({loadError}). Recarregue a página.
        </div>
      )}
      <div className="legend">
        <div className="title">Preço por noite</div>
        <div
          className="ramp"
          style={{
            background: `linear-gradient(90deg, ${PRICE_RAMP.map(([, c]) => c).join(", ")})`,
          }}
        />
        <div className="ticks">
          <span>R$100</span>
          <span>R$600</span>
          <span>R$2.500+</span>
        </div>
      </div>
    </>
  );
}
