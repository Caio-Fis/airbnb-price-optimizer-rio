"""
Scraper do Airbnb para Rio de Janeiro.

Uso:
    python -m src.ingestion.airbnb_scraper          # salva em data/raw/<hoje>/
    python -m src.ingestion.airbnb_scraper --dry-run # apenas imprime o número de listings encontrados

Requer Playwright instalado:
    playwright install chromium

Cadência recomendada: mensal (não diária — precisamos de variação temporal).
Cada execução produz dois arquivos no formato do Inside Airbnb:
    listings_YYYY-MM-DD.parquet
    calendar_YYYY-MM-DD.parquet

Limitações / notas:
  - Usa a API interna do Airbnb (/api/v3/StaysSearch e PdpAvailabilityCalendar).
    Estrutura pode mudar sem aviso.
  - Rate limiting conservador (2–4s de delay) para não sobrecarregar o servidor.
  - Sem dados de PII — apenas preço, disponibilidade, localização e atributos públicos.
"""
import asyncio
import json
import os
import random
import re
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

RAW_DATA_PATH = Path(os.getenv("RAW_DATA_PATH", "data/raw"))
PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))

# Bounding box do Rio de Janeiro
RIO_BBOX = {
    "ne_lat": -22.65, "ne_lng": -42.90,
    "sw_lat": -23.15, "sw_lng": -43.90,
}

# Número de células no grid para dividir o RJ (100 = ~10×10)
GRID_CELLS = 100

# Delay aleatório entre requests (segundos)
DELAY_MIN = 2.0
DELAY_MAX = 4.0

# Máximo de listings por célula (paginação do Airbnb)
MAX_LISTINGS_PER_CELL = 300

# Dias à frente para pegar disponibilidade
CALENDAR_DAYS = 90

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

INSIDE_AIRBNB_PAGE = "https://insideairbnb.com/get-the-data/"
INSIDE_AIRBNB_PATTERN = re.compile(
    r"data\.insideairbnb\.com/brazil/rj/rio-de-janeiro/(\d{4}-\d{2}-\d{2})/data"
)


def check_inside_airbnb(known_dates: list[str] | None = None) -> list[str]:
    """
    Verifica se Inside Airbnb publicou novos snapshots para o Rio.
    Retorna lista de datas (YYYY-MM-DD) ainda não baixadas.
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(
            INSIDE_AIRBNB_PAGE,
            timeout=20,
            headers={"User-Agent": USER_AGENTS[0]},
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        found = set()
        for tag in soup.find_all(href=INSIDE_AIRBNB_PATTERN):
            m = INSIDE_AIRBNB_PATTERN.search(tag["href"])
            if m:
                found.add(m.group(1))

        known = set(known_dates or [])
        new_dates = sorted(found - known)
        if new_dates:
            logger.info(f"Inside Airbnb: {len(new_dates)} novos snapshots encontrados: {new_dates}")
        else:
            logger.info("Inside Airbnb: nenhum novo snapshot disponível para o RJ")
        return new_dates
    except Exception as e:
        logger.warning(f"Falha ao verificar Inside Airbnb: {e}")
        return []


def _grid_cells(bbox: dict, n_cells: int) -> list[dict]:
    """Divide o bounding box em n_cells células para paginação."""
    side = int(np.sqrt(n_cells))
    lat_steps = np.linspace(bbox["sw_lat"], bbox["ne_lat"], side + 1)
    lng_steps = np.linspace(bbox["sw_lng"], bbox["ne_lng"], side + 1)
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append({
                "sw_lat": lat_steps[i], "sw_lng": lng_steps[j],
                "ne_lat": lat_steps[i + 1], "ne_lng": lng_steps[j + 1],
            })
    return cells


async def _fetch_listings_cell(page, cell: dict, check_in: str, check_out: str) -> list[dict]:
    """Faz requests à API do Airbnb para uma célula do grid e retorna listings."""
    results = []
    cursor = None
    fetch_url = "https://www.airbnb.com/api/v3/StaysSearch"

    for _ in range(5):  # máximo 5 páginas por célula
        params = {
            "operationName": "StaysSearch",
            "locale": "pt-BR",
            "currency": "BRL",
        }
        variables = {
            "staysSearchRequest": {
                "rawParams": [
                    {"filterName": "cdnCacheSafe", "filterValues": ["false"]},
                    {"filterName": "categoryTag", "filterValues": ["Tag:8678"]},
                    {"filterName": "channel", "filterValues": ["EXPLORE"]},
                    {"filterName": "checkin", "filterValues": [check_in]},
                    {"filterName": "checkout", "filterValues": [check_out]},
                    {"filterName": "ne_lat", "filterValues": [str(cell["ne_lat"])]},
                    {"filterName": "ne_lng", "filterValues": [str(cell["ne_lng"])]},
                    {"filterName": "sw_lat", "filterValues": [str(cell["sw_lat"])]},
                    {"filterName": "sw_lng", "filterValues": [str(cell["sw_lng"])]},
                    {"filterName": "itemsPerGrid", "filterValues": ["50"]},
                ],
            }
        }
        if cursor:
            variables["staysSearchRequest"]["rawParams"].append(
                {"filterName": "cursor", "filterValues": [cursor]}
            )

        try:
            resp = await page.evaluate(
                """async ([url, params, variables]) => {
                    const r = await fetch(url + '?' + new URLSearchParams(params), {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json',
                                  'X-Airbnb-API-Key': 'd306zoyjsyarp7ifhu67rjxn52tv0t20'},
                        body: JSON.stringify({variables, extensions: {}})
                    });
                    return r.json();
                }""",
                [fetch_url, params, variables],
            )

            listings_data = (
                resp.get("data", {})
                    .get("presentation", {})
                    .get("staysSearch", {})
                    .get("results", {})
                    .get("searchResults", [])
            )

            for item in listings_data:
                listing = item.get("listing", {})
                pricing = item.get("pricingQuote", {})
                if not listing.get("id"):
                    continue
                results.append({
                    "id": listing["id"],
                    "name": listing.get("name", ""),
                    "latitude": listing.get("coordinate", {}).get("latitude"),
                    "longitude": listing.get("coordinate", {}).get("longitude"),
                    "room_type": listing.get("roomTypeCategory", ""),
                    "neighbourhood_cleansed": listing.get("neighborhood", ""),
                    "accommodates": listing.get("maxGuestCapacity", 0),
                    "bedrooms": listing.get("bedrooms", 0),
                    "beds": listing.get("beds", 0),
                    "bathrooms": listing.get("bathrooms"),
                    "review_scores_rating": listing.get("avgRating"),
                    "number_of_reviews": listing.get("reviewsCount", 0),
                    "price": pricing.get("rate", {}).get("amount"),
                    "host_is_superhost": listing.get("isSuperhost", False),
                })

            # Verificar se há próxima página
            pagination = (
                resp.get("data", {})
                    .get("presentation", {})
                    .get("staysSearch", {})
                    .get("results", {})
                    .get("paginationInfo", {})
            )
            cursor = pagination.get("nextPageCursor")
            if not cursor or len(results) >= MAX_LISTINGS_PER_CELL:
                break

        except Exception as e:
            logger.warning(f"Erro ao buscar célula {cell}: {e}")
            break

        await asyncio.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    return results


async def _fetch_calendar(page, listing_id: str, start_date: str) -> list[dict]:
    """Busca disponibilidade do calendário para os próximos CALENDAR_DAYS dias."""
    url = "https://www.airbnb.com/api/v3/PdpAvailabilityCalendar"
    params = {
        "operationName": "PdpAvailabilityCalendar",
        "locale": "pt-BR",
        "currency": "BRL",
    }
    variables = {
        "request": {
            "count": CALENDAR_DAYS,
            "listingId": listing_id,
            "month": int(start_date[5:7]),
            "year": int(start_date[:4]),
        }
    }
    rows = []
    try:
        resp = await page.evaluate(
            """async ([url, params, variables]) => {
                const r = await fetch(url + '?' + new URLSearchParams(params), {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json',
                              'X-Airbnb-API-Key': 'd306zoyjsyarp7ifhu67rjxn52tv0t20'},
                    body: JSON.stringify({variables, extensions: {}})
                });
                return r.json();
            }""",
            [url, params, variables],
        )
        days = (
            resp.get("data", {})
                .get("merlin", {})
                .get("pdpAvailabilityCalendar", {})
                .get("calendarMonths", [])
        )
        for month in days:
            for day in month.get("days", []):
                rows.append({
                    "listing_id": listing_id,
                    "date": day.get("calendarDate"),
                    "available": "t" if day.get("available") else "f",
                })
    except Exception as e:
        logger.debug(f"Calendário não disponível para listing {listing_id}: {e}")
    return rows


async def _scrape_async(output_dir: Path, dry_run: bool = False) -> tuple[Path, Path]:
    """Executa o scraper de forma assíncrona com Playwright."""
    from playwright.async_api import async_playwright

    today = date.today().isoformat()
    check_in = today
    check_out = (date.today() + timedelta(days=3)).isoformat()

    cells = _grid_cells(RIO_BBOX, GRID_CELLS)
    all_listings: list[dict] = []
    all_calendar: list[dict] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            locale="pt-BR",
            timezone_id="America/Sao_Paulo",
        )
        page = await context.new_page()

        # Warm-up — visitar a homepage para obter cookies/tokens
        await page.goto("https://www.airbnb.com", wait_until="domcontentloaded")
        await asyncio.sleep(2)

        logger.info(f"Iniciando scraping de {len(cells)} células do grid...")

        for i, cell in enumerate(cells):
            listings = await _fetch_listings_cell(page, cell, check_in, check_out)
            # Deduplicar por id
            seen_ids = {l["id"] for l in all_listings}
            new_listings = [l for l in listings if l["id"] not in seen_ids]
            all_listings.extend(new_listings)

            if (i + 1) % 10 == 0:
                logger.info(f"Células processadas: {i + 1}/{len(cells)} | Listings únicos: {len(all_listings)}")

            if dry_run and len(all_listings) >= 100:
                logger.info("Dry-run: parando após 100 listings")
                break

        if not dry_run:
            # Buscar calendário para cada listing (com throttling)
            logger.info(f"Buscando calendário para {len(all_listings)} listings...")
            for j, listing in enumerate(all_listings):
                cal_rows = await _fetch_calendar(page, str(listing["id"]), today)
                all_calendar.extend(cal_rows)
                if (j + 1) % 500 == 0:
                    logger.info(f"Calendário: {j + 1}/{len(all_listings)} listings processados")
                await asyncio.sleep(random.uniform(0.5, 1.5))

        await browser.close()

    # Salvar resultados
    listings_df = pd.DataFrame(all_listings).drop_duplicates(subset="id")
    # Normalizar room_type para o formato do Inside Airbnb
    room_type_map = {
        "entire_home": "Entire home/apt",
        "private_room": "Private room",
        "shared_room": "Shared room",
        "hotel_room": "Hotel room",
    }
    if "room_type" in listings_df.columns:
        listings_df["room_type"] = listings_df["room_type"].map(
            lambda x: room_type_map.get(str(x).lower(), x)
        )

    listings_path = output_dir / f"listings_{today}.parquet"
    listings_df.to_parquet(listings_path, index=False)
    logger.info(f"Listings salvos: {listings_path} — {len(listings_df)} rows")

    calendar_path = output_dir / f"calendar_{today}.parquet"
    if all_calendar:
        cal_df = pd.DataFrame(all_calendar)
        cal_df.to_parquet(calendar_path, index=False)
        logger.info(f"Calendário salvo: {calendar_path} — {len(cal_df)} rows")
    else:
        pd.DataFrame(columns=["listing_id", "date", "available"]).to_parquet(calendar_path, index=False)

    return listings_path, calendar_path


def scrape_rio_listings(output_dir: Path | None = None, dry_run: bool = False) -> tuple[Path, Path]:
    """
    Scrapa listings e calendário do Airbnb para o Rio de Janeiro.
    Salva em output_dir (padrão: data/raw/<hoje>/).
    Retorna (listings_path, calendar_path).
    """
    if output_dir is None:
        output_dir = RAW_DATA_PATH / date.today().isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)

    return asyncio.run(_scrape_async(output_dir, dry_run=dry_run))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Airbnb scraper para Rio de Janeiro")
    parser.add_argument("--dry-run", action="store_true",
                        help="Apenas coleta ~100 listings sem calendário")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None
    listings_p, calendar_p = scrape_rio_listings(out_dir, dry_run=args.dry_run)
    print(f"Listings: {listings_p}")
    print(f"Calendar: {calendar_p}")
