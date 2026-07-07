"""
Baixa a foto de capa de cada listing (picture_url do Inside Airbnb) para
data/images/{listing_id}.jpg.

- im_w=480 no CDN do Airbnb: suficiente para CLIP (que reduz a 224px) e ~5× menor
- idempotente: pula arquivos já baixados (re-execução segura)
- concorrência limitada para não estressar o CDN
"""
import asyncio
import os
import sys
from pathlib import Path

import httpx
import pandas as pd
from loguru import logger

PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))
IMAGES_PATH = Path(os.getenv("IMAGES_PATH", "data/images"))

CONCURRENCY = 24
TIMEOUT = 20.0


def _sized_url(url: str) -> str:
    """Variante 480px: o CDN do Airbnb só redimensiona em caminhos /im/pictures/."""
    if "muscache.com/pictures/" in url:
        url = url.replace("muscache.com/pictures/", "muscache.com/im/pictures/")
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}im_w=480"


async def _download(client: httpx.AsyncClient, sem: asyncio.Semaphore,
                    listing_id: int, url: str, counters: dict):
    out = IMAGES_PATH / f"{listing_id}.jpg"
    if out.exists() and out.stat().st_size > 0:
        counters["skipped"] += 1
        return
    async with sem:
        try:
            resp = await client.get(_sized_url(url))
            if resp.status_code != 200 or not resp.content:
                resp = await client.get(url)  # fallback: original sem resize
            if resp.status_code == 200 and resp.content:
                out.write_bytes(resp.content)
                counters["ok"] += 1
            else:
                counters["failed"] += 1
        except Exception:
            counters["failed"] += 1

    done = counters["ok"] + counters["failed"] + counters["skipped"]
    if done % 2000 == 0:
        logger.info(f"{done:,} processadas — ok={counters['ok']:,} "
                    f"skip={counters['skipped']:,} fail={counters['failed']:,}")


async def main():
    IMAGES_PATH.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(
        PROCESSED_DATA_PATH / "listings_set2025.parquet",
        columns=["id", "picture_url"],
    ).dropna(subset=["picture_url"])
    logger.info(f"{len(df):,} listings com foto")

    counters = {"ok": 0, "failed": 0, "skipped": 0}
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(
        timeout=TIMEOUT,
        headers={"User-Agent": "Mozilla/5.0 (research; airbnb-price-optimizer)"},
        follow_redirects=True,
    ) as client:
        tasks = [
            _download(client, sem, int(row.id), str(row.picture_url), counters)
            for row in df.itertuples()
        ]
        await asyncio.gather(*tasks)

    logger.info(f"FIM — ok={counters['ok']:,} skip={counters['skipped']:,} "
                f"fail={counters['failed']:,}")
    if counters["ok"] + counters["skipped"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
