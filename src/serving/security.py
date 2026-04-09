"""
Segurança da API:
  - Autenticação por API Key (header X-API-Key)
  - Rate limiting por IP via slowapi
  - Validação de entrada específica do domínio
"""
import re
from typing import Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from loguru import logger
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.serving.settings import settings

# ── Rate Limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[settings.rate_limit_predict],
    headers_enabled=True,   # expõe X-RateLimit-* headers na resposta
)

# ── API Key Auth ──────────────────────────────────────────────────────────────
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: Optional[str] = Security(_api_key_header)) -> str:
    """
    Dependency que valida o header X-API-Key.
    Se API_KEYS_RAW não estiver configurada, a API fica aberta
    (útil em dev — em produção sempre configure as chaves).
    """
    valid_keys = settings.api_keys

    if not valid_keys:
        # Sem chaves configuradas: modo aberto (dev)
        logger.warning("API running without authentication — set API_KEYS_RAW in production")
        return "anonymous"

    if not api_key or api_key not in valid_keys:
        logger.warning(f"Rejected request — invalid or missing API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key


# ── Validação de domínio ──────────────────────────────────────────────────────
# Coordenadas do Rio de Janeiro (bbox com margem)
_RIO_LAT = (-23.15, -22.65)
_RIO_LON = (-43.90, -42.90)

# Apenas letras, espaços, hífens e acentos — previne injeções de lookup
_NEIGHBOURHOOD_PATTERN = re.compile(r"^[\w\s\-À-ÿ]{2,80}$")


def validate_coordinates(lat: Optional[float], lon: Optional[float]) -> None:
    """Rejeita coordenadas fora do bounding box do Rio de Janeiro."""
    if lat is None and lon is None:
        return
    if lat is None or lon is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="latitude e longitude devem ser fornecidos juntos",
        )
    if not (_RIO_LAT[0] <= lat <= _RIO_LAT[1]):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"latitude fora do Rio de Janeiro: {lat}",
        )
    if not (_RIO_LON[0] <= lon <= _RIO_LON[1]):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"longitude fora do Rio de Janeiro: {lon}",
        )


def validate_neighbourhood(neighbourhood: str) -> None:
    """Valida formato do nome do bairro."""
    if not _NEIGHBOURHOOD_PATTERN.match(neighbourhood):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="neighbourhood contém caracteres inválidos",
        )


VALID_ROOM_TYPES = {"Entire home/apt", "Private room", "Shared room", "Hotel room"}


def validate_room_type(room_type: str) -> None:
    if room_type not in VALID_ROOM_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"room_type inválido. Valores aceitos: {sorted(VALID_ROOM_TYPES)}",
        )
