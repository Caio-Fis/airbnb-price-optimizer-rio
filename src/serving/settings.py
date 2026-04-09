"""
Configuração centralizada via variáveis de ambiente.
Nunca hardcode secrets — tudo vem do .env ou do ambiente.
"""
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Auth -----------------------------------------------------------------
    # Múltiplas chaves separadas por vírgula: "key1,key2"
    api_keys_raw: str = ""

    @property
    def api_keys(self) -> set[str]:
        keys = {k.strip() for k in self.api_keys_raw.split(",") if k.strip()}
        return keys

    # --- Rate limiting --------------------------------------------------------
    rate_limit_predict: str = "30/minute"
    rate_limit_health: str = "120/minute"

    # --- CORS -----------------------------------------------------------------
    # Produção: defina explicitamente. Dev: "*" é aceitável.
    allowed_origins: list[str] = ["*"]
    allowed_methods: list[str] = ["GET", "POST"]
    allowed_headers: list[str] = ["X-API-Key", "Content-Type"]

    # --- Docs -----------------------------------------------------------------
    # Desabilitar em produção (ENVIRONMENT=production)
    environment: Literal["development", "production"] = "development"

    @property
    def docs_url(self) -> str | None:
        return "/docs" if self.environment == "development" else None

    @property
    def redoc_url(self) -> str | None:
        return "/redoc" if self.environment == "development" else None

    @property
    def openapi_url(self) -> str | None:
        return "/openapi.json" if self.environment == "development" else None

    # --- Caminhos -------------------------------------------------------------
    model_path: str = "models/model.joblib"
    encoders_path: str = "models/encoders.joblib"
    processed_data_path: str = "data/processed"

    # --- Servidor -------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    workers: int = 1

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
