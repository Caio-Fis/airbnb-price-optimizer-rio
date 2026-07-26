#!/usr/bin/env python3
"""Empacota o campeão em um tarball versionado e publica como GitHub Release.

`models/` e `data/processed/` são gitignored: o modelo em produção não tem
histórico de versões no repo. Este script fecha esse buraco sem depender de
infraestrutura externa (o remote DVC apontaria para o GCS, que está sendo
descomissionado).

O bundle contém TODOS os artefatos que o serving carrega — um model.joblib
sozinho não reconstrói a inferência. A lista espelha o bloco COPY do Dockerfile,
que é a fonte da verdade do que vai para a imagem.

Uso:
    python scripts/release_champion.py --tag model-v3
    python scripts/release_champion.py --tag model-v3 --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent

# Espelha o bloco COPY do Dockerfile. Se um artefato entrar lá, entra aqui.
REQUIRED_ARTIFACTS = [
    "models/model.joblib",
    "models/encoders.joblib",
    "models/baseline_metrics.joblib",
    "data/processed/feature_names.joblib",
    "data/processed/competition_stats.joblib",
    "data/processed/demand_params.joblib",
    "data/processed/hw_seasonal_by_dow.joblib",
    "data/processed/prediction_intervals.joblib",
    "data/processed/seasonal_factors.joblib",
    "data/processed/geo_metro_cache.json",
    "data/processed/geo_poi_bar.json",
    "data/processed/geo_poi_restaurant.json",
    "data/processed/geo_poi_cafe_bakery.json",
    "data/processed/geo_poi_gym.json",
    "data/processed/geo_poi_park.json",
    "data/processed/geo_poi_supermarket.json",
    "data/processed/geo_poi_attraction.json",
    "data/processed/geo_poi_nightclub.json",
]

# Grandes e reconstrutíveis a partir dos snapshots brutos; ficam fora do bundle
# para não estourar o tamanho da release.
OPTIONAL_ARTIFACTS = [
    "data/processed/listings_slim.parquet",
    "data/processed/listings_map.json.gz",
]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(include_optional: bool) -> list[Path]:
    names = REQUIRED_ARTIFACTS + (OPTIONAL_ARTIFACTS if include_optional else [])
    paths, missing = [], []
    for name in names:
        path = REPO_ROOT / name
        if path.exists():
            paths.append(path)
        elif name in REQUIRED_ARTIFACTS:
            missing.append(name)
    if missing:
        raise SystemExit(
            "Artefatos obrigatórios ausentes — bundle seria inútil:\n  "
            + "\n  ".join(missing)
        )
    return paths


def build_manifest(tag: str, paths: list[Path]) -> str:
    commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    metrics_path = REPO_ROOT / "models/baseline_metrics.joblib"
    metrics = joblib.load(metrics_path) if metrics_path.exists() else {}

    lines = [
        f"Campeão: {tag}",
        f"Gerado em: {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"Commit: {commit}",
        f"OOF RMSE: {metrics.get('oof_rmse', 'n/d')}",
        f"Registrado em: {metrics.get('registered_at', 'n/d')}",
        "",
        "md5                               tamanho  arquivo",
    ]
    for path in paths:
        rel = path.relative_to(REPO_ROOT)
        lines.append(f"{md5(path)}  {path.stat().st_size:>9}  {rel}")
    lines += [
        "",
        "Restaurar: extrair na raiz do repo (o tar preserva os caminhos relativos).",
        "    tar xzf <bundle>.tar.gz -C /caminho/do/repo",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="tag da release, ex: model-v3")
    parser.add_argument("--notes", default="", help="corpo da release")
    parser.add_argument("--include-optional", action="store_true",
                        help="inclui listings_slim/listings_map (+8 MB)")
    parser.add_argument("--dry-run", action="store_true",
                        help="monta o bundle e mostra o manifesto, sem publicar")
    args = parser.parse_args()

    paths = collect(args.include_optional)
    manifest = build_manifest(args.tag, paths)

    out_dir = REPO_ROOT / "dist"
    out_dir.mkdir(exist_ok=True)
    bundle = out_dir / f"champion-{args.tag}.tar.gz"
    manifest_path = out_dir / f"champion-{args.tag}.MANIFEST.txt"
    manifest_path.write_text(manifest)

    with tarfile.open(bundle, "w:gz") as tar:
        for path in paths:
            tar.add(path, arcname=str(path.relative_to(REPO_ROOT)))
        tar.add(manifest_path, arcname="MANIFEST.txt")

    print(manifest)
    print(f"Bundle: {bundle} ({bundle.stat().st_size / 1e6:.1f} MB)")

    if args.dry_run:
        print("\n[dry-run] nada publicado.")
        return 0

    subprocess.run(
        ["gh", "release", "create", args.tag, str(bundle), str(manifest_path),
         "--title", f"Modelo campeão {args.tag}",
         "--notes", args.notes or manifest],
        cwd=REPO_ROOT, check=True,
    )
    print(f"\nRelease {args.tag} publicada.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
