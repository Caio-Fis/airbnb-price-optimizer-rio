"""
Recupera runs presos no banco-sombra do MLflow.

Contexto: enquanto `configure_mlflow()` não existia, o default do MLflow fazia
URL-quote do caminho do projeto (que tem espaços) e o SQLAlchemy abria o caminho
literal, criando `../Airbnb%20value%20of%20homes/mlflow.db`. Métricas e
parâmetros de 17 runs (Jul/2026) foram parar lá; o `mlflow.db` real parou em
12/Abr/2026. Os artefatos nunca se perderam — sempre foram para `mlruns/1/`.

Este script copia as linhas que faltam do banco-sombra para o real, casando os
schemas do MLflow. É idempotente: rodar duas vezes não duplica nada.

Uso:
    python scripts/migrate_mlflow_shadow_db.py --dry-run
    python scripts/migrate_mlflow_shadow_db.py
"""
import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

REAL_DB = Path("mlflow.db").resolve()
# Mesmo diretório do projeto, mas com o nome percent-encoded — é assim que o
# MLflow o criou, então é assim que o encontramos.
SHADOW_DB = Path(str(REAL_DB.parent).replace(" ", "%20")) / "mlflow.db"

# Ordem importa: experiments antes de runs, runs antes das tabelas filhas.
TABLES = ["experiments", "runs", "metrics", "params", "tags", "latest_metrics"]


def _table_info(conn: sqlite3.Connection, schema: str, table: str) -> list[tuple]:
    # O qualificador de schema vai no PRAGMA, não no nome da tabela:
    # `PRAGMA shadow.table_info(runs)`, nunca `PRAGMA table_info(shadow.runs)`.
    return list(conn.execute(f"PRAGMA {schema}.table_info({table})"))


def _columns(conn: sqlite3.Connection, schema: str, table: str) -> list[str]:
    return [r[1] for r in _table_info(conn, schema, table)]


def _primary_key(conn: sqlite3.Connection, schema: str, table: str) -> list[str]:
    return [r[1] for r in _table_info(conn, schema, table) if r[5]]


def migrate(dry_run: bool) -> int:
    if not SHADOW_DB.exists():
        print(f"Nada a fazer: banco-sombra não existe em {SHADOW_DB}")
        return 0
    if not REAL_DB.exists():
        print(f"ERRO: banco real não encontrado em {REAL_DB}", file=sys.stderr)
        return 1

    if not dry_run:
        backup = REAL_DB.with_suffix(f".db.bak-{datetime.now():%Y%m%d%H%M%S}")
        shutil.copy2(REAL_DB, backup)
        print(f"Backup do banco real: {backup.name}")

    real = sqlite3.connect(REAL_DB)
    real.execute(f"ATTACH DATABASE '{SHADOW_DB}' AS shadow")

    total = 0
    for table in TABLES:
        shadow_cols = _columns(real, "shadow", table)
        real_cols = _columns(real, "main", table)
        if not shadow_cols or not real_cols:
            print(f"  {table}: ausente em um dos bancos, pulando")
            continue

        # Só as colunas que existem nos dois lados (versões de schema diferentes).
        cols = [c for c in shadow_cols if c in real_cols]
        col_list = ", ".join(f'"{c}"' for c in cols)
        pk = _primary_key(real, "main", table)

        if pk:
            # OR IGNORE resolveria colisão de PK, mas queremos contar o que entra.
            where = " AND ".join(
                f'main."{table}"."{k}" = shadow."{table}"."{k}"' for k in pk
            )
            sql_select = (
                f'SELECT {col_list} FROM shadow."{table}" '
                f'WHERE NOT EXISTS (SELECT 1 FROM main."{table}" WHERE {where})'
            )
        else:
            sql_select = f'SELECT {col_list} FROM shadow."{table}"'

        n = real.execute(f"SELECT COUNT(*) FROM ({sql_select})").fetchone()[0]
        if n and not dry_run:
            real.execute(f'INSERT INTO main."{table}" ({col_list}) {sql_select}')
        print(f"  {table}: {n} linha(s) {'a migrar' if dry_run else 'migradas'}")
        total += n

    if dry_run:
        real.rollback()
        print(f"\nDRY-RUN — {total} linha(s) seriam migradas. Nada foi escrito.")
    else:
        real.commit()
        n_runs = real.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        print(f"\n{total} linha(s) migradas. Banco real agora tem {n_runs} runs.")
        print(f"Banco-sombra mantido em {SHADOW_DB} — remova quando conferir.")
    real.close()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="só relata, não escreve")
    sys.exit(migrate(ap.parse_args().dry_run))
