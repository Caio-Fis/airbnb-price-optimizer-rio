"""
Ablação pareada em 5 seeds das colunas NÃO-NUMÉRICAS descartadas por
`select_dtypes(number)` no train.py.

Contexto: room_type e host_is_superhost já foram testados (variante D da rodada
anterior, as 4 colunas bool) e deram EFEITO NULO em 5 seeds. Ficam de fora aqui.
As candidatas abaixo nunca foram testadas.

Blocos:
  H  host quality  : response_rate, acceptance_rate, response_time (ordinal),
                     host_since_days, identity_verified            (+5)
  W  banheiro      : is_shared_bath, is_private_bath, is_half_bath (+3)
                     — o `bathrooms` numérico perde "compartilhado vs privativo"
  P  property_type : top-12 one-hot + "outros"                     (+13)
  ALL              : H + W + P                                    (+21)

Comparação PAREADA: o mesmo seed define a partição de todas as variantes.
NaN vai cru para o XGBoost (que aprende a direção do missing) em vez de
fillna(median) — para as candidatas, "não informado" é sinal, não ruído.

Regra do projeto: nada entra por parecer promissor. Só com 5 seeds e a diferença
média superando 2× o erro padrão.

Grava incrementalmente — se a máquina cair, o que já rodou não se perde.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.training.train import (  # noqa: E402
    EXCLUDE_COLS,
    XGBOOST_PARAMS,
    PROCESSED_DATA_PATH,
    _rmse_original_space,
    _mae_original_space,
)

SEEDS = [42, 7, 123, 2024, 99]
OUT = Path(__file__).resolve().parent.parent / "resultados_naonumericas.csv"
SCRAPE_REF = pd.Timestamp("2025-09-27")  # calendar_last_scraped

df = pd.read_parquet(PROCESSED_DATA_PATH / "final_features.parquet")
y = df["log_price"]

# ---------------------------------------------------------------- baseline (A)
X_base = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns]).select_dtypes(
    include=[np.number]
)
X_base = X_base.dropna(axis=1, how="all")
X_base = X_base.fillna(X_base.median())

# --------------------------------------------------------- bloco H: host quality
def _pct(col):
    return pd.to_numeric(df[col].str.rstrip("%"), errors="coerce")

RESPONSE_ORDER = {
    "within an hour": 0,
    "within a few hours": 1,
    "within a day": 2,
    "a few days or more": 3,
}
H = pd.DataFrame(index=df.index)
H["host_response_rate_num"] = _pct("host_response_rate")
H["host_acceptance_rate_num"] = _pct("host_acceptance_rate")
H["host_response_time_ord"] = df["host_response_time"].map(RESPONSE_ORDER)
H["host_since_days"] = (SCRAPE_REF - pd.to_datetime(df["host_since"], errors="coerce")).dt.days
H["host_identity_verified_bin"] = (df["host_identity_verified"] == "t").astype(int)

# ------------------------------------------------------------- bloco W: banheiro
bt = df["bathrooms_text"].fillna("").str.lower()
W = pd.DataFrame(index=df.index)
W["bath_is_shared"] = bt.str.contains("shared").astype(int)
W["bath_is_private"] = bt.str.contains("private").astype(int)
W["bath_is_half"] = (bt.str.contains("half") | bt.str.contains(r"\.5")).astype(int)

# -------------------------------------------------------- bloco P: property_type
top = df["property_type"].value_counts().head(12).index
P = pd.get_dummies(
    df["property_type"].where(df["property_type"].isin(top), "outros"),
    prefix="ptype",
).astype(int)

VARIANTS = {
    "A_ref": X_base,
    "H_host": pd.concat([X_base, H], axis=1),
    "W_bath": pd.concat([X_base, W], axis=1),
    "P_ptype": pd.concat([X_base, P], axis=1),
    "ALL": pd.concat([X_base, H, W, P], axis=1),
}
for name, X in VARIANTS.items():
    print(f"{name:9s} {X.shape[1]:3d} feats", flush=True)
print(flush=True)

rows = []
for seed in SEEDS:
    for name, X in VARIANTS.items():
        params = dict(XGBOOST_PARAMS)
        params["random_state"] = seed
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.zeros(len(y))
        iters = []
        for tr, va in kf.split(X):
            m = xgb.XGBRegressor(**params)
            m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])],
                  verbose=False)
            oof[va] = m.predict(X.iloc[va])
            iters.append(m.best_iteration)
        rmse = _rmse_original_space(y.values, oof)
        mae = _mae_original_space(y.values, oof)
        med = float(np.median(np.abs(np.expm1(oof) - np.expm1(y.values))))
        rows.append({"seed": seed, "variante": name, "n_feat": X.shape[1],
                     "rmse": rmse, "mae": mae, "medae": med,
                     "best_iter": int(np.mean(iters))})
        pd.DataFrame(rows).to_csv(OUT, index=False)  # checkpoint incremental
        print(f"RESULTADO seed={seed} {name:9s} | RMSE=R${rmse:7.2f} | "
              f"MAE=R${mae:6.2f} | MedAE=R${med:6.2f} | best_iter={int(np.mean(iters)):5d}",
              flush=True)

res = pd.DataFrame(rows)
print("\n=== MÉDIA ± DESVIO ENTRE OS 5 SEEDS ===", flush=True)
print(res.groupby("variante").agg(
    rmse_m=("rmse", "mean"), rmse_sd=("rmse", "std"),
    mae_m=("mae", "mean"), mae_sd=("mae", "std"),
    medae_m=("medae", "mean"), medae_sd=("medae", "std"),
).round(3).to_string(), flush=True)

print("\n=== DIFERENÇAS PAREADAS vs A_ref (negativo = melhora) ===", flush=True)
piv = res.pivot(index="seed", columns="variante")
for metric in ["rmse", "mae", "medae"]:
    for name in VARIANTS:
        if name == "A_ref":
            continue
        d = piv[metric][name] - piv[metric]["A_ref"]
        wins = int((d < 0).sum())
        se = d.std() / np.sqrt(len(d))
        sig = "SIM" if abs(d.mean()) > 2 * se else "nao"
        print(f"  {metric:5s} {name:9s}: media={d.mean():+7.3f} sd={d.std():6.3f} "
              f"| vence em {wins}/5 seeds | supera 2*EP: {sig}", flush=True)

print("\n=== NAONUMERICAS CONCLUIDO ===", flush=True)
