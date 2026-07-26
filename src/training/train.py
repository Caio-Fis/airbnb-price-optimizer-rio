"""
Treinamento com XGBoost e LightGBM.
Usa MLflow para tracking de experimentos.
Target: log(price) → RMSE + MAE no espaço original.
"""
import os
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.model_selection import KFold

from src.training.mlflow_setup import MLFLOW_EXPERIMENT_NAME, configure_mlflow

PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))

# Colunas a excluir do treinamento
EXCLUDE_COLS = [
    "id", "listing_id", "listing_id_yolo", "price", "log_price",
    "name", "description", "amenities", "amenities_list",
    "host_name", "last_scraped", "neighbourhood_cleansed",
    "picture_url", "host_url", "listing_url",
    # Reviews removidos — não devem influenciar a previsão de preço
    "number_of_reviews", "number_of_reviews_ltm", "number_of_reviews_l30d",
    "number_of_reviews_ly", "reviews_per_month",
    "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_checkin", "review_scores_communication",
    "review_scores_location", "review_scores_value",
    "review_velocity", "days_since_last_review", "total_reviews",
    "n_neg_keywords", "neg_keyword_ratio",
    # IDs sem significado preditivo
    "scrape_id", "host_id",
    # Derivadas de preço/ocupação do painel — vazamento de alvo e
    # indisponíveis na inferência (o serving zerava, criando skew)
    "occupancy_rate", "price_premium", "revenue_optimal_price",
    "expected_occupancy_at_optimal",
    "estimated_occupancy_l365d", "estimated_revenue_l365d",
    # Colunas do scrape que o serving não recebe do usuário
    "host_listings_count", "host_total_listings_count",
    "minimum_minimum_nights", "maximum_minimum_nights",
    "minimum_maximum_nights", "maximum_maximum_nights",
    "minimum_nights_avg_ntm", "maximum_nights_avg_ntm",
    "availability_30", "availability_60", "availability_90", "availability_eoy",
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms",
    # Sazonalidade do snapshot: variância ~0 no treino (data única de scrape);
    # o ajuste por data é a camada sazonal pós-modelo no serving
    "hw_level", "hw_trend", "hw_seasonal", "hw_fitted", "hw_residual",
    "month", "day_of_week", "is_weekend",
]

# n_estimators=3000: com 1000 o early stopping nunca acionava (best_iteration ~996
# em todas as variantes testadas) — o modelo não convergia. Validado em 5 seeds:
# ganho consistente 5/5 em RMSE (-0,62), MAE (-0,75) e MedAE (-0,49), com desvio
# 4-6x menor que a média. Ver tasks/todo.md — Fase 6.
XGBOOST_PARAMS = {
    "n_estimators": 3000,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
    "eval_metric": "rmse",
}

LIGHTGBM_PARAMS = {
    "n_estimators": 3000,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
    "verbose": -1,
}


def _load_features() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED_DATA_PATH / "final_features.parquet")
    y = df["log_price"]

    drop_cols = [c for c in EXCLUDE_COLS if c in df.columns]
    candidates = df.drop(columns=drop_cols)
    X = candidates.select_dtypes(include=[np.number])

    # select_dtypes(number) descarta bool e object silenciosamente — foi assim que
    # room_type_* e host_is_superhost_t sumiram do modelo sem ninguém notar.
    # Manter visível: usar uma delas exige converter na origem E garantir paridade
    # com o serving (ver src/serving/predict.py::_build_features).
    ignored = [c for c in candidates.columns if c not in X.columns]
    if ignored:
        logger.warning(f"{len(ignored)} colunas não-numéricas ignoradas no treino: {ignored}")
    # Drop columns where all values are NaN, then median-impute the rest
    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median())
    return X, y


def _rmse_original_space(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae_original_space(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return float(np.mean(np.abs(y_true - y_pred)))


def train_model(model_type: str = "xgboost") -> str:
    configure_mlflow()
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    X, y = _load_features()

    logger.info(f"Training {model_type} — X shape: {X.shape}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))

    with mlflow.start_run(run_name=model_type) as run:
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_samples", X.shape[0])

        params = XGBOOST_PARAMS if model_type == "xgboost" else LIGHTGBM_PARAMS
        mlflow.log_params({k: v for k, v in params.items() if k != "early_stopping_rounds"})

        models = []
        best_iters = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if model_type == "xgboost":
                model = xgb.XGBRegressor(**XGBOOST_PARAMS)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                best_iters.append(model.best_iteration)
            else:
                model = lgb.LGBMRegressor(**LIGHTGBM_PARAMS)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                )
                best_iters.append(model.best_iteration_)

            oof_preds[val_idx] = model.predict(X_val)
            models.append(model)
            logger.info(f"Fold {fold + 1}/5 done")

        rmse = _rmse_original_space(y.values, oof_preds)
        mae = _mae_original_space(y.values, oof_preds)

        mlflow.log_metric("oof_rmse", rmse)
        mlflow.log_metric("oof_mae", mae)
        logger.info(f"{model_type} | OOF RMSE: R${rmse:.2f} | MAE: R${mae:.2f}")

        # Intervalos de confiança via residuais OOF (P10/P90)
        oof_residuals = np.expm1(oof_preds) - np.expm1(y.values)
        p10_res = float(np.percentile(oof_residuals, 10))
        p90_res = float(np.percentile(oof_residuals, 90))
        median_pred = float(np.median(np.expm1(oof_preds)))
        intervals = {"p10_pct": p10_res / median_pred, "p90_pct": p90_res / median_pred}
        intervals_path = PROCESSED_DATA_PATH / "prediction_intervals.joblib"
        joblib.dump(intervals, intervals_path)
        mlflow.log_metric("interval_p10_pct", intervals["p10_pct"])
        mlflow.log_metric("interval_p90_pct", intervals["p90_pct"])
        try:
            mlflow.log_artifact(str(intervals_path))
        except Exception as e:
            logger.warning(f"MLflow artifact logging skipped: {e}")
        logger.info(f"Prediction intervals: P10={intervals['p10_pct']:.2%} P90={intervals['p90_pct']:.2%}")

        # Treinar modelo final com n_estimators = média dos best_iteration dos folds
        avg_best = int(round(np.mean(best_iters))) if best_iters else 500
        mlflow.log_param("final_n_estimators", avg_best)
        logger.info(f"Final model n_estimators={avg_best} (avg best_iteration across folds)")

        final_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
        final_params["n_estimators"] = avg_best

        if model_type == "xgboost":
            final_model = xgb.XGBRegressor(**final_params)
        else:
            final_model = lgb.LGBMRegressor(**final_params)

        final_model.fit(X, y)

        model_path = PROCESSED_DATA_PATH / f"model_{model_type}.joblib"
        joblib.dump(final_model, model_path)
        try:
            mlflow.log_artifact(str(model_path), artifact_path="model")
            mlflow.sklearn.log_model(final_model, artifact_path="sklearn_model")
        except Exception as e:
            logger.warning(f"MLflow model artifact logging skipped: {e}")

        feature_names_path = PROCESSED_DATA_PATH / "feature_names.joblib"
        joblib.dump(list(X.columns), feature_names_path)
        try:
            mlflow.log_artifact(str(feature_names_path))
        except Exception as e:
            logger.warning(f"MLflow feature names artifact logging skipped: {e}")

        return run.info.run_id
