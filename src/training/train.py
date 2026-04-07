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

PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "airbnb-price-optimizer")

# Colunas a excluir do treinamento
EXCLUDE_COLS = [
    "id", "listing_id", "listing_id_yolo", "price", "log_price",
    "name", "description", "amenities", "amenities_list",
    "host_name", "last_scraped", "neighbourhood_cleansed",
    "picture_url", "host_url", "listing_url",
]

XGBOOST_PARAMS = {
    "n_estimators": 1000,
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
    "n_estimators": 1000,
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
    X = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
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
            else:
                model = lgb.LGBMRegressor(**LIGHTGBM_PARAMS)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                )

            oof_preds[val_idx] = model.predict(X_val)
            models.append(model)
            logger.info(f"Fold {fold + 1}/5 done")

        rmse = _rmse_original_space(y.values, oof_preds)
        mae = _mae_original_space(y.values, oof_preds)

        mlflow.log_metric("oof_rmse", rmse)
        mlflow.log_metric("oof_mae", mae)
        logger.info(f"{model_type} | OOF RMSE: R${rmse:.2f} | MAE: R${mae:.2f}")

        # Treinar modelo final em todos os dados
        if model_type == "xgboost":
            final_model = xgb.XGBRegressor(**{k: v for k, v in XGBOOST_PARAMS.items() if k != "early_stopping_rounds"})
        else:
            final_model = lgb.LGBMRegressor(**{k: v for k, v in LIGHTGBM_PARAMS.items() if k != "early_stopping_rounds"})

        final_model.fit(X, y)

        model_path = PROCESSED_DATA_PATH / f"model_{model_type}.joblib"
        joblib.dump(final_model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.sklearn.log_model(final_model, artifact_path="sklearn_model")

        feature_names_path = PROCESSED_DATA_PATH / "feature_names.joblib"
        joblib.dump(list(X.columns), feature_names_path)
        mlflow.log_artifact(str(feature_names_path))

        return run.info.run_id
