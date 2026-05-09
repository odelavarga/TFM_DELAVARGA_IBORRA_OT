"""
models.py
---------
Definició i entrenament dels models d'aprenentatge automàtic.
Models: Regressió Lineal, Ridge, Lasso, Random Forest, XGBoost, LightGBM.
"""

import numpy as np
import logging
import joblib
import os

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


def get_models() -> dict:
    """
    Retorna un diccionari amb tots els models a comparar.
    Les claus són els noms dels models.
    """
    models = {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=100.0, max_iter=5000)),
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
    }

    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=0,
        )
    else:
        logger.warning("XGBoost no disponible. Instal·la'l amb: pip install xgboost")

    if HAS_LGB:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    else:
        logger.warning("LightGBM no disponible. Instal·la'l amb: pip install lightgbm")

    return models


def cross_validate_models(models: dict, X, y, cv: int = 5) -> dict:
    """
    Valida creuada de tots els models amb CV estratificat.

    Returns
    -------
    cv_results : dict
        {nom_model: {"rmse_mean": float, "rmse_std": float, "r2_mean": float}}
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}

    for name, model in models.items():
        logger.info(f"  Validació creuada: {name}...")
        neg_mse = cross_val_score(model, X, y, cv=kf,
                                  scoring="neg_mean_squared_error", n_jobs=-1)
        r2 = cross_val_score(model, X, y, cv=kf,
                             scoring="r2", n_jobs=-1)
        rmse_scores = np.sqrt(-neg_mse)
        cv_results[name] = {
            "rmse_mean": rmse_scores.mean(),
            "rmse_std": rmse_scores.std(),
            "r2_mean": r2.mean(),
            "r2_std": r2.std(),
        }
        logger.info(
            f"    RMSE={rmse_scores.mean():,.0f} ± {rmse_scores.std():,.0f} | "
            f"R²={r2.mean():.4f} ± {r2.std():.4f}"
        )

    return cv_results


def train_model(model, X_train, y_train):
    """Entrena un model i retorna el model entrenat."""
    model.fit(X_train, y_train)
    return model


def save_model(model, name: str, models_dir: str):
    """Serialitza el model amb joblib."""
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{name}.joblib")
    joblib.dump(model, path)
    logger.info(f"Model guardat: {path}")
    return path


def load_model(name: str, models_dir: str):
    """Carrega un model serialitzat."""
    path = os.path.join(models_dir, f"{name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model no trobat: {path}")
    return joblib.load(path)
