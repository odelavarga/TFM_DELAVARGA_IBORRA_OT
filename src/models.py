"""
models.py
---------
Definició, entrenament i optimització dels models d'aprenentatge automàtic.

Models: Regressió Lineal, Ridge, Lasso, Random Forest, XGBoost, LightGBM.
Optimització d'hiperparàmetres: Optuna (cerca bayesiana).

Nota sobre la transformació logarítmica:
  Els models s'entrenen amb log1p(Precio) com a target.
  Les prediccions retornades estan en escala logarítmica;
  cal aplicar np.expm1() per obtenir valors en euros.
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

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
OPTUNA_N_TRIALS = 50      # Nombre de trials per a la cerca Optuna
OPTUNA_CV_FOLDS = 3       # Folds per avaluar cada trial (ràpid)
OPTUNA_TIMEOUT = 300      # Temps màxim per model (segons), None = sense límit


def get_models() -> dict:
    """
    Retorna un diccionari amb tots els models base a comparar.
    Les claus són els noms dels models.
    Aquests models s'utilitzen per a la validació creuada inicial i
    com a punt de partida per a l'optimització d'hiperparàmetres.
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
            ("model", Lasso(alpha=0.01, max_iter=5000)),
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


# ── Funcions objectiu d'Optuna per a cada model ──────────────────────────────

def _objective_xgboost(trial, X, y, cv_folds: int):
    """Funció objectiu Optuna per a XGBoost."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "verbosity": 0,
    }
    model = XGBRegressor(**params)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=kf,
                             scoring="neg_mean_squared_error", n_jobs=1)
    return np.sqrt(-scores.mean())  # RMSE (en escala log1p)


def _objective_lightgbm(trial, X, y, cv_folds: int):
    """Funció objectiu Optuna per a LightGBM."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "verbose": -1,
    }
    model = LGBMRegressor(**params)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=kf,
                             scoring="neg_mean_squared_error", n_jobs=1)
    return np.sqrt(-scores.mean())


def _objective_randomforest(trial, X, y, cv_folds: int):
    """Funció objectiu Optuna per a Random Forest."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }
    model = RandomForestRegressor(**params)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=kf,
                             scoring="neg_mean_squared_error", n_jobs=1)
    return np.sqrt(-scores.mean())


def _objective_ridge(trial, X, y, cv_folds: int):
    """Funció objectiu Optuna per a Ridge."""
    alpha = trial.suggest_float("alpha", 1e-2, 1000.0, log=True)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=kf,
                             scoring="neg_mean_squared_error", n_jobs=1)
    return np.sqrt(-scores.mean())


# ── Cerca d'hiperparàmetres amb Optuna ───────────────────────────────────────

def optimize_hyperparameters(
    model_name: str,
    X_train,
    y_train,
    n_trials: int = OPTUNA_N_TRIALS,
    cv_folds: int = OPTUNA_CV_FOLDS,
    timeout: int = OPTUNA_TIMEOUT,
) -> dict:
    """
    Cerca d'hiperparàmetres amb Optuna (cerca bayesiana TPE).

    Optuna utilitza el sampler Tree-structured Parzen Estimator (TPE),
    que és una forma d'optimització bayesiana que modela la distribució
    dels hiperparàmetres que donen bons resultats i en suggereix de nous
    basant-se en aquesta informació.

    Parameters
    ----------
    model_name : str
        Nom del model. Accepta: 'XGBoost', 'LightGBM', 'RandomForest', 'Ridge'.
    X_train, y_train : dades d'entrenament (target en escala log1p)
    n_trials : int
        Nombre de combinacions d'hiperparàmetres a explorar.
    cv_folds : int
        Nombre de folds per a la validació creuada interna.
    timeout : int
        Temps màxim en segons (None = sense límit).

    Returns
    -------
    best_params : dict
        Millors hiperparàmetres trobats.
    """
    if not HAS_OPTUNA:
        logger.warning("Optuna no disponible. Instal·la'l amb: pip install optuna")
        return {}

    objective_map = {
        "XGBoost": _objective_xgboost,
        "LightGBM": _objective_lightgbm,
        "RandomForest": _objective_randomforest,
        "Ridge": _objective_ridge,
    }

    if model_name not in objective_map:
        logger.warning(f"No hi ha funció objectiu Optuna per a '{model_name}'.")
        return {}

    if model_name == "XGBoost" and not HAS_XGB:
        logger.warning("XGBoost no disponible.")
        return {}
    if model_name == "LightGBM" and not HAS_LGB:
        logger.warning("LightGBM no disponible.")
        return {}

    logger.info(
        f"  [Optuna] Cercant hiperparàmetres per a {model_name} "
        f"({n_trials} trials, {cv_folds}-fold CV)..."
    )

    objective_fn = objective_map[model_name]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.optimize(
        lambda trial: objective_fn(trial, X_train, y_train, cv_folds),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
        n_jobs=1,  # Paral·lelisme a nivell de CV, no de trials
    )

    best_params = study.best_params
    best_value = study.best_value
    logger.info(
        f"  [Optuna] {model_name} — Millors paràmetres: {best_params}"
    )
    logger.info(
        f"  [Optuna] {model_name} — Millor RMSE (log1p): {best_value:.6f}"
    )
    return best_params


def build_optimized_model(model_name: str, best_params: dict):
    """
    Construeix el model amb els millors hiperparàmetres trobats per Optuna.

    Parameters
    ----------
    model_name : str
    best_params : dict

    Returns
    -------
    model sklearn-compatible
    """
    if model_name == "XGBoost" and HAS_XGB:
        return XGBRegressor(
            **best_params,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=0,
        )
    elif model_name == "LightGBM" and HAS_LGB:
        return LGBMRegressor(
            **best_params,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    elif model_name == "RandomForest":
        return RandomForestRegressor(
            **best_params,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    elif model_name == "Ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(**best_params)),
        ])
    else:
        logger.warning(f"No s'ha pogut construir el model optimitzat per a '{model_name}'.")
        return None


def run_optuna_optimization(
    X_train,
    y_train,
    models_to_optimize: list = None,
    n_trials: int = OPTUNA_N_TRIALS,
    cv_folds: int = OPTUNA_CV_FOLDS,
    timeout: int = OPTUNA_TIMEOUT,
) -> dict:
    """
    Executa la cerca d'hiperparàmetres amb Optuna per als models indicats.

    Parameters
    ----------
    X_train, y_train : dades d'entrenament (target en escala log1p)
    models_to_optimize : list
        Llista de noms de models a optimitzar.
        Per defecte: ['XGBoost', 'LightGBM', 'RandomForest'].
    n_trials : int
    cv_folds : int
    timeout : int

    Returns
    -------
    optimized_models : dict
        {nom_model: model_optimitzat}
    optuna_results : dict
        {nom_model: best_params}
    """
    if not HAS_OPTUNA:
        logger.warning(
            "Optuna no disponible. Instal·la'l amb: pip install optuna\n"
            "S'entrenaran els models amb els hiperparàmetres per defecte."
        )
        return {}, {}

    # Models prioritaris per a optimització
    candidates = []
    if models_to_optimize is None:
        if HAS_XGB:
            candidates.append("XGBoost")
        if HAS_LGB:
            candidates.append("LightGBM")
        candidates.append("RandomForest")
    else:
        candidates = models_to_optimize

    optimized_models = {}
    optuna_results = {}

    for model_name in candidates:
        best_params = optimize_hyperparameters(
            model_name, X_train, y_train,
            n_trials=n_trials,
            cv_folds=cv_folds,
            timeout=timeout,
        )
        if best_params:
            opt_model = build_optimized_model(model_name, best_params)
            if opt_model is not None:
                optimized_models[f"{model_name}_Optuna"] = opt_model
                optuna_results[model_name] = best_params

    return optimized_models, optuna_results


# ── Funcions d'entrenament i persistència ────────────────────────────────────

def cross_validate_models(models: dict, X, y, cv: int = 5) -> dict:
    """
    Valida creuada de tots els models.

    Nota: si el target y és en escala log1p, les mètriques RMSE aquí
    estan en escala logarítmica. La comparació és vàlida (menor = millor),
    però les magnituds no corresponen a euros directament.

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
            f"    RMSE(log)={rmse_scores.mean():.4f} ± {rmse_scores.std():.4f} | "
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
