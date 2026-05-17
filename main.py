"""
main.py
-------
Pipeline principal per a la valoració immobiliària a Catalunya.

Millores implementades:
  1. Transformació logarítmica del target (np.log1p / np.expm1)
     - Els models s'entrenen amb log1p(Precio)
     - Les prediccions es converteixen a euros amb np.expm1()
     - Redueix dràsticament el MAPE i la influència d'outliers
  2. Cerca d'hiperparàmetres amb Optuna (cerca bayesiana TPE)
     - Optimitza XGBoost, LightGBM i RandomForest
     - 50 trials per model, CV 3-fold intern
  3. XAI complet amb SHAP
     - Summary (beeswarm), Bar, Dependence plots, Waterfall
     - Feature Importance intrínseca + heatmap combinat
  4. Eliminació d'outliers en dos passos
     - Rang fix (regles de negoci)
     - IQR estadístic (k=3, criteri Tukey estricte)

Ús:
    python main.py --data ../Data/DatosViviendas1.csv
    python main.py --data ../Data/DatosViviendas1.csv --skip-eda
    python main.py --data ../Data/DatosViviendas1.csv --skip-eda --skip-cv
    python main.py --data ../Data/DatosViviendas1.csv --skip-eda --skip-optuna
    python main.py --data ../Data/DatosViviendas1.csv --no-log-transform
"""

import argparse
import logging
import os
import sys
import time

# Forçar UTF-8 a stdout/stderr per evitar errors en consoles Windows (CP1252)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass  # Python < 3.7 no té reconfigure

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Afegir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_and_filter
from preprocessing import preprocess, get_feature_target
from eda import run_eda
from models import (
    get_models,
    cross_validate_models,
    train_model,
    save_model,
    run_optuna_optimization,
)
from evaluation import (
    evaluate_all,
    plot_metrics_comparison,
    plot_predictions_vs_actual,
    plot_residuals,
    plot_cv_results,
    save_metrics_csv,
    plot_log_transform_effect,
)
from explainability import run_explainability

# ── Configuració de logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")

# ── Paths per defecte ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline de valoració immobiliària a Catalunya"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join(BASE_DIR, "Data", "DatosViviendas1.csv"),
        help="Ruta al fitxer CSV original (DatosViviendas1.csv)",
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Ometre l'anàlisi exploratòria (EDA)",
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Ometre la validació creuada (més ràpid)",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Ometre l'anàlisi SHAP",
    )
    parser.add_argument(
        "--skip-optuna",
        action="store_true",
        help="Ometre la cerca d'hiperparàmetres amb Optuna",
    )
    parser.add_argument(
        "--no-log-transform",
        action="store_true",
        help="No aplicar transformació logarítmica al target (no recomanat)",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=50,
        help="Nombre de trials Optuna per model (default: 50)",
    )
    parser.add_argument(
        "--optuna-timeout",
        type=int,
        default=300,
        help="Temps màxim Optuna per model en segons (default: 300)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    # Transformació logarítmica activada per defecte
    log_transform = not args.no_log_transform

    logger.info("=" * 65)
    logger.info("PIPELINE DE VALORACIÓ IMMOBILIÀRIA — CATALUNYA")
    logger.info("=" * 65)
    logger.info(f"  Transformació log1p/expm1: {'SÍ' if log_transform else 'NO'}")
    logger.info(f"  Optimització Optuna:       {'NO (--skip-optuna)' if args.skip_optuna else 'SÍ'}")
    logger.info(f"  Trials Optuna:             {args.optuna_trials}")
    logger.info("=" * 65)

    # ── 1. Càrrega i filtratge ───────────────────────────────────────────────
    logger.info("\n[1/7] Càrrega i filtratge de dades...")
    data_path = os.path.abspath(args.data)
    cat_csv = os.path.join(DATA_DIR, "catalunya_clean.csv")
    df_raw = load_and_filter(data_path, save_path=cat_csv)

    # ── 2. EDA ───────────────────────────────────────────────────────────────
    if not args.skip_eda:
        logger.info("\n[2/7] Anàlisi exploratòria de dades (EDA)...")
        run_eda(df_raw, OUTPUTS_DIR)
    else:
        logger.info("\n[2/7] EDA omesa (--skip-eda)")

    # ── 3. Preprocessament ──────────────────────────────────────────────────
    logger.info("\n[3/7] Preprocessament (outliers + imputació + features)...")
    df_clean, encoders = preprocess(df_raw.copy())
    X, y, feature_names, log_transform = get_feature_target(
        df_clean, log_transform=log_transform
    )

    # Visualitzar efecte de la transformació logarítmica
    if log_transform:
        y_raw = df_clean["Precio"] if "Precio" in df_clean.columns else np.expm1(y)
        try:
            plot_log_transform_effect(
                df_clean["Precio"] if "Precio" in df_clean.columns
                else pd.Series(np.expm1(y.values)),
                OUTPUTS_DIR
            )
        except Exception as e:
            logger.warning(f"No s'ha pogut generar el gràfic de transformació: {e}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(
        f"  Train: {len(X_train):,} | Test: {len(X_test):,} | "
        f"Features: {len(feature_names)}"
    )
    if log_transform:
        logger.info(
            f"  Target (log1p): train=[{y_train.min():.3f}, {y_train.max():.3f}] | "
            f"test=[{y_test.min():.3f}, {y_test.max():.3f}]"
        )

    # ── 4. Validació creuada (models base) ───────────────────────────────────
    models = get_models()
    cv_results = {}

    if not args.skip_cv:
        logger.info(f"\n[4/7] Validació creuada ({CV_FOLDS}-fold) — models base...")
        logger.info("  (mètriques en escala log1p, no en euros)")
        cv_results = cross_validate_models(models, X_train, y_train, cv=CV_FOLDS)
        plot_cv_results(cv_results, OUTPUTS_DIR)
    else:
        logger.info("\n[4/7] Validació creuada omesa (--skip-cv)")

    # ── 5. Cerca d'hiperparàmetres amb Optuna ────────────────────────────────
    logger.info("\n[5/7] Cerca d'hiperparàmetres amb Optuna...")
    if not args.skip_optuna:
        logger.info(
            f"  Mètode: Cerca bayesiana TPE | "
            f"{args.optuna_trials} trials per model | "
            f"Timeout: {args.optuna_timeout}s"
        )
        optimized_models, optuna_results = run_optuna_optimization(
            X_train, y_train,
            models_to_optimize=None,  # XGBoost + LightGBM + RandomForest
            n_trials=args.optuna_trials,
            cv_folds=3,
            timeout=args.optuna_timeout,
        )
        if optimized_models:
            models.update(optimized_models)
            logger.info(
                f"  Models optimitzats afegits: {list(optimized_models.keys())}"
            )
            # Guardar hiperparàmetres Optuna
            _save_optuna_results(optuna_results, OUTPUTS_DIR)
        else:
            logger.info("  No s'han pogut optimitzar models amb Optuna.")
    else:
        logger.info("  Optuna omès (--skip-optuna). S'utilitzen hiperparàmetres per defecte.")

    # ── 6. Entrenament i avaluació ───────────────────────────────────────────
    logger.info("\n[6/7] Entrenament i avaluació sobre test (mètriques en euros)...")
    trained_models = {}
    for name, model in models.items():
        logger.info(f"  Entrenant: {name}...")
        t_start = time.time()
        trained = train_model(model, X_train, y_train)
        trained_models[name] = trained
        save_model(trained, name, MODELS_DIR)
        logger.info(f"    Temps: {time.time() - t_start:.1f}s")

    # Avaluar amb mètriques en euros (np.expm1 aplicat internament)
    metrics_df = evaluate_all(
        trained_models, X_test, y_test, log_transform=log_transform
    )
    plot_metrics_comparison(metrics_df, OUTPUTS_DIR)
    save_metrics_csv(metrics_df, OUTPUTS_DIR)

    # Identificar el millor model (menor MAPE, ja que és la mètrica relativa)
    best_name = metrics_df["MAPE"].idxmin()
    logger.info(
        f"\n  [*] Millor model (menor MAPE): {best_name}\n"
        f"      RMSE = {metrics_df.loc[best_name,'RMSE']:,.0f} EUR\n"
        f"      MAE  = {metrics_df.loc[best_name,'MAE']:,.0f} EUR\n"
        f"      MAPE = {metrics_df.loc[best_name,'MAPE']:.2f}%\n"
        f"      R²   = {metrics_df.loc[best_name,'R2']:.4f}"
    )

    # Gràfics detallats per al millor model
    plot_predictions_vs_actual(
        trained_models[best_name], X_test, y_test, best_name, OUTPUTS_DIR,
        log_transform=log_transform
    )
    plot_residuals(
        trained_models[best_name], X_test, y_test, best_name, OUTPUTS_DIR,
        log_transform=log_transform
    )

    # ── 7. Interpretabilitat XAI ─────────────────────────────────────────────
    if not args.skip_shap:
        logger.info("\n[7/7] Interpretabilitat XAI (SHAP + Feature Importance)...")
        run_explainability(
            trained_models,
            X_test,
            feature_names,
            OUTPUTS_DIR,
            best_model_name=best_name,
            log_transform=log_transform,
        )
    else:
        logger.info("\n[7/7] SHAP omès (--skip-shap). Calculant Feature Importance...")
        from explainability import plot_feature_importance
        for name, model in trained_models.items():
            plot_feature_importance(model, feature_names, name, OUTPUTS_DIR)

    # ── Resum final ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("\n" + "=" * 65)
    logger.info("RESUM DE RESULTATS (Test set — mètriques en euros reals)")
    logger.info("=" * 65)
    logger.info(f"\n{metrics_df.to_string()}")
    logger.info(f"\nTransformació log1p/expm1: {'APLICADA' if log_transform else 'NO APLICADA'}")
    logger.info(f"Temps total: {elapsed:.1f}s")
    logger.info(f"Outputs guardats a: {OUTPUTS_DIR}")
    logger.info(f"Models guardats a:  {MODELS_DIR}")
    logger.info("=" * 65)

    print("\n" + "=" * 65)
    print("[OK] Pipeline completat correctament.")
    print(f"  Millor model (menor MAPE): {best_name}")
    print(f"  RMSE : {metrics_df.loc[best_name,'RMSE']:>12,.0f} EUR")
    print(f"  MAE  : {metrics_df.loc[best_name,'MAE']:>12,.0f} EUR")
    print(f"  MAPE : {metrics_df.loc[best_name,'MAPE']:>11.2f}%")
    print(f"  R²   : {metrics_df.loc[best_name,'R2']:>14.4f}")
    print(f"  Log1p/expm1 aplicat: {'SÍ' if log_transform else 'NO'}")
    print(f"  Temps total: {elapsed:.1f}s")
    print(f"  Outputs: {OUTPUTS_DIR}")
    print("=" * 65)


def _save_optuna_results(optuna_results: dict, output_dir: str):
    """Guarda els millors hiperparàmetres Optuna en un CSV."""
    if not optuna_results:
        return
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for model_name, params in optuna_results.items():
        for param_name, value in params.items():
            rows.append({
                "Model": model_name,
                "Hiperparàmetre": param_name,
                "Valor": value,
            })
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "optuna_best_params.csv")
    df.to_csv(path, index=False)
    logger.info(f"Millors hiperparàmetres Optuna guardats: {path}")


if __name__ == "__main__":
    main()
