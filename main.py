"""
main.py
-------
Pipeline principal per a la valoració immobiliària a Catalunya.

Ús:
    python main.py --data ../Data/DatosViviendas1.csv
    python main.py --data ../Data/DatosViviendas1.csv --skip-eda
    python main.py --data ../Data/DatosViviendas1.csv --skip-eda --skip-cv
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

import pandas as pd
from sklearn.model_selection import train_test_split

# Afegir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_and_filter
from preprocessing import preprocess, get_feature_target
from eda import run_eda
from models import get_models, cross_validate_models, train_model, save_model
from evaluation import (
    evaluate_all,
    plot_metrics_comparison,
    plot_predictions_vs_actual,
    plot_residuals,
    plot_cv_results,
    save_metrics_csv,
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
        default=os.path.join(BASE_DIR, "..", "Data", "DatosViviendas1.csv"),
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
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    logger.info("=" * 60)
    logger.info("PIPELINE DE VALORACIÓ IMMOBILIÀRIA — CATALUNYA")
    logger.info("=" * 60)

    # ── 1. Càrrega i filtratge ───────────────────────────────────────────────
    logger.info("\n[1/6] Càrrega i filtratge de dades...")
    data_path = os.path.abspath(args.data)
    cat_csv = os.path.join(DATA_DIR, "catalunya_clean.csv")
    df_raw = load_and_filter(data_path, save_path=cat_csv)

    # ── 2. EDA ───────────────────────────────────────────────────────────────
    if not args.skip_eda:
        logger.info("\n[2/6] Anàlisi exploratòria de dades (EDA)...")
        run_eda(df_raw, OUTPUTS_DIR)
    else:
        logger.info("\n[2/6] EDA omesa (--skip-eda)")

    # ── 3. Preprocessament ──────────────────────────────────────────────────
    logger.info("\n[3/6] Preprocessament...")
    df_clean, encoders = preprocess(df_raw.copy())
    X, y, feature_names = get_feature_target(df_clean)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(
        f"  Train: {len(X_train):,} | Test: {len(X_test):,} | "
        f"Features: {len(feature_names)}"
    )

    # ── 4. Validació creuada ─────────────────────────────────────────────────
    models = get_models()
    cv_results = {}

    if not args.skip_cv:
        logger.info(f"\n[4/6] Validació creuada ({CV_FOLDS}-fold)...")
        cv_results = cross_validate_models(models, X_train, y_train, cv=CV_FOLDS)
        plot_cv_results(cv_results, OUTPUTS_DIR)
    else:
        logger.info("\n[4/6] Validació creuada omesa (--skip-cv)")

    # ── 5. Entrenament i avaluació ───────────────────────────────────────────
    logger.info("\n[5/6] Entrenament i avaluació sobre test...")
    trained_models = {}
    for name, model in models.items():
        logger.info(f"  Entrenant: {name}...")
        t_start = time.time()
        trained = train_model(model, X_train, y_train)
        trained_models[name] = trained
        save_model(trained, name, MODELS_DIR)
        logger.info(f"    Temps: {time.time() - t_start:.1f}s")

    metrics_df = evaluate_all(trained_models, X_test, y_test)
    plot_metrics_comparison(metrics_df, OUTPUTS_DIR)
    save_metrics_csv(metrics_df, OUTPUTS_DIR)

    # Identificar el millor model (menor RMSE)
    best_name = metrics_df["RMSE"].idxmin()
    logger.info(f"\n  [*] Millor model: {best_name} "
                f"(RMSE={metrics_df.loc[best_name,'RMSE']:,.0f} EUR, "
                f"R2={metrics_df.loc[best_name,'R2']:.4f})")

    # Gràfics detallats per al millor model
    plot_predictions_vs_actual(
        trained_models[best_name], X_test, y_test, best_name, OUTPUTS_DIR
    )
    plot_residuals(
        trained_models[best_name], X_test, y_test, best_name, OUTPUTS_DIR
    )

    # ── 6. Interpretabilitat ─────────────────────────────────────────────────
    if not args.skip_shap:
        logger.info("\n[6/6] Interpretabilitat (SHAP + Feature Importance)...")
        run_explainability(
            trained_models,
            X_test,
            feature_names,
            OUTPUTS_DIR,
            best_model_name=best_name,
        )
    else:
        logger.info("\n[6/6] Interpretabilitat omesa (--skip-shap)")
        # Feature importance sense SHAP
        from explainability import plot_feature_importance
        for name, model in trained_models.items():
            plot_feature_importance(model, feature_names, name, OUTPUTS_DIR)

    # ── Resum final ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("\n" + "=" * 60)
    logger.info("RESUM DE RESULTATS (Test set)")
    logger.info("=" * 60)
    logger.info(f"\n{metrics_df.to_string()}")
    logger.info(f"\nTemps total: {elapsed:.1f}s")
    logger.info(f"Outputs guardats a: {OUTPUTS_DIR}")
    logger.info(f"Models guardats a:  {MODELS_DIR}")
    logger.info("=" * 60)

    print("\n[OK] Pipeline completat correctament.")
    print(f"  Millor model: {best_name}")
    print(f"  RMSE: {metrics_df.loc[best_name,'RMSE']:,.0f} EUR")
    print(f"  R2:   {metrics_df.loc[best_name,'R2']:.4f}")
    print(f"  Outputs: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
