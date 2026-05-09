"""
explainability.py
-----------------
Interpretabilitat dels models amb SHAP i importància de variables.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

logger = logging.getLogger(__name__)


def plot_feature_importance(model, feature_names: list, name: str, output_dir: str, top_n: int = 20):
    """
    Importància de variables per a models basats en arbres
    (RandomForest, GradientBoosting, XGBoost, LightGBM).
    """
    # Obtenir importàncies (suporta Pipeline de sklearn)
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model", model)

    if not hasattr(estimator, "feature_importances_"):
        logger.warning(f"{name}: no té feature_importances_, saltant.")
        return

    importances = estimator.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_features[::-1], top_importances[::-1], color="steelblue")
    ax.set_title(f"Importància de Variables — {name} (Top {top_n})")
    ax.set_xlabel("Importància")
    plt.tight_layout()

    path = os.path.join(output_dir, f"12_feature_importance_{name}.png")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figura guardada: {path}")


def run_shap_analysis(model, X_sample: pd.DataFrame, name: str, output_dir: str, max_samples: int = 500):
    """
    Anàlisi SHAP per al model indicat.
    Genera summary plot i bar plot.

    Parameters
    ----------
    model : model entrenat (pot ser Pipeline)
    X_sample : pd.DataFrame — mostra de dades per calcular SHAP
    name : str — nom del model
    output_dir : str
    max_samples : int — nombre màxim de mostres per SHAP (per velocitat)
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP no disponible. Instal·la'l amb: pip install shap")
        return

    # Reduir mostra si cal
    if len(X_sample) > max_samples:
        X_sample = X_sample.sample(max_samples, random_state=42)

    # Obtenir estimador base (per a Pipelines)
    estimator = model
    if hasattr(model, "named_steps"):
        # Transformar X amb els passos previs al model
        steps = list(model.named_steps.items())
        for step_name, step in steps[:-1]:
            X_sample = pd.DataFrame(
                step.transform(X_sample),
                columns=X_sample.columns,
            )
        estimator = steps[-1][1]

    os.makedirs(output_dir, exist_ok=True)

    try:
        # TreeExplainer per a models basats en arbres
        if hasattr(estimator, "feature_importances_"):
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_sample)
        else:
            # LinearExplainer per a models lineals
            explainer = shap.LinearExplainer(estimator, X_sample)
            shap_values = explainer.shap_values(X_sample)

        # Summary plot (beeswarm)
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
        plt.title(f"SHAP Summary Plot — {name}")
        path = os.path.join(output_dir, f"13_shap_summary_{name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP summary guardat: {path}")

        # Bar plot (importància global)
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
        plt.title(f"SHAP Feature Importance — {name}")
        path = os.path.join(output_dir, f"14_shap_bar_{name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP bar guardat: {path}")

    except Exception as e:
        logger.error(f"Error en SHAP per {name}: {e}")


def run_explainability(trained_models: dict, X_test: pd.DataFrame,
                       feature_names: list, output_dir: str,
                       best_model_name: str = None):
    """
    Executa tota l'anàlisi d'interpretabilitat.

    Parameters
    ----------
    trained_models : dict {nom: model entrenat}
    X_test : pd.DataFrame
    feature_names : list
    output_dir : str
    best_model_name : str, optional
        Si s'especifica, fa l'anàlisi SHAP completa només per al millor model.
    """
    logger.info("Iniciant anàlisi d'interpretabilitat...")

    # Importància de variables per a tots els models
    for name, model in trained_models.items():
        plot_feature_importance(model, feature_names, name, output_dir)

    # SHAP: per al millor model (o tots si no s'especifica)
    shap_models = (
        {best_model_name: trained_models[best_model_name]}
        if best_model_name and best_model_name in trained_models
        else trained_models
    )
    for name, model in shap_models.items():
        logger.info(f"  SHAP per a: {name}")
        run_shap_analysis(model, X_test.copy(), name, output_dir)

    logger.info("Anàlisi d'interpretabilitat completada.")
