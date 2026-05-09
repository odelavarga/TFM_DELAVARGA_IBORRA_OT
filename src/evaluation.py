"""
evaluation.py
-------------
Avaluació dels models i visualització de resultats.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")


def compute_metrics(y_true, y_pred) -> dict:
    """Calcula MAE, RMSE, R², MAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def evaluate_all(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Avalua tots els models sobre el conjunt de test.

    Returns
    -------
    pd.DataFrame amb les mètriques per model.
    """
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        metrics["Model"] = name
        rows.append(metrics)
        logger.info(
            f"  {name:<20s} MAE={metrics['MAE']:>10,.0f}€  "
            f"RMSE={metrics['RMSE']:>10,.0f}€  "
            f"R²={metrics['R2']:.4f}  MAPE={metrics['MAPE']:.2f}%"
        )
    df = pd.DataFrame(rows).set_index("Model")
    return df[["MAE", "RMSE", "R2", "MAPE"]]


def plot_metrics_comparison(metrics_df: pd.DataFrame, output_dir: str):
    """Gràfic de barres comparant RMSE i R² de tots els models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metrics_df["RMSE"].sort_values().plot(kind="barh", ax=axes[0], color="steelblue")
    axes[0].set_title("RMSE per Model (menor = millor)")
    axes[0].set_xlabel("RMSE (€)")

    metrics_df["R2"].sort_values().plot(kind="barh", ax=axes[1], color="coral")
    axes[1].set_title("R² per Model (major = millor)")
    axes[1].set_xlabel("R²")

    fig.suptitle("Comparació de Models", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "08_metrics_comparison.png")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figura guardada: {path}")


def plot_predictions_vs_actual(model, X_test, y_test, name: str, output_dir: str):
    """Scatter de prediccions vs valors reals."""
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_test / 1000, y_pred / 1000, alpha=0.3, s=8, color="steelblue")
    lims = [
        min(y_test.min(), y_pred.min()) / 1000,
        max(y_test.max(), y_pred.max()) / 1000,
    ]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Predicció perfecta")
    ax.set_xlabel("Preu Real (milers €)")
    ax.set_ylabel("Preu Predit (milers €)")
    ax.set_title(f"Prediccions vs Valors Reals — {name}")
    ax.legend()
    path = os.path.join(output_dir, f"09_pred_vs_real_{name}.png")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figura guardada: {path}")


def plot_residuals(model, X_test, y_test, name: str, output_dir: str):
    """Distribució dels residus."""
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_pred / 1000, residuals / 1000, alpha=0.3, s=8, color="teal")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Preu Predit (milers €)")
    axes[0].set_ylabel("Residu (milers €)")
    axes[0].set_title(f"Residus vs Prediccions — {name}")

    axes[1].hist(residuals / 1000, bins=60, color="teal", edgecolor="white")
    axes[1].set_xlabel("Residu (milers €)")
    axes[1].set_ylabel("Freqüència")
    axes[1].set_title(f"Distribució dels Residus — {name}")

    plt.tight_layout()
    path = os.path.join(output_dir, f"10_residuals_{name}.png")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figura guardada: {path}")


def plot_cv_results(cv_results: dict, output_dir: str):
    """Gràfic de resultats de la validació creuada."""
    names = list(cv_results.keys())
    rmse_means = [cv_results[n]["rmse_mean"] for n in names]
    rmse_stds = [cv_results[n]["rmse_std"] for n in names]
    r2_means = [cv_results[n]["r2_mean"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(names, rmse_means, xerr=rmse_stds, color="steelblue", capsize=4)
    axes[0].set_title("RMSE (CV 5-fold)")
    axes[0].set_xlabel("RMSE (€)")

    axes[1].barh(names, r2_means, color="coral", capsize=4)
    axes[1].set_title("R² (CV 5-fold)")
    axes[1].set_xlabel("R²")

    fig.suptitle("Resultats Validació Creuada", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "11_cv_results.png")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figura guardada: {path}")


def save_metrics_csv(metrics_df: pd.DataFrame, output_dir: str):
    """Guarda les mètriques en CSV."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "metrics_summary.csv")
    metrics_df.to_csv(path)
    logger.info(f"Mètriques guardades: {path}")
