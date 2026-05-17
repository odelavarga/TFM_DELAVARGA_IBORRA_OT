"""
evaluation.py
-------------
Avaluació dels models i visualització de resultats.

Transformació logarítmica:
  Quan log_transform=True, els models prediuen log1p(Precio).
  Aquesta funció aplica np.expm1() per tornar les prediccions a euros
  abans de calcular les mètriques (MAE, RMSE, R², MAPE en euros reals).
  Això permet comparar mètriques entre models amb escales comprensibles.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
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


def predict_in_euros(model, X, log_transform: bool = True) -> np.ndarray:
    """
    Fa la predicció i retorna els valors en euros.

    Si log_transform=True, el model prediu log1p(Precio) i cal aplicar
    np.expm1() per convertir de nou a euros.

    Parameters
    ----------
    model : model entrenat
    X : dades de features
    log_transform : bool
        Si True, aplica np.expm1() a la predicció.

    Returns
    -------
    np.ndarray — prediccions en euros
    """
    y_pred_raw = model.predict(X)
    if log_transform:
        return np.expm1(y_pred_raw)
    return y_pred_raw


def y_in_euros(y, log_transform: bool = True) -> np.ndarray:
    """
    Converteix el target de volta a euros si estava en escala log1p.

    Parameters
    ----------
    y : array-like — target (log1p(Precio) o Precio)
    log_transform : bool

    Returns
    -------
    np.ndarray — target en euros
    """
    if log_transform:
        return np.expm1(np.array(y))
    return np.array(y)


def compute_metrics(y_true, y_pred) -> dict:
    """
    Calcula MAE, RMSE, R², MAPE.

    Els arrays d'entrada han d'estar en la mateixa escala (euros).
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def evaluate_all(models: dict, X_test, y_test,
                 log_transform: bool = True) -> pd.DataFrame:
    """
    Avalua tots els models sobre el conjunt de test.

    Si log_transform=True:
      - Les prediccions (log1p) es converteixen a euros amb np.expm1().
      - El target y_test (log1p) es converteix a euros amb np.expm1().
      - Totes les mètriques (MAE, RMSE, MAPE) estan en escala d'euros reals.

    Parameters
    ----------
    models : dict {nom: model entrenat}
    X_test : pd.DataFrame
    y_test : pd.Series — target (log1p o euros, segons log_transform)
    log_transform : bool

    Returns
    -------
    pd.DataFrame amb les mètriques per model.
    """
    y_true_eur = y_in_euros(y_test, log_transform)
    rows = []

    for name, model in models.items():
        y_pred_eur = predict_in_euros(model, X_test, log_transform)
        # Assegurar que no hi ha prediccions negatives
        y_pred_eur = np.maximum(y_pred_eur, 0)
        metrics = compute_metrics(y_true_eur, y_pred_eur)
        metrics["Model"] = name
        rows.append(metrics)
        logger.info(
            f"  {name:<25s} MAE={metrics['MAE']:>10,.0f}€  "
            f"RMSE={metrics['RMSE']:>10,.0f}€  "
            f"R²={metrics['R2']:.4f}  MAPE={metrics['MAPE']:.2f}%"
        )

    df = pd.DataFrame(rows).set_index("Model")
    return df[["MAE", "RMSE", "R2", "MAPE"]]


def plot_metrics_comparison(metrics_df: pd.DataFrame, output_dir: str):
    """Gràfic de barres comparant RMSE, MAE, MAPE i R² de tots els models."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # RMSE
    metrics_df["RMSE"].sort_values().plot(kind="barh", ax=axes[0, 0], color="steelblue")
    axes[0, 0].set_title("RMSE per Model (menor = millor)")
    axes[0, 0].set_xlabel("RMSE (€)")
    _add_value_labels(axes[0, 0], metrics_df["RMSE"].sort_values(), fmt="{:,.0f}€")

    # R²
    metrics_df["R2"].sort_values().plot(kind="barh", ax=axes[0, 1], color="coral")
    axes[0, 1].set_title("R² per Model (major = millor)")
    axes[0, 1].set_xlabel("R²")
    _add_value_labels(axes[0, 1], metrics_df["R2"].sort_values(), fmt="{:.4f}")

    # MAE
    metrics_df["MAE"].sort_values().plot(kind="barh", ax=axes[1, 0], color="teal")
    axes[1, 0].set_title("MAE per Model (menor = millor)")
    axes[1, 0].set_xlabel("MAE (€)")
    _add_value_labels(axes[1, 0], metrics_df["MAE"].sort_values(), fmt="{:,.0f}€")

    # MAPE
    metrics_df["MAPE"].sort_values().plot(kind="barh", ax=axes[1, 1], color="darkorange")
    axes[1, 1].set_title("MAPE per Model — % error mitjà (menor = millor)")
    axes[1, 1].set_xlabel("MAPE (%)")
    _add_value_labels(axes[1, 1], metrics_df["MAPE"].sort_values(), fmt="{:.1f}%")

    fig.suptitle("Comparació de Models (mètriques en euros reals)", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "08_metrics_comparison.png")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figura guardada: {path}")


def _add_value_labels(ax, series, fmt="{:.2f}"):
    """Afegeix etiquetes de valor a les barres horitzontals."""
    x_max = series.max()
    for i, (label, val) in enumerate(series.items()):
        ax.text(
            val + x_max * 0.01, i,
            fmt.format(val),
            va="center", ha="left", fontsize=8,
        )


def plot_predictions_vs_actual(model, X_test, y_test, name: str,
                                output_dir: str, log_transform: bool = True):
    """Scatter de prediccions vs valors reals (en euros)."""
    y_pred_eur = predict_in_euros(model, X_test, log_transform)
    y_true_eur = y_in_euros(y_test, log_transform)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_true_eur / 1000, y_pred_eur / 1000, alpha=0.3, s=8, color="steelblue")
    lims = [
        min(y_true_eur.min(), y_pred_eur.min()) / 1000,
        max(y_true_eur.max(), y_pred_eur.max()) / 1000,
    ]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Predicció perfecta")
    ax.set_xlabel("Preu Real (milers €)")
    ax.set_ylabel("Preu Predit (milers €)")
    ax.set_title(f"Prediccions vs Valors Reals — {name}")
    ax.legend()

    # Mètriques al títol
    mape = mean_absolute_percentage_error(y_true_eur, y_pred_eur) * 100
    rmse = np.sqrt(mean_squared_error(y_true_eur, y_pred_eur))
    ax.set_title(f"Prediccions vs Valors Reals — {name}\n"
                 f"RMSE={rmse:,.0f}€  MAPE={mape:.1f}%")

    path = os.path.join(output_dir, f"09_pred_vs_real_{name}.png")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figura guardada: {path}")


def plot_residuals(model, X_test, y_test, name: str,
                   output_dir: str, log_transform: bool = True):
    """Distribució dels residus (en euros)."""
    y_pred_eur = predict_in_euros(model, X_test, log_transform)
    y_true_eur = y_in_euros(y_test, log_transform)
    residuals = y_true_eur - y_pred_eur

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_pred_eur / 1000, residuals / 1000, alpha=0.3, s=8, color="teal")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Preu Predit (milers €)")
    axes[0].set_ylabel("Residu (milers €)")
    axes[0].set_title(f"Residus vs Prediccions — {name}")

    axes[1].hist(residuals / 1000, bins=60, color="teal", edgecolor="white")
    axes[1].set_xlabel("Residu (milers €)")
    axes[1].set_ylabel("Freqüència")
    axes[1].set_title(f"Distribució dels Residus — {name}")

    # Estadístiques dels residus
    mean_res = residuals.mean() / 1000
    std_res = residuals.std() / 1000
    axes[1].axvline(mean_res, color="red", linestyle="--",
                    label=f"Mitjana: {mean_res:.1f}k€")
    axes[1].legend(fontsize=9)

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
    axes[0].set_title("RMSE (CV 5-fold, escala log1p)")
    axes[0].set_xlabel("RMSE log1p")

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


def plot_log_transform_effect(y_raw: pd.Series, output_dir: str):
    """
    Visualitza l'efecte de la transformació logarítmica sobre el target.
    Mostra la distribució original (Precio) vs. log1p(Precio).
    """
    y_log = np.log1p(y_raw)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(y_raw / 1000, bins=80, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Precio (milers €)")
    axes[0].set_ylabel("Freqüència")
    axes[0].set_title(f"Distribució original — Skewness: {y_raw.skew():.2f}")

    axes[1].hist(y_log, bins=80, color="coral", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("log1p(Precio)")
    axes[1].set_ylabel("Freqüència")
    axes[1].set_title(f"Distribució log1p — Skewness: {y_log.skew():.2f}")

    fig.suptitle(
        "Efecte de la Transformació Logarítmica sobre el Target (Precio)",
        fontsize=13
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "00_log_transform_target.png")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figura transformació log guardada: {path}")
