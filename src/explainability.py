"""
explainability.py
-----------------
Interpretabilitat dels models amb SHAP (XAI) i importància de variables.

"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import logging

logger = logging.getLogger(__name__)

# Nombre màxim de mostres per calcular SHAP (per eficiència)
SHAP_MAX_SAMPLES = 500
# Nombre de features al summary plot
SHAP_TOP_N = 20


def plot_feature_importance(model, feature_names: list, name: str,
                             output_dir: str, top_n: int = SHAP_TOP_N):
    """
    Importància de variables intrínseca per a models basats en arbres
    (RandomForest, GradientBoosting, XGBoost, LightGBM).

    Utilitza feature_importances_ (Gini impurity / Gain):
    mesura quant redueix la impuresa cada variable als arbres de decisió.
    """
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
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_features)))
    ax.barh(top_features[::-1], top_importances[::-1], color=colors[::-1])
    ax.set_title(f"Importància de Variables (Gini/Gain) — {name} (Top {top_n})")
    ax.set_xlabel("Importància normalitzada")

    # Etiquetes de valor
    for i, (feat, imp) in enumerate(zip(top_features[::-1], top_importances[::-1])):
        ax.text(imp + 0.001, i, f"{imp:.4f}", va="center", ha="left", fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, f"12_feature_importance_{name}.png")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figura guardada: {path}")


def run_shap_analysis(model, X_sample: pd.DataFrame, name: str,
                       output_dir: str, feature_names: list = None,
                       max_samples: int = SHAP_MAX_SAMPLES,
                       log_transform: bool = True):
    """
    Anàlisi SHAP completa per al model indicat.

    Genera els següents gràfics XAI:
      1. Summary plot (beeswarm) — distribució d'impacte per variable
      2. Bar plot — importància global SHAP (|SHAP| mitja)
      3. Dependence plots — per les 3 variables més importants
      4. Waterfall plot — explicació d'una predicció individual

    Interpretació dels valors SHAP:
      - Valor SHAP positiu: la feature augmenta la predicció del preu.
      - Valor SHAP negatiu: la feature redueix la predicció del preu.
      - Si log_transform=True, els valors SHAP estan en escala log1p.
        Exemple: SHAP = 0.5 significa que la feature contribueix +0.5 unitats
        de log1p(Precio), que equival aproximadament a un factor e^0.5 ≈ 1.65
        (65% d'augment en el preu).

    Parameters
    ----------
    model : model entrenat (pot ser Pipeline de sklearn)
    X_sample : pd.DataFrame — mostres per calcular SHAP
    name : str — nom del model
    output_dir : str
    feature_names : list, optional — noms de les columnes
    max_samples : int — nombre màxim de mostres (per velocitat)
    log_transform : bool — si el target estava en log1p
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP no disponible. Instal·la'l amb: pip install shap")
        return

    # Reduir mostra si cal
    if len(X_sample) > max_samples:
        X_sample = X_sample.sample(max_samples, random_state=42)

    if feature_names is None:
        feature_names = list(X_sample.columns)

    # Assegurar que X_sample és un DataFrame amb els noms correctes
    if not isinstance(X_sample, pd.DataFrame):
        X_sample = pd.DataFrame(X_sample, columns=feature_names)
    else:
        X_sample = X_sample.copy()
        X_sample.columns = feature_names

    # Obtenir estimador base (per a Pipelines)
    estimator = model
    X_for_shap = X_sample.copy()
    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.items())
        for step_name, step in steps[:-1]:
            if hasattr(step, "transform"):
                X_transformed = step.transform(X_for_shap)
                X_for_shap = pd.DataFrame(X_transformed, columns=feature_names)
        estimator = steps[-1][1]

    os.makedirs(output_dir, exist_ok=True)

    try:
        # ── Crear Explainer ──────────────────────────────────────────────────
        if hasattr(estimator, "feature_importances_"):
            # TreeExplainer: eficient per a models basats en arbres
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_for_shap)
            expected_value = explainer.expected_value
        else:
            # LinearExplainer per a models lineals
            explainer = shap.LinearExplainer(estimator, X_for_shap)
            shap_values = explainer.shap_values(X_for_shap)
            expected_value = explainer.expected_value

        # Gestionar arrays 2D (regressions retornen array 2D en SHAP >= 0.40)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        if hasattr(shap_values, "values"):
            # Nou format Explanation object
            shap_arr = shap_values.values
        else:
            shap_arr = shap_values

        # Assegurar 2D
        if shap_arr.ndim == 1:
            shap_arr = shap_arr.reshape(1, -1)

        log_note = " [log1p(€)]" if log_transform else " [€]"

        # ── 1. Summary Plot (Beeswarm) ──────────────────────────────────────
        # Mostra distribució dels valors SHAP per a cada feature
        # Color: valor de la feature (vermell=alt, blau=baix)
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_arr, X_for_shap,
            feature_names=feature_names,
            show=False,
            max_display=SHAP_TOP_N,
        )
        plt.title(
            f"SHAP Summary (Beeswarm) — {name}{log_note}\n"
            "Vermell=valor alt de la feature | Blau=valor baix\n"
            "Dreta(+)=augmenta el preu | Esquerra(-)=redueix el preu"
        )
        path = os.path.join(output_dir, f"13_shap_summary_{name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP summary guardat: {path}")

        # ── 2. Bar Plot (Importància Global) ────────────────────────────────
        # Mostra la importància global = |SHAP| mitja per a cada feature
        fig = plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_arr, X_for_shap,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            max_display=SHAP_TOP_N,
        )
        plt.title(
            f"SHAP Feature Importance (|SHAP| mitja) — {name}{log_note}"
        )
        path = os.path.join(output_dir, f"14_shap_bar_{name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP bar guardat: {path}")

        # ── 3. Dependence Plots (top 3 variables) ───────────────────────────
        # Mostra la relació entre el valor d'una feature i el seu SHAP
        mean_abs_shap = np.abs(shap_arr).mean(axis=0)
        top3_idx = np.argsort(mean_abs_shap)[::-1][:3]

        for rank, feat_idx in enumerate(top3_idx):
            feat_name = feature_names[feat_idx]
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.dependence_plot(
                feat_idx,
                shap_arr,
                X_for_shap,
                feature_names=feature_names,
                ax=ax,
                show=False,
            )
            ax.set_title(
                f"SHAP Dependence Plot — {feat_name} ({name}){log_note}\n"
                f"Importància #{rank+1}: relació entre valor de '{feat_name}' i impacte SHAP"
            )
            ax.set_ylabel(f"SHAP value ({feat_name})")
            path = os.path.join(
                output_dir,
                f"15_shap_dependence_{name}_{feat_name.replace('/', '_')}.png"
            )
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"SHAP dependence ({feat_name}) guardat: {path}")

        # ── 4. Waterfall Plot (predicció individual) ─────────────────────────
        # Explica per quantes unitats cada feature contribueix a la predicció
        # d'una instància concreta (la instància del test set amb preu més alt)
        try:
            best_idx = int(np.argmax(np.abs(X_for_shap).sum(axis=1)))
            if isinstance(expected_value, (list, np.ndarray)):
                ev = float(expected_value[0])
            else:
                ev = float(expected_value)

            explanation = shap.Explanation(
                values=shap_arr[best_idx],
                base_values=ev,
                data=X_for_shap.iloc[best_idx].values,
                feature_names=feature_names,
            )
            fig, ax = plt.subplots(figsize=(10, 7))
            shap.waterfall_plot(explanation, max_display=15, show=False)
            plt.title(
                f"SHAP Waterfall — {name}{log_note}\n"
                "Contribució de cada feature a la predicció individual"
            )
            path = os.path.join(output_dir, f"16_shap_waterfall_{name}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"SHAP waterfall guardat: {path}")
        except Exception as e:
            logger.warning(f"No s'ha pogut generar el waterfall plot per {name}: {e}")

        # ── 5. Guardar valors SHAP en CSV ────────────────────────────────────
        shap_df = pd.DataFrame(shap_arr, columns=feature_names)
        shap_csv_path = os.path.join(output_dir, f"shap_values_{name}.csv")
        shap_df.describe().T.to_csv(shap_csv_path)
        logger.info(f"Estadístiques SHAP guardades: {shap_csv_path}")

        # ── 6. Resum textual de les variables més importants ─────────────────
        mean_abs = np.abs(shap_arr).mean(axis=0)
        feat_importance = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
        logger.info(f"\n  [SHAP] Variables més influents per a {name}:")
        for feat, imp in feat_importance.head(10).items():
            logger.info(f"    {feat:<20s}: SHAP mitja = {imp:.4f}{log_note}")

    except Exception as e:
        logger.error(f"Error en SHAP per {name}: {e}", exc_info=True)


def run_explainability(trained_models: dict, X_test: pd.DataFrame,
                        feature_names: list, output_dir: str,
                        best_model_name: str = None,
                        log_transform: bool = True):
    """
    Executa tota l'anàlisi d'interpretabilitat XAI.

    Tècniques aplicades:
      1. Feature Importance intrínseca (tots els models basats en arbres)
      2. SHAP TreeExplainer (millor model o tots si no s'especifica):
         - Summary plot (beeswarm)
         - Bar plot (importància global)
         - Dependence plots (top 3 features)
         - Waterfall plot (predicció individual)

    Parameters
    ----------
    trained_models : dict {nom: model entrenat}
    X_test : pd.DataFrame
    feature_names : list
    output_dir : str
    best_model_name : str, optional
        Si s'especifica, fa l'anàlisi SHAP completa només per al millor model.
    log_transform : bool
        Si True, indica que els models estan entrenats amb log1p(Precio).
        Impacta en les etiquetes dels gràfics SHAP.
    """
    logger.info("Iniciant anàlisi d'interpretabilitat XAI...")

    # ── Importància de variables intrínseca (tots els models) ────────────────
    logger.info("  1. Feature Importance (Gini/Gain) per a tots els models...")
    for name, model in trained_models.items():
        plot_feature_importance(model, feature_names, name, output_dir)

    # ── SHAP: millor model (o tots si no s'especifica) ───────────────────────
    shap_models = (
        {best_model_name: trained_models[best_model_name]}
        if best_model_name and best_model_name in trained_models
        else trained_models
    )

    logger.info(f"  2. SHAP XAI per a: {list(shap_models.keys())}...")
    for name, model in shap_models.items():
        logger.info(f"  Executant SHAP per a: {name}")
        run_shap_analysis(
            model,
            X_test.copy(),
            name,
            output_dir,
            feature_names=feature_names,
            log_transform=log_transform,
        )

    # ── Resum final d'importàncies ───────────────────────────────────────────
    _plot_combined_importance(trained_models, feature_names, output_dir)

    logger.info("Anàlisi d'interpretabilitat XAI completada.")


def _plot_combined_importance(trained_models: dict, feature_names: list,
                              output_dir: str, top_n: int = 15):
    """
    Gràfic combinat: importàncies de tots els models basats en arbres
    en un únic heatmap per comparar quines variables coincideixen.
    """
    importance_data = {}
    for name, model in trained_models.items():
        estimator = model
        if hasattr(model, "named_steps"):
            estimator = model.named_steps.get("model", model)
        
        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
            
            # --- MODIFICACIÓ CLAU: Normalització ---
            # Dividim per la suma total perquè tots els valors estiguin entre 0 i 1
            total_importance = importances.sum()
            if total_importance > 0:
                importances = importances / total_importance
            # ---------------------------------------
            
            importance_data[name] = importances

    if len(importance_data) < 2:
        return

    import_df = pd.DataFrame(importance_data, index=feature_names)

    # Ordenar per importància mitjana
    import_df["mean"] = import_df.mean(axis=1)
    import_df = import_df.sort_values("mean", ascending=False).head(top_n)
    import_df = import_df.drop(columns=["mean"])

    try:
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(max(8, len(import_df.columns) * 2), 8))
        sns.heatmap(
            import_df,
            annot=True, fmt=".3f",
            cmap="YlOrRd",
            ax=ax,
            linewidths=0.5,
            # Afegim vmin i vmax per assegurar que l'escala de colors sigui idèntica
            vmin=0.0, vmax=import_df.max().max() 
        )
        ax.set_title(
            f"Comparació d'Importàncies de Variables — Top {top_n}\n"
            "(Valors normalitzats per model en percentatge [0-1])"
        )
        ax.set_ylabel("Variable")
        ax.set_xlabel("Model")
        plt.tight_layout()
        path = os.path.join(output_dir, "17_combined_feature_importance.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Heatmap importàncies combinat guardat: {path}")
    except Exception as e:
        logger.warning(f"No s'ha pogut generar el heatmap combinat: {e}")
