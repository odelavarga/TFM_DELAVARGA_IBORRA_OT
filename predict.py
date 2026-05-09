"""
predict.py
----------
Estimació del preu d'un habitatge a Catalunya a partir de les seves
característiques, utilitzant tots els models entrenats.

Ús interactiu (demana les dades per teclat):
    python predict.py

Ús amb arguments (sense interacció):
    python predict.py --metros 80 --habitaciones 3 --aseos 2 \
                      --terraza 1 --piscina 0 --garaje 1 \
                      --provincia Barcelona --municipio "Sant Cugat del Valles" \
                      --latitud 41.47 --longitud 2.08 \
                      --anyo 2023 --mes 6

Ús amb fitxer JSON d'entrada:
    python predict.py --input habitatge.json

Ús en mode batch (múltiples habitatges des d'un CSV):
    python predict.py --batch habitatges.csv
"""

import argparse
import json
import logging
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Forçar UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("predict")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# ── Mapeig de províncies a codis (LabelEncoder de NPRO) ─────────────────────
# Aquests valors corresponen al LabelEncoder entrenat sobre les dades de Catalunya
PROVINCE_MAP = {
    "barcelona": 0,
    "girona": 1,
    "lleida": 2,
    "tarragona": 3,
}

# Valors de referència per al municipio (NMUN_enc) si no es coneix el municipi
# S'utilitza la mediana del dataset com a valor per defecte
DEFAULT_NMUN_ENC = 412  # mediana aproximada dels 825 municipis


def load_models() -> dict:
    """Carrega tots els models .joblib disponibles."""
    models = {}
    if not os.path.isdir(MODELS_DIR):
        raise FileNotFoundError(
            f"No s'ha trobat la carpeta de models: {MODELS_DIR}\n"
            "Executa primer: python main.py --data ../Data/DatosViviendas1.csv"
        )
    for fname in sorted(os.listdir(MODELS_DIR)):
        if fname.endswith(".joblib"):
            name = fname.replace(".joblib", "")
            path = os.path.join(MODELS_DIR, fname)
            models[name] = joblib.load(path)
            logger.info(f"  Model carregat: {name}")
    if not models:
        raise FileNotFoundError(
            "No s'han trobat models a la carpeta 'models/'.\n"
            "Executa primer: python main.py --data ../Data/DatosViviendas1.csv"
        )
    return models


def get_province_enc(provincia: str) -> int:
    """Retorna el codi numèric de la província."""
    key = provincia.strip().lower()
    # Cerca parcial
    for k, v in PROVINCE_MAP.items():
        if key in k or k in key:
            return v
    logger.warning(
        f"Provincia '{provincia}' no reconeguda. "
        f"Opcions: {list(PROVINCE_MAP.keys())}. S'utilitza Barcelona (0)."
    )
    return 0


def get_nmun_enc(municipio: str) -> int:
    """
    Intenta trobar el codi del municipi al dataset de Catalunya.
    Si no el troba, retorna la mediana.
    """
    cat_csv = os.path.join(DATA_DIR, "catalunya_clean.csv")
    if not os.path.isfile(cat_csv):
        return DEFAULT_NMUN_ENC

    try:
        df = pd.read_csv(cat_csv, usecols=["NMUN"], nrows=5000)
        # Construir mapeig nom -> codi (LabelEncoder ordena alfabèticament)
        unique_muns = sorted(df["NMUN"].dropna().unique())
        mun_map = {m: i for i, m in enumerate(unique_muns)}

        key = municipio.strip().lower()
        for mun, code in mun_map.items():
            if key in mun.lower() or mun.lower() in key:
                logger.info(f"  Municipi trobat: '{mun}' -> codi {code}")
                return code
        logger.warning(
            f"Municipi '{municipio}' no trobat. S'utilitza la mediana ({DEFAULT_NMUN_ENC})."
        )
    except Exception:
        pass
    return DEFAULT_NMUN_ENC


def build_feature_vector(params: dict) -> pd.DataFrame:
    """
    Construeix el vector de features a partir dels paràmetres de l'habitatge.

    Paràmetres esperats (dict):
        metros        : float  — superfície en m²
        habitaciones  : int    — nombre d'habitacions
        aseos         : int    — nombre de banys/lavabos
        terraza       : int    — 1 si té terrassa, 0 si no
        piscina       : int    — 1 si té piscina, 0 si no
        garaje        : int    — 1 si té garatge, 0 si no
        latitud       : float  — latitud geogràfica
        longitud      : float  — longitud geogràfica
        anyo          : int    — any de l'anunci
        mes           : int    — mes de l'anunci (1-12)
        provincia     : str    — nom de la província (Barcelona/Girona/Lleida/Tarragona)
        municipio     : str    — nom del municipi (opcional)
    """
    metros = float(params.get("metros", 80))
    habitaciones = float(params.get("habitaciones", 3))
    aseos = float(params.get("aseos", 1))
    terraza = int(params.get("terraza", 0))
    piscina = int(params.get("piscina", 0))
    garaje = int(params.get("garaje", 0))
    latitud = float(params.get("latitud", 41.39))   # Barcelona per defecte
    longitud = float(params.get("longitud", 2.17))
    anyo = int(params.get("anyo", 2023))
    mes = int(params.get("mes", 6))
    provincia = str(params.get("provincia", "Barcelona"))
    municipio = str(params.get("municipio", ""))

    # Features derivades
    hab_per_m2 = habitaciones / metros if metros > 0 else 0
    serveis = terraza + piscina + garaje
    inmueble_enc = 0  # Pis (únic tipus al dataset de Catalunya filtrat)
    npro_enc = get_province_enc(provincia)
    nmun_enc = get_nmun_enc(municipio) if municipio else DEFAULT_NMUN_ENC

    # Ordre de features igual que al model entrenat
    feature_names = [
        "Habitaciones", "Aseos", "Terraza", "Piscina", "Garaje",
        "Metros", "Latitud", "Longitud", "Anyo", "Mes",
        "Hab_per_m2", "Serveis", "Inmueble_enc", "NPRO_enc", "NMUN_enc",
    ]
    values = [
        habitaciones, aseos, terraza, piscina, garaje,
        metros, latitud, longitud, anyo, mes,
        hab_per_m2, serveis, inmueble_enc, npro_enc, nmun_enc,
    ]
    return pd.DataFrame([values], columns=feature_names)


def predict_all_models(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    """
    Fa la predicció amb tots els models i retorna un DataFrame de resultats.

    Returns
    -------
    pd.DataFrame amb columnes: Model, Preu_Estimat, Diferencia_vs_Millor
    """
    results = {}
    for name, model in models.items():
        try:
            pred = float(model.predict(X)[0])
            pred = max(pred, 0)  # No pot ser negatiu
            results[name] = pred
        except Exception as e:
            logger.warning(f"  Error predint amb {name}: {e}")

    df = pd.DataFrame(
        list(results.items()), columns=["Model", "Preu_Estimat"]
    ).sort_values("Preu_Estimat")

    # Diferència respecte a la mediana de les prediccions
    mediana = df["Preu_Estimat"].median()
    df["Diferencia_vs_Mediana"] = df["Preu_Estimat"] - mediana
    df["Diferencia_pct"] = (df["Diferencia_vs_Mediana"] / mediana * 100).round(1)

    return df.reset_index(drop=True)


def print_results(results_df: pd.DataFrame, params: dict):
    """Imprimeix els resultats de forma llegible."""
    print("\n" + "=" * 60)
    print("  ESTIMACIO DEL PREU DE L'HABITATGE")
    print("=" * 60)
    print("\nCaracteristiques introduides:")
    print(f"  Superficie    : {params.get('metros', '?')} m2")
    print(f"  Habitacions   : {params.get('habitaciones', '?')}")
    print(f"  Banys         : {params.get('aseos', '?')}")
    print(f"  Terrassa      : {'Si' if params.get('terraza', 0) else 'No'}")
    print(f"  Piscina       : {'Si' if params.get('piscina', 0) else 'No'}")
    print(f"  Garatge       : {'Si' if params.get('garaje', 0) else 'No'}")
    print(f"  Provincia     : {params.get('provincia', 'Barcelona')}")
    if params.get("municipio"):
        print(f"  Municipi      : {params.get('municipio')}")
    print(f"  Latitud       : {params.get('latitud', 41.39)}")
    print(f"  Longitud      : {params.get('longitud', 2.17)}")
    print(f"  Any/Mes       : {params.get('anyo', 2023)}/{params.get('mes', 6)}")

    print("\nPrediccions per model:")
    print(f"  {'Model':<22} {'Preu Estimat':>14}  {'Dif. vs Mediana':>16}")
    print("  " + "-" * 56)
    for _, row in results_df.iterrows():
        signe = "+" if row["Diferencia_vs_Mediana"] >= 0 else ""
        print(
            f"  {row['Model']:<22} {row['Preu_Estimat']:>12,.0f} EUR  "
            f"  {signe}{row['Diferencia_pct']:>+.1f}%"
        )

    mediana = results_df["Preu_Estimat"].median()
    mitjana = results_df["Preu_Estimat"].mean()
    min_pred = results_df["Preu_Estimat"].min()
    max_pred = results_df["Preu_Estimat"].max()

    print("\n" + "-" * 60)
    print(f"  Mediana de prediccions : {mediana:>12,.0f} EUR")
    print(f"  Mitjana de prediccions : {mitjana:>12,.0f} EUR")
    print(f"  Rang                   : {min_pred:,.0f} - {max_pred:,.0f} EUR")
    print(f"  Preu/m2 (mediana)      : {mediana/float(params.get('metros',80)):>12,.0f} EUR/m2")
    print("=" * 60)


def plot_predictions(results_df: pd.DataFrame, params: dict, output_path: str):
    """Genera un gràfic de barres comparant les prediccions de tots els models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Estimacio del Preu — {params.get('metros')} m2, "
        f"{params.get('habitaciones')} hab., {params.get('provincia')}",
        fontsize=13, fontweight="bold"
    )

    # ── Gràfic 1: Barres horitzontals ────────────────────────────────────────
    ax = axes[0]
    colors = plt.cm.RdYlGn(
        np.linspace(0.2, 0.8, len(results_df))
    )
    bars = ax.barh(
        results_df["Model"],
        results_df["Preu_Estimat"],
        color=colors,
        edgecolor="white",
        height=0.6,
    )
    # Etiquetes de valor
    for bar, val in zip(bars, results_df["Preu_Estimat"]):
        ax.text(
            bar.get_width() + results_df["Preu_Estimat"].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f} EUR",
            va="center", ha="left", fontsize=9,
        )
    # Línia de mediana
    mediana = results_df["Preu_Estimat"].median()
    ax.axvline(mediana, color="navy", linestyle="--", linewidth=1.5,
               label=f"Mediana: {mediana:,.0f} EUR")
    ax.set_xlabel("Preu Estimat (EUR)")
    ax.set_title("Prediccio per Model")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax.legend(fontsize=9)
    ax.set_xlim(0, results_df["Preu_Estimat"].max() * 1.25)

    # ── Gràfic 2: Punt central amb interval ──────────────────────────────────
    ax2 = axes[1]
    models_list = results_df["Model"].tolist()
    preus = results_df["Preu_Estimat"].tolist()
    y_pos = range(len(models_list))

    ax2.scatter(preus, y_pos, s=120, zorder=5,
                c=colors, edgecolors="black", linewidths=0.5)
    ax2.axvline(mediana, color="navy", linestyle="--", linewidth=1.5,
                label=f"Mediana: {mediana:,.0f} EUR")
    ax2.axvspan(
        results_df["Preu_Estimat"].min(),
        results_df["Preu_Estimat"].max(),
        alpha=0.08, color="blue", label="Rang de prediccions"
    )
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels(models_list)
    ax2.set_xlabel("Preu Estimat (EUR)")
    ax2.set_title("Dispersio de Prediccions")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax2.legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Grafic guardat: {output_path}")


def plot_batch_predictions(all_results: list, output_path: str):
    """
    Gràfic per a mode batch: comparació de prediccions per a múltiples habitatges.
    all_results: llista de (label, results_df)
    """
    n = len(all_results)
    fig, ax = plt.subplots(figsize=(max(10, n * 1.5), 6))

    model_names = all_results[0][1]["Model"].tolist()
    x = np.arange(n)
    width = 0.8 / len(model_names)
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

    for i, (mname, color) in enumerate(zip(model_names, colors)):
        preus = [res["Preu_Estimat"][res["Model"] == mname].values[0]
                 for _, res in all_results]
        offset = (i - len(model_names) / 2) * width + width / 2
        ax.bar(x + offset, preus, width, label=mname, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in all_results], rotation=15, ha="right")
    ax.set_ylabel("Preu Estimat (EUR)")
    ax.set_title("Comparacio de Prediccions — Multiples Habitatges")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Grafic batch guardat: {output_path}")


def interactive_input() -> dict:
    """Demana les característiques de l'habitatge per teclat."""
    print("\n" + "=" * 60)
    print("  ESTIMADOR DE PREU D'HABITATGE — CATALUNYA")
    print("  (prem Enter per acceptar el valor per defecte)")
    print("=" * 60)

    def ask(prompt, default, cast=float):
        val = input(f"  {prompt} [{default}]: ").strip()
        return cast(val) if val else cast(default)

    def ask_yn(prompt, default=0):
        val = input(f"  {prompt} [{'S' if default else 'N'}] (S/N): ").strip().lower()
        if val in ("s", "si", "1", "y", "yes"):
            return 1
        if val in ("n", "no", "0"):
            return 0
        return default

    params = {}
    params["metros"] = ask("Superficie (m2)", 80, float)
    params["habitaciones"] = ask("Nombre d'habitacions", 3, int)
    params["aseos"] = ask("Nombre de banys/lavabos", 1, int)
    params["terraza"] = ask_yn("Te terrassa?", 0)
    params["piscina"] = ask_yn("Te piscina?", 0)
    params["garaje"] = ask_yn("Te garatge?", 0)

    print("\n  Provincies disponibles: Barcelona, Girona, Lleida, Tarragona")
    prov = input("  Provincia [Barcelona]: ").strip()
    params["provincia"] = prov if prov else "Barcelona"

    mun = input("  Municipi (opcional, Enter per ometre): ").strip()
    if mun:
        params["municipio"] = mun

    lat = input("  Latitud (opcional, Enter per ometre): ").strip()
    lon = input("  Longitud (opcional, Enter per ometre): ").strip()
    if lat:
        params["latitud"] = float(lat)
    if lon:
        params["longitud"] = float(lon)

    anyo = input("  Any de l'anunci [2023]: ").strip()
    params["anyo"] = int(anyo) if anyo else 2023
    mes = input("  Mes de l'anunci (1-12) [6]: ").strip()
    params["mes"] = int(mes) if mes else 6

    return params


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimador de preu d'habitatge a Catalunya"
    )
    parser.add_argument("--metros", type=float, help="Superficie en m2")
    parser.add_argument("--habitaciones", type=int, help="Nombre d'habitacions")
    parser.add_argument("--aseos", type=int, help="Nombre de banys")
    parser.add_argument("--terraza", type=int, choices=[0, 1], help="Terrassa (0/1)")
    parser.add_argument("--piscina", type=int, choices=[0, 1], help="Piscina (0/1)")
    parser.add_argument("--garaje", type=int, choices=[0, 1], help="Garatge (0/1)")
    parser.add_argument("--provincia", type=str, help="Provincia (Barcelona/Girona/Lleida/Tarragona)")
    parser.add_argument("--municipio", type=str, help="Nom del municipi")
    parser.add_argument("--latitud", type=float, help="Latitud")
    parser.add_argument("--longitud", type=float, help="Longitud")
    parser.add_argument("--anyo", type=int, help="Any de l'anunci")
    parser.add_argument("--mes", type=int, help="Mes de l'anunci (1-12)")
    parser.add_argument("--input", type=str, help="Fitxer JSON amb les caracteristiques")
    parser.add_argument("--batch", type=str, help="Fitxer CSV amb multiples habitatges")
    parser.add_argument("--no-plot", action="store_true", help="No generar grafics")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Carregant models entrenats...")
    models = load_models()
    logger.info(f"  {len(models)} models carregats: {list(models.keys())}")

    # ── Mode batch ───────────────────────────────────────────────────────────
    if args.batch:
        logger.info(f"Mode batch: {args.batch}")
        batch_df = pd.read_csv(args.batch)
        all_results = []
        all_rows = []

        for i, row in batch_df.iterrows():
            params = row.to_dict()
            label = params.pop("label", f"Habitatge {i+1}")
            X = build_feature_vector(params)
            results_df = predict_all_models(models, X)
            all_results.append((label, results_df))
            print_results(results_df, {**params, "label": label})

            # Afegir a la taula resum
            for _, r in results_df.iterrows():
                all_rows.append({
                    "Habitatge": label,
                    "Model": r["Model"],
                    "Preu_Estimat": r["Preu_Estimat"],
                })

        # Guardar CSV resum
        summary_path = os.path.join(OUTPUTS_DIR, "batch_predictions.csv")
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        pd.DataFrame(all_rows).to_csv(summary_path, index=False)
        logger.info(f"Resum batch guardat: {summary_path}")

        if not args.no_plot:
            plot_batch_predictions(
                all_results,
                os.path.join(OUTPUTS_DIR, "batch_predictions_comparison.png")
            )
        return

    # ── Mode fitxer JSON ─────────────────────────────────────────────────────
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            params = json.load(f)
        logger.info(f"Parametres carregats des de: {args.input}")

    # ── Mode arguments de línia de comandes ──────────────────────────────────
    elif args.metros is not None:
        params = {k: v for k, v in vars(args).items()
                  if v is not None and k not in ("input", "batch", "no_plot")}

    # ── Mode interactiu ──────────────────────────────────────────────────────
    else:
        params = interactive_input()

    # ── Predicció ────────────────────────────────────────────────────────────
    logger.info("Construint vector de features...")
    X = build_feature_vector(params)
    logger.info(f"  Features: {X.to_dict(orient='records')[0]}")

    logger.info("Calculant prediccions...")
    results_df = predict_all_models(models, X)

    # ── Resultats ────────────────────────────────────────────────────────────
    print_results(results_df, params)

    # ── Guardar CSV ──────────────────────────────────────────────────────────
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUTS_DIR, "prediction_result.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Resultats guardats: {csv_path}")

    # ── Gràfic ───────────────────────────────────────────────────────────────
    if not args.no_plot:
        plot_path = os.path.join(OUTPUTS_DIR, "prediction_comparison.png")
        plot_predictions(results_df, params, plot_path)
        print(f"\n  Grafic guardat: {plot_path}")

    print(f"  CSV guardat:   {csv_path}")


if __name__ == "__main__":
    main()
