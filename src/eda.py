"""
eda.py
------
Anàlisi exploratòria de dades (EDA) del mercat immobiliari de Catalunya.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import logging

import pyproj
import contextily as ctx
import geopandas as gpd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted")
FIGSIZE = (12, 6)


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figura guardada: {path}")


def plot_price_distribution(df: pd.DataFrame, output_dir: str):
    """Distribució del preu (histograma + KDE) i del log-preu."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

    axes[0].hist(df["Precio"] / 1000, bins=80, color="steelblue", edgecolor="white")
    axes[0].set_title("Distribució del Preu")
    axes[0].set_xlabel("Preu (milers €)")
    axes[0].set_ylabel("Freqüència")

    log_price = np.log1p(df["Precio"])
    axes[1].hist(log_price, bins=80, color="coral", edgecolor="white")
    axes[1].set_title("Distribució del Log(Preu)")
    axes[1].set_xlabel("log(Preu + 1)")
    axes[1].set_ylabel("Freqüència")

    fig.suptitle("Distribució del Preu dels Habitatges a Catalunya", fontsize=14)
    _save(fig, os.path.join(output_dir, "01_price_distribution.png"))


def plot_price_by_province(df: pd.DataFrame, output_dir: str):
    """Boxplot del preu per província."""
    if "NPRO" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    order = df.groupby("NPRO")["Precio"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="NPRO", y="Precio", order=order, ax=ax,
                showfliers=False, hue="NPRO", legend=False, palette="Set2")
    ax.set_title("Preu per Província (sense outliers extrems)")
    ax.set_xlabel("Província")
    ax.set_ylabel("Preu (€)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    _save(fig, os.path.join(output_dir, "02_price_by_province.png"))


def plot_price_vs_metros(df: pd.DataFrame, output_dir: str):
    # Filtra l'1% extrem de cada variable
    q_metros = df["Metros"].quantile(0.99)
    q_preu   = df["Precio"].quantile(0.99)
    clean    = df[(df["Metros"] > 0) & (df["Metros"] < q_metros) &
                  (df["Precio"] > 0) & (df["Precio"] < q_preu)]

    sample = clean.sample(min(5000, len(clean)), random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Esquerra: filtrat sense log
    axes[0].scatter(sample["Metros"], sample["Precio"] / 1000,
                    alpha=0.3, s=10, color="steelblue")
    axes[0].set_title("Filtrat (sense log)")
    axes[0].set_xlabel("Superfície (m²)")
    axes[0].set_ylabel("Preu (milers €)")

    # Dreta: escala log (revela l'estructura)
    axes[1].scatter(sample["Metros"], sample["Precio"] / 1000,
                    alpha=0.3, s=10, color="steelblue")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("Escala logarítmica")
    axes[1].set_xlabel("Superfície (m²)")
    axes[1].set_ylabel("Preu (milers €)")

    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "03_price_vs_metros.png"))


def plot_correlation_matrix(df: pd.DataFrame, output_dir: str):
    """Matriu de correlació de les variables numèriques."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("Matriu de Correlació")
    _save(fig, os.path.join(output_dir, "04_correlation_matrix.png"))


def plot_price_by_type(df: pd.DataFrame, output_dir: str):
    """Preu mitjà per tipus d'immoble."""
    col = "Inmueble" if "Inmueble" in df.columns else None
    if col is None:
        return
    means = df.groupby(col)["Precio"].median().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    means.plot(kind="bar", ax=ax, color="teal", edgecolor="white")
    ax.set_title("Preu Medià per Tipus d'Immoble (Top 10)")
    ax.set_xlabel("Tipus d'Immoble")
    ax.set_ylabel("Preu Medià (€)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    plt.xticks(rotation=30, ha="right")
    _save(fig, os.path.join(output_dir, "05_price_by_type.png"))


def plot_amenities_impact(df: pd.DataFrame, output_dir: str):
    """Impacte de terrassa, piscina i garatge en el preu."""
    amenities = ["Terraza", "Piscina", "Garaje"]
    amenities = [a for a in amenities if a in df.columns]
    if not amenities:
        return
    fig, axes = plt.subplots(1, len(amenities), figsize=(14, 5), sharey=True)
    for ax, col in zip(axes, amenities):
        sns.boxplot(data=df, x=col, y="Precio", ax=ax, showfliers=False,
                    hue=col, legend=False, palette=["#d9534f", "#5cb85c"])
        ax.set_title(col)
        ax.set_xlabel(f"{col} (0=No, 1=Sí)")
        ax.set_ylabel("Preu (€)" if col == amenities[0] else "")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    fig.suptitle("Impacte dels Serveis en el Preu", fontsize=13)
    _save(fig, os.path.join(output_dir, "06_amenities_impact.png"))


def plot_price_per_m2_map(df: pd.DataFrame, output_dir: str):
    if "Latitud" not in df.columns or "Longitud" not in df.columns:
        return

    df = df.copy()
    df["Latitud"]  = pd.to_numeric(df["Latitud"],  errors="coerce")
    df["Longitud"] = pd.to_numeric(df["Longitud"], errors="coerce")

    if "Precio_m2" not in df.columns:
        df["Precio_m2"] = df["Precio"] / df["Metros"].replace(0, np.nan)

    # Diagnòstic
    print(f"Lat:  {df['Latitud'].min():.2f} – {df['Latitud'].max():.2f}")
    print(f"Lon:  {df['Longitud'].min():.2f} – {df['Longitud'].max():.2f}")

    # Filtre bbox Catalunya completa
    sample = df.dropna(subset=["Latitud", "Longitud", "Precio_m2"])
    sample = sample[
        sample["Latitud"].between(40.5, 42.9) &
        sample["Longitud"].between(0.15, 3.35)
    ]

    # Filtre outliers preu
    p99 = sample["Precio_m2"].quantile(0.99)
    sample = sample[sample["Precio_m2"] < p99]
    sample = sample.sample(min(8000, len(sample)), random_state=42)

    if sample.empty:
        print("⚠️  Cap punt dins del bbox de Catalunya — revisa les coordenades!")
        return

    # Diagnòstic Barcelona
    bcn = sample[sample["Longitud"].between(1.9, 2.5) & sample["Latitud"].between(41.2, 41.6)]
    print(f"Punts zona Barcelona: {len(bcn)}")

    # Convertir a GeoDataFrame i projectar a Web Mercator
    gdf = gpd.GeoDataFrame(
        sample,
        geometry=gpd.points_from_xy(sample["Longitud"], sample["Latitud"]),
        crs="EPSG:4326"
    ).to_crs("EPSG:3857")

    fig, ax = plt.subplots(figsize=(12, 10))

    norm = Normalize(vmin=sample["Precio_m2"].quantile(0.05),
                     vmax=sample["Precio_m2"].quantile(0.95))

    gdf.plot(
        ax=ax,
        column="Precio_m2",
        cmap="YlOrRd",
        norm=norm,
        markersize=8,
        alpha=0.6,
        legend=False
    )

    # Forçar el viewport a tot Catalunya
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_min, y_min = transformer.transform(0.15, 40.5)
    x_max, y_max = transformer.transform(3.35, 42.9)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Mapa base
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=8)

    sm = ScalarMappable(cmap="YlOrRd", norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="€/m²", fraction=0.03, pad=0.04)

    ax.set_title("Distribució Geogràfica del Preu per m² a Catalunya", fontsize=14)
    ax.set_axis_off()

    _save(fig, os.path.join(output_dir, "07_price_map.png"))

def print_summary(df: pd.DataFrame):
    """Imprimeix un resum estadístic del dataset."""
    print("\n" + "="*60)
    print("RESUM DEL DATASET DE CATALUNYA")
    print("="*60)
    print(f"Registres totals : {len(df):,}")
    print(f"Variables        : {df.shape[1]}")
    print(f"\nEstadístiques del Preu (€):")
    stats = df["Precio"].describe()
    for k, v in stats.items():
        print(f"  {k:8s}: {v:>12,.0f}")
    if "NPRO" in df.columns:
        print(f"\nRegistres per Província:")
        for prov, cnt in df["NPRO"].value_counts().items():
            print(f"  {prov:<25s}: {cnt:>6,}")
    print("="*60 + "\n")


def run_eda(df_raw: pd.DataFrame, output_dir: str):
    """
    Executa tota l'anàlisi exploratòria i guarda les figures.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Dataset de Catalunya (abans del preprocessament complet,
        però amb Precio i Metros ja nets).
    output_dir : str
        Carpeta on guardar les figures.
    """
    logger.info("Iniciant EDA...")
    print_summary(df_raw)
    plot_price_distribution(df_raw, output_dir)
    plot_price_by_province(df_raw, output_dir)
    plot_price_vs_metros(df_raw, output_dir)
    plot_correlation_matrix(df_raw, output_dir)
    plot_price_by_type(df_raw, output_dir)
    plot_amenities_impact(df_raw, output_dir)
    plot_price_per_m2_map(df_raw, output_dir)
    logger.info("EDA completat.")
