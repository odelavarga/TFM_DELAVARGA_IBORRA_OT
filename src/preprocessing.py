"""
preprocessing.py
----------------
Neteja i preprocessament del dataset immobiliari de Catalunya.

Inclou:
  - Correcció de formats (lat/lon, dates)
  - Eliminació d'outliers (rang fix + IQR estadístic)
  - Imputació de valors nuls
  - Enginyeria de variables (features derivades)
  - Codificació de variables categòriques (LabelEncoder)
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ── Límits absoluts (rang fix) ────────────────────────────────────────────────
PRECIO_MIN = 10_000
PRECIO_MAX = 5_000_000
METROS_MIN = 10
METROS_MAX = 1_000
HABITACIONES_MAX = 20
ASEOS_MAX = 15

# ── Factor IQR per a outliers estadístics ────────────────────────────────────
IQR_FACTOR = 3.0   # k=3 conservador: elimina outliers extrems, no valors rars


def fix_latlon(df: pd.DataFrame) -> pd.DataFrame:
    """Converteix Latitud i Longitud a float (poden tenir punts com separadors de milers)."""
    for col in ["Latitud", "Longitud"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace(r"[^\d.\-]", "", regex=True)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fix_fecha(df: pd.DataFrame) -> pd.DataFrame:
    """Extreu any i mes de la columna Fecha."""
    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")
        df["Anyo"] = df["Fecha"].dt.year
        df["Mes"] = df["Fecha"].dt.month
        df = df.drop(columns=["Fecha"])
    return df


def remove_outliers_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pas 1: Eliminació d'outliers per rang fix (regles de negoci).
    Filtra preus i metres fora dels límits absoluts definits per domini.
    """
    n_before = len(df)
    df = df[df["Precio"].between(PRECIO_MIN, PRECIO_MAX)]
    df = df[df["Metros"].between(METROS_MIN, METROS_MAX)]
    df = df[df["Habitaciones"].fillna(0) <= HABITACIONES_MAX]
    df = df[df["Aseos"].fillna(0) <= ASEOS_MAX]
    n_removed = n_before - len(df)
    logger.info(
        f"  [Outliers rang fix] Eliminats: {n_removed:,} registres "
        f"({n_before:,} -> {len(df):,})"
    )
    return df


def remove_outliers_iqr(df: pd.DataFrame,
                         cols: list = None,
                         factor: float = IQR_FACTOR) -> pd.DataFrame:
    """
    Pas 2: Eliminació d'outliers estadístics amb criteri IQR.

    Per a cada columna indicada, elimina els registres on:
        valor < Q1 - factor * IQR   o   valor > Q3 + factor * IQR

    Amb factor=3.0 es conserven pràcticament tots els valors excepte els
    extremadament atípics (equivalent al criteri de Tukey estricte).

    Parameters
    ----------
    df : pd.DataFrame
    cols : list, optional
        Columnes sobre les quals aplicar IQR. Per defecte: Precio i Metros.
    factor : float
        Multiplicador de l'IQR. 1.5 = criteri Tukey estàndard,
        3.0 = criteri Tukey estricte (outliers extrems).

    Returns
    -------
    pd.DataFrame sense outliers extrems.
    """
    if cols is None:
        cols = ["Precio", "Metros"]

    n_before = len(df)
    mask = pd.Series(True, index=df.index)

    stats_log = []
    for col in cols:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        col_mask = df[col].between(lower, upper)
        n_removed_col = (~col_mask).sum()
        stats_log.append(
            f"    {col}: Q1={Q1:,.0f}, Q3={Q3:,.0f}, IQR={IQR:,.0f} "
            f"| Rang [{lower:,.0f}, {upper:,.0f}] "
            f"| Eliminats: {n_removed_col:,}"
        )
        mask = mask & col_mask

    df = df[mask]
    n_removed = n_before - len(df)
    logger.info(
        f"  [Outliers IQR k={factor}] Eliminats: {n_removed:,} registres "
        f"({n_before:,} -> {len(df):,})"
    )
    for line in stats_log:
        logger.info(line)
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline complet d'eliminació d'outliers en dos passos:
      1. Rang fix (regles de negoci)
      2. IQR estadístic (k=3, outliers extrems)

    Tractament estadístic aplicat:
    - Les variables Precio i Metros segueixen distribucions fortament asimètriques
      (skew positiu) en mercats immobiliaris.
    - El criteri IQR k=3 (Tukey estricte) elimina únicament els valors que estan
      a més de 3 vegades el rang interquartílic per sota de Q1 o per sobre de Q3.
    - Alternativa a z-score: l'IQR és robust davant distribucions no normals.
    - La transformació log1p posterior al target redueix l'efecte dels outliers
      residuals sobre l'entrenament del model.
    """
    logger.info("Eliminació d'outliers:")
    df = remove_outliers_range(df)
    df = remove_outliers_iqr(df, cols=["Precio", "Metros"], factor=IQR_FACTOR)
    return df


def log_outlier_stats(df: pd.DataFrame):
    """Registra estadístiques descriptives clau per a auditoria."""
    for col in ["Precio", "Metros"]:
        if col in df.columns:
            logger.info(
                f"  {col}: min={df[col].min():,.0f}  "
                f"p1={df[col].quantile(0.01):,.0f}  "
                f"mediana={df[col].median():,.0f}  "
                f"p99={df[col].quantile(0.99):,.0f}  "
                f"max={df[col].max():,.0f}  "
                f"skew={df[col].skew():.2f}"
            )


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa valors nuls."""
    # Numèriques: mediana
    for col in ["Habitaciones", "Aseos"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Binàries: 0 (absent)
    for col in ["Terraza", "Piscina", "Garaje"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # Categòriques: 'Desconegut'
    for col in ["Inmueble", "NPRO", "NMUN", "CodigoPostal"]:
        if col in df.columns:
            df[col] = df[col].fillna("Desconegut")

    # Coordenades: mediana
    for col in ["Latitud", "Longitud"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica variables categòriques amb Label Encoding."""
    cat_cols = ["Inmueble", "NPRO", "NMUN"]
    encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            logger.info(f"  {col}: {df[col].nunique()} categories -> codificat")
    return df, encoders


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea noves variables derivades."""
    # Preu per metre quadrat (s'exclou del model per evitar data leakage)
    df["Precio_m2"] = df["Precio"] / df["Metros"].replace(0, np.nan)

    # Ràtio habitacions/metres
    df["Hab_per_m2"] = df["Habitaciones"] / df["Metros"].replace(0, np.nan)

    # Nombre total de serveis (terrassa + piscina + garatge)
    df["Serveis"] = df[["Terraza", "Piscina", "Garaje"]].sum(axis=1)

    return df


def preprocess(df: pd.DataFrame):
    """
    Pipeline complet de preprocessament.

    Passos:
      1. Correcció de formats (lat/lon, dates)
      2. Eliminació de nuls en target i metres
      3. Eliminació d'outliers (rang fix + IQR estadístic k=3)
      4. Estadístiques descriptives post-neteja
      5. Imputació de valors nuls restants
      6. Enginyeria de variables
      7. Codificació de variables categòriques

    Returns
    -------
    df_clean : pd.DataFrame
        Dataset net i preparat per al modelatge.
    encoders : dict
        Diccionari amb els LabelEncoders per a cada variable categòrica.
    """
    logger.info("Iniciant preprocessament...")
    df = fix_latlon(df)
    df = fix_fecha(df)
    df = df.dropna(subset=["Precio", "Metros"])
    df = remove_outliers(df)

    logger.info("Estadístiques post-neteja:")
    log_outlier_stats(df)

    df = fill_missing(df)
    df = add_features(df)
    df, encoders = encode_categoricals(df)

    # Eliminar columnes originals categòriques (ja codificades)
    cols_orig = [c for c in ["Inmueble", "NPRO", "NMUN", "NCA", "ID", "CodigoPostal"] if c in df.columns]
    df = df.drop(columns=cols_orig)

    logger.info(f"Preprocessament completat: {df.shape[0]:,} registres, {df.shape[1]} variables")
    return df, encoders


def get_feature_target(df: pd.DataFrame, log_transform: bool = True):
    """
    Separa features (X) i target (y = Precio).

    Parameters
    ----------
    df : pd.DataFrame
    log_transform : bool
        Si True, aplica np.log1p() al target (Precio).
        La transformació logarítmica redueix la influència d'outliers
        residuals i fa que l'error es mesuri en escala relativa (MAPE).
        Per obtenir prediccions en euros cal aplicar np.expm1().

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series  (log1p(Precio) si log_transform=True, Precio si False)
    feature_names : list
    log_transform : bool  (passat tal qual per a referència futura)
    """
    target = "Precio"
    # Excloure Precio_m2 (derivat del target, causaria data leakage)
    exclude = [target, "Precio_m2"]
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols]
    y_raw = df[target]

    if log_transform:
        y = np.log1p(y_raw)
        logger.info(
            f"Target: log1p(Precio) aplicat | "
            f"Rang original: [{y_raw.min():,.0f}, {y_raw.max():,.0f}] EUR -> "
            f"log1p: [{y.min():.3f}, {y.max():.3f}]"
        )
    else:
        y = y_raw
        logger.info(f"Target: {target} (sense transformació logarítmica)")

    logger.info(f"Features: {feature_cols}")
    logger.info(f"Shape X={X.shape}, y={y.shape}")
    return X, y, feature_cols, log_transform
