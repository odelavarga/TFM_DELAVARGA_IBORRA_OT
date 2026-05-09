"""
preprocessing.py
----------------
Neteja i preprocessament del dataset immobiliari de Catalunya.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# Límits per filtrar outliers de preu i metres
PRECIO_MIN = 10_000
PRECIO_MAX = 5_000_000
METROS_MIN = 10
METROS_MAX = 1_000
HABITACIONES_MAX = 20
ASEOS_MAX = 15


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


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina registres amb valors fora de rang per Precio i Metros."""
    n_before = len(df)
    df = df[df["Precio"].between(PRECIO_MIN, PRECIO_MAX)]
    df = df[df["Metros"].between(METROS_MIN, METROS_MAX)]
    df = df[df["Habitaciones"].fillna(0) <= HABITACIONES_MAX]
    df = df[df["Aseos"].fillna(0) <= ASEOS_MAX]
    logger.info(f"Outliers eliminats: {n_before - len(df):,} registres ({n_before:,} -> {len(df):,})")
    return df


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
    # Preu per metre quadrat
    df["Precio_m2"] = df["Precio"] / df["Metros"].replace(0, np.nan)

    # Ràtio habitacions/metres
    df["Hab_per_m2"] = df["Habitaciones"] / df["Metros"].replace(0, np.nan)

    # Nombre total de serveis (terrassa + piscina + garatge)
    df["Serveis"] = df[["Terraza", "Piscina", "Garaje"]].sum(axis=1)

    return df


def preprocess(df: pd.DataFrame):
    """
    Pipeline complet de preprocessament.

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
    df = fill_missing(df)
    df = add_features(df)
    df, encoders = encode_categoricals(df)

    # Eliminar columnes originals categòriques (ja codificades)
    cols_orig = [c for c in ["Inmueble", "NPRO", "NMUN", "NCA", "ID", "CodigoPostal"] if c in df.columns]
    df = df.drop(columns=cols_orig)

    logger.info(f"Preprocessament completat: {df.shape[0]:,} registres, {df.shape[1]} variables")
    return df, encoders


def get_feature_target(df: pd.DataFrame):
    """
    Separa features (X) i target (y = Precio).

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    feature_names : list
    """
    target = "Precio"
    # Excloure Precio_m2 (derivat del target, causaria data leakage)
    exclude = [target, "Precio_m2"]
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols]
    y = df[target]
    logger.info(f"Features: {feature_cols}")
    logger.info(f"Target: {target} | Shape X={X.shape}, y={y.shape}")
    return X, y, feature_cols
