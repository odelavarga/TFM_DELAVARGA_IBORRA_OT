"""
data_loader.py
--------------
Càrrega, filtratge per Catalunya i neteja inicial del dataset immobiliari.
"""

import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

# Columnes a eliminar (no aporten valor predictiu o estan excloses per disseny)
COLS_TO_DROP = [
    "Relacion",       # exclosa per requisit del projecte
    "Unnamed: 0",     # índex redundant
    "URL",            # no predictiva
    "URL_Cliente",    # no predictiva
    "ID_Cliente",     # identificador intern
    "Caracteristicas",# text lliure, no estructurat
    "Precision",      # metadada de geocodificació
    "CMUN", "CPRO", "CCA", "CUDIS",  # codis numèrics redundants amb NPRO/NMUN
]

NCA_CATALUNYA = "Catalu\xf1a"  # 'Cataluña' en latin-1


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Carrega el fitxer CSV original amb totes les comunitats autònomes.

    Parameters
    ----------
    filepath : str
        Ruta al fitxer DatosViviendas1.csv

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No s'ha trobat el fitxer: {filepath}")

    logger.info(f"Carregant dades des de: {filepath}")
    df = pd.read_csv(
        filepath,
        sep=";",
        encoding="latin-1",
        low_memory=False,
    )
    logger.info(f"Dataset complet carregat: {df.shape[0]:,} files x {df.shape[1]} columnes")
    return df


def filter_catalunya(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra el DataFrame per quedar-se només amb els registres de Catalunya.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df_cat = df[df["NCA"] == NCA_CATALUNYA].copy()
    logger.info(f"Registres de Catalunya: {len(df_cat):,} ({len(df_cat)/len(df)*100:.1f}% del total)")
    return df_cat


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina columnes no necessàries per al modelatge.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    cols_present = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_present)
    logger.info(f"Columnes eliminades: {cols_present}")
    logger.info(f"Columnes restants: {df.columns.tolist()}")
    return df


def load_and_filter(filepath: str, save_path: str = None) -> pd.DataFrame:
    """
    Pipeline complet: càrrega → filtre Catalunya → eliminació de columnes.

    Parameters
    ----------
    filepath : str
        Ruta al CSV original.
    save_path : str, optional
        Si s'especifica, guarda el resultat en aquest path.

    Returns
    -------
    pd.DataFrame
    """
    df = load_raw_data(filepath)
    df = filter_catalunya(df)
    df = drop_unused_columns(df)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False, encoding="utf-8")
        logger.info(f"Dataset de Catalunya guardat a: {save_path}")

    return df
