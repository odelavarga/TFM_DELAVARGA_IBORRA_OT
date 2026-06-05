# Ciència de dades aplicada al sector immobiliari: anàlisi del mercat de Catalunya

**Treball Final de Màster (TFM) — Màster en Ciència de Dades (UOC)**  
**Autor:** Ot De la Varga Iborra  
**Director:** Jorge Segura Gisbert  

---

## Descripció del projecte

Aquest projecte desenvolupa un sistema de valoració immobiliària automàtica (AVM) per al mercat de Catalunya, implementant i comparant múltiples models d'aprenentatge automàtic per predir el preu dels habitatges a partir de les seves característiques físiques, de localització i de l'entorn.

L'enfocament segueix la metodologia descrita a l'estat de l'art del TFM:
- **Regressió lineal** (baseline / model hedònic)
- **Random Forest**
- **XGBoost**
- **LightGBM**
- **Xarxa neuronal (MLP)**
- **Anàlisi d'interpretabilitat amb SHAP (XAI)**

---

## Estructura del projecte

```
real_estate_catalunya/
│
├── data/                        # Dades processades (generades per l'script)
│   ├── catalunya_clean.csv      # Dataset filtrat i net de Catalunya
│   └── .gitkeep
│
├── models/                      # Models entrenats serialitzats (.pkl)
│   └── .gitkeep
│
├── notebooks/                   # Jupyter Notebooks exploratòris
│   └── exploratory_analysis.ipynb
│
├── outputs/                     # Figures, gràfiques i resultats
│   └── .gitkeep
│
├── src/                         # Codi font modular
│   ├── __init__.py
│   ├── data_loader.py           # Càrrega i filtratge de dades
│   ├── preprocessing.py         # Neteja i preprocessament
│   ├── eda.py                   # Anàlisi exploratòria de dades
│   ├── models.py                # Definició i entrenament de models
│   ├── evaluation.py            # Mètriques i avaluació comparativa
│   └── explainability.py        # Interpretabilitat SHAP (XAI)
│
├── main.py                      # Pipeline principal
├── requirements.txt             # Dependències Python
├── .gitignore                   # Fitxers a ignorar per Git
└── README.md                    # Aquest fitxer
```

---

## Instal·lació

### Prerequisits
- Python 3.9+
- pip

### Instal·lar dependències

```bash
pip install -r requirements.txt
```

---

## Ús

### Executar el pipeline complet

```bash
python main.py
```

Això executarà:
1. Càrrega i filtratge de les dades per Catalunya
2. Preprocessament i neteja
3. Anàlisi exploratòria (EDA) amb gràfiques
4. Entrenament i comparació de models
5. Avaluació amb mètriques (RMSE, MAE, R², MAPE)
6. Anàlisi d'interpretabilitat SHAP

### Opcions de configuració

Es pot modificar el fitxer `main.py` per ajustar:
- Ruta del fitxer de dades original
- Models a entrenar
- Paràmetres de validació creuada

---

## Dades

Les dades provenen del fitxer `DatosViviendas1.csv`, que conté anuncis immobiliaris d'Espanya. El projecte filtra automàticament per **Catalunya** (`NCA == 'Cataluña'`) i elimina la columna `Relacion`.

### Variables principals

| Variable | Descripció |
|---|---|
| `Precio` | Preu de l'habitatge (€) — **variable objectiu** |
| `Metros` | Superfície en m² |
| `Habitaciones` | Nombre d'habitacions |
| `Aseos` | Nombre de banys/lavabos |
| `Terraza` | Té terrassa (0/1) |
| `Piscina` | Té piscina (0/1) |
| `Garaje` | Té garatge (0/1) |
| `Inmueble` | Tipus d'immoble |
| `CodigoPostal` | Codi postal |
| `Latitud` / `Longitud` | Coordenades geogràfiques |
| `NPRO` | Nom de la província |
| `NMUN` | Nom del municipi |

---

## Models implementats

| Model | Descripció |
|---|---|
| `LinearRegression` | Regressió lineal (baseline hedònic) |
| `RandomForest` | Random Forest Regressor |
| `XGBoost` | Gradient Boosting (XGBoost) |
| `LightGBM` | Gradient Boosting (LightGBM) |
| `MLP` | Xarxa neuronal multicapa |

### Mètriques d'avaluació
- **RMSE** — Root Mean Squared Error
- **MAE** — Mean Absolute Error
- **R²** — Coeficient de determinació
- **MAPE** — Mean Absolute Percentage Error

---

## Resultats

Els resultats es guarden a la carpeta `outputs/`:
- Gràfiques EDA (distribució de preus, correlacions, etc.)
- Comparativa de models
- Gràfiques SHAP (importància de variables)
- Taula resum de mètriques

---

## Llicència

Aquest projecte és part d'un Treball Final de Màster acadèmic. Llicència CC BY-NC 3.0 ES.
