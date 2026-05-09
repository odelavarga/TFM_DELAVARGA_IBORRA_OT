# Documentació del Projecte: Valoració Immobiliària a Catalunya

## Descripció General

Aquest projecte implementa un pipeline complet d'aprenentatge automàtic per a la **valoració immobiliària a Catalunya**. A partir d'un dataset de prop de 955.000 habitatges d'Espanya, es filtren els 89.205 registres de Catalunya i s'entrenen 7 models de regressió per predir el preu de venda d'un habitatge.

---

## Estructura del Projecte

```
real_estate_catalunya/
│
├── main.py                  ← Pipeline principal (punt d'entrada)
├── predict.py               ← Estimador de preu per a nous habitatges
├── run_predict.bat          ← Script d'ajuda per executar predict.py (Windows)
├── requirements.txt         ← Dependències Python
├── README.md                ← Guia ràpida d'ús
├── DOCUMENTACIO.md          ← Aquest fitxer (documentació detallada)
├── .gitignore               ← Fitxers exclosos del repositori Git
├── pipeline.log             ← Log d'execució (generat automàticament)
│
├── src/                     ← Codi font modular
│   ├── __init__.py
│   ├── data_loader.py       ← Càrrega i filtratge de dades
│   ├── preprocessing.py     ← Neteja i preprocessament
│   ├── eda.py               ← Anàlisi exploratòria (gràfics)
│   ├── models.py            ← Definició i entrenament de models
│   ├── evaluation.py        ← Mètriques i gràfics d'avaluació
│   └── explainability.py    ← Interpretabilitat (SHAP, Feature Importance)
│
├── examples/                ← Fitxers d'exemple per a predict.py
│   ├── habitatge_exemple.json   ← Exemple d'un habitatge (format JSON)
│   └── habitatges_batch.csv     ← Exemple de múltiples habitatges (format CSV)
│
├── data/                    ← Dades processades (generat automàticament)
│   └── catalunya_clean.csv  ← Dataset filtrat de Catalunya (89.205 registres)
│
├── models/                  ← Models entrenats serialitzats (generat automàticament)
│   ├── LinearRegression.joblib
│   ├── Ridge.joblib
│   ├── Lasso.joblib
│   ├── RandomForest.joblib
│   ├── GradientBoosting.joblib
│   ├── XGBoost.joblib
│   └── LightGBM.joblib
│
├── outputs/                 ← Gràfics i resultats (generat automàticament)
│   ├── 01_price_distribution.png
│   ├── 02_price_by_province.png
│   ├── 03_price_vs_metros.png
│   ├── 04_price_by_type.png
│   ├── 05_amenities_boxplot.png
│   ├── 06_correlation_heatmap.png
│   ├── 07_geo_price_map.png
│   ├── 08_metrics_comparison.png
│   ├── 09_pred_vs_real_<millor_model>.png
│   ├── 10_residuals_<millor_model>.png
│   ├── 11_cv_results.png
│   ├── 12_feature_importance_<model>.png  (un per cada model d'arbres)
│   ├── 13_shap_beeswarm_<model>.png       (si s'executa amb SHAP)
│   ├── 14_shap_dependence_<feature>.png   (si s'executa amb SHAP)
│   ├── metrics_summary.csv               ← Taula resum de mètriques
│   ├── prediction_result.csv             ← Resultat de l'última predicció individual
│   ├── prediction_comparison.png         ← Gràfic de l'última predicció individual
│   ├── batch_predictions.csv             ← Resultats del mode batch
│   └── batch_predictions_comparison.png  ← Gràfic comparatiu del mode batch
│
└── notebooks/               ← Carpeta per a Jupyter Notebooks (opcional)
```

---

## Fitxers de Codi Font (`src/`)

### `data_loader.py`
**Funció:** Càrrega el CSV original i filtra les dades de Catalunya.

**Què fa:**
- Llegeix el fitxer `DatosViviendas1.csv` (954.157 registres, 27 columnes)
- Filtra per `NCA == "Cataluña"` → 89.205 registres
- **Elimina la columna `Relacion`** (tal com s'especifica al TFM)
- Elimina altres columnes innecessàries: `Unnamed: 0`, `URL`, `URL_Cliente`, `ID_Cliente`, `Caracteristicas`, `Precision`, `CMUN`, `CPRO`, `CCA`, `CUDIS`
- Guarda el dataset net a `data/catalunya_clean.csv`

**Columnes resultants (16):**
`Fecha`, `ID`, `Inmueble`, `Habitaciones`, `Aseos`, `Terraza`, `Piscina`, `Garaje`, `Precio`, `Metros`, `CodigoPostal`, `Latitud`, `Longitud`, `NPRO`, `NCA`, `NMUN`

---

### `preprocessing.py`
**Funció:** Neteja, transformació i preparació de les dades per al modelatge.

**Passos del pipeline:**
1. **`fix_latlon()`** — Converteix Latitud/Longitud a float (gestiona comes com separadors decimals)
2. **`fix_fecha()`** — Extreu `Anyo` i `Mes` de la columna `Fecha` i elimina la columna original
3. **`dropna()`** — Elimina registres sense `Precio` o `Metros` (valors obligatoris)
4. **`remove_outliers()`** — Filtra valors fora de rang:
   - Preu: entre 10.000€ i 5.000.000€
   - Metres: entre 10 m² i 1.000 m²
   - Habitacions: màxim 20
   - Banys: màxim 15
   - Resultat: elimina 1.419 registres → 86.866 registres finals
5. **`fill_missing()`** — Imputa valors nuls:
   - Numèriques (`Habitaciones`, `Aseos`): mediana
   - Binàries (`Terraza`, `Piscina`, `Garaje`): 0 (absent)
   - Categòriques (`Inmueble`, `NPRO`, `NMUN`, `CodigoPostal`): "Desconegut"
   - Coordenades: mediana
6. **`add_features()`** — Crea noves variables derivades:
   - `Precio_m2`: preu per metre quadrat (s'exclou del model per evitar data leakage)
   - `Hab_per_m2`: ràtio habitacions per metre quadrat
   - `Serveis`: suma de Terraza + Piscina + Garaje
7. **`encode_categoricals()`** — Label Encoding per a `Inmueble`, `NPRO`, `NMUN`

**Variables finals del model (15 features):**
`Habitaciones`, `Aseos`, `Terraza`, `Piscina`, `Garaje`, `Metros`, `Latitud`, `Longitud`, `Anyo`, `Mes`, `Hab_per_m2`, `Serveis`, `Inmueble_enc`, `NPRO_enc`, `NMUN_enc`

**Target:** `Precio`

---

### `eda.py`
**Funció:** Genera gràfics d'anàlisi exploratòria de dades (EDA).

**Gràfics generats:**

| Fitxer | Descripció |
|--------|-----------|
| `01_price_distribution.png` | Histograma de la distribució del preu + boxplot |
| `02_price_by_province.png` | Boxplot del preu per província (Barcelona, Girona, Lleida, Tarragona) |
| `03_price_vs_metros.png` | Scatter plot preu vs superfície (m²) amb línia de tendència |
| `04_price_by_type.png` | Preu medià per tipus d'immoble (pis, casa, àtic, etc.) |
| `05_amenities_boxplot.png` | Comparació de preus amb/sense terrassa, piscina i garatge |
| `06_correlation_heatmap.png` | Mapa de calor de correlació entre variables numèriques |
| `07_geo_price_map.png` | Mapa geogràfic de Catalunya amb el preu per m² (scatter per coordenades) |

**Funció `print_summary()`:** Imprimeix estadístiques bàsiques del dataset (registres, variables, estadístiques del preu).

---

### `models.py`
**Funció:** Defineix, entrena i guarda els models de machine learning.

**Models implementats (`get_models()`):**

| Model | Tipus | Descripció |
|-------|-------|-----------|
| `LinearRegression` | Lineal | Regressió lineal estàndard (baseline) |
| `Ridge` | Lineal regularitzat | Regressió Ridge (L2, α=1.0) |
| `Lasso` | Lineal regularitzat | Regressió Lasso (L1, α=1.0) |
| `RandomForest` | Ensemble (bagging) | 200 arbres, max_depth=15 |
| `GradientBoosting` | Ensemble (boosting) | sklearn GBM, 200 estimadors, lr=0.1 |
| `XGBoost` | Ensemble (boosting) | XGBoost, 300 estimadors, lr=0.05 |
| `LightGBM` | Ensemble (boosting) | LightGBM, 300 estimadors, lr=0.05 |

**Funcions principals:**
- `get_models()` → diccionari de models instanciats
- `train_model(model, X_train, y_train)` → entrena i retorna el model
- `cross_validate_models(models, X, y, cv=5)` → validació creuada 5-fold (RMSE, R²)
- `save_model(model, name, dir)` → serialitza el model amb joblib (`.joblib`)

---

### `evaluation.py`
**Funció:** Calcula mètriques d'avaluació i genera gràfics de resultats.

**Mètriques calculades (`compute_metrics()`):**
- **MAE** (Mean Absolute Error): error absolut mitjà en euros
- **RMSE** (Root Mean Squared Error): arrel de l'error quadràtic mitjà
- **R²** (coeficient de determinació): proporció de variança explicada (0-1)
- **MAPE** (Mean Absolute Percentage Error): error percentual mitjà

**Gràfics generats:**

| Fitxer | Descripció |
|--------|-----------|
| `08_metrics_comparison.png` | Barres horitzontals comparant RMSE i R² de tots els models |
| `09_pred_vs_real_<model>.png` | Scatter plot prediccions vs valors reals (línia ideal en vermell) |
| `10_residuals_<model>.png` | Residus vs prediccions + histograma de residus |
| `11_cv_results.png` | RMSE i R² de la validació creuada (amb barres d'error) |
| `metrics_summary.csv` | Taula CSV amb MAE, RMSE, R², MAPE de tots els models |

---

### `explainability.py`
**Funció:** Interpretabilitat dels models (per entendre quines variables influeixen més).

**Dues modalitats:**

**1. Feature Importance (sense SHAP, ràpid):**
- Disponible per a models d'arbres (RandomForest, GradientBoosting, XGBoost, LightGBM)
- Utilitza `model.feature_importances_`
- Genera: `12_feature_importance_<model>.png`

**2. SHAP (SHapley Additive exPlanations, complet):**
- Explica les prediccions a nivell individual
- Genera: `13_shap_beeswarm_<model>.png` — impacte global de cada variable
- Genera: `14_shap_dependence_<feature>.png` — relació entre una variable i el seu impacte SHAP

**Funció `run_explainability()`:** Executa SHAP per al millor model i Feature Importance per a tots.

---

### `predict.py`
**Funció:** Permet estimar el preu d'un habitatge nou introduint les seves característiques, i compara les prediccions de tots els models entrenats.

**Prerequisit:** Cal haver executat `main.py` primer per tenir els models entrenats a `models/`.

**Quatre modes d'ús:**

**1. Mode interactiu** (sense arguments — demana les dades per teclat):
```bash
run_predict.bat
```
El programa pregunta pas a pas: superfície, habitacions, banys, terrassa, piscina, garatge, província, municipi, coordenades, any i mes.

**2. Mode arguments** (tot en una línia de comandes):
```bash
run_predict.bat --metros 90 --habitaciones 3 --aseos 2 --terraza 1 --piscina 0 --garaje 1 --provincia Barcelona --municipio Barcelona --latitud 41.39 --longitud 2.17 --anyo 2023 --mes 6
```

**3. Mode fitxer JSON** (a partir d'un fitxer de configuració):
```bash
run_predict.bat --input examples/habitatge_exemple.json
```
Format del JSON (`examples/habitatge_exemple.json`):
```json
{
    "metros": 90, "habitaciones": 3, "aseos": 2,
    "terraza": 1, "piscina": 0, "garaje": 1,
    "provincia": "Barcelona", "municipio": "Barcelona",
    "latitud": 41.3851, "longitud": 2.1734,
    "anyo": 2023, "mes": 6
}
```

**4. Mode batch** (múltiples habitatges d'un CSV):
```bash
run_predict.bat --batch examples/habitatges_batch.csv
```
Format del CSV (`examples/habitatges_batch.csv`): columna `label` + les mateixes columnes que el JSON.

**Paràmetres d'entrada:**

| Paràmetre | Tipus | Descripció | Valor per defecte |
|-----------|-------|-----------|-------------------|
| `metros` | float | Superfície en m² | 80 |
| `habitaciones` | int | Nombre d'habitacions | 3 |
| `aseos` | int | Nombre de banys/lavabos | 1 |
| `terraza` | 0/1 | Té terrassa? | 0 |
| `piscina` | 0/1 | Té piscina? | 0 |
| `garaje` | 0/1 | Té garatge? | 0 |
| `provincia` | str | Barcelona / Girona / Lleida / Tarragona | Barcelona |
| `municipio` | str | Nom del municipi (opcional) | — |
| `latitud` | float | Latitud geogràfica | 41.39 |
| `longitud` | float | Longitud geogràfica | 2.17 |
| `anyo` | int | Any de l'anunci | 2023 |
| `mes` | int | Mes de l'anunci (1-12) | 6 |

**Output per consola** (exemple real per un pis de 90m² a Barcelona):
```
  ESTIMACIO DEL PREU DE L'HABITATGE
Caracteristiques introduides:
  Superficie    : 90 m2
  Habitacions   : 3
  Banys         : 2
  Terrassa      : Si
  Piscina       : No
  Garatge       : Si
  Provincia     : Barcelona
  Municipi      : Barcelona

Prediccions per model:
  Model                    Preu Estimat   Dif. vs Mediana
  --------------------------------------------------------
  RandomForest                230,963 EUR    -21.6%
  XGBoost                     267,868 EUR     -9.1%
  LightGBM                    271,356 EUR     -7.9%
  GradientBoosting            294,723 EUR     +0.0%
  Lasso                       463,121 EUR    +57.1%
  Ridge                       474,260 EUR    +60.9%
  LinearRegression            474,509 EUR    +61.0%

------------------------------------------------------------
  Mediana de prediccions :      294,723 EUR
  Mitjana de prediccions :      353,829 EUR
  Rang                   : 230,963 - 474,509 EUR
  Preu/m2 (mediana)      :        3,275 EUR/m2
```

**Fitxers generats:**
- `outputs/prediction_result.csv` — taula amb Model, Preu_Estimat, Diferencia_vs_Mediana, Diferencia_pct
- `outputs/prediction_comparison.png` — gràfic de barres + scatter de les prediccions de tots els models
- `outputs/batch_predictions.csv` — (mode batch) taula resum de tots els habitatges
- `outputs/batch_predictions_comparison.png` — (mode batch) gràfic comparatiu multi-habitatge

**Gràfic generat (`prediction_comparison.png`):**
- **Esquerra:** Barres horitzontals amb el preu estimat per cada model + línia de mediana
- **Dreta:** Scatter plot mostrant la dispersió de les prediccions + rang (zona blava)

**Argument addicional:**
- `--no-plot` — Ometre la generació del gràfic (més ràpid, útil en scripts)

---

### `run_predict.bat`
**Funció:** Script d'ajuda per Windows que activa l'entorn virtual i executa `predict.py`.

**Per què existeix:** La ruta del projecte conté espais (OneDrive), cosa que dificulta l'execució directa des de la línia de comandes. Aquest script resol el problema fent `cd` a la carpeta del projecte primer.

**Ús:**
```bat
run_predict.bat --input examples/habitatge_exemple.json
run_predict.bat --metros 80 --habitaciones 3 --provincia Girona
run_predict.bat --batch examples/habitatges_batch.csv
run_predict.bat  (mode interactiu)
```

---

### `main.py`
**Funció:** Orquestra tot el pipeline de principi a fi.

**Passos del pipeline:**
1. **[1/6] Càrrega** — Llegeix el CSV i filtra Catalunya
2. **[2/6] EDA** — Genera gràfics exploratoris (opcional: `--skip-eda`)
3. **[3/6] Preprocessament** — Neteja i prepara les dades; divisió train/test (80%/20%)
4. **[4/6] Validació creuada** — 5-fold CV per comparar models (opcional: `--skip-cv`)
5. **[5/6] Entrenament i avaluació** — Entrena tots els models, avalua al test set, identifica el millor
6. **[6/6] Interpretabilitat** — SHAP i Feature Importance (opcional: `--skip-shap`)

**Arguments de línia de comandes:**
```
python main.py --data <ruta_csv>   # Ruta al fitxer de dades
               --skip-eda          # Ometre EDA (més ràpid)
               --skip-cv           # Ometre validació creuada
               --skip-shap         # Ometre SHAP (Feature Importance igualment)
```

**Exemple d'ús ràpid:**
```bash
python main.py --data ../Data/DatosViviendas1.csv --skip-eda --skip-cv --skip-shap
```

**Exemple complet (triga ~10-15 min):**
```bash
python main.py --data ../Data/DatosViviendas1.csv
```

---

## Fitxers de Configuració

### `requirements.txt`
Llista de dependències Python necessàries:
- `pandas`, `numpy` — manipulació de dades
- `scikit-learn` — models lineals, RandomForest, GradientBoosting, mètriques
- `xgboost` — model XGBoost
- `lightgbm` — model LightGBM
- `shap` — interpretabilitat SHAP
- `matplotlib`, `seaborn` — visualitzacions
- `joblib` — serialització de models

**Instal·lació:**
```bash
pip install -r requirements.txt
```

### `.gitignore`
Exclou del repositori Git:
- Entorn virtual (`env/`, `venv/`)
- Fitxers de dades grans (`*.csv`, `data/`)
- Models serialitzats (`models/*.joblib`)
- Outputs generats (`outputs/*.png`, `outputs/*.csv`)
- Fitxers de log (`*.log`)
- Fitxers temporals de Python (`__pycache__/`, `*.pyc`)

> **Nota:** Les carpetes `data/`, `models/` i `outputs/` existeixen al repositori però estan buides (amb `.gitkeep`). Els fitxers es generen en executar el pipeline.

### `README.md`
Guia ràpida d'instal·lació i ús del projecte per a GitHub.

### `pipeline.log`
Fitxer de log generat automàticament en cada execució. Conté tots els missatges INFO/WARNING/ERROR amb timestamp. Útil per a depuració i traçabilitat.

---

## Resultats Obtinguts

### Dataset
- **Total registres Espanya:** 954.157
- **Registres Catalunya (NCA=Cataluña):** 89.205 (9.3%)
- **Registres finals (després de neteja):** 86.866
- **Features del model:** 15
- **Train set:** 69.492 registres (80%)
- **Test set:** 17.374 registres (20%)

### Mètriques al Test Set

| Model | MAE (€) | RMSE (€) | R² | MAPE (%) |
|-------|---------|---------|-----|---------|
| LinearRegression | 107.395 | 202.040 | 0.466 | 65.70% |
| Ridge | 107.391 | 202.041 | 0.466 | 65.69% |
| Lasso | 107.355 | 202.061 | 0.466 | 65.64% |
| RandomForest | 63.553 | 144.342 | 0.727 | 30.76% |
| GradientBoosting | 71.783 | 150.554 | 0.703 | 37.32% |
| XGBoost | 67.163 | 143.009 | 0.732 | 34.47% |
| **LightGBM** ⭐ | **63.976** | **137.420** | **0.753** | **32.99%** |

**Millor model: LightGBM** amb RMSE=137.420€ i R²=0.753

### Interpretació dels Resultats
- Els models lineals (R²≈0.47) capturen poc la variabilitat del preu, indicant relacions no lineals
- Els models d'ensemble (R²≈0.73-0.75) milloren substancialment les prediccions
- LightGBM és el millor model en totes les mètriques
- Un MAPE del 33% indica que, de mitjana, el model s'equivoca un 33% en el preu predit
- Les variables més importants (Feature Importance) solen ser: `Metros`, `Latitud`, `Longitud`, `NMUN_enc`

---

## Com Executar el Projecte

### Prerequisits
```bash
# Crear entorn virtual
python -m venv env
env\Scripts\activate  # Windows

# Instal·lar dependències
pip install -r requirements.txt
```

### Execució
```bash
cd real_estate_catalunya

# Execució ràpida (sense EDA, CV ni SHAP) — ~30 segons
python main.py --data ../Data/DatosViviendas1.csv --skip-eda --skip-cv --skip-shap

# Execució amb EDA i Feature Importance — ~2-3 minuts
python main.py --data ../Data/DatosViviendas1.csv --skip-cv

# Execució completa (amb CV 5-fold i SHAP) — ~15-20 minuts
python main.py --data ../Data/DatosViviendas1.csv
```

### Outputs
Tots els resultats es guarden automàticament a:
- `outputs/` — gràfics PNG i `metrics_summary.csv`
- `models/` — models entrenats (`.joblib`)
- `data/` — dataset filtrat de Catalunya (`catalunya_clean.csv`)
- `pipeline.log` — log complet de l'execució
