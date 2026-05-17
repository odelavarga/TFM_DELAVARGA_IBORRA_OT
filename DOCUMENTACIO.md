# Documentació del Projecte: Valoració Immobiliària a Catalunya

## Descripció General

Aquest projecte implementa un pipeline complet d'aprenentatge automàtic per a la **valoració immobiliària a Catalunya**. A partir d'un dataset de prop de 955.000 habitatges d'Espanya, es filtren els 89.205 registres de Catalunya i s'entrenen múltiples models de regressió per predir el preu de venda d'un habitatge.

### Millores implementades (versió actual)
1. **Transformació logarítmica del target** (`np.log1p` / `np.expm1`) per reduir el MAPE
2. **Cerca d'hiperparàmetres amb Optuna** (cerca bayesiana TPE, 50 trials per model)
3. **XAI complet amb SHAP** (Summary, Bar, Dependence, Waterfall plots)
4. **Eliminació d'outliers en dos passos** (rang fix + IQR estadístic k=3)

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
│   ├── preprocessing.py     ← Neteja, outliers i preprocessament
│   ├── eda.py               ← Anàlisi exploratòria (gràfics)
│   ├── models.py            ← Models + optimització Optuna
│   ├── evaluation.py        ← Mètriques i gràfics d'avaluació
│   └── explainability.py    ← XAI: SHAP + Feature Importance
│
├── examples/                ← Fitxers d'exemple per a predict.py
│   ├── habitatge_exemple.json
│   └── habitatges_batch.csv
│
├── data/                    ← Dades processades (generat automàticament)
│   └── catalunya_clean.csv
│
├── models/                  ← Models entrenats serialitzats (generat automàticament)
│   ├── LinearRegression.joblib
│   ├── Ridge.joblib
│   ├── Lasso.joblib
│   ├── RandomForest.joblib
│   ├── GradientBoosting.joblib
│   ├── XGBoost.joblib
│   ├── LightGBM.joblib
│   ├── XGBoost_Optuna.joblib      ← Model optimitzat per Optuna
│   ├── LightGBM_Optuna.joblib     ← Model optimitzat per Optuna
│   └── RandomForest_Optuna.joblib ← Model optimitzat per Optuna
│
├── outputs/                 ← Gràfics i resultats (generat automàticament)
│   ├── 00_log_transform_target.png      ← Efecte log1p al target
│   ├── 01_price_distribution.png
│   ├── 02_price_by_province.png
│   ├── 03_price_vs_metros.png
│   ├── 04_price_by_type.png
│   ├── 05_amenities_boxplot.png
│   ├── 06_correlation_heatmap.png
│   ├── 07_geo_price_map.png
│   ├── 08_metrics_comparison.png        ← RMSE, MAE, MAPE, R² (4 gràfics)
│   ├── 09_pred_vs_real_<model>.png
│   ├── 10_residuals_<model>.png
│   ├── 11_cv_results.png
│   ├── 12_feature_importance_<model>.png
│   ├── 13_shap_summary_<model>.png      ← SHAP beeswarm
│   ├── 14_shap_bar_<model>.png          ← SHAP importància global
│   ├── 15_shap_dependence_<model>_<feat>.png ← SHAP dependence (top 3)
│   ├── 16_shap_waterfall_<model>.png    ← SHAP waterfall (predicció individual)
│   ├── 17_combined_feature_importance.png ← Heatmap comparatiu tots models
│   ├── metrics_summary.csv
│   ├── optuna_best_params.csv           ← Millors hiperparàmetres Optuna
│   ├── shap_values_<model>.csv          ← Estadístiques SHAP per feature
│   ├── prediction_result.csv
│   ├── prediction_comparison.png
│   ├── batch_predictions.csv
│   └── batch_predictions_comparison.png
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
1. **`fix_latlon()`** — Converteix Latitud/Longitud a float
2. **`fix_fecha()`** — Extreu `Anyo` i `Mes` de la columna `Fecha`
3. **`dropna()`** — Elimina registres sense `Precio` o `Metros`
4. **`remove_outliers()`** — Eliminació en **dos passos** (veure secció específica)
5. **`log_outlier_stats()`** — Registra estadístiques post-neteja per a auditoria
6. **`fill_missing()`** — Imputa valors nuls
7. **`add_features()`** — Crea variables derivades
8. **`encode_categoricals()`** — Label Encoding

**Variables finals del model (15 features):**
`Habitaciones`, `Aseos`, `Terraza`, `Piscina`, `Garaje`, `Metros`, `Latitud`, `Longitud`, `Anyo`, `Mes`, `Hab_per_m2`, `Serveis`, `Inmueble_enc`, `NPRO_enc`, `NMUN_enc`

**Target:** `Precio` (o `log1p(Precio)` si `log_transform=True`)

---

#### Tractament d'Outliers (2 passos)

**Pas 1 — Rang fix (regles de negoci):**
| Variable | Mínim | Màxim |
|----------|-------|-------|
| Precio | 10.000 € | 5.000.000 € |
| Metros | 10 m² | 1.000 m² |
| Habitaciones | — | 20 |
| Aseos | — | 15 |

Raó: elimina errors de dades (preus de 0€, habitatges de 2m²) i propietats luxoses excepcionals que distorsionarien el model per al mercat residencial típic.

**Pas 2 — IQR estadístic (criteri Tukey estricte, k=3):**

Per a cada columna `c` en {`Precio`, `Metros`}:
```
lower = Q1 - 3 * IQR
upper = Q3 + 3 * IQR
```
on `IQR = Q3 - Q1`.

**Raó tècnica:**
- Les distribucions de preus immobiliaris segueixen una distribució log-normal amb fort **skew positiu** (cua dreta llarga).
- El criteri **z-score** assumeix normalitat i no és adequat per a distribucions asimètriques.
- L'**IQR és robust** davant distribucions no normals perquè no depèn de la mitjana ni la desviació estàndard.
- Amb k=3 (Tukey estricte) s'eliminen únicament els valors *extremadament* atípics:
  - Preu: habitatges que costen més de ~3× el preu del tercer quartil
  - En una distribució normal equivalent, k=3 retindria el 99.7% dels valors
- La **transformació log1p** posterior redueix l'efecte dels outliers residuals.

**Estadístiques loggades post-neteja:**
```
Precio: min=10.000  p1=30.000  mediana=180.000  p99=800.000  max=... skew=...
Metros: min=10      p1=40      mediana=90        p99=250      max=... skew=...
```

---

#### Transformació Logarítmica del Target (`log_transform=True`)

**Funció `get_feature_target(df, log_transform=True)`:**

Quan `log_transform=True` (valor per defecte):
- El target `y` = `np.log1p(Precio)` en lloc de `Precio`
- Els models s'entrenen sobre valors en escala logarítmica
- Per obtenir prediccions en euros: `np.expm1(y_pred)`

**Per què redueix el MAPE:**

El MAPE penalitza per igual errors relatius independentment del preu absolut:
```
MAPE = mean(|y_real - y_pred| / y_real) × 100%
```

Amb la transformació log:
1. **Comprima la distribució asimètrica**: habitatges de 50k€ i 500k€ queden a distàncies similars en escala log, mentre que en escala lineal el model prioritzaria minimitzar l'error absolut dels habitatges cars.
2. **Redueix la variança dels residus**: els arbres de decisió i les regressions funcionen millor quan el target és aproximadament simètric.
3. **L'error en log1p correspon aproximadament a l'error relatiu (%):** minimitzar RMSE(log) és similar a minimitzar MAPE(euros).

**Gràfic generat:** `00_log_transform_target.png` — mostra la distribució original vs. log1p amb el skewness de cadascuna.

---

### `models.py`
**Funció:** Defineix, entrena i optimitza els models de machine learning.

**Models base (`get_models()`):**

| Model | Tipus | Descripció |
|-------|-------|-----------|
| `LinearRegression` | Lineal | Baseline |
| `Ridge` | Lineal + L2 | α=10.0 |
| `Lasso` | Lineal + L1 | α=0.01 |
| `RandomForest` | Ensemble bagging | 200 arbres, max_depth=20 |
| `GradientBoosting` | Ensemble boosting | 300 estimadors, lr=0.05 |
| `XGBoost` | Boosting avançat | 500 estimadors, lr=0.05 |
| `LightGBM` | Boosting avançat | 500 estimadors, lr=0.05 |

**Nota:** Tots els models s'entrenen amb `log1p(Precio)` com a target.

---

#### Cerca d'Hiperparàmetres amb Optuna

**Algorisme:** Tree-structured Parzen Estimator (TPE) — **cerca bayesiana**

**Com funciona:**
1. Optuna manté un model probabilístic de quins hiperparàmetres donen bons resultats
2. En cada trial, el sampler TPE suggereix paràmetres que probablement millorin el resultat
3. Contrasta amb Grid Search (explora totes les combinacions) i Random Search (aleatori): TPE aprèn de les execucions anteriors i dirigeix la cerca cap a zones prometedores de l'espai d'hiperparàmetres

**Configuració:**
- Trials per model: 50 (configurable amb `--optuna-trials`)
- CV interna: 3-fold (per velocitat)
- Mètrica: RMSE en escala log1p (minimitzar)
- Timeout: 300 segons per model (configurable amb `--optuna-timeout`)
- Pruner: MedianPruner — atura trials que clarament van malament

**Espai de cerca per model:**

*XGBoost:*
| Hiperparàmetre | Rang |
|----------------|------|
| n_estimators | [200, 1000] |
| max_depth | [3, 10] |
| learning_rate | [0.01, 0.3] (log) |
| subsample | [0.6, 1.0] |
| colsample_bytree | [0.6, 1.0] |
| reg_alpha | [1e-4, 10.0] (log) |
| reg_lambda | [1e-4, 10.0] (log) |
| min_child_weight | [1, 10] |
| gamma | [0.0, 1.0] |

*LightGBM:*
| Hiperparàmetre | Rang |
|----------------|------|
| n_estimators | [200, 1000] |
| num_leaves | [20, 150] |
| learning_rate | [0.01, 0.3] (log) |
| subsample | [0.6, 1.0] |
| min_child_samples | [5, 50] |
| reg_alpha, reg_lambda | [1e-4, 10.0] (log) |

*RandomForest:*
| Hiperparàmetre | Rang |
|----------------|------|
| n_estimators | [100, 600] |
| max_depth | [5, 30] |
| min_samples_leaf | [1, 20] |
| max_features | {sqrt, log2, 0.5, 0.8} |

**Output:** Models `XGBoost_Optuna`, `LightGBM_Optuna`, `RandomForest_Optuna` afegits a la comparació.
**CSV generat:** `outputs/optuna_best_params.csv`

**Funcions principals:**
- `get_models()` → models base
- `run_optuna_optimization(X, y)` → models optimitzats
- `cross_validate_models(models, X, y, cv=5)` → CV 5-fold
- `train_model(model, X, y)` → entrena
- `save_model / load_model` → persistència joblib

---

### `evaluation.py`
**Funció:** Calcula mètriques i genera gràfics. Gestiona la transformació logarítmica inversa.

**Funció clau `predict_in_euros(model, X, log_transform=True)`:**
```python
y_pred_raw = model.predict(X)       # predicció en escala log1p
return np.expm1(y_pred_raw)         # convertir a euros
```

**Funció `y_in_euros(y, log_transform=True)`:**
```python
return np.expm1(np.array(y))        # target de volta a euros
```

**Mètriques calculades (totes en euros reals):**
- **MAE** (Mean Absolute Error): error absolut mitjà en euros
- **RMSE** (Root Mean Squared Error): penalitza errors grans
- **R²**: proporció de variança explicada (0-1)
- **MAPE**: error percentual mitjà (independent de l'escala)

**Gràfics generats:**

| Fitxer | Descripció |
|--------|-----------|
| `00_log_transform_target.png` | Distribució Precio vs log1p(Precio) amb skewness |
| `08_metrics_comparison.png` | 4 subgràfics: RMSE, R², MAE, MAPE de tots els models |
| `09_pred_vs_real_<model>.png` | Scatter prediccions vs reals amb RMSE i MAPE al títol |
| `10_residuals_<model>.png` | Residus vs prediccions + histograma |
| `11_cv_results.png` | RMSE i R² de la validació creuada |
| `metrics_summary.csv` | Taula CSV amb MAE, RMSE, R², MAPE per model |

---

### `explainability.py` — XAI (Explainable AI)
**Funció:** Interpretabilitat dels models per respondre: *quines variables influeixen en el preu i com?*

#### Tècniques XAI implementades

**1. Feature Importance intrínseca (Gini/Gain impurity)**
- Disponible per a models d'arbres (RF, GBM, XGBoost, LightGBM)
- Mesura quant redueix la impuresa de Gini (o el Gain) cada variable en els arbres
- Limitació: pot sobreestimar variables amb molts valors únics (com NMUN_enc)
- Fitxer: `12_feature_importance_<model>.png`

**2. SHAP Summary Plot (Beeswarm)**
- Mostra la distribució dels valors SHAP per a cada feature per a totes les mostres
- **Color:** vermell = valor alt de la feature, blau = valor baix
- **Posició horitzontal:** positiu = augmenta el preu, negatiu = el redueix
- Avantatge vs. Feature Importance: mostra la *direcció* de l'impacte
- Fitxer: `13_shap_summary_<model>.png`

**3. SHAP Bar Plot (Importància Global)**
- Importància global = `|SHAP|` mitjana per feature
- Equivalent a Feature Importance però basat en SHAP (més fiable)
- Fitxer: `14_shap_bar_<model>.png`

**4. SHAP Dependence Plots (top 3 variables)**
- Mostra la relació entre el valor d'una variable i el seu valor SHAP
- Permet detectar relacions no lineals (ex. preu augmenta amb metres fins a un punt)
- Colorejat per la variable d'interacció detectada automàticament
- Fitxers: `15_shap_dependence_<model>_<feature>.png`

**5. SHAP Waterfall Plot (predicció individual)**
- Explica la predicció d'un habitatge concret
- Mostra com cada feature "empeny" la predicció cap amunt o cap avall des del valor base
- Format: `base_value + SHAP₁ + SHAP₂ + ... = predicció final`
- Fitxer: `16_shap_waterfall_<model>.png`

**6. Heatmap combinat d'importàncies**
- Compara les importàncies de tots els models d'arbres en una matriu
- Permet identificar quines variables són consistentment importants
- Fitxer: `17_combined_feature_importance.png`

**Interpretació dels valors SHAP amb log_transform:**
- Els valors SHAP estan en escala log1p si el model s'ha entrenat amb log1p(Precio)
- Un SHAP = 0.5 per a `Metros` significa que aquells metres contribueixen +0.5 a log1p(Precio)
- Això equival a multiplicar el preu base per e^0.5 ≈ 1.65 (augment del ~65%)
- La interpretació qualitativa (signe i rànquing) és equivalent a escala en euros

**CSV generat:** `outputs/shap_values_<model>.csv` — estadístiques descriptives dels valors SHAP per feature

---

### `predict.py`
**Funció:** Estima el preu d'un habitatge nou amb tots els models entrenats.

**Nova funcionalitat — Transformació inversa automàtica:**
```python
# Si els models estan entrenats amb log1p:
pred_raw = model.predict(X)         # valor en log1p
pred_euros = np.expm1(pred_raw)     # valor en euros ← nou
```

**Cinc modes d'ús:**

1. **Interactiu** (per teclat): `python predict.py`
2. **Arguments**: `python predict.py --metros 80 --habitaciones 3 ...`
3. **JSON**: `python predict.py --input examples/habitatge_exemple.json`
4. **Batch CSV**: `python predict.py --batch examples/habitatges_batch.csv`
5. **Sense transformació** (models antics): `python predict.py --no-log ...`

**Nou argument `--no-log`:**
- Desactiva la transformació expm1
- Usar únicament per a models entrenats sense log1p (versions antigues)
- Per defecte: activat (models actuals usen log1p)

---

### `main.py`
**Funció:** Orquestra tot el pipeline de principi a fi.

**Passos del pipeline (7 etapes):**
1. **[1/7] Càrrega** — Llegeix el CSV i filtra Catalunya
2. **[2/7] EDA** — Gràfics exploratoris (opcional: `--skip-eda`)
3. **[3/7] Preprocessament** — Neteja, outliers (rang+IQR), features, log1p(Precio)
4. **[4/7] Validació creuada** — 5-fold CV models base (opcional: `--skip-cv`)
5. **[5/7] Optuna** — Cerca bayesiana d'hiperparàmetres (opcional: `--skip-optuna`)
6. **[6/7] Entrenament i avaluació** — Tots els models, mètriques en euros (expm1)
7. **[7/7] XAI** — SHAP + Feature Importance (opcional: `--skip-shap`)

**Arguments de línia de comandes:**
```
python main.py --data <ruta_csv>        # Ruta al fitxer de dades
               --skip-eda               # Ometre EDA
               --skip-cv                # Ometre validació creuada
               --skip-shap              # Ometre SHAP
               --skip-optuna            # Ometre cerca Optuna
               --no-log-transform       # No aplicar log1p (no recomanat)
               --optuna-trials N        # Trials Optuna per model (default: 50)
               --optuna-timeout S       # Timeout Optuna en segons (default: 300)
```

**Exemples d'ús:**
```bash
# Molt ràpid (~30 seg): sense EDA, CV, Optuna ni SHAP
python main.py --data ../Data/DatosViviendas1.csv --skip-eda --skip-cv --skip-optuna --skip-shap

# Ràpid (~5 min): amb SHAP però sense CV ni Optuna
python main.py --data ../Data/DatosViviendas1.csv --skip-cv --skip-optuna

# Complet (~30-60 min): amb tot
python main.py --data ../Data/DatosViviendas1.csv

# Optuna ràpid (10 trials per model):
python main.py --data ../Data/DatosViviendas1.csv --skip-cv --optuna-trials 10
```

---

## Fitxers de Configuració

### `requirements.txt`
```
pandas>=2.0.0, numpy>=1.24.0, scikit-learn>=1.3.0
xgboost>=2.0.0, lightgbm>=4.0.0
matplotlib>=3.7.0, seaborn>=0.12.0
shap>=0.44.0
optuna>=3.4.0          ← NOU: cerca d'hiperparàmetres
joblib>=1.3.0, tqdm>=4.65.0
```

**Instal·lació:**
```bash
pip install -r requirements.txt
```

---

## Tractament Estadístic Complet

### 1. Eliminació d'Outliers

| Pas | Mètode | Variables | Criteri |
|-----|--------|-----------|---------|
| 1 | Rang fix (negoci) | Precio, Metros, Habitaciones, Aseos | Límits absoluts raonables |
| 2 | IQR Tukey estricte (k=3) | Precio, Metros | `Q1 - 3·IQR` ↔ `Q3 + 3·IQR` |

**Per què IQR i no Z-score:**
- Z-score assumeix normalitat; els preus immobiliaris no són normals (skew >>0)
- IQR és robust davant distribucions asimètriques i log-normals
- k=3 és conservador: manté habitatges cars legítims, elimina errors de dades

### 2. Transformació del Target

| Pas | Mètode | Quan |
|-----|--------|------|
| Training | `y_log = np.log1p(Precio)` | `get_feature_target(df, log_transform=True)` |
| Predicció | `euros = np.expm1(y_pred)` | `predict_in_euros()`, `predict_all_models()` |

**Efecte esperat:** reducció del MAPE del ~30-35% al ~20-25% per a models d'ensemble.

### 3. Cerca d'Hiperparàmetres

| Mètode | Tipus | Avantatge |
|--------|-------|-----------|
| Optuna TPE | Bayesiana | Aprèn de trials anteriors |
| Grid Search (alternativa) | Exhaustiva | Cobertura completa però lenta |
| Random Search (alternativa) | Aleatòria | Ràpida però no dirigida |

### 4. Imputació de Valors Nuls

| Variable | Estratègia | Raó |
|----------|-----------|-----|
| Habitaciones, Aseos | Mediana | Robust davant outliers |
| Terraza, Piscina, Garaje | 0 (absent) | Interpretació lògica |
| Latitud, Longitud | Mediana | Valor geogràfic central |
| Variables categòriques | "Desconegut" | Evita pèrdua d'informació |

---

## Interpretabilitat XAI — Quines Variables Influeixen?

Les variables més importants identificades (resultats típics):

| Rànquing | Variable | Interpretació |
|----------|----------|--------------|
| 1 | `Metros` | La superfície és el predictor principal del preu |
| 2 | `NMUN_enc` | El municipi és clau (Barcelona vs. zones rurals) |
| 3 | `Latitud` | Localització geogràfica (Barcelona més cara al sud) |
| 4 | `Longitud` | Localització (costa est més cara) |
| 5 | `NPRO_enc` | Província (Barcelona > Girona > Tarragona > Lleida) |
| 6 | `Habitaciones` | Nombre d'habitacions |
| 7 | `Aseos` | Nombre de banys |
| 8 | `Garaje` | Garatge té impacte positiu significant |
| 9 | `Anyo` | Evolució temporal del mercat |
| 10 | `Serveis` | Combinació amenities (terrassa+piscina+garatge) |

**SHAP vs. Feature Importance:**
- **Feature Importance** mesura l'impacte global i pot estar esbiaixada cap a variables amb molts valors únics
- **SHAP** és teòricament fonamentada (teoria de jocs de Shapley), mesura la contribució marginal de cada feature i mostra la *direcció* de l'impacte
- Per a decisions del TFM: **prioritzar els resultats SHAP**

---

## Resultats Esperats (amb log1p)

### Dataset (post-outliers IQR k=3)
- **Registres Catalunya originals:** 89.205
- **Post-rang fix:** ~87.786
- **Post-IQR k=3:** ~87.000-87.500 (variable)
- **Features del model:** 15
- **Train set (80%):** ~70.000 registres
- **Test set (20%):** ~17.500 registres

### Mètriques Esperades (models sense Optuna, sense log1p — referència original)

| Model | MAE (€) | RMSE (€) | R² | MAPE (%) |
|-------|---------|---------|-----|---------|
| LinearRegression | 107.395 | 202.040 | 0.466 | 65.70% |
| Ridge | 107.391 | 202.041 | 0.466 | 65.69% |
| Lasso | 107.355 | 202.061 | 0.466 | 65.64% |
| RandomForest | 63.553 | 144.342 | 0.727 | 30.76% |
| GradientBoosting | 71.783 | 150.554 | 0.703 | 37.32% |
| XGBoost | 67.163 | 143.009 | 0.732 | 34.47% |
| **LightGBM** ⭐ | **63.976** | **137.420** | **0.753** | **32.99%** |

### Millores Esperades amb log1p + Optuna

| Tècnica | Efecte esperat sobre MAPE |
|---------|--------------------------|
| Transformació log1p | -5% a -10% MAPE (reducció relativa ~15-30%) |
| Optuna (50 trials) | -2% a -5% MAPE addicional |
| IQR k=3 (neteja) | Millora la robustesa, menor variança |
| **Total esperat** | **MAPE ~18-25% per al millor model** |

---

## Com Executar el Projecte

### Prerequisits
```bash
python -m venv env
env\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Execució ràpida (recomanada per a proves)
```bash
# ~30 seg: entrena tots els models base amb log1p, sense Optuna ni SHAP
python main.py --data ../Data/DatosViviendas1.csv --skip-eda --skip-cv --skip-optuna --skip-shap
```

### Execució amb Optuna (recomanada per al TFM)
```bash
# ~10-15 min: log1p + Optuna (10 trials ràpid) + SHAP
python main.py --data ../Data/DatosViviendas1.csv --skip-cv --optuna-trials 10
```

### Execució completa
```bash
# ~30-60 min: tot incloent CV 5-fold, Optuna 50 trials, SHAP complet
python main.py --data ../Data/DatosViviendas1.csv
```

### Predicció d'un habitatge nou
```bash
# Amb transformació log1p (per defecte, models actuals)
python predict.py --metros 90 --habitaciones 3 --aseos 2 --terraza 1 --piscina 0 --garaje 1 --provincia Barcelona

# En mode interactiu
python predict.py
```

### Outputs generats
- `outputs/` — tots els gràfics PNG, `metrics_summary.csv`, `optuna_best_params.csv`
- `models/` — models serialitzats (`.joblib`), incloent els optimitzats per Optuna
- `data/` — `catalunya_clean.csv`
- `pipeline.log` — log complet de l'execució
