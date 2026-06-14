---
title: "Forest Fire Susceptibility Mapping in Bagmati Province, Nepal"
subtitle: "A Complete Implementation Guide for Machine Learning-Based Spatial Risk Modelling"
author: "Master's Dissertation — Data Analytics"
date: "2026"
toc: true
toc-depth: 3
number-sections: true
geometry: margin=1in
fontsize: 11pt
bibliography: references.bib
csl: https://www.zotero.org/styles/apa
link-citations: true
colorlinks: true
---

\newpage

# Introduction

## Research Question

**Which areas of Bagmati Province, Nepal face the highest probability of forest fire ignition and spread, given vegetation conditions, topography, and weather patterns?**

This question is a spatial prediction problem. The goal is to produce a continuous probability map at 100-metre resolution, where each pixel represents the likelihood of fire occurring under a given set of environmental conditions. Such a map — a **fire susceptibility map** — does not predict when fire will occur, but rather identifies which locations are structurally predisposed to burning.

## Why This Matters

Nepal experiences severe pre-monsoon fires annually, concentrated between March and May, with April alone accounting for over 43% of annual burned area [@mishra2022]. Approximately 65% of forested area in key Nepalese landscapes falls within high fire risk zones [@parajuli2020]. As climate change intensifies dry seasons and increases vapour pressure deficit (VPD), these patterns are projected to worsen [@nepal2025rf].

Bagmati Province — encompassing the Chure hills, mid-hills, and parts of the high Himalaya — presents a complex gradient of vegetation types, topographic variability, and human activity that makes it an ideal study for multi-factor fire susceptibility modelling.

## What This Guide Covers

This document is a structured implementation guide progressing through:

1. Understanding the **inputs** (features) and **outputs** (maps, predictions)
2. **Exploratory Data Analysis** — understanding distributions, imbalance, and spatial patterns
3. **Feature Engineering** — transforming raw data into model-ready features
4. **Train-Test Splitting** — why standard random splits fail for spatial data
5. **Model Selection** — seven model families with trade-offs and recommended use cases
6. **Validation and Metrics** — choosing the right evaluation strategy for imbalanced fire data
7. **Interpretability** — SHAP, LIME, and partial dependence plots
8. **Output Maps** — generating, classifying, and presenting the final susceptibility raster
9. **Error Analysis** — understanding where and why models fail

\newpage

# Project Overview: Inputs, Outputs, and Pipeline

## The Core Problem

This is a **binary spatial classification** task:

- **Input:** A set of environmental and anthropogenic features measured at each 100m grid cell across Bagmati Province
- **Output:** A probability score $p \in [0, 1]$ for each cell, representing the likelihood that a fire occurred there given those conditions

A secondary output is a **classified risk map** that groups continuous probabilities into human-readable risk categories (Low / Moderate / High / Very High).

## Input Features

Your processed dataset (`data/processed/forest_fire_dataset_100m.parquet`) contains the following feature groups:

### Topographic Features

| Feature | Description | Why It Matters |
|---|---|---|
| `elevation_m` | Elevation in metres (SRTM 30m) | Controls temperature and vegetation type |
| `slope_deg` | Slope angle in degrees | Steeper slopes dry faster; fire spreads faster upslope |
| `aspect_deg` | Slope aspect (0–360°) | South-facing slopes are drier, more fire-prone |
| `tri` | Terrain Ruggedness Index | Complex terrain creates erratic fire spread |
| `twi` | Topographic Wetness Index | Low TWI = drier sites, higher ignition risk |
| `solar_radiation` | Modelled annual solar radiation (Wh/m²) | Drives evapotranspiration and drying |

### Land Use and Land Cover

| Feature | Description |
|---|---|
| `lulc_code` | ESA WorldCover class code (2021) |
| `lulc_class` | Human-readable class name |
| `is_flammable` | Binary: 1 if class is burnable (forest, shrub, grassland) |

### Distance / Human Proximity Features

| Feature | Description |
|---|---|
| `dist_to_road_m` | Distance to nearest road (OSM) |
| `dist_to_settlement_m` | Distance to nearest settlement |
| `dist_to_forest_edge_m` | Distance to nearest forest boundary |
| `dist_to_cropland_m` | Distance to nearest cropland (burn-clearing source) |
| `dist_to_water_m` | Distance to nearest water body |

### Climate and Weather Features

| Feature | Description |
|---|---|
| `temp_max_mean_c` | Mean daily maximum temperature (fire season) |
| `wind_max_mean_kmh` | Mean daily maximum wind speed (fire season) |
| `precip_fire_season_mm` | Total precipitation (March–May) |
| `drought_factor` | Derived drought index (Keetch-Byram proxy) |
| `fwi_proxy` | Fire Weather Index proxy |
| `consec_dry_days_max` | Maximum consecutive dry days |

> **Key research finding:** Vapour pressure deficit (VPD) — not precipitation — is the strongest fire predictor in Nepal [@nepal2025rf]. Your `drought_factor` and `fwi_proxy` capture this indirectly. Consider adding VPD directly if gridded climate data supports it.

### Vegetation Index Features (MODIS/Sentinel-2)

| Feature | Description |
|---|---|
| `ndvi_annual` | Mean annual NDVI |
| `ndvi_fire_season` | Mean NDVI during March–May |
| `ndvi_premonsoon` | Pre-monsoon NDVI (Feb–Mar) |
| `ndvi_peak` | Peak growing-season NDVI |
| `ndvi_postmonsoon` | Post-monsoon NDVI |
| `evi_*` | Enhanced Vegetation Index variants |
| `nbr_*` | Normalised Burn Ratio variants |
| `ndwi_*` | Normalised Difference Water Index |
| `lst_fire_season_mean_c` | Land Surface Temperature (fire season) |

### Human Pressure

| Feature | Description |
|---|---|
| `population_density` | WorldPop 100m population density (2020) |

## Target Variables

| Variable | Type | Description |
|---|---|---|
| `fire_occurred` | Binary (0/1) | Did any fire hotspot fall within this cell, 2015–2024? **Primary target** |
| `fire_count` | Integer | Number of fire hotspot detections in this cell |
| `burn_year` | Integer / NaN | Year of most recent confirmed burn (MCD64A1) |

**Recommended primary target:** `fire_occurred`. This frames the problem as binary classification, which is the most natural formulation for susceptibility mapping and aligns with the majority of the literature.

## The Output

The model produces:

1. **Susceptibility probability raster:** A GeoTIFF where each pixel contains $p(\text{fire} | \mathbf{x})$, the model's predicted probability given feature vector $\mathbf{x}$.
2. **Classified risk map:** The probability raster thresholded into four risk classes.
3. **Feature importance map:** SHAP-derived spatial attribution showing which features drive predictions in different areas.

\newpage

# Related Work

## Studies in Nepal and the Himalayan Region

### GIS-Based Risk Mapping (Parajuli et al., 2020)

@parajuli2020 produced the first comprehensive GIS-based fire risk map for two major Nepal landscapes (Terai Arc Landscape and Churia Hills). Their weighted index model integrated eight variables — land cover (40% weight), land surface temperature (20%), elevation, slope, proximity to roads, and settlements — and validated against 18 years of MODIS hotspot data, achieving AUC ~0.83. A critical methodological note: all eight variables passed variance inflation factor (VIF) testing with VIF < 2, establishing the baseline for multi-collinearity checks in Nepal fire studies.

### Deep Learning vs. MaxEnt in Nepal (Mishra et al., 2022)

@mishra2022 compared Deep Neural Network (DNN) and MaxEnt models for fire vulnerability mapping across Nepal. DNN achieved a probability of detection of 0.71 versus MaxEnt's 0.64, but classified 2.64% of forest area as very-high-risk compared to MaxEnt's 0.27%, illustrating the divergence in spatial output footprints between ML and ecological modelling approaches. This paper also established Nepal's seasonal fire concentration: >78% of burned area in March–May, peak in April.

### Random Forest with VPD Predictor (IOP, 2025)

@nepal2025rf is the most recent and methodologically strongest Nepal-specific study. Testing RF, ANN, RBFNN, and SVM on 13 conditioning factors, RF achieved accuracy 88.60%, AUC 0.95. Crucially, vapour pressure deficit ranked as the strongest predictor — outranking precipitation, which had been emphasised in prior Nepal studies. The authors attribute this to temporal mismatch: fires occur in the dry pre-monsoon season when precipitation is low by definition, making it a weak discriminator between fire and non-fire cells.

### Five-Country Himalayan Comparison (Khadke et al., 2025)

@khadke2025himalaya extended fire susceptibility modelling across India, Bhutan, Nepal, China/Tibet, and Pakistan, comparing RF, SVM, Boosted Regression Trees (BRT), and Generalised Linear Model (GLM). RF outperformed all alternatives with AUC 0.91 at the regional scale, suggesting that RF's robustness to correlated features and non-linear interactions generalises well across the Himalayan elevation gradient.

## International Comparative Studies

### Explainable GeoAI in the Indian Himalaya (Scientific Reports, 2026)

@scientific2026india applied a stacking ensemble of RF, XGBoost, LightGBM, and CatBoost to wildfire susceptibility in the Upper Ravi sub-basin. The stacked model achieved AUC 0.95, outperforming all individual components (RF: 0.92, XGBoost: 0.91). Monte Carlo uncertainty analysis (1,000 iterations) produced a mean AUC of ~0.85, important as it demonstrates that point estimates can overstate stability. SHAP was used for interpretability, identifying elevation and land cover as the dominant factors.

### ML with SHAP in Türkiye (Tonbul & Veraverbeke, 2025)

@tonbul2025turkey compared three ensemble classifiers for fire susceptibility across Türkiye's fire-prone Mediterranean regions, applying SHAP to 21 conditioning factors to quantify individual variable contributions and pairwise interaction effects (notably solar radiation × wind speed). This paper demonstrates SHAP's ability to capture interaction terms that traditional variable importance metrics miss.

### CNN vs. ConvLSTM for Southeast Asia (Eaturu & Vadrevu, 2025)

@eaturu2025sea benchmarked MLP, LSTM, CNN, CNN-LSTM, and ConvLSTM across Southeast Asian countries for fire count prediction. CNN achieved the best performance in countries with strong spatial dependencies (e.g., Indonesia, Malaysia), while ConvLSTM outperformed for regions with complex spatiotemporal dynamics (Laos, Myanmar). No single deep learning architecture was universally optimal, motivating a multi-model comparison strategy.

### Western Ghats, Southern India (Ecological Informatics, 2021)

@ecological2021india modelled fire susceptibility in a forest-agriculture mosaic in Kerala's Wayanad district — a landscape structurally similar to Bagmati's Chure hills, with mixed forests, plantations, and cropland. Using Sentinel-2A and NASA fire archives with geospatial predictors including slope, TWI, aspect, and land cover, this paper provides a methodological template applicable to Bagmati's mid-hill and foothill zones.

### Mediterranean XGBoost vs. RF (Bilucan et al., 2024)

@bilucan2024mediterranean systematically compared XGBoost, RF, and AdaBoost across a Mediterranean Turkish ecosystem, using McNemar's test to establish statistical significance of pairwise differences. XGBoost slightly outperformed RF (85.4% vs 84.6% accuracy), but McNemar's test confirmed the two are statistically equivalent; both significantly outperformed AdaBoost. This provides a template for statistical hypothesis testing between models in your thesis.

## Methodological Literature

### Spatial Cross-Validation (Roberts et al., 2017)

@roberts2017cv established the foundational framework for cross-validation in ecological data with autocorrelation structure. Standard random cross-validation produces misleadingly optimistic error estimates when training and test samples are spatially autocorrelated. Spatially blocked cross-validation — where geographically contiguous blocks are held out as test folds — is the methodologically correct approach for fire susceptibility modelling. This finding was empirically confirmed by @ploton2020spatialcv, who demonstrated that random CV indicated >50% variance explained in a tropical forest model while spatial CV revealed near-zero predictive power.

### Area of Applicability (Meyer & Pebesma, 2021)

@meyer2021aoa extend the spatial CV framework by defining an "area of applicability" (AOA): the spatial domain within which a model can reasonably be expected to predict. Pixels outside the AOA — whose feature vectors are dissimilar to the training data — should be excluded from the susceptibility map or flagged with high uncertainty.

\newpage

# Exploratory Data Analysis

EDA for a spatial fire dataset has three distinct objectives: (1) understand the **marginal distributions** of features, (2) characterise **class imbalance** between fire and non-fire pixels, and (3) examine **spatial and temporal structure** that will constrain modelling choices.

## Loading the Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_parquet("data/processed/forest_fire_training_100m.parquet")

print(df.shape)           # rows × columns
print(df["fire_occurred"].value_counts())
print(df.dtypes)
print(df.isnull().sum().sort_values(ascending=False).head(20))
```

## Class Imbalance Analysis

Fire cells are rare — typically 1–5% of all land area burns in any given year. Imbalance ratios of 20:1 to 100:1 are common in fire susceptibility datasets.

```python
fire_rate = df["fire_occurred"].mean()
print(f"Fire occurrence rate: {fire_rate:.2%}")

# Visual
ax = df["fire_occurred"].value_counts().plot(kind="bar")
ax.set_title("Class Distribution: Fire vs Non-Fire")
ax.set_xticklabels(["Non-Fire (0)", "Fire (1)"], rotation=0)
plt.tight_layout()
plt.savefig("docs/figures/class_imbalance.png", dpi=150)
```

**Implication:** Never use accuracy as a metric. A model that predicts "no fire" everywhere achieves 95%+ accuracy but is useless. Use AUC-PR (precision-recall area under curve) as the primary metric.

## Feature Distributions

```python
CONTINUOUS_FEATURES = [
    "elevation_m", "slope_deg", "aspect_deg", "tri", "twi",
    "temp_max_mean_c", "wind_max_mean_kmh", "precip_fire_season_mm",
    "drought_factor", "fwi_proxy", "consec_dry_days_max",
    "ndvi_fire_season", "ndvi_premonsoon", "lst_fire_season_mean_c",
    "dist_to_road_m", "dist_to_settlement_m", "population_density",
]

fig, axes = plt.subplots(6, 3, figsize=(15, 20))
for ax, feat in zip(axes.flat, CONTINUOUS_FEATURES):
    df[feat].dropna().hist(bins=50, ax=ax, color="steelblue", alpha=0.7)
    ax.set_title(feat, fontsize=9)
    ax.set_xlabel("")
plt.tight_layout()
plt.savefig("docs/figures/feature_distributions.png", dpi=150)
```

Look for:

- **Heavily skewed features** (e.g., `dist_to_road_m`, `population_density`) — these should be log-transformed.
- **Bimodal features** — may indicate different landscape units (lowland vs. highland).
- **Clipped or capped values** — sentinel values indicating no-data (e.g., -9999, 0).

## Fire vs. Non-Fire Distributions

Compare feature distributions between the two classes to identify discriminating variables:

```python
fig, axes = plt.subplots(6, 3, figsize=(15, 20))
for ax, feat in zip(axes.flat, CONTINUOUS_FEATURES):
    for label, colour in [(0, "steelblue"), (1, "tomato")]:
        subset = df[df["fire_occurred"] == label][feat].dropna()
        subset.plot.kde(ax=ax, label=["Non-Fire", "Fire"][label], color=colour)
    ax.set_title(feat, fontsize=9)
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig("docs/figures/fire_vs_nonfire_kde.png", dpi=150)
```

Features with strong class separation (e.g., `ndvi_premonsoon`, `slope_deg`, `lst_fire_season_mean_c`) will be important model predictors.

## Temporal Fire Pattern

```python
# Fires by year
fires = df[df["fire_occurred"] == 1]
fires["burn_year"].value_counts().sort_index().plot(kind="bar")
plt.title("Fire Events by Year (2015–2024)")
plt.ylabel("Count of 100m Cells")
plt.savefig("docs/figures/fires_by_year.png", dpi=150)
```

This reveals inter-annual variability driven by drought cycles, which motivates the temporal train-test split strategy.

## Correlation Matrix

```python
corr = df[CONTINUOUS_FEATURES].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(corr, mask=mask, annot=False, cmap="RdBu_r",
            vmin=-1, vmax=1, ax=ax)
ax.set_title("Pearson Correlation Matrix — Continuous Features")
plt.tight_layout()
plt.savefig("docs/figures/correlation_matrix.png", dpi=150)
```

High correlations (|r| > 0.7) between features that you expect to be independent (e.g., multiple NDVI variants) indicate redundancy. Use VIF to detect multicollinearity for logistic regression; tree-based models are generally robust to it.

## Spatial Distribution of Fires

```python
import geopandas as gpd
from shapely.geometry import Point

# Create GeoDataFrame from lat/lon columns (adjust column names as needed)
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["lon"], df["lat"]),
    crs="EPSG:4326"
)
fires_gdf = gdf[gdf["fire_occurred"] == 1]

ax = gdf.sample(50000).plot(markersize=0.1, color="lightgrey", figsize=(10, 8))
fires_gdf.plot(ax=ax, markersize=0.3, color="red", alpha=0.3)
ax.set_title("Fire Occurrence Locations — Bagmati Province")
plt.savefig("docs/figures/spatial_fire_distribution.png", dpi=150)
```

Look for spatial clustering — most fires will be in the Chure (southern) belt, along ridgelines, and near agricultural boundaries. This clustering is why you need spatial cross-validation.

\newpage

# Feature Engineering

## Principles for Geospatial Fire Modelling

1. **Temporal alignment:** Features should represent conditions at the time of fire, not annual averages. The `ndvi_premonsoon` (February–March) feature is more predictive than `ndvi_annual` because it captures pre-fire vegetation dryness.

2. **Log-transform skewed distance features:** Distance-to-road, distance-to-settlement, and population density typically follow right-skewed distributions with many near-zero values.

3. **Circular encoding for aspect:** Aspect is a circular variable (0° = North = 360°). Never use raw degree values in linear models. Encode as `sin(aspect_rad)` and `cos(aspect_rad)`.

4. **Remove non-flammable pixels:** Cells classified as water, snow, or built-up area cannot burn. Filter on `is_flammable == 1` before modelling, or include `lulc_code` as a feature and let the model learn it.

## Implementation

```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- Filter to flammable pixels only ---
df_model = df[df["is_flammable"] == 1].copy()

# --- Log-transform skewed features ---
LOG_FEATURES = [
    "dist_to_road_m", "dist_to_settlement_m", "dist_to_forest_edge_m",
    "dist_to_cropland_m", "dist_to_water_m", "population_density",
]
for feat in LOG_FEATURES:
    df_model[f"log_{feat}"] = np.log1p(df_model[feat])

# --- Circular encoding for aspect ---
df_model["aspect_sin"] = np.sin(np.radians(df_model["aspect_deg"]))
df_model["aspect_cos"] = np.cos(np.radians(df_model["aspect_deg"]))

# --- NDVI anomaly: pre-monsoon NDVI relative to annual mean ---
# Low pre-monsoon NDVI relative to mean signals vegetation stress
df_model["ndvi_anomaly"] = df_model["ndvi_premonsoon"] - df_model["ndvi_annual"]

# --- Define final feature list ---
FEATURES = [
    # Topography
    "elevation_m", "slope_deg", "aspect_sin", "aspect_cos",
    "tri", "twi", "solar_radiation",
    # Climate
    "temp_max_mean_c", "wind_max_mean_kmh", "precip_fire_season_mm",
    "drought_factor", "fwi_proxy", "consec_dry_days_max",
    # Vegetation
    "ndvi_fire_season", "ndvi_premonsoon", "ndvi_anomaly",
    "evi_fire_season", "nbr_fire_season", "ndwi_fire_season",
    "lst_fire_season_mean_c",
    # Human / distance (log-transformed)
    "log_dist_to_road_m", "log_dist_to_settlement_m",
    "log_dist_to_forest_edge_m", "log_dist_to_cropland_m",
    "log_population_density",
    # Land cover
    "lulc_code",
]
TARGET = "fire_occurred"

X = df_model[FEATURES]
y = df_model[TARGET]

print(f"Feature matrix: {X.shape}")
print(f"Class balance: {y.mean():.3%} fire cells")
```

## Handling Missing Values

```python
# Check missingness per feature
missing = X.isnull().mean().sort_values(ascending=False)
print(missing[missing > 0])

# For tree-based models: impute with median (or let the model handle NaN)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)
```

## VIF Check (for Logistic Regression)

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame({
    "Feature": FEATURES,
    "VIF": [variance_inflation_factor(X_imputed, i) for i in range(X_imputed.shape[1])]
}).sort_values("VIF", ascending=False)

print(vif_data)
# Remove features with VIF > 10 for logistic regression
```

\newpage

# Train-Test Split: Why Random Splits Fail

## The Problem: Spatial Autocorrelation

In a spatial dataset, nearby pixels share similar feature values (topography, climate, vegetation) because the underlying physical processes are continuous. If you randomly split your 100m pixel dataset into 80% train / 20% test, your test set will contain pixels that are geographically adjacent — and therefore nearly identical — to training pixels.

This means your model "remembers" local spatial patterns rather than learning generalisable physical relationships. The result: **inflated AUC scores that don't hold on genuinely new areas**.

@roberts2017cv demonstrated this conceptually; @ploton2020spatialcv showed it empirically: a tropical forest model showed >50% variance explained under random CV, but near-zero under spatial CV.

## Recommended Strategy: Spatial Block Cross-Validation

Divide the study area into spatial blocks (e.g., 10km × 10km or administrative districts), then use each block as the held-out fold:

```python
import geopandas as gpd
import numpy as np
from sklearn.model_selection import BaseCrossValidator

class SpatialBlockCV(BaseCrossValidator):
    """
    Assigns each sample to a spatial block and folds on blocks.
    Requires 'block_id' column in the DataFrame.
    """
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def _iter_test_indices(self, X, y=None, groups=None):
        blocks = np.unique(groups)
        np.random.shuffle(blocks)
        folds = np.array_split(blocks, self.n_splits)
        for fold in folds:
            mask = np.isin(groups, fold)
            yield np.where(mask)[0]

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# Create block IDs from grid coordinates
# Assuming df_model has 'x' and 'y' columns in metres (projected CRS)
block_size_m = 10_000  # 10 km blocks
df_model["block_x"] = (df_model["x"] // block_size_m).astype(int)
df_model["block_y"] = (df_model["y"] // block_size_m).astype(int)
df_model["block_id"] = df_model["block_x"].astype(str) + "_" + df_model["block_y"].astype(str)

block_groups = df_model["block_id"].values
cv = SpatialBlockCV(n_splits=5)
```

## Temporal Holdout for Final Evaluation

For the **final test set** (used only once at the end), hold out the most recent years:

```python
TRAIN_YEARS = range(2015, 2023)   # 2015–2022
TEST_YEARS  = [2023, 2024]         # Final evaluation on unseen recent years

train_mask = df_model["burn_year"].isin(TRAIN_YEARS) | (df_model["fire_occurred"] == 0)
test_mask  = df_model["burn_year"].isin(TEST_YEARS)

# For non-fire cells, assign to train (they have no burn_year)
X_train = X[train_mask]
y_train = y[train_mask]
X_test  = X[test_mask]
y_test  = y[test_mask]
```

This simulates real-world deployment: the model is trained on historical data and evaluated on its ability to predict fires in years it has never seen.

\newpage

# Model Selection

Seven model families are relevant to this problem. Each is described below with its mechanism, strengths and weaknesses for fire susceptibility mapping, and key hyperparameters to tune.

## Logistic Regression (Baseline)

**How it works:** Models the log-odds of fire occurrence as a linear combination of features:
$$\log\frac{p}{1-p} = \beta_0 + \beta_1 x_1 + \ldots + \beta_n x_n$$

**Strengths:**
- Fast and interpretable — coefficients directly show feature direction and magnitude
- Probability outputs are well-calibrated
- Useful as a baseline to measure improvement from more complex models

**Weaknesses:**
- Cannot capture non-linear relationships (e.g., fire risk peaks at intermediate elevations)
- Requires feature scaling and VIF checks for multicollinearity

**Key hyperparameters:** `C` (inverse regularisation strength), `penalty` (L1 for sparsity, L2 default)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

lr_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced"))
])
```

## Random Forest

**How it works:** An ensemble of decision trees where each tree is trained on a bootstrap sample of the data, and each split considers a random subset of features. Predictions are averaged across all trees.

**Strengths:**
- Handles non-linear interactions without explicit feature engineering
- Built-in feature importance (mean decrease in impurity)
- Robust to outliers and redundant features
- `class_weight="balanced"` handles imbalance directly

**Weaknesses:**
- Does not extrapolate beyond the training data range
- Large memory footprint for high-dimensional rasters
- Can overfit on small datasets

**Performance in Nepal context:** AUC 0.91–0.95 [@khadke2025himalaya; @nepal2025rf]

**Key hyperparameters:** `n_estimators` (100–500), `max_depth` (None or 10–30), `min_samples_leaf` (5–50 for spatial data to reduce overfitting)

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=10,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
```

## XGBoost and LightGBM

**How it works:** Gradient boosted trees build an ensemble sequentially, where each new tree corrects the residual errors of the previous ensemble. XGBoost and LightGBM are two highly optimised implementations that differ in their tree-building strategy (level-wise vs. leaf-wise).

**Strengths:**
- Consistently top performance on tabular data in competitions and literature [@bilucan2024mediterranean]
- `scale_pos_weight` parameter for class imbalance
- Native handling of missing values (XGBoost)
- LightGBM is ~10× faster than XGBoost on large datasets

**Weaknesses:**
- More hyperparameters to tune
- Leaf-wise growth in LightGBM can overfit on small data

**Key hyperparameters:** `n_estimators`, `learning_rate` (0.01–0.1), `max_depth` (3–8), `subsample`, `colsample_bytree`, `scale_pos_weight` (ratio of negative to positive class)

```python
import xgboost as xgb

neg_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=neg_pos_ratio,
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1
)
```

## MaxEnt (Maximum Entropy)

**How it works:** MaxEnt is a presence-only modelling technique that estimates the distribution of fire locations by finding the probability distribution of maximum entropy (closest to uniform) subject to constraints that expected feature values match observed values at fire locations. It does not require absence data — non-fire pixels are treated as background.

**Strengths:**
- Well-established in the ecological niche modelling literature
- Handles presence-only data (useful if FIRMS detection probabilities are uncertain)
- Produces biologically interpretable response curves
- Common reference model in Nepal fire literature [@mishra2022]

**Weaknesses:**
- Conceptually different from binary classification — results are not directly comparable to AUC from RF/XGBoost
- Requires the `maxent.jar` Java package or Python alternatives (`elapid`)
- Prone to overfitting with many features (regularisation is critical)

```python
# Using elapid (Python MaxEnt implementation)
import elapid

presence = df_model[df_model["fire_occurred"] == 1][FEATURES].dropna()
background = df_model[df_model["fire_occurred"] == 0][FEATURES].sample(10000, random_state=42).dropna()

maxent = elapid.MaxentModel(
    feature_types=["linear", "quadratic", "hinge"],
    regularization_multiplier=1.5,
    clamp=True
)
maxent.fit(presence, background)
```

## Boosted Regression Trees (BRT)

**How it works:** BRT (also called Gradient Boosted Regression/Classification Trees, or GBM) is conceptually similar to XGBoost but was popularised in ecology by @elith2008brt as an interpretable species distribution modelling method. The ecological literature favours BRT because it natively produces **partial dependence plots** — smooth response curves showing the marginal effect of each predictor.

**Strengths:**
- Well-suited to ecological datasets with complex non-linearities
- Partial dependence plots provide direct ecological interpretation
- Handles interactions through tree structure
- Reference standard in ecological and fire hazard literature

**Weaknesses:**
- Slower to train than LightGBM
- Requires careful tuning of `learning_rate` × `n_trees` trade-off (Elith recommends 1000+ trees at learning rate 0.01 or lower)

```python
from sklearn.ensemble import GradientBoostingClassifier

brt_model = GradientBoostingClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.75,
    random_state=42
)
```

## Multi-Layer Perceptron (MLP)

**How it works:** A fully-connected feedforward neural network that learns non-linear feature transformations through multiple layers of weighted connections. For tabular features (one vector per pixel), a 2–4 layer MLP is standard.

**Strengths:**
- Can learn arbitrary non-linear interactions
- Easily extended to multi-task learning (predict `fire_occurred` and `fire_count` jointly)
- Can incorporate NDVI time-series sequences as input (via embedding layer or LSTM branch)

**Weaknesses:**
- Requires more data than tree-based models to avoid overfitting
- Less interpretable natively (SHAP is needed for post-hoc explanation)
- Sensitive to feature scaling — always normalise inputs

```python
import torch
import torch.nn as nn

class FireMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)
```

## Convolutional Neural Network (CNN)

**How it works:** Instead of treating each pixel independently, CNN extracts local spatial context by sliding a convolutional kernel over the multi-channel feature stack raster. Each 100m pixel is represented as the centre of a $k \times k$ spatial patch (e.g., $7 \times 7$ = 700m × 700m neighbourhood), and the CNN learns spatial patterns of fire risk.

**Strengths:**
- Captures spatial context explicitly — fire spread patterns, edge effects, neighbourhood vegetation
- Can learn multi-scale features via pooling
- @eaturu2025sea showed CNN outperforms MLP and LSTM for regions with strong spatial fire dependencies

**Weaknesses:**
- Requires the feature rasters to be in aligned grid format (they are, as GeoTIFFs in `data/processed/feature_stack_100m/`)
- Patch extraction is memory-intensive for province-scale rasters
- Harder to interpret than tree-based models

```python
import torch
import torch.nn as nn

class FireCNN(nn.Module):
    def __init__(self, in_channels, patch_size=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, 1)

    def forward(self, x):
        # x shape: (batch, channels, patch_H, patch_W)
        return torch.sigmoid(self.classifier(self.features(x).flatten(1))).squeeze(-1)
```

## Model Comparison Summary

| Model | Complexity | Interpretability | Data Need | Handles Imbalance | Spatial Context | Literature AUC (Nepal/Himalaya) |
|---|---|---|---|---|---|---|
| Logistic Regression | Low | High | Low | Via class weight | No | ~0.75–0.83 |
| Random Forest | Medium | Medium | Medium | Via class weight | No | 0.88–0.95 |
| XGBoost / LightGBM | Medium-High | Medium | Medium | `scale_pos_weight` | No | 0.91–0.95 |
| MaxEnt | Low-Medium | High | Presence-only | N/A | No | ~0.64–0.83 |
| BRT | Medium | High | Medium | Via subsampling | No | 0.88–0.91 |
| MLP | High | Low (needs SHAP) | High | Via loss weight | No | — |
| CNN | High | Low (needs SHAP) | High | Via loss weight | **Yes** | — |

**Recommended approach for a master's project:** Train RF and XGBoost as primary models, LR as a baseline, and one deep learning model (MLP) as an extension. Present the CNN as a discussion of future work or run it on a small region as a proof-of-concept.

\newpage

# Validation and Evaluation Metrics

## Why Accuracy is Misleading

If 3% of pixels are fire cells, a model predicting "never fire" achieves 97% accuracy. Accuracy is useless for imbalanced binary classification. Use the following metrics instead.

## Recommended Metrics

### AUC-ROC (Area Under the ROC Curve)

The ROC curve plots True Positive Rate against False Positive Rate at all classification thresholds. AUC-ROC summarises discrimination ability. An AUC of 0.5 is random; 1.0 is perfect.

**Limitation:** AUC-ROC can be optimistic for highly imbalanced datasets because it accounts for True Negatives (the large non-fire class), which inflates performance appearance.

### AUC-PR (Area Under the Precision-Recall Curve)

Plots Precision (PPV) against Recall (TPR). More informative than AUC-ROC for imbalanced data because it focuses on the rare positive class.

```python
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, classification_report,
    RocCurveDisplay, PrecisionRecallDisplay
)

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

print(f"AUC-ROC:  {roc_auc_score(y_test, y_prob):.4f}")
print(f"AUC-PR:   {average_precision_score(y_test, y_prob):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["Non-Fire", "Fire"]))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[0])
PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=axes[1])
axes[0].set_title("ROC Curve")
axes[1].set_title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig("docs/figures/roc_pr_curves.png", dpi=150)
```

### Optimal Threshold Selection

The default 0.5 threshold is rarely optimal. Find the threshold that maximises F1 on the validation set:

```python
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
optimal_threshold = thresholds[np.argmax(f1_scores)]
print(f"Optimal threshold: {optimal_threshold:.3f}")
```

## Spatial Cross-Validation Workflow

```python
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

cv_scores = {"auc_roc": [], "auc_pr": []}
block_groups = df_model["block_id"].values

for train_idx, test_idx in cv.split(X, y, groups=block_groups):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]

    cv_scores["auc_roc"].append(roc_auc_score(y_te, probs))
    cv_scores["auc_pr"].append(average_precision_score(y_te, probs))

print(f"CV AUC-ROC: {np.mean(cv_scores['auc_roc']):.4f} ± {np.std(cv_scores['auc_roc']):.4f}")
print(f"CV AUC-PR:  {np.mean(cv_scores['auc_pr']):.4f} ± {np.std(cv_scores['auc_pr']):.4f}")
```

## Statistical Comparison Between Models

Following @bilucan2024mediterranean, use McNemar's test to determine whether observed differences in accuracy between models are statistically significant:

```python
from statsmodels.stats.contingency_tables import mcnemar

# Compare RF vs XGBoost predictions on same test set
y_pred_rf  = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

# McNemar contingency table
b = ((y_pred_rf == y_test) & (y_pred_xgb != y_test)).sum()
c = ((y_pred_rf != y_test) & (y_pred_xgb == y_test)).sum()

result = mcnemar([[0, b], [c, 0]], exact=True)
print(f"McNemar p-value: {result.pvalue:.4f}")
```

\newpage

# Model Interpretability

## Why Interpretability Matters for Fire Susceptibility

A fire susceptibility model that identifies high-risk areas without explaining why those areas are risky is of limited value to forest managers and policy makers. Interpretability answers: *Which environmental factors drive fire probability in different parts of Bagmati Province?*

## SHAP (SHapley Additive exPlanations)

SHAP [@lundberg2017shap] decomposes each prediction into additive contributions from individual features, grounded in cooperative game theory. For a prediction $f(x)$:

$$f(x) = \phi_0 + \sum_{i=1}^{n} \phi_i$$

where $\phi_i$ is the SHAP value for feature $i$ — positive values push the prediction towards fire, negative values away from fire.

### Global Feature Importance

```python
import shap

# For tree-based models — fast TreeExplainer
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)  # shape: (n_samples, n_features)

# If rf_model outputs [class_0, class_1], take class_1 SHAP values
if isinstance(shap_values, list):
    shap_vals = shap_values[1]
else:
    shap_vals = shap_values

# Beeswarm plot — shows distribution of SHAP values per feature
shap.summary_plot(shap_vals, X_test, feature_names=FEATURES, show=False)
plt.savefig("docs/figures/shap_beeswarm.png", dpi=150, bbox_inches="tight")
```

### Interpreting the Beeswarm Plot

In the beeswarm plot:
- Each dot is one sample (pixel)
- Horizontal position = SHAP value (positive = pushes prediction towards fire)
- Colour = feature value (red = high, blue = low)
- Features are sorted by mean absolute SHAP value

**Expected findings for Bagmati Province (based on literature):**
- `slope_deg` — high slopes (red) should show positive SHAP (fire-prone)
- `ndvi_premonsoon` — low NDVI (blue = dry vegetation) → positive SHAP
- `lst_fire_season_mean_c` — high LST (red) → positive SHAP
- `dist_to_settlement_m` — short distance (blue) → positive SHAP (human ignition)
- `elevation_m` — complex pattern; may be negative (high elevation = cold, less fire) or show a hump shape

### SHAP Dependence Plots

Examine the relationship between a specific feature and its SHAP contribution, revealing threshold effects and non-linearities:

```python
# Example: NDVI fire season dependence
shap.dependence_plot(
    "ndvi_fire_season", shap_vals, X_test,
    interaction_index="lst_fire_season_mean_c",  # colour by interaction feature
    feature_names=FEATURES, show=False
)
plt.savefig("docs/figures/shap_dependence_ndvi.png", dpi=150, bbox_inches="tight")
```

### Spatial SHAP Maps

Map SHAP values back to pixel locations for spatial attribution:

```python
import rasterio
from rasterio.transform import from_bounds
import numpy as np

# Assuming df_model has 'row', 'col' indices mapping to the raster grid
# Create a SHAP raster for the most important feature
feature_idx = FEATURES.index("ndvi_fire_season")
shap_raster = np.full((n_rows, n_cols), np.nan)
shap_raster[df_model["row"], df_model["col"]] = shap_vals[:, feature_idx]

# Save as GeoTIFF
with rasterio.open(
    "data/processed/shap_ndvi_fire_season.tif", "w",
    driver="GTiff", height=n_rows, width=n_cols,
    count=1, dtype="float32", crs=crs, transform=transform
) as dst:
    dst.write(shap_raster.astype("float32"), 1)
```

## Partial Dependence Plots

PDPs show the marginal effect of a feature on prediction probability, averaged over all other features:

```python
from sklearn.inspection import PartialDependenceDisplay

fig, ax = plt.subplots(figsize=(14, 10))
PartialDependenceDisplay.from_estimator(
    rf_model, X_test, features=["slope_deg", "ndvi_fire_season",
    "lst_fire_season_mean_c", "dist_to_road_m",
    "elevation_m", "drought_factor"],
    feature_names=FEATURES, ax=ax
)
plt.suptitle("Partial Dependence Plots — Random Forest")
plt.tight_layout()
plt.savefig("docs/figures/partial_dependence.png", dpi=150)
```

## LIME (Local Interpretable Model-agnostic Explanations)

LIME explains individual pixel predictions by fitting a locally linear model around each sample:

```python
import lime
import lime.lime_tabular

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=FEATURES,
    class_names=["Non-Fire", "Fire"],
    mode="classification"
)

# Explain a specific high-risk pixel
sample_idx = y_test[y_test == 1].index[0]
sample = X_test.loc[sample_idx].values

explanation = lime_explainer.explain_instance(
    sample, rf_model.predict_proba, num_features=10
)
explanation.save_to_file("docs/figures/lime_explanation_sample.html")
```

LIME is most useful for auditing individual predictions — e.g., understanding why a particular village boundary or forest patch was flagged as high-risk.

\newpage

# Output Maps

## Generating the Susceptibility Probability Raster

After training and selecting the best model, apply it to the full province raster to produce a spatially continuous susceptibility map:

```python
import rasterio
import numpy as np

# Load full dataset (all pixels, not just training sample)
df_full = pd.read_parquet("data/processed/forest_fire_dataset_100m.parquet")
df_full_flammable = df_full[df_full["is_flammable"] == 1].copy()

# Apply same feature engineering as training
# ... (log transforms, circular encoding, etc.) ...

X_full = df_full_flammable[FEATURES]
X_full_imputed = imputer.transform(X_full)

# Predict probabilities
probs = best_model.predict_proba(X_full_imputed)[:, 1]
df_full_flammable["fire_prob"] = probs

# Reconstruct raster
prob_raster = np.full((n_rows, n_cols), np.nan)
prob_raster[df_full_flammable["row"], df_full_flammable["col"]] = probs

# Save as GeoTIFF (use same CRS and transform as input rasters)
with rasterio.open(
    "data/processed/fire_susceptibility_prob.tif", "w",
    driver="GTiff", height=n_rows, width=n_cols,
    count=1, dtype="float32", crs=target_crs, transform=target_transform,
    compress="lzw"
) as dst:
    dst.write(prob_raster.astype("float32"), 1)
```

## Risk Classification

Convert the continuous probability map into four risk categories. Use **Jenks natural breaks** to find thresholds that minimise within-class variance:

```python
import jenkspy

valid_probs = probs[~np.isnan(probs)]
breaks = jenkspy.jenks_breaks(valid_probs, n_classes=4)

risk_classes = np.digitize(probs, bins=breaks[1:-1])  # 0=Low, 1=Mod, 2=High, 3=Very High

risk_labels = {0: "Low", 1: "Moderate", 2: "High", 3: "Very High"}
for cls, label in risk_labels.items():
    pct = (risk_classes == cls).sum() / len(risk_classes) * 100
    print(f"{label}: {pct:.1f}% of flammable area")
```

**Expected output (based on @parajuli2020):** ~65% of flammable area in the High or Very High category for Bagmati Province.

## Visualising the Susceptibility Map

```python
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from rasterio.plot import show

cmap = mcolors.ListedColormap(["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"])
bounds = [0, 1, 2, 3, 4]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

with rasterio.open("data/processed/fire_susceptibility_risk_class.tif") as src:
    risk_data = src.read(1)

fig, ax = plt.subplots(figsize=(12, 10))
img = ax.imshow(risk_data, cmap=cmap, norm=norm)
plt.colorbar(img, ax=ax, ticks=[0.5, 1.5, 2.5, 3.5],
             label="Risk Class").set_ticklabels(["Low", "Moderate", "High", "Very High"])
ax.set_title("Forest Fire Susceptibility — Bagmati Province, Nepal\n(100m Resolution)")
ax.axis("off")
plt.savefig("docs/figures/susceptibility_map.png", dpi=200, bbox_inches="tight")
```

## Uncertainty Quantification

Quantify prediction uncertainty using the variance across RF tree predictions (proxy for epistemic uncertainty):

```python
# For RandomForest: collect individual tree probabilities
tree_probs = np.array([tree.predict_proba(X_full_imputed)[:, 1]
                        for tree in rf_model.estimators_])
uncertainty = tree_probs.std(axis=0)

uncertainty_raster = np.full((n_rows, n_cols), np.nan)
uncertainty_raster[df_full_flammable["row"], df_full_flammable["col"]] = uncertainty

# Save uncertainty raster
with rasterio.open("data/processed/fire_susceptibility_uncertainty.tif", "w",
                   driver="GTiff", height=n_rows, width=n_cols,
                   count=1, dtype="float32", crs=target_crs, transform=target_transform) as dst:
    dst.write(uncertainty_raster.astype("float32"), 1)
```

High uncertainty areas (where trees disagree) should be treated with caution in policy decisions.

\newpage

# Error Analysis

## Types of Errors

For binary fire susceptibility classification, there are two error types with asymmetric costs:

| Error Type | Definition | Cost |
|---|---|---|
| **False Positive (Commission)** | Model predicts fire; no fire occurred | Wasted intervention resources |
| **False Negative (Omission)** | Model misses a fire that occurred | Missed warning — potentially severe |

For fire risk management, **omission errors are costlier**: missing a high-risk area means no preventive action. This motivates setting the classification threshold lower than 0.5 to reduce omission rate, accepting more commission errors.

## Spatial Error Analysis

Map where errors occur to identify systematic failure modes:

```python
# On test set
y_pred_test = (y_prob_test >= optimal_threshold).astype(int)
errors = pd.DataFrame({
    "true": y_test.values,
    "pred": y_pred_test,
    "row": df_model.loc[y_test.index, "row"].values,
    "col": df_model.loc[y_test.index, "col"].values,
})

fp = errors[(errors["true"] == 0) & (errors["pred"] == 1)]
fn = errors[(errors["true"] == 1) & (errors["pred"] == 0)]

print(f"False Positives (commission): {len(fp)} ({len(fp)/len(errors)*100:.1f}%)")
print(f"False Negatives (omission):   {len(fn)} ({len(fn)/len(errors)*100:.1f}%)")

# Map false negatives — where is the model missing fires?
fn_raster = np.zeros((n_rows, n_cols))
fn_raster[fn["row"], fn["col"]] = 1
```

## Common Failure Patterns

**1. Understorey fires in dense forest:** MODIS active fire data misses many low-intensity understorey fires. If your fire labels come from FIRMS, you are training on fires that were detectable from space — the model may systematically miss areas with historically frequent but low-intensity burning.

**2. Agricultural burning near settlement edges:** Cropland burning is common in Bagmati Province but often excluded from "forest fire" definitions. If `lulc_code` mixes cropland and forest pixels at 100m resolution, these edges will be a source of label noise.

**3. Elevation gradient effects:** The model may perform well in the Chure belt (well-represented in training data) but poorly in the higher Himalayan forests (fewer historical fire records). Examine errors stratified by elevation band.

**4. Year-to-year NDVI variability:** An anomalously dry pre-monsoon season may shift NDVI distributions enough that the model underperforms in the holdout years. Check model calibration across test years separately.

## Calibration Curve

A well-calibrated model should have predicted probability $p = 0.3$ match an empirical fire rate of 30%:

```python
from sklearn.calibration import CalibrationDisplay

fig, ax = plt.subplots(figsize=(7, 6))
CalibrationDisplay.from_predictions(y_test, y_prob_test, n_bins=10, ax=ax)
ax.set_title("Calibration Curve — Random Forest")
plt.savefig("docs/figures/calibration_curve.png", dpi=150)
```

Tree-based models are often overconfident (predictions cluster near 0 and 1). If miscalibrated, apply Platt scaling or isotonic regression:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(rf_model, method="isotonic", cv="prefit")
calibrated_model.fit(X_val, y_val)
```

\newpage

# Recommended Implementation Workflow

The following is the recommended sequence for executing this project end-to-end:

1. **EDA** → understand class imbalance, feature distributions, fire seasonality, spatial clustering
2. **Feature engineering** → log-transforms, circular aspect encoding, NDVI anomaly, VIF check for LR
3. **Baseline** → train Logistic Regression with spatial block CV; record AUC-PR
4. **Primary models** → train Random Forest and XGBoost with spatial block CV; compare with baseline
5. **Hyperparameter tuning** → use `Optuna` or `RandomizedSearchCV` with spatial CV scoring
6. **Final evaluation** → apply best model to temporal holdout (2023–2024); report final AUC-ROC, AUC-PR, F1
7. **Interpretability** → SHAP beeswarm + dependence plots; identify top 5 predictors; spatial SHAP maps
8. **Output maps** → generate full-province susceptibility probability raster and classified risk map
9. **Error analysis** → map false negatives; examine errors by elevation band and land cover class
10. **Discussion** → compare predictor rankings with @nepal2025rf and @parajuli2020; interpret differences for Bagmati Province specifically

\newpage

# References

::: {#refs}
:::
