#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# =========================
# PATHS
# =========================
DATA_STANDARD = Path("results/ml_dataset_standard.csv")
DATA_PURE_SHIFT = Path("results/ml_dataset_pure_shift.csv")


# =========================
# LOAD DATA
# =========================
df_std = pd.read_csv(DATA_STANDARD)
df_ps = pd.read_csv(DATA_PURE_SHIFT)

df = pd.concat([df_std, df_ps], ignore_index=True)


# =========================
# FEATURES / TARGETS
# =========================
X = df.filter(like="ppm_").values

y_cols = [
    col for col in df.columns
    if col not in df.filter(like="ppm_").columns
    and col not in ["sample_name", "sample_kind", "nmr_type"]
]

y = df[y_cols].values


# =========================
# STRATIFICATION
# =========================
# équilibre mixture vs reference + type NMR
stratify_labels = df["sample_kind"].astype(str) + "_" + df["nmr_type"].astype(str)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=stratify_labels
)


# =========================
# SCALING
# =========================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =========================
# PCA (réduction dimension)
# =========================
pca = PCA(n_components=50)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


print(f"PCA explained variance: {np.sum(pca.explained_variance_ratio_):.3f}")


# =========================
# ===== PLS MODEL =====
# =========================
pls = PLSRegression(n_components=10)
pls.fit(X_train_pca, y_train)

y_pred_pls = pls.predict(X_test_pca)


# =========================
# ===== RANDOM FOREST =====
# =========================
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_pca, y_train)
y_pred_rf = rf.predict(X_test_pca)


# =========================
# EVALUATION
# =========================
def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n===== {name} =====")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")


evaluate("PLS", y_test, y_pred_pls)
evaluate("Random Forest", y_test, y_pred_rf)