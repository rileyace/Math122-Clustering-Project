#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 00:53:03 2026

@author: dengyuhe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-means clustering for MLB batter platoon splits
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# 1. Load data
# ============================================================

df = pd.read_csv("all_batters.csv")


# ============================================================
# 2. Construct additional variables
# ============================================================

df["woba_diff"] = df["wOBA_left"] - df["wOBA_right"]
df["ops_diff"] = df["OPS_left"] - df["OPS_right"]


# ============================================================
# 3. Create next-season target variable
# ============================================================
# Use current-season statistics to predict next-season wOBA.

df = df.sort_values(["Name", "Year"])

df["next_year"] = df.groupby("Name")["Year"].shift(-1)
df["wOBA_next"] = df.groupby("Name")["wOBA"].shift(-1)

# Keep only true consecutive-year pairs.
# Example: 2023 -> 2024 is kept.
# Example: 2019 -> 2021 is removed if 2020 is missing.
df = df[df["next_year"] == df["Year"] + 1].copy()


# ============================================================
# 4. Define clustering variables and regression variables
# ============================================================

# New clustering features:
# We cluster hitters based on their performance against left-handed
# and right-handed pitchers.
cluster_vars = ["wOBA_left", "wOBA_right"]

# Regression features.
# I exclude wOBA_left and wOBA_right here because they are already used
# to define clusters. This gives a cleaner test of whether clustering helps.
feature_vars = [
    "PA_left", "HR_left", "BB%_left", "K%_left", "ISO_left",
    "BABIP_left", "AVG_left", "wRC+_left",

    "PA_right", "HR_right", "BB%_right", "K%_right", "ISO_right",
    "BABIP_right", "AVG_right", "wRC+_right",

    "OPS_left", "OPS_right",
    "woba_diff", "ops_diff",

    "wOBA"
]

target_var = "wOBA_next"

# Avoid duplicate column names
all_vars = ["Year", "Name", target_var] + cluster_vars + feature_vars
all_vars = list(dict.fromkeys(all_vars))

model_df = df[all_vars].dropna().copy()

print("Modeling sample size:", model_df.shape)
print(model_df[["Year", "Name", "wOBA", "wOBA_next"]].head())


# ============================================================
# 5. Train-test split by season
# ============================================================
# The model trains on earlier seasons and tests on the latest season.
# If the latest usable row is 2024, then the target is 2025 wOBA.
# If you want to predict 2024 wOBA from 2023 stats, set test_year = 2023.

test_year = model_df["Year"].max()

# Uncomment this line if you specifically want to predict 2024 from 2023:
# test_year = 2023

train_df = model_df[model_df["Year"] < test_year].copy()
test_df = model_df[model_df["Year"] == test_year].copy()

print("\nTraining years:", sorted(train_df["Year"].unique()))
print("Testing year:", test_year)
print("Train size:", train_df.shape[0])
print("Test size:", test_df.shape[0])


# ============================================================
# 6. K-means clustering using wOBA_left and wOBA_right
# ============================================================
# Important: fit clustering only on the training data.
# Then assign test observations to the nearest training centroids.

cluster_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=3, random_state=0, n_init=10))
])

train_df["cluster"] = cluster_pipeline.fit_predict(train_df[cluster_vars])
test_df["cluster"] = cluster_pipeline.predict(test_df[cluster_vars])


# Plot clusters
plt.figure(figsize=(7, 6))

plt.scatter(
    train_df["wOBA_left"],
    train_df["wOBA_right"],
    c=train_df["cluster"].astype(int),
    alpha=0.75
)

plt.xlabel("wOBA against left-handed pitchers")
plt.ylabel("wOBA against right-handed pitchers")
plt.title("KMeans Clusters Using wOBA_left and wOBA_right")

plt.show()


# Optional: print cluster averages for interpretation
cluster_summary = train_df.groupby("cluster")[cluster_vars + ["wOBA", "wOBA_next"]].mean()
print("\nCluster summary:")
print(cluster_summary)


# ============================================================
# 7. Evaluation helper
# ============================================================

def evaluate_predictions(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "Model": model_name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }


results = []


# ============================================================
# 8. Baseline model: full-sample ridge regression
# ============================================================

alphas = np.logspace(-4, 4, 100)

baseline_model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", RidgeCV(alphas=alphas, cv=5))
])

X_train = train_df[feature_vars]
y_train = train_df[target_var]

X_test = test_df[feature_vars]
y_test = test_df[target_var]

baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)

results.append(
    evaluate_predictions(y_test, baseline_pred, "Full-sample Ridge")
)

print("\nBest alpha for full-sample ridge:")
print(baseline_model.named_steps["ridge"].alpha_)


# ============================================================
# 9. Cluster-specific ridge regression
# ============================================================

cluster_predictions = np.zeros(len(test_df))
cluster_models = {}

for c in sorted(train_df["cluster"].unique()):
    train_c = train_df[train_df["cluster"] == c]
    test_c = test_df[test_df["cluster"] == c]

    print(f"\nCluster {c}")
    print("Train observations:", train_c.shape[0])
    print("Test observations:", test_c.shape[0])

    if test_c.shape[0] == 0:
        continue

    cv_folds = min(5, train_c.shape[0])

    cluster_model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=alphas, cv=cv_folds))
    ])

    cluster_model.fit(train_c[feature_vars], train_c[target_var])
    pred_c = cluster_model.predict(test_c[feature_vars])

    cluster_predictions[test_df["cluster"].values == c] = pred_c
    cluster_models[c] = cluster_model

    print("Best alpha:")
    print(cluster_model.named_steps["ridge"].alpha_)

results.append(
    evaluate_predictions(y_test, cluster_predictions, "wOBA-cluster Ridge")
)


# ============================================================
# 10. PCA diagnostic: identify major predictor patterns
# ============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_df[feature_vars])

pca = PCA()
X_pca = pca.fit_transform(X_train_scaled)

explained_variance = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
    "Explained Variance Ratio": pca.explained_variance_ratio_,
    "Cumulative Explained Variance": np.cumsum(pca.explained_variance_ratio_)
})

print("\nPCA explained variance:")
print(explained_variance.head(10))


plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(pca.explained_variance_ratio_) + 1),
    np.cumsum(pca.explained_variance_ratio_),
    marker="o"
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA: Cumulative Explained Variance")
plt.grid(True)

plt.show()


# PCA loadings
loadings = pd.DataFrame(
    pca.components_.T,
    index=feature_vars,
    columns=[f"PC{i+1}" for i in range(len(feature_vars))]
)

print("\nTop variables contributing to PC1:")
print(loadings["PC1"].abs().sort_values(ascending=False).head(10))

print("\nTop variables contributing to PC2:")
print(loadings["PC2"].abs().sort_values(ascending=False).head(10))

print("\nTop variables contributing to PC3:")
print(loadings["PC3"].abs().sort_values(ascending=False).head(10))


# ============================================================
# 11. Full-sample PCA + Ridge
# ============================================================

pca_ridge_model = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.90)),    #keep enough components to explain 90% of the variation among regressors
    ("ridge", RidgeCV(alphas=alphas, cv=5))
])

pca_ridge_model.fit(X_train, y_train)
pca_ridge_pred = pca_ridge_model.predict(X_test)

results.append(
    evaluate_predictions(y_test, pca_ridge_pred, "Full-sample PCA + Ridge")
)

print("\nNumber of PCA components used in full-sample PCA + Ridge:")
print(pca_ridge_model.named_steps["pca"].n_components_)

print("Best alpha for full-sample PCA + Ridge:")
print(pca_ridge_model.named_steps["ridge"].alpha_)


# ============================================================
# 12. Cluster-specific PCA + Ridge
# ============================================================

cluster_pca_predictions = np.zeros(len(test_df))
cluster_pca_models = {}

for c in sorted(train_df["cluster"].unique()):
    train_c = train_df[train_df["cluster"] == c]
    test_c = test_df[test_df["cluster"] == c]

    if test_c.shape[0] == 0:
        continue

    cv_folds = min(5, train_c.shape[0])

    cluster_pca_model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.90)),
        ("ridge", RidgeCV(alphas=alphas, cv=cv_folds))
    ])

    cluster_pca_model.fit(train_c[feature_vars], train_c[target_var])
    pred_c = cluster_pca_model.predict(test_c[feature_vars])

    cluster_pca_predictions[test_df["cluster"].values == c] = pred_c
    cluster_pca_models[c] = cluster_pca_model

    print(f"\nCluster {c} PCA + Ridge")
    print("Number of PCA components:")
    print(cluster_pca_model.named_steps["pca"].n_components_)
    print("Best alpha:")
    print(cluster_pca_model.named_steps["ridge"].alpha_)

results.append(
    evaluate_predictions(y_test, cluster_pca_predictions, "wOBA-cluster PCA + Ridge")
)


# ============================================================
# 13. Compare model performance
# ============================================================

results_df = pd.DataFrame(results)

print("\nModel comparison:")
print(results_df.sort_values("MSE"))


plt.figure(figsize=(8, 5))

plt.bar(results_df["Model"], results_df["RMSE"])

plt.ylabel("RMSE")
plt.title("Prediction Error Comparison")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.show()


# ============================================================
# 14. Actual vs predicted plot
# ============================================================

prediction_df = test_df[["Year", "Name", "cluster", "wOBA", "wOBA_next"]].copy()

prediction_df["pred_full_ridge"] = baseline_pred
prediction_df["pred_cluster_ridge"] = cluster_predictions
prediction_df["pred_full_pca_ridge"] = pca_ridge_pred
prediction_df["pred_cluster_pca_ridge"] = cluster_pca_predictions

print("\nPrediction sample:")
print(prediction_df.head(15))


plt.figure(figsize=(6, 6))

plt.scatter(
    y_test,
    baseline_pred,
    label="Full Ridge",
    alpha=0.7
)

plt.scatter(
    y_test,
    cluster_predictions,
    label="wOBA-cluster Ridge",
    alpha=0.7
)

min_val = min(
    y_test.min(),
    baseline_pred.min(),
    cluster_predictions.min()
)

max_val = max(
    y_test.max(),
    baseline_pred.max(),
    cluster_predictions.max()
)

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--"
)

plt.xlabel("Actual next-season wOBA")
plt.ylabel("Predicted next-season wOBA")
plt.title("Actual vs Predicted next-season wOBA")
plt.legend()

plt.show()


# ============================================================
# 15. Save predictions and results
# ============================================================

prediction_df.to_csv("woba_prediction_results.csv", index=False)
results_df.to_csv("model_comparison_results.csv", index=False)