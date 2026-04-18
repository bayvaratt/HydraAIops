"""
TON_IoT Multiclass IDS — Random Forest Training & Evaluation
Steps 1–7: baseline, grid search, final model, confusion matrix,
feature importance, 10-fold CV, save.

Requirements:
    pip install pandas numpy scikit-learn seaborn matplotlib joblib
"""

import pickle
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------
print("Loading data …")
X_train = pd.read_csv("X_train_processed.csv").values
y_train = pd.read_csv("y_train_processed.csv").values.ravel()
X_test  = pd.read_csv("X_test_processed.csv").values
y_test  = pd.read_csv("y_test_processed.csv").values.ravel()

with open("preprocessing_metadata.pkl", "rb") as f:
    meta = pickle.load(f)

# Build reverse map: int → class name
LABEL_MAP_INV = {v: k for k, v in meta["LABEL_MAP"].items()}
class_names = [LABEL_MAP_INV[i] for i in sorted(LABEL_MAP_INV)]
feature_names = meta.get("top15_features",
                          [f"feat_{i}" for i in range(X_train.shape[1])])

print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")
print(f"Classes: {class_names}\n")

# ---------------------------------------------------------------------------
# STEP 1 — BASELINE MODEL
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 1 — BASELINE MODEL")
print("=" * 60)

rf_base = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf_base.fit(X_train, y_train)
y_pred_base = rf_base.predict(X_test)

print(classification_report(y_test, y_pred_base, target_names=class_names, digits=4))
print(f"Accuracy  : {accuracy_score(y_test, y_pred_base):.4f}")
print(f"Macro F1  : {f1_score(y_test, y_pred_base, average='macro'):.4f}")
print(f"Weighted F1: {f1_score(y_test, y_pred_base, average='weighted'):.4f}")

# ---------------------------------------------------------------------------
# STEP 2 — HYPERPARAMETER TUNING
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2 — GRID SEARCH (5-fold, scoring=f1_macro)")
print("=" * 60)

param_grid = {
    "n_estimators":    [100, 200, 300],
    "max_depth":       [None, 10, 20, 30],
    "min_samples_leaf":[1, 2, 4],
    "max_features":    ["sqrt", "log2"],
}

rf_grid = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
gs = GridSearchCV(
    rf_grid,
    param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1,
)
gs.fit(X_train, y_train)

print(f"\nbest_params_ : {gs.best_params_}")
print(f"best_score_  : {gs.best_score_:.4f}  (CV macro F1 on train)")

# ---------------------------------------------------------------------------
# STEP 3 — FINAL MODEL
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3 — FINAL MODEL")
print("=" * 60)

best_params = gs.best_params_
rf_final = RandomForestClassifier(
    **best_params,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf_final.fit(X_train, y_train)

# Inference timing
t0 = time.perf_counter()
y_pred = rf_final.predict(X_test)
t1 = time.perf_counter()
inference_ms_per_sample = (t1 - t0) / len(X_test) * 1000

print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

acc      = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
wt_f1    = f1_score(y_test, y_pred, average="weighted")
kappa    = cohen_kappa_score(y_test, y_pred)

print(f"Accuracy        : {acc:.4f}")
print(f"Macro F1        : {macro_f1:.4f}")
print(f"Weighted F1     : {wt_f1:.4f}")
print(f"Cohen's Kappa   : {kappa:.4f}")
print(f"Inference       : {inference_ms_per_sample:.4f} ms/sample")

# ---------------------------------------------------------------------------
# STEP 4 — CONFUSION MATRIX
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4 — CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row-normalised

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=0.5,
    ax=ax,
)
ax.set_title("Random Forest — Normalised Confusion Matrix", fontsize=14)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.tick_params(axis="x", rotation=35)
ax.tick_params(axis="y", rotation=0)
plt.tight_layout()
plt.savefig("rf_confusion_matrix.png", dpi=150)
plt.close()
print("Saved: rf_confusion_matrix.png")

# ---------------------------------------------------------------------------
# STEP 5 — FEATURE IMPORTANCE
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5 — FEATURE IMPORTANCE")
print("=" * 60)

importances = pd.Series(rf_final.feature_importances_, index=feature_names)
importances_sorted = importances.sort_values(ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(importances_sorted.index, importances_sorted.values,
               color="steelblue", edgecolor="black")
ax.set_title("Random Forest — Top 15 Feature Importances", fontsize=13)
ax.set_xlabel("Importance (mean decrease in impurity)")
for bar, val in zip(bars, importances_sorted.values):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig("rf_feature_importance.png", dpi=150)
plt.close()
print("Saved: rf_feature_importance.png")

print("\nTop 15 by RF importance:")
for feat, imp in importances.sort_values(ascending=False).head(15).items():
    print(f"  {feat:<35s} {imp:.4f}")

# ---------------------------------------------------------------------------
# STEP 6 — 10-FOLD STRATIFIED CROSS-VALIDATION
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6 — 10-FOLD STRATIFIED CV (full dataset)")
print("=" * 60)

X_full = np.vstack([X_train, X_test])
y_full = np.concatenate([y_train, y_test])

cv_rf = RandomForestClassifier(
    **best_params,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(cv_rf, X_full, y_full,
                            cv=skf, scoring="f1_macro", n_jobs=-1)

print(f"CV macro F1 per fold: {[f'{s:.4f}' for s in cv_scores]}")
print(f"Mean ± Std          : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ---------------------------------------------------------------------------
# STEP 7 — SAVE MODEL
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7 — SAVE MODEL")
print("=" * 60)

joblib.dump(rf_final, "rf_model.pkl")
print("Saved: rf_model.pkl")

# ---------------------------------------------------------------------------
# FINAL SUMMARY TABLE
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"| {'Metric':<22} | {'Score':>10} |")
print(f"|{'-'*24}|{'-'*12}|")
print(f"| {'Accuracy':<22} | {acc*100:>9.2f}% |")
print(f"| {'Macro F1':<22} | {macro_f1:>10.4f} |")
print(f"| {'Weighted F1':<22} | {wt_f1:>10.4f} |")
kappa_label = "Cohen's Kappa"
print(f"| {kappa_label:<22} | {kappa:>10.4f} |")
print(f"| {'Inference (ms/sample)':<22} | {inference_ms_per_sample:>10.4f} |")
print(f"| {'CV Macro F1 (mean)':<22} | {cv_scores.mean():>10.4f} |")
print(f"| {'CV Macro F1 (std)':<22} | {cv_scores.std():>10.4f} |")
print(f"|{'-'*24}|{'-'*12}|")
print(f"\nBest hyperparameters: {best_params}")
