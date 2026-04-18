"""
TON_IoT Multiclass IDS — Full Preprocessing Pipeline
Steps 1–12: load, clean, encode, split, scale, select, resample, save.

Requirements:
    pip install pandas numpy scikit-learn imbalanced-learn matplotlib
"""

import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# STEP 1 — LOAD & INSPECT
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 1 — LOAD & INSPECT")
print("=" * 60)

df = pd.read_csv("data/ton_iot/ton_iot.csv")
print(f"\nShape: {df.shape}")
print(f"\nDtypes:\n{df.dtypes}")

# Use 'type' (multiclass attack type) as target; 'label' is binary and dropped
LABEL_COL = "type" if "type" in df.columns else "label"
# Drop the binary label column if both exist to avoid leakage
if "label" in df.columns and LABEL_COL == "type":
    df = df.drop(columns=["label"])
print(f"\nLabel column detected: '{LABEL_COL}'")
print(f"\nClass distribution:\n{df[LABEL_COL].value_counts()}")
print(f"\nNull counts per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# Class distribution bar chart
fig, ax = plt.subplots(figsize=(10, 5))
counts = df[LABEL_COL].value_counts().sort_values(ascending=False)
ax.bar(counts.index, counts.values, color="steelblue", edgecolor="black")
ax.set_title("TON_IoT Class Distribution", fontsize=14)
ax.set_xlabel("Class")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=30)
for bar, val in zip(ax.patches, counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 50,
        f"{val:,}",
        ha="center",
        va="bottom",
        fontsize=8,
    )
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.close()
print("\nSaved: class_distribution.png")

# ---------------------------------------------------------------------------
# STEP 2 — DROP BIASED COLUMNS
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2 — DROP BIASED COLUMNS")
print("=" * 60)

IDENTITY_COLS = ["src_ip", "dst_ip", "src_port", "dst_port", "ts"]
drop_identity = [c for c in IDENTITY_COLS if c in df.columns]
df = df.drop(columns=drop_identity, errors="ignore")
print(f"Dropped identity columns: {drop_identity}")

# Drop columns with >30% nulls
null_ratio = df.isnull().mean()
high_null = null_ratio[null_ratio > 0.30].index.tolist()
df = df.drop(columns=high_null, errors="ignore")
print(f"Dropped high-null columns (>30%): {high_null}")
print(f"Shape after drops: {df.shape}")

# ---------------------------------------------------------------------------
# STEP 3 — HANDLE MISSING VALUES
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3 — HANDLE MISSING VALUES")
print("=" * 60)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_all = df.select_dtypes(exclude=[np.number]).columns.tolist()

# Remove label from both lists
feature_num = [c for c in num_cols if c != LABEL_COL]
feature_cat = [c for c in cat_cols_all if c != LABEL_COL]

for c in feature_num:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].median())

for c in feature_cat:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].mode()[0])

print(f"Remaining nulls after imputation: {df.isnull().sum().sum()}")

# ---------------------------------------------------------------------------
# STEP 4 — ENCODE CATEGORICALS
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4 — ENCODE CATEGORICALS")
print("=" * 60)

cat_encoders: dict[str, LabelEncoder] = {}
for c in feature_cat:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c].astype(str))
    cat_encoders[c] = le

print(f"Label-encoded {len(cat_encoders)} categorical columns: {list(cat_encoders.keys())}")

# After encoding, rebuild numeric feature list
feature_cols = [c for c in df.columns if c != LABEL_COL]

# ---------------------------------------------------------------------------
# STEP 5 — ENCODE TARGET LABEL
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5 — ENCODE TARGET LABEL")
print("=" * 60)

LABEL_MAP = {
    "normal": 0,
    "backdoor": 1,
    "dos": 2,
    "ddos": 3,
    "injection": 4,
    "mitm": 5,
    "password": 6,
    "ransomware": 7,
    "scanning": 8,
    "xss": 9,
}

df["label_enc"] = df[LABEL_COL].map(LABEL_MAP)
unmapped = df["label_enc"].isnull().sum()
if unmapped > 0:
    print(f"WARNING: {unmapped} rows have unrecognised label values:")
    print(df[df["label_enc"].isnull()][LABEL_COL].value_counts())
    df = df.dropna(subset=["label_enc"])

df["label_enc"] = df["label_enc"].astype(int)
print(f"LABEL_MAP: {LABEL_MAP}")
print(f"Encoded distribution:\n{df['label_enc'].value_counts().sort_index()}")

# ---------------------------------------------------------------------------
# STEP 6 — LOG TRANSFORM SKEWED FEATURES
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6 — LOG TRANSFORM SKEWED FEATURES")
print("=" * 60)

# Feature columns are everything except the original label and label_enc
X_cols = [c for c in df.columns if c not in (LABEL_COL, "label_enc")]
skewed_cols = []
for c in X_cols:
    if pd.api.types.is_numeric_dtype(df[c]):
        sk = df[c].skew()
        if abs(sk) > 1.0:
            df[c] = np.log1p(np.clip(df[c], 0, None))
            skewed_cols.append(c)

print(f"Log1p-transformed {len(skewed_cols)} skewed columns (|skew| > 1.0):")
for c in skewed_cols:
    print(f"  {c}")

# ---------------------------------------------------------------------------
# STEP 7 — TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7 — TRAIN / TEST SPLIT")
print("=" * 60)

X = df[X_cols].values
y = df["label_enc"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")

train_dist = pd.Series(y_train).value_counts().sort_index()
test_dist  = pd.Series(y_test).value_counts().sort_index()
dist_df = pd.DataFrame({"Train": train_dist, "Test": test_dist})
dist_df.index = [list(LABEL_MAP.keys())[i] for i in dist_df.index]
print(f"\nClass distribution after split:\n{dist_df}")

# ---------------------------------------------------------------------------
# STEP 8 — SCALING (two versions)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 8 — SCALING")
print("=" * 60)

# Version A — StandardScaler (tree models)
std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std  = std_scaler.transform(X_test)

# Version B — MinMaxScaler (neural models)
mm_scaler = MinMaxScaler()
X_train_mm = mm_scaler.fit_transform(X_train)
X_test_mm  = mm_scaler.transform(X_test)

print("StandardScaler fitted on X_train → X_train_std, X_test_std")
print("MinMaxScaler  fitted on X_train → X_train_mm,  X_test_mm")

# ---------------------------------------------------------------------------
# STEP 9 — FEATURE SELECTION (top 15 by RF importance)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 9 — FEATURE SELECTION")
print("=" * 60)

# SelectKBest as reference ranking
skb = SelectKBest(f_classif, k="all")
skb.fit(X_train_std, y_train)
skb_scores = pd.Series(skb.scores_, index=X_cols).sort_values(ascending=False)
print(f"\nTop 15 by SelectKBest (f_classif):\n{skb_scores.head(15)}")

# RandomForest importance for final selection
print("\nFitting RandomForest(n_estimators=50) for feature importance …")
rf_sel = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf_sel.fit(X_train_std, y_train)
rf_importance = pd.Series(rf_sel.feature_importances_, index=X_cols)
top15_features = rf_importance.nlargest(15).index.tolist()
top15_idx = [X_cols.index(f) for f in top15_features]

print(f"\nTop 15 features by RF importance:")
for rank, (feat, imp) in enumerate(
    rf_importance[top15_features].sort_values(ascending=False).items(), 1
):
    print(f"  {rank:2d}. {feat:<35s} {imp:.4f}")

# Slice to top 15
X_train_sel    = X_train_std[:, top15_idx]
X_test_sel     = X_test_std[:, top15_idx]
X_train_sel_mm = X_train_mm[:, top15_idx]
X_test_sel_mm  = X_test_mm[:, top15_idx]

print(f"\nX_train_sel shape: {X_train_sel.shape}")

# ---------------------------------------------------------------------------
# STEP 10 — SMOTE-TOMEK RESAMPLING (train only)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 10 — SMOTE-TOMEK RESAMPLING")
print("=" * 60)

print("Class distribution BEFORE resampling:")
before = pd.Series(y_train).value_counts().sort_index()
before.index = [list(LABEL_MAP.keys())[i] for i in before.index]
print(before.to_string())

print("\nRunning SMOTETomek … (this may take a few minutes)")
smt = SMOTETomek(random_state=42)
X_train_res, y_train_res = smt.fit_resample(X_train_sel, y_train)

print("\nClass distribution AFTER resampling:")
after = pd.Series(y_train_res).value_counts().sort_index()
after.index = [list(LABEL_MAP.keys())[i] for i in after.index]
print(after.to_string())

comparison = pd.DataFrame({"Before": before, "After": after})
print(f"\nSide-by-side comparison:\n{comparison}")
print(f"\nX_train_res shape: {X_train_res.shape}  (was {X_train_sel.shape})")

# ---------------------------------------------------------------------------
# STEP 11 — COMPUTE CLASS WEIGHTS
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 11 — CLASS WEIGHTS")
print("=" * 60)

classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes.tolist(), weights.tolist()))
print(f"class_weight_dict: {class_weight_dict}")

# Sample weights for y_train_res (used by XGBoost sample_weight param)
sample_weights = np.array([class_weight_dict.get(int(c), 1.0) for c in y_train_res])
print(f"sample_weights shape: {sample_weights.shape}  "
      f"min={sample_weights.min():.4f}  max={sample_weights.max():.4f}")

# ---------------------------------------------------------------------------
# STEP 12 — SAVE OUTPUTS
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 12 — SAVE OUTPUTS")
print("=" * 60)

pd.DataFrame(X_train_res, columns=top15_features).to_csv(
    "X_train_processed.csv", index=False
)
pd.Series(y_train_res, name="label_enc").to_csv(
    "y_train_processed.csv", index=False
)

pd.DataFrame(X_test_sel, columns=top15_features).to_csv(
    "X_test_processed.csv", index=False
)
pd.Series(y_test, name="label_enc").to_csv(
    "y_test_processed.csv", index=False
)

pd.DataFrame(X_train_sel_mm, columns=top15_features).to_csv(
    "X_train_mm.csv", index=False
)
pd.DataFrame(X_test_sel_mm, columns=top15_features).to_csv(
    "X_test_mm.csv", index=False
)

metadata = {
    "LABEL_MAP": LABEL_MAP,
    "cat_encoders": cat_encoders,
    "top15_features": top15_features,
    "std_scaler": std_scaler,
    "mm_scaler": mm_scaler,
    "skb": skb,
    "rf_importance": rf_importance,
    "class_weight_dict": class_weight_dict,
    "X_cols": X_cols,
    "LABEL_COL": LABEL_COL,
}
with open("preprocessing_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("Saved: X_train_processed.csv")
print("Saved: y_train_processed.csv")
print("Saved: X_test_processed.csv")
print("Saved: y_test_processed.csv")
print("Saved: X_train_mm.csv")
print("Saved: X_test_mm.csv")
print("Saved: preprocessing_metadata.pkl")
print("\nPreprocessing complete. All files saved.")
