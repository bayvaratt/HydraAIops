import json
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import warnings


from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

from hydra.utils.io import ensure_dir
from hydra.utils.metrics import print_header, save_json


# ----------------------------
# config
# ----------------------------

@dataclass
class TonConfig:
    target: str = "type"            # "type" (multiclass) or "label" (binary)
    test_size: float = 0.2
    seed: int = 42

    # sampling
    max_rows: Optional[int] = None  # cap after loading (sample after full load)
    min_per_class: int = 0          # 0 disables

    # split behaviour
    use_group_split: bool = False   # default OFF
    group_col: str = "src_ip"

    # feature leakage drops
    drop_ip_cols: bool = True
    drop_ports: bool = True

    # cat handling
    cat_top_k: int = 200            # cap high-cardinality categories

    # sanity thresholds (soft)
    min_test_rows: int = 200        # if test smaller than this, reduce test_size
    min_test_classes: int = 2       # only for multiclass


# ----------------------------
# utils
# ----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_ton(csv_path: str, max_rows: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    """
    Loads the full TON CSV, then samples if max_rows is set.
    This avoids the "sorted by type" problem when using nrows=.
    """
    df = pd.read_csv(csv_path)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return df


def make_ohe():
    # sklearn >= 1.2 uses sparse_output; older uses sparse
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def ton_sanity(df: pd.DataFrame) -> dict:
    info = {
        "shape": df.shape,
        "cols": df.columns.tolist(),
        "dtypes_count": df.dtypes.astype(str).value_counts().to_dict(),
        "missing_any": bool(df.isna().any().any()),
    }
    if "label" in df.columns:
        info["label_counts"] = df["label"].value_counts(dropna=False).to_dict()
    if "type" in df.columns:
        info["type_counts_top"] = df["type"].value_counts(dropna=False).head(20).to_dict()
    return info


def cap_cardinality(series: pd.Series, top_k: int) -> pd.Series:
    if series.nunique(dropna=False) <= top_k:
        return series
    vc = series.value_counts(dropna=False)
    keep = set(vc.head(top_k).index)
    return series.where(series.isin(keep), "__OTHER__")


def enforce_min_per_class(df: pd.DataFrame, y_col: str, min_per_class: int, seed: int) -> pd.DataFrame:
    if min_per_class <= 0:
        return df
    parts = []
    for cls, g in df.groupby(y_col):
        if len(g) <= min_per_class:
            parts.append(g)
        else:
            parts.append(g.sample(n=min_per_class, random_state=seed))
    out = pd.concat(parts, axis=0)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def stratified_cap(df: pd.DataFrame, y_col: str, max_rows: Optional[int], seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    if y_col not in df.columns or df[y_col].nunique() < 2:
        return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    n_classes = df[y_col].nunique()
    per_class = max(1, max_rows // n_classes)

    out = (
        df.groupby(y_col, group_keys=False)
          .apply(lambda g: g.sample(n=min(len(g), per_class), random_state=seed))
    )

    # top-up if short
    if len(out) < max_rows:
        need = max_rows - len(out)
        rest = df.drop(out.index, errors="ignore")
        if len(rest) > 0:
            out = pd.concat([out, rest.sample(n=min(need, len(rest)), random_state=seed)])

    out = out.sample(n=min(max_rows, len(out)), random_state=seed).reset_index(drop=True)
    return out


def prepare_xy(
    df: pd.DataFrame,
    target: str,
    drop_ip_cols: bool,
    drop_ports: bool,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str], List[str]]:
    """
    Returns X, y, num_cols, cat_cols, dropped_cols
    """
    df = df.copy()
    if target not in df.columns:
        raise ValueError(f"Missing target column: {target}")

    drop_cols: List[str] = []

    # always drop the other label to prevent leakage
    if target == "type" and "label" in df.columns:
        drop_cols.append("label")
    if target == "label" and "type" in df.columns:
        drop_cols.append("type")

    if drop_ip_cols:
        for c in ["src_ip", "dst_ip"]:
            if c in df.columns:
                drop_cols.append(c)

    if drop_ports:
        for c in ["src_port", "dst_port"]:
            if c in df.columns:
                drop_cols.append(c)

    y = df[target]
    X = df.drop(columns=[target] + drop_cols, errors="ignore")

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if X[c].dtype != "object"]
    return X, y, num_cols, cat_cols, drop_cols


def pick_test_size(n_rows: int, cfg: TonConfig) -> float:
    ts = cfg.test_size
    if int(n_rows * ts) < cfg.min_test_rows:
        ts = min(0.5, max(cfg.min_test_rows / max(n_rows, 1), 0.05))
    return ts


# ----------------------------
# main training
# ----------------------------

def train_ton(df: pd.DataFrame, out_dir: str, cfg: TonConfig) -> None:
    ensure_dir(out_dir)
    set_seed(cfg.seed)

    print_header("TON_IoT: sanity")
    sanity = ton_sanity(df)
    print(json.dumps(sanity, indent=2))
    save_json(sanity, f"{out_dir}/ton_sanity.json")

    # sampling
    if cfg.target == "type":
        if cfg.max_rows is not None:
            df = stratified_cap(df, "type", cfg.max_rows, cfg.seed)
        df = enforce_min_per_class(df, "type", cfg.min_per_class, cfg.seed)
    else:
        if cfg.max_rows is not None and len(df) > cfg.max_rows:
            df = df.sample(n=cfg.max_rows, random_state=cfg.seed).reset_index(drop=True)

    X, y, num_cols, cat_cols, dropped = prepare_xy(
        df=df,
        target=cfg.target,
        drop_ip_cols=cfg.drop_ip_cols,
        drop_ports=cfg.drop_ports,
    )

    # cap high-cardinality cats
    X2 = X.copy()
    for c in cat_cols:
        if X2[c].nunique(dropna=False) > cfg.cat_top_k:
            X2[c] = cap_cardinality(X2[c], cfg.cat_top_k)

    test_size = pick_test_size(len(X2), cfg)
    use_group = bool(cfg.use_group_split and cfg.group_col in df.columns)

    print_header("TON_IoT: feature plan")
    print(f"target: {cfg.target}")
    print(f"rows used: {len(X2)}")
    print(f"use_group_split: {use_group}")
    print(f"test_size: {test_size:.3f}")
    print(f"dropped: {sorted(set(dropped))}")
    print(f"X shape: {X2.shape}")
    print(f"num_cols: {len(num_cols)}  cat_cols: {len(cat_cols)}")

    if cfg.target == "type":
        print("type counts (after sampling):")
        print(df["type"].value_counts().head(20).to_string())

    # split
    if use_group:
        groups = df[cfg.group_col]
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=cfg.seed)
        train_idx, test_idx = next(splitter.split(X2, y, groups=groups))
        X_train, X_test = X2.iloc[train_idx], X2.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        stratify = y if y.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X2, y, test_size=test_size, random_state=cfg.seed, stratify=stratify
        )

    # if multiclass and test collapses, force stratified split
    if cfg.target == "type" and y_test.nunique() < cfg.min_test_classes:
        print_header("TON_IoT: split warning")
        print(f"Test set has only {y_test.nunique()} classes. Forcing stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X2, y, test_size=test_size, random_state=cfg.seed, stratify=y
        )

    # preprocess
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", make_ohe(), cat_cols),
        ]
    )

    # model selection (default baseline)
    model_name = "sklearn_logreg"
    model = LogisticRegression(max_iter=2000)

    # Prefer LightGBM if available
    try:
        import lightgbm as lgb

        model_name = "lightgbm"
        objective = "multiclass" if cfg.target == "type" else "binary"

        model = lgb.LGBMClassifier(
            objective=objective,
            n_estimators=600,
            learning_rate=0.05,
            max_depth=10,
            num_leaves=63,
            min_child_samples=50,
            min_split_gain=1e-3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=cfg.seed,
            n_jobs=-1,
            verbosity=-1,
        )
    except Exception:
        # try xgboost; else keep logreg
        try:
            import xgboost as xgb

            model_name = "xgboost"
            if cfg.target == "type":
                model = xgb.XGBClassifier(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=8,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    n_jobs=-1,
                    random_state=cfg.seed,
                )
            else:
                model = xgb.XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=8,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    n_jobs=-1,
                    random_state=cfg.seed,
                )
        except Exception:
            pass

    # ALWAYS pipeline end-to-end
    clf = Pipeline([
        ("pre", pre),
        ("model", model),
    ])

    print_header(f"TON_IoT training ({model_name})")
    clf.fit(X_train, y_train)

    # Predict (silence LightGBM "feature names" warning only when using lightgbm)
    if model_name == "lightgbm":
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"X does not have valid feature names.*LGBMClassifier was fitted with feature names",
                category=UserWarning,
            )
            preds = clf.predict(X_test)
    else:
        preds = clf.predict(X_test)

    print_header(f"TON_IoT report ({model_name})")
    print(classification_report(y_test, preds, digits=4, zero_division=0))

    report = classification_report(y_test, preds, digits=4, zero_division=0, output_dict=True)
    save_json(report, f"{out_dir}/ton_report_{model_name}.json")

    cm = confusion_matrix(y_test, preds)
    np.save(f"{out_dir}/ton_cm_{model_name}.npy", cm)

    try:
        import joblib
        joblib.dump(clf, f"{out_dir}/ton_model_{model_name}.joblib")
    except Exception:
        pass


# ----------------------------
# public API used by CLI
# ----------------------------

def train_ton_multiclass(df: pd.DataFrame, out_dir: str, seed: int = 42) -> None:
    cfg = TonConfig(target="type", seed=seed)
    train_ton(df, out_dir=out_dir, cfg=cfg)


def train_ton_binary(df: pd.DataFrame, out_dir: str, seed: int = 42) -> None:
    cfg = TonConfig(target="label", seed=seed)
    train_ton(df, out_dir=out_dir, cfg=cfg)
