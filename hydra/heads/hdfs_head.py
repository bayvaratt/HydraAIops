import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from hydra.utils.io import ensure_dir
from hydra.utils.metrics import print_header, save_json


BLOCK_RE = re.compile(r"(blk_-?\d+)")


def _must_exist(p: str) -> str:
    if not Path(p).exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return p


def resolve_hdfs_paths(hdfs_dir: str) -> Tuple[str, str]:
    """
    Expected structure (your screenshot):
      hdfs_dir/
        raw/HDFS.log
        preprocessed/anomaly_label.csv
    """
    root = Path(hdfs_dir)

    log_path = root / "raw" / "HDFS.log"
    label_path = root / "preprocessed" / "anomaly_label.csv"

    # fallback: search if user renamed folders
    if not log_path.exists():
        found = list(root.rglob("HDFS.log"))
        if found:
            log_path = found[0]

    if not label_path.exists():
        found = list(root.rglob("anomaly_label.csv"))
        if found:
            label_path = found[0]

    return _must_exist(str(log_path)), _must_exist(str(label_path))


def load_hdfs_labels(label_csv: str) -> pd.DataFrame:
    df = pd.read_csv(label_csv)

    # normalize column names
    blk_col = None
    for c in df.columns:
        if c.lower() in ("blockid", "block_id", "blkid"):
            blk_col = c
            break
    if blk_col is None:
        blk_col = df.columns[0]

    lab_col = None
    for c in df.columns:
        if c.lower() in ("label", "anomaly", "result"):
            lab_col = c
            break
    if lab_col is None:
        lab_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    out = df[[blk_col, lab_col]].copy()
    out.columns = ["BlockId", "LabelRaw"]

    def to_y(v) -> int:
        s = str(v).strip().lower()
        if s in ("anomaly", "fail", "1", "true", "abnormal"):
            return 1
        if s in ("normal", "success", "0", "false"):
            return 0
        # unknown -> treat as normal (conservative)
        return 0

    out["y"] = out["LabelRaw"].map(to_y).astype(int)
    return out[["BlockId", "y"]]


def load_hdfs_lines(log_path: str, max_lines: Optional[int] = None) -> List[str]:
    lines = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def template_line(s: str) -> str:
    s = re.sub(r"\b\d+\b", "<NUM>", s)
    s = re.sub(r"0x[0-9a-fA-F]+", "<HEX>", s)
    s = re.sub(r"blk_-?\d+", "<BLK>", s)
    return s


def build_block_sequences(lines: List[str]) -> Dict[str, List[str]]:
    blocks: Dict[str, List[str]] = {}
    for line in lines:
        m = BLOCK_RE.search(line)
        if not m:
            continue
        blk = m.group(1)
        blocks.setdefault(blk, []).append(template_line(line))
    return blocks


def make_dataset(blocks: Dict[str, List[str]], labels: pd.DataFrame) -> pd.DataFrame:
    seq_df = pd.DataFrame({"BlockId": list(blocks.keys())})
    seq_df["text"] = seq_df["BlockId"].map(lambda b: " || ".join(blocks[b]))
    merged = seq_df.merge(labels, on="BlockId", how="inner")
    return merged


def train_hdfs_semantic(
    hdfs_dir: str,
    out_dir: str,
    max_lines: int = 200000,
    seed: int = 42,
    max_features: int = 50000,
    epochs: int = 1,
) -> None:
    # baseline model ignores epochs; kept for CLI compatibility
    _ = epochs

    ensure_dir(out_dir)

    log_path, label_path = resolve_hdfs_paths(hdfs_dir)

    print_header("HDFS: paths")
    print("log_path:", log_path)
    print("label_path:", label_path)
    save_json({"log_path": log_path, "label_path": label_path}, f"{out_dir}/hdfs_paths.json")

    labels = load_hdfs_labels(label_path)

    print_header("HDFS: label counts")
    print(labels["y"].value_counts().to_string())
    save_json({"y_counts": labels["y"].value_counts().to_dict()}, f"{out_dir}/hdfs_label_counts.json")

    lines = load_hdfs_lines(log_path, max_lines=max_lines)

    print_header("HDFS: loaded lines")
    print("lines:", len(lines))

    blocks = build_block_sequences(lines)

    print_header("HDFS: blocks found")
    print("blocks:", len(blocks))

    ds = make_dataset(blocks, labels)

    print_header("HDFS: merged dataset")
    print("shape:", ds.shape)
    save_json(
        {
            "lines": len(lines),
            "blocks": len(blocks),
            "dataset_rows": int(len(ds)),
            "dataset_y_counts": ds["y"].value_counts().to_dict(),
        },
        f"{out_dir}/hdfs_sanity.json",
    )

    if ds["y"].nunique() < 2:
        raise ValueError("HDFS merged dataset has <2 classes. Something is wrong with labels or merge.")

    X_train, X_test, y_train, y_test = train_test_split(
        ds["text"], ds["y"], test_size=0.2, random_state=seed, stratify=ds["y"]
    )

    vec = CountVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)

    print_header("HDFS report (logreg baseline)")
    print(classification_report(y_test, pred, digits=4))
    report = classification_report(y_test, pred, digits=4, output_dict=True)
    save_json(report, f"{out_dir}/hdfs_report_logreg.json")
    np.save(f"{out_dir}/hdfs_cm_logreg.npy", confusion_matrix(y_test, pred))
