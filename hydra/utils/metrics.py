import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def print_header(title: str) -> None:
    print("
" + "=" * 80)
    print(title)
    print("=" * 80)


def save_json(obj: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_classification_report(y_true, y_pred, out_path: str, digits: int = 4) -> dict:
    report = classification_report(y_true, y_pred, digits=digits, output_dict=True)
    save_json(report, out_path)
    return report


def save_confusion_matrix(y_true, y_pred, out_path: str) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, cm)
    return cm
