from pathlib import Path

import pandas as pd

from hydra.data.io import load_dataset


def test_load_dataset_supports_legacy_to_raw_path_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    raw_dir = Path("data/raw/demo")
    raw_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0, 4.0],
            "label": [0, 1, 0, 1],
        }
    ).to_csv(raw_dir / "demo.csv", index=False)

    cfg_path = tmp_path / "datasets.yaml"
    cfg_path.write_text(
        "demo:\n"
        "  path: data/demo/demo.csv\n"
        "  label_col: label\n"
        "  positive_label: 1\n",
        encoding="utf-8",
    )

    df, cfg = load_dataset(str(cfg_path), "demo")

    assert list(df["label"]) == [0, 1, 0, 1]
    assert cfg.path == "data/demo/demo.csv"
