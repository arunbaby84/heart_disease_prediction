from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
DATASET_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def download_dataset(output_path: str | Path | None = None) -> Path:
    root_dir = Path(__file__).resolve().parents[1]
    if output_path is None:
        output_path = root_dir / "data" / "heart.csv"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_URL, header=None, names=DATASET_COLUMNS)
    df = df.replace("?", np.nan)
    for col in df.columns[:-1]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    df["target"] = (df["target"] > 0).astype(int)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    return output_path


if __name__ == "__main__":
    download_dataset()
