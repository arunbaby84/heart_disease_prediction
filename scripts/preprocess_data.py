from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model_pipeline import DATA_PATH, PROCESSED_DATA_PATH, preprocess_heart_data  # noqa: E402


if __name__ == "__main__":
    processed_path = preprocess_heart_data(DATA_PATH, PROCESSED_DATA_PATH)
    print(f"Preprocessed dataset saved to: {PROCESSED_DATA_PATH}")
    print(processed_path.head())
