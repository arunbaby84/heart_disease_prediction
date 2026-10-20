from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model_pipeline import train_and_evaluate  # noqa: E402


if __name__ == "__main__":
    best_model_name, best_model, report = train_and_evaluate()
    print(f"Best model selected: {best_model_name}")
    print(best_model)
    print(report[best_model_name]["metrics"])
