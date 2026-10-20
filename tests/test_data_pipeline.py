from __future__ import annotations

import numpy as np
import pandas as pd
import src.model_pipeline as model_pipeline

from src.model_pipeline import (
    FEATURE_COLUMNS,
    build_full_model_pipeline,
    load_heart_data,
    make_splits,
    preprocess_heart_data,
)


def test_preprocess_heart_data_handles_missing_values_and_encodes_categories(tmp_path):
    raw = pd.DataFrame(
        {
            "age": [52, 58, 49, 60],
            "sex": [1, 0, 1, 1],
            "cp": ["0", "2", "3", "1"],
            "trestbps": [125, 140, 130, 145],
            "chol": [212, 211, 266, 282],
            "fbs": [0, 0, 0, 0],
            "restecg": [1, 0, 1, 0],
            "thalach": [168, 165, 171, 142],
            "exang": [0, 0, 0, 1],
            "oldpeak": [1.0, np.nan, 0.6, 2.8],
            "slope": [2, 1, 2, 2],
            "ca": [2, 1, 0, 2],
            "thal": ["3", "7", "6", "3"],
            "target": [1, 0, 1, 1],
        }
    )
    csv_path = tmp_path / "raw_heart.csv"
    raw.to_csv(csv_path, index=False)

    processed = preprocess_heart_data(csv_path)

    assert processed.isna().sum().sum() == 0
    assert set(processed["target"].unique()) == {0, 1}
    assert processed["cp"].dtype.kind in {"i", "u", "f"}
    assert processed["thal"].dtype.kind in {"i", "u", "f"}
    assert list(processed.columns) == FEATURE_COLUMNS + ["target"]


def test_load_heart_data_handles_missing_values_and_binary_target(tmp_path):
    raw = pd.DataFrame(
        [
            [52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3, 1],
            [58, 0, 2, 140, 211, 0, 0, 165, 0, 2.5, 1, 1, 7, 0],
            [49, 1, 3, 130, 266, 0, 1, 171, 0, 0.6, 2, 0, 6, 1],
            [60, 1, 1, 145, 282, 0, 0, 142, 1, 2.8, 2, 2, 3, 1],
        ],
        columns=FEATURE_COLUMNS + ["target"],
    )
    raw.loc[1, "thal"] = "?"
    csv_path = tmp_path / "sample_heart.csv"
    raw.to_csv(csv_path, index=False)

    df = load_heart_data(csv_path)
    assert list(df.columns) == FEATURE_COLUMNS + ["target"]
    assert df["target"].isin([0, 1]).all()
    assert df.isna().sum().sum() == 0


def test_load_heart_data_preprocesses_raw_file_when_processed_file_is_missing(tmp_path, monkeypatch):
    raw = pd.DataFrame(
        [
            [52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3, 1],
            [58, 0, 2, 140, 211, 0, 0, 165, 0, 2.5, 1, 1, 7, 0],
            [49, 1, 3, 130, 266, 0, 1, 171, 0, 0.6, 2, 0, 6, 1],
            [60, 1, 1, 145, 282, 0, 0, 142, 1, 2.8, 2, 2, 3, 1],
        ],
        columns=FEATURE_COLUMNS + ["target"],
    )
    raw_path = tmp_path / "raw.csv"
    processed_path = tmp_path / "generated" / "processed.csv"
    requested_path = tmp_path / "missing.csv"
    raw.to_csv(raw_path, index=False)
    monkeypatch.setattr(model_pipeline, "DATA_PATH", raw_path)
    monkeypatch.setattr(model_pipeline, "PROCESSED_DATA_PATH", processed_path)

    df = load_heart_data(requested_path)

    assert processed_path.exists()
    assert list(df.columns) == FEATURE_COLUMNS + ["target"]
    assert len(df) == len(raw)
    assert df.isna().sum().sum() == 0


def test_make_splits_returns_stratified_train_test_sets():
    data = pd.DataFrame(
        {
            "age": [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
            "sex": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "cp": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "trestbps": [120, 121, 122, 123, 124, 125, 126, 127, 128, 129],
            "chol": [200, 210, 220, 230, 240, 250, 260, 270, 280, 290],
            "fbs": [0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
            "restecg": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "thalach": [150, 155, 160, 165, 170, 175, 180, 185, 190, 195],
            "exang": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "oldpeak": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "slope": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "ca": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "thal": [2, 3, 2, 3, 2, 3, 2, 3, 2, 3],
            "target": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )

    X_train, X_test, y_train, y_test = make_splits(data)
    assert len(X_train) == 8
    assert len(X_test) == 2
    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})


def test_build_full_model_pipeline_includes_preprocessing():
    pipeline = build_full_model_pipeline("logistic_regression")

    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == "preprocessor"
    assert pipeline.steps[1][0] == "model"
    assert pipeline.named_steps["preprocessor"] is not None
