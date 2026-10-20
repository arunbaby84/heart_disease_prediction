from __future__ import annotations

import pandas as pd
import pytest

from src.model_pipeline import (
    FEATURE_COLUMNS,
    build_model_pipeline,
    compute_metrics,
    load_model,
    make_splits,
    predict_record,
)


def test_model_pipeline_trains_and_predicts():
    raw = pd.DataFrame(
        [
            [52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3, 1],
            [58, 0, 2, 140, 211, 0, 0, 165, 0, 2.5, 1, 1, 7, 0],
            [49, 1, 3, 130, 266, 0, 1, 171, 0, 0.6, 2, 0, 6, 1],
            [60, 1, 1, 145, 282, 0, 0, 142, 1, 2.8, 2, 2, 3, 1],
            [45, 1, 2, 120, 190, 0, 1, 160, 0, 1.2, 2, 0, 3, 1],
            [55, 0, 0, 130, 230, 1, 1, 150, 0, 0.8, 2, 0, 6, 0],
            [54, 1, 0, 128, 230, 0, 1, 155, 1, 2.2, 2, 1, 3, 1],
            [50, 0, 1, 112, 198, 0, 0, 175, 0, 0.9, 1, 0, 2, 0],
            [48, 1, 1, 122, 180, 0, 1, 170, 0, 1.5, 2, 0, 5, 0],
            [62, 0, 2, 130, 260, 0, 1, 140, 1, 3.5, 2, 2, 3, 1],
            [44, 1, 0, 123, 180, 0, 0, 160, 0, 1.1, 1, 0, 2, 0],
            [66, 0, 1, 150, 244, 0, 0, 162, 0, 1.1, 2, 0, 6, 1],
        ],
        columns=FEATURE_COLUMNS + ["target"],
    )

    X_train, X_test, y_train, y_test = make_splits(raw)
    pipeline = build_model_pipeline("logistic_regression")
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)

    assert len(predictions) == len(y_test)
    assert probabilities.shape == (len(y_test), 2)
    assert set(predictions).issubset({0, 1})


def test_build_model_pipeline_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unsupported model 'unknown'"):
        build_model_pipeline("unknown")


def test_compute_metrics_returns_expected_classification_scores():
    metrics = compute_metrics(
        y_true=pd.Series([0, 0, 1, 1]),
        y_pred=[0, 1, 1, 1],
        y_proba=[0.1, 0.7, 0.8, 0.9],
    )

    assert metrics == {
        "accuracy": 0.75,
        "precision": pytest.approx(2 / 3),
        "recall": 1.0,
        "roc_auc": 1.0,
    }


def test_saved_model_can_be_loaded_and_used_for_record_prediction(tmp_path):
    raw = pd.DataFrame(
        [
            [52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3, 1],
            [58, 0, 2, 140, 211, 0, 0, 165, 0, 2.5, 1, 1, 7, 0],
            [49, 1, 3, 130, 266, 0, 1, 171, 0, 0.6, 2, 0, 6, 1],
            [60, 1, 1, 145, 282, 0, 0, 142, 1, 2.8, 2, 2, 3, 1],
            [45, 1, 2, 120, 190, 0, 1, 160, 0, 1.2, 2, 0, 3, 1],
            [55, 0, 0, 130, 230, 1, 1, 150, 0, 0.8, 2, 0, 6, 0],
            [54, 1, 0, 128, 230, 0, 1, 155, 1, 2.2, 2, 1, 3, 1],
            [50, 0, 1, 112, 198, 0, 0, 175, 0, 0.9, 1, 0, 2, 0],
            [48, 1, 1, 122, 180, 0, 1, 170, 0, 1.5, 2, 0, 5, 0],
            [62, 0, 2, 130, 260, 0, 1, 140, 1, 3.5, 2, 2, 3, 1],
            [44, 1, 0, 123, 180, 0, 0, 160, 0, 1.1, 1, 0, 2, 0],
            [66, 0, 1, 150, 244, 0, 0, 162, 0, 1.1, 2, 0, 6, 1],
        ],
        columns=FEATURE_COLUMNS + ["target"],
    )
    X_train, _, y_train, _ = make_splits(raw)
    pipeline = build_model_pipeline("logistic_regression").fit(X_train, y_train)
    model_path = tmp_path / "model.joblib"
    import joblib

    joblib.dump(pipeline, model_path)

    prediction = predict_record(load_model(model_path), raw.iloc[0][FEATURE_COLUMNS].to_dict())

    assert prediction["prediction"] in {0, 1}
    assert 0.0 <= prediction["probability"] <= 1.0
