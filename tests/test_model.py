from __future__ import annotations

import pandas as pd

from src.model_pipeline import FEATURE_COLUMNS, build_model_pipeline, make_splits


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
