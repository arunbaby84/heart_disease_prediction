from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "heart.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "heart_preprocessed.csv"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
PICKLE_MODEL_PATH = ARTIFACTS_DIR / "best_model.pkl"
REPORT_PATH = ARTIFACTS_DIR / "model_report.json"

FEATURE_COLUMNS = [
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
]
TARGET_COLUMN = "target"


def preprocess_heart_data(
    raw_path: str | Path = DATA_PATH,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Apply the project EDA-informed preprocessing pipeline.

    This stage standardizes raw download output, converts string-coded values to numeric,
    fills missing values using dataset statistics, and encodes categorical features before
    training. The output is a clean, model-ready dataset.
    """
    df = pd.read_csv(raw_path)
    df = df.replace("?", np.nan).copy()

    for col in df.columns:
        if col == TARGET_COLUMN:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].astype("string").fillna("missing")
            df[col] = pd.Categorical(df[col]).codes.astype(np.int64)

    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
        df[TARGET_COLUMN] = (df[TARGET_COLUMN] > 0).astype(int)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col == TARGET_COLUMN:
            continue
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    df = df.dropna().reset_index(drop=True)
    df = df.loc[:, FEATURE_COLUMNS + [TARGET_COLUMN]]

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)

    return df


def load_heart_data(path: str | Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return preprocess_heart_data(DATA_PATH, PROCESSED_DATA_PATH)
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        return preprocess_heart_data(path, PROCESSED_DATA_PATH)
    df = df.replace("?", np.nan)
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = (df[TARGET_COLUMN] > 0).astype(int)
    return df


def build_preprocessing_pipeline() -> Pipeline:
    """Create the reusable preprocessing stage used before model fitting."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def build_model_pipeline(model_name: str) -> Pipeline:
    """Build a full, serializable sklearn pipeline containing preprocessing and model."""
    if model_name == "logistic_regression":
        estimator = LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
        )
    else:
        raise ValueError(f"Unsupported model '{model_name}'")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessing_pipeline()),
            ("model", estimator),
        ]
    )
    return pipeline


def build_full_model_pipeline(model_name: str) -> Pipeline:
    """Backward-compatible alias for a full preprocessing + modeling pipeline."""
    return build_model_pipeline(model_name)


def make_splits(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    from sklearn.model_selection import train_test_split

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def build_mlflow_input_example(X: pd.DataFrame) -> pd.DataFrame:
    """Create a realistic sample for schema inference with float dtypes and missing values.

    MLflow warns when an inferred schema contains integer columns because those cannot safely
    represent missing values during inference. Casting the feature example to float64 and
    inserting a few NaNs mirrors the real-world input distribution and avoids schema
    enforcement errors.
    """
    example = X.head(5).copy().astype(np.float64)
    example.iloc[0, 0] = np.nan
    example.iloc[1, -1] = np.nan
    return example


def train_and_evaluate() -> tuple[str, Pipeline, dict[str, Any]]:
    if not PROCESSED_DATA_PATH.exists():
        preprocess_heart_data(DATA_PATH, PROCESSED_DATA_PATH)

    df = load_heart_data(PROCESSED_DATA_PATH)
    X_train, X_test, y_train, y_test = make_splits(df)

    model_names = ["logistic_regression", "random_forest"]
    run_records: dict[str, Any] = {}

    for model_name in model_names:
        model = build_model_pipeline(model_name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        metrics["cv_roc_auc_mean"] = float(cv_scores.mean())
        metrics["cv_roc_auc_std"] = float(cv_scores.std())
        run_records[model_name] = {
            "metrics": metrics,
            "model": model,
            "classification_report": classification_report(y_test, y_pred, zero_division=0),
        }

    best_model_name = max(model_names, key=lambda name: run_records[name]["metrics"]["roc_auc"])
    best_model = run_records[best_model_name]["model"]

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    with open(PICKLE_MODEL_PATH, "wb") as pickle_file:
        import pickle

        pickle.dump(best_model, pickle_file)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump({name: details["metrics"] for name, details in run_records.items()}, file, indent=2)

    mlflow.set_tracking_uri((ROOT_DIR / "mlruns").as_uri())
    mlflow.set_experiment("heart_disease_prediction")

    for name, details in run_records.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params(details["model"].get_params())
            for metric_name, value in details["metrics"].items():
                mlflow.log_metric(metric_name, value)

            input_example = build_mlflow_input_example(X_train)
            signature = mlflow.models.infer_signature(input_example, y_train.head(5).to_numpy())
            mlflow.sklearn.log_model(
                details["model"],
                artifact_path="model",
                input_example=input_example,
                signature=signature,
            )

    # Persist the final full pipeline including preprocessing so it can be re-used later.
    best_model_path = MODEL_PATH
    joblib.dump(best_model, best_model_path)

    return best_model_name, best_model, run_records


def load_model(path: str | Path = MODEL_PATH) -> Pipeline:
    return joblib.load(path)


def predict_record(model: Pipeline, record: dict[str, Any]) -> dict[str, Any]:
    frame = pd.DataFrame([record], columns=FEATURE_COLUMNS)
    prediction = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0][1])
    return {"prediction": prediction, "probability": probability}


if __name__ == "__main__":
    best_model_name, best_model, report = train_and_evaluate()
    print(f"Best model: {best_model_name}")
    print(json.dumps(report[best_model_name]["metrics"], indent=2))
