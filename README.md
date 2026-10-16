# Heart Disease Prediction MLOps Project

This repository implements an end-to-end machine learning workflow for predicting heart disease risk using the Cleveland Heart Disease dataset from the UCI Machine Learning Repository. It includes data acquisition, exploratory data analysis, reproducible preprocessing, model training, experiment tracking with MLflow, FastAPI deployment, Docker packaging, CI/CD automation, and Kubernetes-ready configuration.

## Repository layout

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── app/
│   ├── __init__.py
│   └── main.py
├── data/
│   └── .gitkeep
├── docs/
│   ├── report.md
│   └── screenshots/
├── k8s/
│   └── deployment.yaml
├── notebooks/
│   └── 01_eda.py
├── scripts/
│   ├── download_data.py
│   ├── train_model.py
│   └── __init__.py
├── src/
│   ├── __init__.py
│   └── model_pipeline.py
├── tests/
│   ├── test_data_pipeline.py
│   └── test_model.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── LICENSE
├── Makefile
├── README.md
├── requirements.txt
├── pyproject.toml
└── pytest.ini
```

## Local setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the raw dataset:

```bash
python scripts/download_data.py
```

4. Run the preprocessing stage to clean missing values and encode feature columns:

```bash
python scripts/preprocess_data.py
```

5. Train the model and generate MLflow artifacts:

```bash
python scripts/train_model.py
```

6. Open the MLflow tracking UI locally:

```powershell
& .\.venv\Scripts\python.exe -m mlflow ui --host 127.0.0.1 --port 5000
```

Then open http://127.0.0.1:5000 in your browser to inspect runs, metrics, and model artifacts.

7. Start the API locally (Windows PowerShell-safe version):

```powershell
# If script execution is blocked, run the venv Python directly instead of Activate.ps1
& .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

If PowerShell blocks the activation script, you can either bypass it for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

8. Call the API:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":52,"sex":1,"cp":0,"trestbps":125,"chol":212,"fbs":0,"restecg":1,"thalach":168,"exang":0,"oldpeak":1.0,"slope":2,"ca":2,"thal":3}'
```

## Model summary

The project evaluates two production candidates:

- Logistic Regression
- Random Forest Classifier

Each model is validated with a stratified train/test split and cross-validation on ROC-AUC. The selected model is stored in two reusable formats:

- `artifacts/best_model.joblib` for sklearn-native reuse via `joblib`
- `artifacts/best_model.pkl` for a plain pickle export that can be loaded in any Python environment

The saved artifact is a full sklearn `Pipeline`, so it includes the preprocessing steps (median imputation and scaling) as well as the final estimator. That makes the model reproducible and ready for reuse without rerunning the preprocessing logic separately.

### Loading the saved model

```python
import joblib

model = joblib.load("artifacts/best_model.joblib")
# or
import pickle
with open("artifacts/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Example prediction
sample = {
    "age": 52,
    "sex": 1,
    "cp": 0,
    "trestbps": 125,
    "chol": 212,
    "fbs": 0,
    "restecg": 1,
    "thalach": 168,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 2,
    "thal": 3,
}

# The saved pipeline accepts the same feature schema as the API request
prediction = model.predict([[
    sample["age"], sample["sex"], sample["cp"], sample["trestbps"], sample["chol"],
    sample["fbs"], sample["restecg"], sample["thalach"], sample["exang"], sample["oldpeak"],
    sample["slope"], sample["ca"], sample["thal"]
]])
print(prediction)
```

## Containerized local stack

For a full local MLOps stack, use Docker Compose to run the tracking server and prediction API together:

```bash
docker compose up --build
```

This starts:

- MLflow UI at http://localhost:5000
- FastAPI prediction service at http://localhost:8000

The Compose file is defined in [docker-compose.yml](docker-compose.yml).

## CI/CD and deployment

- Automated CI is defined in [.github/workflows/ci.yml](.github/workflows/ci.yml)
- Docker packaging is defined in [Dockerfile](Dockerfile)
- Docker Compose local stack is defined in [docker-compose.yml](docker-compose.yml)
- Kubernetes deployment is defined in [k8s/deployment.yaml](k8s/deployment.yaml)
- Report documentation is in [docs/report.md](docs/report.md)

## Contributing

Contributions are welcome. Please use a feature branch, keep tests passing, and update documentation when changing the training or serving logic.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
