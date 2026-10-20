import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

payload = {
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

client = TestClient(app)
response = client.post("/predict", json=payload)
result = {
    "status_code": response.status_code,
    "body": response.json(),
}
output_file = ROOT_DIR / "sample_api_output.json"
output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
