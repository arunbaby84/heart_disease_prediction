install:
	python -m pip install -r requirements.txt

train:
	python scripts/download_data.py
	python scripts/train_model.py

test:
	python -m pytest -q

run-api:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

build-image:
	docker build -t heart-disease-api:latest .

run-image:
	docker run --rm -p 8000:8000 heart-disease-api:latest
