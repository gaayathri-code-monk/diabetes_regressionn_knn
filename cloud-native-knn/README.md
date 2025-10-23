# Cloud‑Native KNN AI Service

Lightweight, production-ready ML service demonstrating the full train → evaluate → serve lifecycle using K-Nearest Neighbors (KNN).

- FastAPI service with a simple browser UI at `/ui` and Swagger at `/docs`
- Two example problems:
  - Iris Classification (multiclass)
  - Diabetes Regression (continuous)
- Training scripts persist artifacts under `training/` and `app/models/`

Quick summary

- Frameworks: FastAPI, scikit-learn, Jinja2 (UI)
- Packaging: Docker (optional)
- Model artifacts: saved as `.joblib` in `app/models/`
- UI: `/ui` (Chart.js visualizations)

Quickstart (local)

1. Create & activate virtualenv
   - Windows PowerShell:
     . .venv\Scripts\Activate.ps1
   - macOS / Linux:
     source .venv/bin/activate
2. Install dependencies:
   python -m pip install --upgrade pip
   pip install -r requirements.txt
3. Train models (creates artifacts: joblib, confusion matrix image, CSV/JSON metrics)
   python training/train_iris.py
   python training/train_diabetes.py
4. (Optional) Aggregate metrics & export CSVs:
   python training/evaluate.py
5. Run the service
   uvicorn app.main:app --host 0.0.0.0 --port 8000
6. Open in your browser:
   - UI: http://localhost:8000/ui
   - Swagger: http://localhost:8000/docs
   - Metrics JSON: http://localhost:8000/metrics
   - Example samples: http://localhost:8000/samples

Endpoints (summary)

- POST /predict/iris?model=knn|logreg|rf
  - Body: {"features":[sepal_len,sepal_wid,petal_len,petal_wid]}
  - Returns predicted class, probabilities, model used
- POST /predict/diabetes?model=knn|linreg|rf
  - Body: {"features":[10 numeric features]}
  - Returns predicted value and model used
- GET /metrics
  - Aggregated training/evaluation metrics (JSON)
- GET /samples
  - Example input payloads for the UI and quick testing
- GET /ui
  - Browser UI with charts and KPI

Sample requests

Curl — Iris (RandomForest)

```bash
curl -s -X POST "http://localhost:8000/predict/iris?model=rf" \
  -H "Content-Type: application/json" \
  -d "{\"features\":[6.0,2.9,4.5,1.5]}"
```

Curl — Diabetes (Linear Regression)

```bash
curl -s -X POST "http://localhost:8000/predict/diabetes?model=linreg" \
  -H "Content-Type: application/json" \
  -d "{\"features\":[0.038075906,0.050680118,0.061696207,0.021872355,-0.044223498,-0.03482076,-0.043400846,-0.0025922629,0.01990749,-0.017646125]}"
```

Python example

```python
import requests
# Iris prediction
payload = {"features":[6.0,2.9,4.5,1.5]}
resp = requests.post("http://localhost:8000/predict/iris?model=knn", json=payload)
print(resp.json())

# Diabetes prediction
payload = {"features":[0.038075906,0.050680118,0.061696207,0.021872355,-0.044223498,-0.03482076,-0.043400846,-0.0025922629,0.01990749,-0.017646125]}
resp = requests.post("http://localhost:8000/predict/diabetes?model=linreg", json=payload)
print(resp.json())
```

Project structure (essential)

```
cloud-native-knn/
├─ app/
│  ├─ main.py            # FastAPI app
│  ├─ schemas.py         # Pydantic models
│  ├─ templates/ui.html  # Web UI (Chart.js)
│  ├─ static/            # UI static (css, images)
│  └─ models/            # Saved models (.joblib)
├─ training/
│  ├─ train_iris.py
│  ├─ train_diabetes.py
│  ├─ evaluate.py
│  └─ metrics.json
├─ tests/
│  └─ test_api.py
├─ Dockerfile
├─ requirements.txt
└─ README.md
```

Docker

- Build:
  ```bash
  docker build -t cloud-native-knn:latest .
  ```
- Run:
  ```bash
  docker run -it --rm -p 8000:8000 cloud-native-knn:latest
  ```
  Note: the image can be built to include pre-trained artifacts so the service is ready on first run.

Notes & troubleshooting

- If the UI shows a template error due to missing metrics or variables, ensure `training/` artifacts exist (run the training scripts) or check `app/templates/ui.html` for optional guards.
- Use `uvicorn --reload` during development to pick up code changes.

Testing & CI

- Tests located under `tests/` (e.g. `tests/test_api.py`)
- CI workflow(s) exist under `.github/workflows/` for running tests and building/publishing the Docker image

License & contribution

- Add repository license and contribution guidelines as needed (e.g., LICENSE, CONTRIBUTING.md)

If you want, I can:

- Add a minimal troubleshooting checklist to the UI template so missing variables don't raise template errors
- Add example GitHub Actions workflows or a simple Docker Compose file for local dev
