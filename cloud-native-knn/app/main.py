from fastapi import FastAPI, HTTPException, Request, Form, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from joblib import load
import numpy as np
import os
from .schemas import IrisRequest, IrisResponse, DiabetesRequest, DiabetesResponse

app = FastAPI(title="Cloud-Native KNN Service", version="1.4.0")

# Static and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

def _safe_load(path):
    try:
        return load(path)
    except Exception:
        return None

# Load all available models
MODELS = {
    'iris': {
        'knn': _safe_load('app/models/iris_knn.joblib'),
        'logreg': _safe_load('app/models/iris_logreg.joblib'),
        'rf': _safe_load('app/models/iris_rf.joblib'),
    },
    'diabetes': {
        'knn': _safe_load('app/models/diabetes_knn.joblib'),
        'linreg': _safe_load('app/models/diabetes_linreg.joblib'),
        'rf': _safe_load('app/models/diabetes_rf.joblib'),
    }
}

IRIS_TARGET_NAMES = ['setosa', 'versicolor', 'virginica']

@app.get("/", tags=["meta"])
def root():
    return {
        "service": "Cloud-Native KNN",
        "endpoints": [
            "/predict/iris?model=knn|logreg|rf (POST)",
            "/predict/diabetes?model=knn|linreg|rf (POST)",
            "/metrics (GET)",
            "/samples (GET)",
            "/ui (GET)",
            "/docs"
        ]
    }

@app.get("/samples", tags=["meta"])
def samples():
    return {
        "iris": {"features": [5.1, 3.5, 1.4, 0.2], "models": ["knn","logreg","rf"]},
        "diabetes": {"features": [0.038075906, 0.050680118, 0.061696207, 0.021872355, -0.044223498, -0.03482076, -0.043400846, -0.0025922629, 0.01990749, -0.017646125], "models": ["knn","linreg","rf"]}
    }

@app.get("/ui", response_class=HTMLResponse, tags=["ui"])
def ui(request: Request):
    cm_exists = os.path.exists("app/static/iris_cm.png")
    return templates.TemplateResponse(
        "ui.html",
        {
            "request": request,
            "cm_exists": cm_exists,
            "iris_result": None,
            "diabetes_result": None,
            "diabetes_model": None,
        },
    )

@app.post("/ui/iris", response_class=HTMLResponse, tags=["ui"])
def ui_iris(request: Request,
            model: str = Form('knn'),
            sepal_length: float = Form(...),
            sepal_width: float = Form(...),
            petal_length: float = Form(...),
            petal_width: float = Form(...)):
    mdl = MODELS['iris'].get(model or 'knn')
    if mdl is None:
        raise HTTPException(status_code=503, detail=f"Iris model '{model}' not available. Train first.")
    x = np.array([sepal_length, sepal_width, petal_length, petal_width], dtype=float)
    # Some models (logreg, rf) support predict_proba too
    proba = None
    if hasattr(mdl, 'predict_proba'):
        proba = mdl.predict_proba([x])[0].tolist()
        pred_idx = int(np.argmax(proba))
    else:
        pred_idx = int(mdl.predict([x])[0])
        proba = [0,0,0]  # Fallback
    payload = {"predicted_class": pred_idx, "class_name": IRIS_TARGET_NAMES[pred_idx], "probabilities": proba, "model": model}
    cm_exists = os.path.exists("app/static/iris_cm.png")
    return templates.TemplateResponse("ui.html", {"request": request, "iris_result": payload, "cm_exists": cm_exists})

@app.post("/ui/diabetes", response_class=HTMLResponse, tags=["ui"])
def ui_diabetes(request: Request,
                model: str = Form('knn'),
                f0: float = Form(...), f1: float = Form(...), f2: float = Form(...), f3: float = Form(...),
                f4: float = Form(...), f5: float = Form(...), f6: float = Form(...), f7: float = Form(...),
                f8: float = Form(...), f9: float = Form(...)):
    mdl = MODELS['diabetes'].get(model or 'knn')
    if mdl is None:
        raise HTTPException(status_code=503, detail=f"Diabetes model '{model}' not available. Train first.")
    x = np.array([f0,f1,f2,f3,f4,f5,f6,f7,f8,f9], dtype=float)
    yhat = float(mdl.predict([x])[0])
    cm_exists = os.path.exists("app/static/iris_cm.png")
    return templates.TemplateResponse("ui.html", {"request": request, "diabetes_result": yhat, "diabetes_model": model, "cm_exists": cm_exists})

@app.post("/predict/iris", response_model=IrisResponse, tags=["inference"])
def predict_iris(req: IrisRequest, model: str = Query('knn', pattern="^(knn|logreg|rf)$")):
    mdl = MODELS['iris'].get(model)
    if mdl is None:
        raise HTTPException(status_code=503, detail=f"Iris model '{model}' not available. Train first.")
    x = np.asarray(req.features, dtype=float)
    if x.shape != (4,):
        raise HTTPException(status_code=400, detail="Iris expects 4 features: sepal_length, sepal_width, petal_length, petal_width")
    if hasattr(mdl, 'predict_proba'):
        proba = mdl.predict_proba([x])[0].tolist()
        pred = int(np.argmax(proba))
    else:
        pred = int(mdl.predict([x])[0])
        proba = [0,0,0]
    return IrisResponse(predicted_class=pred, class_name=IRIS_TARGET_NAMES[pred], probabilities=proba)

@app.post("/predict/diabetes", response_model=DiabetesResponse, tags=["inference"])
def predict_diabetes(req: DiabetesRequest, model: str = Query('knn', pattern="^(knn|linreg|rf)$")):
    mdl = MODELS['diabetes'].get(model)
    if mdl is None:
        raise HTTPException(status_code=503, detail=f"Diabetes model '{model}' not available. Train first.")
    x = np.asarray(req.features, dtype=float)
    if x.shape != (10,):
        raise HTTPException(status_code=400, detail="Diabetes expects 10 numeric features as in sklearn.datasets.load_diabetes() (standardized).")
    yhat = float(mdl.predict([x])[0])
    return DiabetesResponse(prediction=yhat)

@app.get("/metrics", tags=["metrics"])
def metrics():
    import json
    metrics_path = os.path.join("training", "metrics.json")
    if not os.path.exists(metrics_path):
        return JSONResponse({"detail": "metrics.json not found. Run `make evaluate` first."}, status_code=404)
    with open(metrics_path, "r") as f:
        data = json.load(f)
    return data
