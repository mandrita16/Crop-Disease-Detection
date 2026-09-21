import sys
import os
import shutil
import uuid
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from database import get_db, Prediction
from schemas import TabularInput

from gradcam import predict_with_gradcam
from tabular_shap import explain_single

app = FastAPI(title="Crop Disease & Quality Analysis API")

# No authentication - open access for local/demo use
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="../static"), name="static")
templates = Jinja2Templates(directory="../templates")

UPLOAD_DIR = "../static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load fusion model if it has been trained; otherwise fall back to a rule-based combiner
import joblib

FUSION_MODEL_PATH = "../models/fusion_meta_model.pkl"
FUSION_LE_PATH = "../models/fusion_label_encoder.pkl"
fusion_model = joblib.load(FUSION_MODEL_PATH) if os.path.exists(FUSION_MODEL_PATH) else None
fusion_le = joblib.load(FUSION_LE_PATH) if os.path.exists(FUSION_LE_PATH) else None


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1]
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    gradcam_out = os.path.join(UPLOAD_DIR, f"gradcam_{unique_name}")
    result = predict_with_gradcam(save_path, output_path=gradcam_out)

    return {
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "image_url": f"/static/uploads/{unique_name}",
        "gradcam_url": f"/static/uploads/gradcam_{unique_name}",
    }


@app.post("/predict/tabular")
async def predict_tabular(data: TabularInput):
    try:
        sample = data.model_dump()
        result = explain_single(sample)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/fusion")
async def predict_fusion(
    file: UploadFile = File(...),
    N: float = Form(...),
    P: float = Form(...),
    K: float = Form(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
    ph: float = Form(...),
    rainfall: float = Form(...),
    db: Session = Depends(get_db),
):
    # --- image branch ---
    ext = os.path.splitext(file.filename)[1]
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    gradcam_out = os.path.join(UPLOAD_DIR, f"gradcam_{unique_name}")
    image_result = predict_with_gradcam(save_path, output_path=gradcam_out)

    # --- tabular branch ---
    tabular_sample = {
        "N": N, "P": P, "K": K, "temperature": temperature,
        "humidity": humidity, "ph": ph, "rainfall": rainfall,
    }
    tabular_result = explain_single(tabular_sample)

    # --- fusion ---
    if fusion_model is not None and fusion_le is not None:
        import numpy as np
        image_probs = np.array([image_result["all_probs"]])
        tabular_probs = np.array([tabular_result["all_probs"]])
        X_meta = np.hstack([image_probs, tabular_probs])
        pred_idx = fusion_model.predict(X_meta)[0]
        pred_proba = fusion_model.predict_proba(X_meta)[0]
        final_prediction = fusion_le.inverse_transform([pred_idx])[0]
        final_confidence = round(float(max(pred_proba)), 4)

        image_top = image_result["predicted_class"]
        tabular_top = tabular_result["predicted_label"]
        needs_review = ("healthy" in image_top.lower()) != (tabular_top == "Good")
    else:
        # rule-based fallback until the fusion meta-model has been trained
        disease_is_healthy = "healthy" in image_result["predicted_class"].lower()
        quality_is_good = tabular_result["predicted_label"] == "Good"

        if disease_is_healthy and quality_is_good:
            final_prediction = "Good"
        elif not disease_is_healthy and not quality_is_good:
            final_prediction = "Poor"
        else:
            final_prediction = "Moderate"

        needs_review = (disease_is_healthy and not quality_is_good) or (not disease_is_healthy and quality_is_good)
        final_confidence = round((image_result["confidence"] + tabular_result["confidence"]) / 2, 4)

    # --- log to DB ---
    record = Prediction(
        image_path=f"/static/uploads/{unique_name}",
        disease_prediction=image_result["predicted_class"],
        disease_confidence=image_result["confidence"],
        gradcam_path=f"/static/uploads/gradcam_{unique_name}",
        n_val=N, p_val=P, k_val=K, temperature=temperature,
        humidity=humidity, ph=ph, rainfall=rainfall,
        quality_prediction=tabular_result["predicted_label"],
        quality_confidence=tabular_result["confidence"],
        fusion_prediction=final_prediction,
        fusion_confidence=final_confidence,
        needs_review=needs_review,
    )
    db.add(record)
    db.commit()

    return {
        "image_result": {
            "predicted_class": image_result["predicted_class"],
            "confidence": image_result["confidence"],
            "gradcam_url": f"/static/uploads/gradcam_{unique_name}",
        },
        "tabular_result": {
            "predicted_label": tabular_result["predicted_label"],
            "confidence": tabular_result["confidence"],
            "feature_contributions": tabular_result["feature_contributions"],
        },
        "final_prediction": final_prediction,
        "final_confidence": final_confidence,
        "needs_review": needs_review,
    }


@app.get("/history")
async def get_history(db: Session = Depends(get_db), limit: int = 20):
    records = db.query(Prediction).order_by(Prediction.timestamp.desc()).limit(limit).all()
    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat(),
            "disease_prediction": r.disease_prediction,
            "quality_prediction": r.quality_prediction,
            "fusion_prediction": r.fusion_prediction,
            "needs_review": r.needs_review,
        }
        for r in records
    ]
