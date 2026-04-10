from __future__ import annotations
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

import re
from pathlib import Path
from typing import List, Optional

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity


MODEL_PATH = Path("root_cause_model.pkl")
APP_TITLE = "Root Cause Prediction API"
APP_VERSION = "1.0.0"


class PredictRequest(BaseModel):
    error_message: str = Field(..., min_length=1, description="Raw incident or integration error message")


class BatchPredictRequest(BaseModel):
    error_messages: List[str] = Field(..., min_length=1, description="List of raw error messages")


class TopMatch(BaseModel):
    rank: int
    similarity_score: float
    matched_error_message: str
    matched_root_cause: str
    matched_incident_category: str


class PredictResponse(BaseModel):
    predicted_root_cause: str
    incident_category: str
    similarity_score: float
    is_anomalous: bool
    anomaly_score: float
    needs_human_attention: bool
    alert_reasons: List[str]
    top_matches: List[TopMatch]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    version: Optional[str] = None
    training_rows: Optional[int] = None


app = FastAPI(title=APP_TITLE, version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=APP_TITLE,
        version=APP_VERSION,
        description="API for incident triage and root cause prediction",
        routes=app.routes,
    )

    openapi_schema["openapi"] = "3.0.3"
    openapi_schema["servers"] = [
        {"url": "http://127.0.0.1:8000"}
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

_model_cache = None


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_last_meaningful_sentence(text: str) -> str:
    text = str(text).replace("\r", " ").replace("\n", " ").strip()
    if not text:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip(" .") for s in sentences if s.strip(" .")]

    if not sentences:
        return text.strip(" .")

    weak_sentences = {
        "validation error occurred",
        "processing error occurred",
        "error occurred",
        "please contact administrator",
        "please contact the administrator",
        "try again later",
    }

    for sentence in reversed(sentences):
        s = sentence.lower().strip()
        if len(s) > 8 and s not in weak_sentences:
            return sentence

    return sentences[-1]




def classify_incident_category(error_message: str, predicted_root_cause: str) -> str:
    text = f"{error_message} {predicted_root_cause}".lower()

    auth_terms = [
        "authentication failed",
        "unauthorized",
        "forbidden",
        "invalid credentials",
        "invalid credential",
        "access denied",
        "oauth",
        "token expired",
        "refresh token",
        "login failed",
        "401",
        "403",
        "identity provider",
    ]
    if any(term in text for term in auth_terms):
        return "authentication"

    schema_terms = [
        "schema",
        "xsd",
        "xml parse",
        "json parse",
        "invalid xml",
        "invalid json",
        "datatype",
        "payload structure",
        "namespace",
        "element",
        "field missing",
        "type mismatch",
        "unexpected character",
        "missing name",
        "row col",
        "unknown-source",
        "parse error",
        "parsing error",
        "malformed xml",
        "malformed json",
    ]
    if any(term in text for term in schema_terms):
        return "schema"

    timeout_terms = [
        "timeout",
        "timed out",
        "time out",
        "socket timeout",
        "read timeout",
        "connection timeout",
        "gateway timeout",
        "request timeout",
    ]
    if any(term in text for term in timeout_terms):
        return "timeout"

    mapping_terms = [
        "mapping",
        "mapped",
        "transformation",
        "crosswalk",
        "lookup failed",
        "source value",
        "target value",
        "translation",
    ]
    if any(term in text for term in mapping_terms):
        return "mapping"

    duplicate_terms = [
        "duplicate",
        "already exists",
        "already processed",
        "reprocessed",
        "duplicate processing",
        "duplicate-processing",
        "idempotent",
        "same event",
        "replay detected",
    ]
    if any(term in text for term in duplicate_terms):
        return "duplicate-processing"

    validation_terms = [
        "validation",
        "validation error",
        "not active",
        "inactive",
        "not valid",
        "cannot be rescinded",
        "effective date",
        "future-dated",
        "future dated",
        "pending transaction",
        "no longer available",
        "not available",
        "invalid value",
        "business process",
        "position must be submitted",
        "contract start date",
        "proposed",
        "supervisory organization",
    ]
    if any(term in text for term in validation_terms):
        return "validation"

    return "other"


def load_model():
    global _model_cache
    if _model_cache is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}. Run train_root_cause_model.py first."
            )
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


def predict_one(error_message: str) -> dict:
    model = load_model()

    word_vectorizer = model["word_vectorizer"]
    char_vectorizer = model["char_vectorizer"]
    X_train = model["X"]
    df = model["df"]
    anomaly_model = model["anomaly_model"]
    similarity_alert_threshold = model.get("similarity_alert_threshold", 0.35)

    normalized_error = normalize_text(error_message)

    xw = word_vectorizer.transform([normalized_error])
    xc = char_vectorizer.transform([normalized_error])
    x = hstack([xw, xc])

    sims = cosine_similarity(x, X_train)[0]
    ranked_indices = sims.argsort()[::-1]
    top_indices = ranked_indices[:3]

    best_idx = int(top_indices[0])
    best_score = float(sims[best_idx])

    retrieved_root_cause = str(df.iloc[best_idx]["Explanation"])

    # Fallback: if the match is weak, use the last meaningful sentence from the input
    if best_score < similarity_alert_threshold:
        predicted_root_cause = extract_last_meaningful_sentence(error_message)
    else:
        predicted_root_cause = retrieved_root_cause

    incident_category = classify_incident_category(error_message, predicted_root_cause)

    anomaly_pred = int(anomaly_model.predict(x.toarray())[0])
    anomaly_score = float(anomaly_model.decision_function(x.toarray())[0])
    is_anomalous = anomaly_pred == -1

    alert_reasons = []
    if is_anomalous:
        alert_reasons.append("anomaly_detector_flag")
    if best_score < similarity_alert_threshold:
        alert_reasons.append("low_similarity_to_known_patterns")

    needs_human_attention = len(alert_reasons) > 0

    top_matches = []
    for rank, idx in enumerate(top_indices, start=1):
        idx = int(idx)
        matched_error_message = str(df.iloc[idx]["Error Message"])
        matched_root_cause = str(df.iloc[idx]["Explanation"])
        matched_category = str(
            df.iloc[idx]["incident_category"]
            if "incident_category" in df.columns
            else classify_incident_category(matched_error_message, matched_root_cause)
        )
        top_matches.append(
            {
                "rank": rank,
                "similarity_score": float(sims[idx]),
                "matched_error_message": matched_error_message,
                "matched_root_cause": matched_root_cause,
                "matched_incident_category": matched_category,
            }
        )

    return {
        "predicted_root_cause": predicted_root_cause,
        "incident_category": incident_category,
        "similarity_score": best_score,
        "is_anomalous": is_anomalous,
        "anomaly_score": anomaly_score,
        "needs_human_attention": needs_human_attention,
        "alert_reasons": alert_reasons,
        "top_matches": top_matches,
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        model = load_model()
        df = model.get("df")
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_type=model.get("model_type"),
            version=model.get("version"),
            training_rows=int(len(df)) if df is not None else None,
        )
    except Exception:
        return HealthResponse(status="error", model_loaded=False)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        result = predict_one(request.error_message)
        return PredictResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/predict-batch")
def predict_batch(request: BatchPredictRequest):
    try:
        results = [predict_one(msg) for msg in request.error_messages]
        return {"count": len(results), "results": results}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}") from exc


@app.get("/")
def home():
    return {
        "message": "Root Cause Prediction API is running",
        "health": "/health",
        "docs": "/docs",
        "predict": "/predict"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("root_cause_api_app:app", host="0.0.0.0", port=8000, reload=True)
