from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
LABEL_ENCODER_PATH = (
    MODELS_DIR
    / "support_priority_label_encoder.joblib"
)


app = FastAPI(
    title="DTMLAK Prediction API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "https://ahyeongk.github.io",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
except Exception as error:
    raise RuntimeError(
        f"Failed to load model files: {error}"
    ) from error


class PredictionInput(BaseModel):
    process: str
    shift: str
    operator_level: str
    training_score: float = Field(ge=0, le=100)
    experience_months: float = Field(ge=0)
    total_units: int = Field(gt=0)
    defect_count: int = Field(ge=0)
    rework_count: int = Field(ge=0)


@app.get("/")
def home():
    return {
        "message": "DTMLAK Backend Running"
    }


@app.get("/health")
def health():
    return {
        "status": "OK",
        "model_loaded": True,
    }


@app.post("/predict")
def predict(data: PredictionInput):
    if data.defect_count > data.total_units:
        raise HTTPException(
            status_code=400,
            detail="Defect count cannot exceed total units.",
        )

    if data.rework_count > data.total_units:
        raise HTTPException(
            status_code=400,
            detail="Rework count cannot exceed total units.",
        )

    defect_rate = (
        data.defect_count
        / data.total_units
    )

    rework_rate = (
        data.rework_count
        / data.total_units
    )

    input_df = pd.DataFrame(
        [
            {
                "Process": data.process,
                "Shift": data.shift,
                "Operator_Level": data.operator_level,
                "Training_Score": data.training_score,
                "Experience_Months": data.experience_months,
                "Defect_Rate": defect_rate,
                "Rework_Rate": rework_rate,
            }
        ]
    )

    try:
        processed_input = preprocessor.transform(
            input_df
        )

        encoded_prediction = model.predict(
            processed_input
        )[0]

        probabilities = model.predict_proba(
            processed_input
        )[0]

        predicted_label = (
            label_encoder.inverse_transform(
                [encoded_prediction]
            )[0]
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        ) from error

    probability_map = {
        label: round(
            float(probability) * 100,
            1,
        )
        for label, probability in zip(
            label_encoder.classes_,
            probabilities,
        )
    }

    return {
        "support_priority": predicted_label,
        "confidence_percent": round(
            float(max(probabilities)) * 100,
            1,
        ),
        "probabilities": probability_map,
        "defect_rate": round(
            defect_rate,
            5,
        ),
        "rework_rate": round(
            rework_rate,
            5,
        ),
    } 