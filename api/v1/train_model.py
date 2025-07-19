from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Any, Dict
from dependencies.auth import verify_sanctum_token
from dependencies.cache import cache
import json
import pandas as pd
from utils.train_model_formula import train_model_formula
import traceback

router = APIRouter()

class TrainResult(BaseModel):
    Model: str
    MAE: float
    RMSE: float
    R2: float

@router.get(
    "/train-model",
    summary="Train a regression model",
)
async def train_model(
    model_algorithm: str = "Gradient Boosting",
    target_variable: str = "Claim_Amount_KES",
    test_set_size: float = 0.2,
    cross_validation_folds: int = 5,
    enable_hyperparameter_tuning: bool = False,
    max_iter: int = 20,
    user_id: int = Depends(verify_sanctum_token),
):
    cache_key = f"cleaned_data:user:{user_id}"
    raw = cache.get(cache_key)
    if not raw:
        raise HTTPException(404, "No cleaned data found in cache.")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(500, "Failed to parse cached data.")

    records = payload.get("data", [])
    records = pd.DataFrame(records)

    # Kick off the training
    try:
        results_df = train_model_formula(
            model_algorithm=model_algorithm,
            target_variable=target_variable,
            test_set_size=test_set_size/100,
            cross_validation_folds=cross_validation_folds,
            enable_hyperparameter_tuning=enable_hyperparameter_tuning,
            max_iter=max_iter,
            records=records
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")

    if results_df is False or results_df is None:
        raise HTTPException(status_code=400, detail="No Result")

    # Convert to JSON-able list of dicts
    try:
        recs = results_df.to_dict(orient="records")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Return results failed: {e}")

    return recs