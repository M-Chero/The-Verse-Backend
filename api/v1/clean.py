from fastapi import APIRouter, Depends, HTTPException
from dependencies.auth import verify_sanctum_token
from dependencies.db import get_db
from services.cleaning_service import clean_and_prepare_data
from dependencies.cache import cache, CACHE_TTL
import pandas as pd
import json

router = APIRouter()

@router.get("/clean-data", summary="Clean cached raw data and cache the result")
async def clean_user_uploaded_data(
    user_id: int = Depends(verify_sanctum_token),
    db = Depends(get_db),
):
    """
    1) Load raw JSON from Redis key `raw_data:user:{user_id}`.
    2) Build a DataFrame, clean it, compute stats.
    3) Cache and return the cleaned‐payload under `cleaned_data:user:{user_id}`.
    """
    try:
        raw_key = f"raw_data:user:{user_id}"
        if not cache.exists(raw_key):
            raise HTTPException(
                status_code=404,
                detail="No uploaded data found; please POST to /api/v1/upload-data first."
            )

        # load raw records
        raw_json = cache.get(raw_key)
        records = json.loads(raw_json)
        df = pd.DataFrame(records)

        # clean
        cleaned = clean_and_prepare_data(df)

        # serialize cleaned records
        cleaned_json = cleaned.reset_index(drop=True) \
                              .to_json(orient="records", date_format="iso")
        records_clean = json.loads(cleaned_json)

        # compute basic stats
        desc = cleaned.describe(include="all").T
        stats_json = (
            desc.reset_index()
                .rename(columns={"index": "column"})
                .to_json(
                    orient="records",
                    date_format="iso",
                    date_unit="s",
                    default_handler=str,
                )
        )
        stats = json.loads(stats_json)

        payload = {
            "message":    "Data cleaned successfully.",
            "rows":       len(cleaned),
            "columns":    cleaned.columns.tolist(),
            "data":       records_clean,
            "statistics": stats,
        }

        # cache cleaned
        clean_key = f"cleaned_data:user:{user_id}"
        cache.set(clean_key, json.dumps(payload), ex=CACHE_TTL)

        return payload

    except HTTPException:
        # re-raise our 404 if no raw data
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))