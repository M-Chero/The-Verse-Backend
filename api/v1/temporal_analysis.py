from fastapi import APIRouter, Depends, HTTPException
from typing import List
from pydantic import BaseModel
import json
import pandas as pd
import numpy as np

from dependencies.auth import verify_sanctum_token
from dependencies.cache import cache

router = APIRouter()

class DayOfWeekCount(BaseModel):
    day: str
    count: int

class RawDataItem(BaseModel):
    name: str
    a: List[float]
    b: float

class TemporalAnalysisResponse(BaseModel):
    day_counts: List[DayOfWeekCount]
    raw_data: List[RawDataItem]

@router.get(
    "/temporal-analysis",
    response_model=TemporalAnalysisResponse,
    summary="Counts of claims by day of week and monthly trends with CI",
)
async def temporal_analysis(user_id: int = Depends(verify_sanctum_token)):
    # 1. fetch cached cleaned data
    cache_key = f"cleaned_data:user:{user_id}"
    raw = cache.get(cache_key)
    if not raw:
        raise HTTPException(404, "No cleaned data found in cache.")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(500, "Failed to parse cached data.")

    records = payload.get("data", [])
    if not records:
        # return both empty lists
        return TemporalAnalysisResponse(day_counts=[], raw_data=[])

    df = pd.DataFrame(records)
    # ensure dates
    if "Submission_Date" not in df.columns:
        return TemporalAnalysisResponse(day_counts=[], raw_data=[])
    df["Submission_Date"] = pd.to_datetime(df["Submission_Date"], errors="coerce")
    df = df.dropna(subset=["Submission_Date"])

    # --- Part A: day-of-week counts ---
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_counts = (
        df["Submission_Date"]
        .dt.day_name()
        .value_counts()
        .reindex(day_order, fill_value=0)
        .reset_index()
    )
    day_counts.columns = ["day", "count"]

    day_counts_result = [
        DayOfWeekCount(day=row["day"], count=int(row["count"]))
        for _, row in day_counts.iterrows()
    ]

    # --- Part B: monthly trends with 95% CI ---
    if "Claim_Amount_KES" in df.columns:
        monthly = (
            df.set_index("Submission_Date")["Claim_Amount_KES"]
            .resample("ME")
            .agg(["sum", "count", "mean", "std"])
            .reset_index()
        )
        monthly.columns = ["Date", "Total", "Count", "Mean", "Std"]

        # compute CI bounds
        monthly["CI_lower"] = monthly["Mean"] - 1.96 * monthly["Std"] / np.sqrt(monthly["Count"])
        monthly["CI_upper"] = monthly["Mean"] + 1.96 * monthly["Std"] / np.sqrt(monthly["Count"])

        raw_data_result: List[RawDataItem] = []
        for row in monthly.itertuples(index=False):
            name = row.Date.strftime("%d %B %Y")
            raw_data_result.append(
                RawDataItem(name=name, a=[row.CI_lower, row.CI_upper], b=row.Mean)
            )
    else:
        raw_data_result = []

    return TemporalAnalysisResponse(
        day_counts=day_counts_result,
        raw_data=raw_data_result,
    )