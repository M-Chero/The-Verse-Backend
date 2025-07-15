from fastapi import APIRouter, Depends, HTTPException
from typing import Any, Dict, List
import json

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from models.schemas import ClaimAmountDensityItem  # you’ll need a new schema if you change the shape
from dependencies.auth import verify_sanctum_token
from dependencies.db import get_db
from dependencies.cache import cache

router = APIRouter()

@router.get(
    "/claims-distribution",
    response_model=Dict[str, Any],  # adjust or create a new Pydantic model if you want strict typing
)
async def get_claim_amount_density_and_correlation(
    user_id: int = Depends(verify_sanctum_token),
    db=Depends(get_db),
):
    # 1. pull cleaned data from cache
    cache_key = f"cleaned_data:user:{user_id}"
    raw = cache.get(cache_key)
    if not raw:
        raise HTTPException(404, "No cleaned data found in cache. POST to /api/v1/clean-data first.")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(500, "Failed to parse cached cleaned data.")

    records = payload.get("data", [])
    if not records:
        return {"density": [], "heatMapDistribution": []}

    # 2. build DataFrame
    df = pd.DataFrame(records)
    # rename as you already do
    df = df.rename(columns={"Category": "category", "Claim_Amount_KES": "claim_amount"})
    tiers = ["Gold", "Platinum", "Silver"]
    df = df[df["category"].isin(tiers)]
    if df.empty:
        return {"density": [], "heatMapDistribution": []}

    # 3. compute claim‐amount density
    min_amt, max_amt = df["claim_amount"].min(), df["claim_amount"].max()
    xs = np.linspace(min_amt, max_amt, 200)
    densities = {}
    for tier in tiers:
        arr = df.loc[df["category"] == tier, "claim_amount"].to_numpy()
        densities[tier] = gaussian_kde(arr)(xs) if arr.size else np.zeros_like(xs)

    density_result = [
        {
            "claim_amount": float(x),
            **{ tier: float(densities[tier][i]) for tier in tiers }
        }
        for i, x in enumerate(xs)
    ]

    # 4. compute correlation matrix for all numeric columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    heat_map_distribution: List[Dict[str, Any]] = []
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr().stack().reset_index(name="correlation")
        corr.columns = ["row", "col", "correlation"]
        for row_var in numeric_cols:
            subset = corr[corr["row"] == row_var]
            heat_map_distribution.append({
                "id": row_var,
                "data": [
                    {"x": col_var, "y": corr_val}
                    for col_var, corr_val in zip(subset["col"], subset["correlation"])
                ]
            })

    # 5. return both in one payload
    return {
        "density": density_result,
        "heatMapDistribution": heat_map_distribution,
    }