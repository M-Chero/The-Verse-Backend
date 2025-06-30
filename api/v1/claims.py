# api/v1/claims.py

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List
import json

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from models.schemas import ClaimAmountDensityItem
from dependencies.auth import verify_sanctum_token
from dependencies.db import get_db
from dependencies.cache import cache

router = APIRouter()

@router.get(
    "/claims-distribution",
    response_model=List[ClaimAmountDensityItem],
)
async def get_claim_amount_density(
    user_id: int = Depends(verify_sanctum_token),
    db=Depends(get_db),
):
    cache_key = f"cleaned_data:user:{user_id}"
    raw = cache.get(cache_key)
    if not raw:
        raise HTTPException(
            status_code=404,
            detail="No cleaned data found in cache. POST to /api/v1/clean-data first.",
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(500, "Failed to parse cached cleaned data.")

    records = payload.get("data", [])
    if not records:
        return []

    df = pd.DataFrame(records)
    if "Category" not in df.columns or "Claim_Amount_KES" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="Cleaned data missing required columns: Category, Claim_Amount_KES",
        )

    # 2. rename and filter
    df = df.rename(columns={"Category": "category", "Claim_Amount_KES": "claim_amount"})
    tiers = ["Gold", "Platinum", "Silver"]
    df = df[df["category"].isin(tiers)]
    if df.empty:
        return []

    # 3. determine global range & evaluation points
    min_amt, max_amt = df["claim_amount"].min(), df["claim_amount"].max()
    n_points = 200
    xs = np.linspace(min_amt, max_amt, n_points)

    # 4. compute KDE for each tier
    densities = {}
    for tier in tiers:
        arr = df.loc[df["category"] == tier, "claim_amount"].to_numpy()
        if arr.size == 0:
            densities[tier] = np.zeros(n_points)
        else:
            kde = gaussian_kde(arr)
            densities[tier] = kde(xs)

    # 5. build response
    result = []
    for i, x in enumerate(xs):
        result.append(
            {
                "claim_amount": float(x),
                "Gold": float(densities["Gold"][i]),
                "Platinum": float(densities["Platinum"][i]),
                "Silver": float(densities["Silver"][i]),
            }
        )

    return result