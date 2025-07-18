import json
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from dependencies.auth import verify_sanctum_token
from dependencies.db import get_db
from dependencies.cache import cache
from fastapi import Query

router = APIRouter()

@router.get(
    "/claims-overview",
)
async def get_claims_overview(
    period: str = Query("monthly", enum=["daily", "weekly", "monthly"]),
    user_id: int = Depends(verify_sanctum_token),
    db=Depends(get_db),
):
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
        return {"total_claims": [], "total_claims_by_employee": [], "claims_by_hierarchy": {}}

    # Build DataFrame
    df = pd.DataFrame(records)

    # Parse Submission_Date to datetime with explicit format
    df['Submission_Date'] = pd.to_datetime(df['Submission_Date'], dayfirst=True, errors='coerce')

    # Ensure Claim_Amount_KES is numeric
    df['Claim_Amount_KES'] = pd.to_numeric(df['Claim_Amount_KES'], errors='coerce').fillna(0)

    # Handle missing values in categorical columns
    df['Visit_Type'] = df['Visit_Type'].fillna('Unknown')
    df['Diagnosis'] = df['Diagnosis'].fillna('Unknown')
    df['Treatment'] = df['Treatment'].fillna('Unknown')

    # Calculate total claims by employee
    employee_group = df.groupby('Employee_ID')['Claim_Amount_KES'].sum().sort_values(ascending=False)
    top_10_employees = employee_group.head(10)
    claims_by_employee = [
        {"Employee_ID": employee_id, "Claim Amount": int(amount)}
        for employee_id, amount in top_10_employees.items()
    ]

    # Calculate trend data for total_claims
    if period == "daily":
        group = df.groupby(df['Submission_Date'].dt.strftime('%Y-%m-%d'))
    elif period == "weekly":
        group = df.groupby(df['Submission_Date'].dt.to_period('W').apply(lambda r: r.start_time.strftime('%Y-%m-%d')))
    elif period == "monthly":
        group = df.groupby(df['Submission_Date'].dt.to_period('M').apply(lambda r: r.start_time.strftime('%Y-%m')))
    else:
        raise HTTPException(400, "Invalid period. Use 'daily', 'weekly', or 'monthly'.")

    trend_data = [
        {"x": date, "y": int(count)}
        for date, count in group.size().sort_index().items()
    ]

    # Calculate hierarchical data for Sunburst (Visit_Type -> Diagnosis -> Treatment)
    hierarchy = []
    for visit_type in df['Visit_Type'].unique():
        visit_df = df[df['Visit_Type'] == visit_type]
        visit_children = []
        for diagnosis in visit_df['Diagnosis'].unique():
            diag_df = visit_df[visit_df['Diagnosis'] == diagnosis]
            diag_children = [
                {"id": treatment, "value": int(count)}
                for treatment, count in diag_df['Treatment'].value_counts().items()
            ]
            visit_children.append({
                "id": diagnosis,
                "value": int(diag_df.shape[0]),
                "children": diag_children
            })
        hierarchy.append({
            "id": visit_type,
            "value": int(visit_df.shape[0]),
            "children": visit_children
        })

    return {
        "total_claims": [
            {
                "id": "Claims",
                "data": trend_data
            }
        ],
        "total_claims_by_employee": claims_by_employee,
        "claims_by_hierarchy": {
            "id": "Claims",
            "children": hierarchy
        }
    }