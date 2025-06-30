from fastapi import APIRouter, Depends, HTTPException
from dependencies.auth import verify_sanctum_token
from dependencies.db import get_db
from services.cleaning_service import clean_and_prepare_data
from utils.file_parser import parse_uploaded_file
from pathlib import Path
import json
from dependencies.cache import cache, CACHE_TTL

router = APIRouter()

@router.get("/clean-data")
async def clean_user_uploaded_data(
    user_id: int = Depends(verify_sanctum_token),
    db = Depends(get_db),
):
    try:
        cache_key = f"cleaned_data:user:{user_id}"

        if cache.exists(cache_key):
            return json.loads(cache.get(cache_key))
        
        user_dir = Path("uploads") / str(user_id)
        if not user_dir.exists():
            raise HTTPException(status_code=404, detail="No uploads found for this user.")

        files = [
            fp for fp in user_dir.iterdir()
            if fp.suffix.lower() in (".csv", ".xls", ".xlsx")
        ]
        if not files:
            raise HTTPException(status_code=404, detail="No valid uploaded files found for this user.")

        latest = max(files, key=lambda f: f.stat().st_mtime)

        contents = latest.read_bytes()
        df = parse_uploaded_file(contents, latest.name)
        cleaned = clean_and_prepare_data(df)

        json_str = cleaned \
                .reset_index(drop=True) \
                .to_json(orient="records", date_format="iso")
        
        records = json.loads(json_str)

        # 5) Compute basic stats
        desc = cleaned.describe(include='all').T

        stats_json = (
            desc
            .reset_index()
            .rename(columns={'index': 'column'})
            .to_json(
                orient='records', 
                date_format='iso',
                date_unit='s',
                default_handler=str
            )
        )
        stats = json.loads(stats_json)

        payload = {
            "message":    "Data cleaned successfully.",
            "filename":   latest.name,
            "rows":       len(cleaned),
            "columns":    cleaned.columns.tolist(),
            "data":       records,
            "statistics": stats,
        }

        cache.set(cache_key, json.dumps(payload), ex=CACHE_TTL)

        return payload

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))