from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from dependencies.auth import verify_sanctum_token
from dependencies.cache import cache, CACHE_TTL
from utils.file_parser import parse_uploaded_file
import logging
import traceback

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/upload-data", summary="Upload raw data and cache it in Redis")
async def upload_data(
    file: UploadFile = File(...),
    user_id: int = Depends(verify_sanctum_token),
):
    """
    Reads the uploaded file into memory, parses it to a DataFrame,
    then caches the raw JSON records under `raw_data:user:{user_id}`.
    """
    try:
        contents = await file.read()
        # parse into DataFrame (same parser used in clean.py)
        df = parse_uploaded_file(contents, file.filename)

        # serialize to JSON records
        json_str = df.reset_index(drop=True) \
                     .to_json(orient="records", date_format="iso")
        cache_key = f"raw_data:user:{user_id}"
        cache.set(cache_key, json_str, ex=CACHE_TTL)

        return {
            "message": "Uploaded data cached successfully",
            "columns": df.columns.tolist(),
        }
    
    except Exception as e:
        # 1) log full traceback to console
        tb = traceback.format_exc()
        logger.error(f"Error parsing uploaded file:\n{tb}")

        # 2) return the exception message in the JSON response
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse uploaded file: {str(e)}"
        )