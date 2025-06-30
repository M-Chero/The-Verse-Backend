from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from dependencies.auth import verify_sanctum_token
from pathlib import Path
import uuid
from services.upload_service import process_uploaded_file

router = APIRouter()

@router.post("/upload-data")
async def upload_data(
    file: UploadFile = File(...),
    user_id: int = Depends(verify_sanctum_token),
):
    try:
        base = Path("uploads") / str(user_id)
        base.mkdir(parents=True, exist_ok=True)
        filename = Path(file.filename).name
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        
        dest = base / unique_name
        contents = await file.read()
        dest.write_bytes(contents)

        df = await process_uploaded_file(dest)
        
        return {
            "message": "File uploaded and processed successfully",
            "filename": unique_name,
            "columns": df.columns.tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))