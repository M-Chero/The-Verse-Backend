from typing import Union
from pathlib import Path
from fastapi import UploadFile
from utils.file_parser import parse_uploaded_file
import pandas as pd

async def process_uploaded_file(file: Union[UploadFile, Path]) -> pd.DataFrame:
    if isinstance(file, UploadFile):
        filename = file.filename
        contents = await file.read()
    elif isinstance(file, Path):
        filename = file.name
        contents = file.read_bytes()
    else:
        raise ValueError(f"Unsupported file type: {type(file)}")

    if not filename.lower().endswith((".xlsx", ".xls", ".csv")):
        raise ValueError("Only CSV and Excel files are supported.")

    return parse_uploaded_file(contents, filename)