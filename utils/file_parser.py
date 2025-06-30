import pandas as pd
import io

def parse_uploaded_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            raise ValueError("Unsupported file format. Only .csv, .xlsx, and .xls are allowed.")
        return df
    
    except Exception as e:
        raise ValueError(f"Failed to parse file: {str(e)}")