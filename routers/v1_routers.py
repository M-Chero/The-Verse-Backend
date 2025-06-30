from fastapi import APIRouter
from api.v1 import upload, clean, claims

v1_router = APIRouter()

v1_router.include_router(upload.router, prefix="/v1", tags=["upload"])
v1_router.include_router(clean.router, prefix="/v1", tags=["clean"])
v1_router.include_router(claims.router, prefix="/v1", tags=["claims"])