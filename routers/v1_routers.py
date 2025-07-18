from fastapi import APIRouter
from api.v1 import upload, clean, claims, temporal_analysis, train_model, claims_overview

v1_router = APIRouter()

v1_router.include_router(upload.router, prefix="/v1", tags=["upload"])
v1_router.include_router(clean.router, prefix="/v1", tags=["clean"])

# Minet Endpoints
v1_router.include_router(claims.router, prefix="/v1", tags=["claims"])
v1_router.include_router(temporal_analysis.router, prefix="/v1", tags=["temporal-analysis"])
v1_router.include_router(train_model.router, prefix="/v1", tags=["train-model"])

# Safaricom Endpoints
v1_router.include_router(claims_overview.router, prefix="/v1", tags=["claims-overview"])