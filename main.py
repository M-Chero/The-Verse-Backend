from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.v1_routers import v1_router 

app = FastAPI(
    title="Fund Management API",
    description="FastAPI application handling the fund management calculations",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(v1_router, prefix="/api")