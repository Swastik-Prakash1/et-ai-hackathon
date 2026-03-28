"""main.py
=================================
FastAPI application entry point for FINANCIAL_INTEL backend.
"""

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as api_router

load_dotenv()

APP_NAME = "FINANCIAL_INTEL API"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "AI signal intelligence backend for Indian retail investors"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/")
def root() -> dict[str, str]:
    """Root route for quick sanity checks.

    Returns:
        Basic service metadata.
    """
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/api/health",
    }
