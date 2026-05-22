"""
FastAPI entry point — run from ASAP_brainwave_classification:

  uvicorn app:app --reload --host 127.0.0.1 --port 8000
"""

from inference.api import app

__all__ = ['app']
