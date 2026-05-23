"""
FastAPI entry point for the local inference API.

Run with:
  uvicorn app:app --reload --host 127.0.0.1 --port 8000
"""

from inference.api import app

__all__ = ['app']
