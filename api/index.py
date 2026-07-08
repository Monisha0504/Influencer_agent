"""
Vercel serverless entry point.

Vercel's Python builder looks for an ASGI/WSGI `app` object inside files under
api/. The actual application lives in main.py at the project root so it can
still be run locally with `uvicorn main:app --reload`; this file just re-exports it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import app  # noqa: E402
