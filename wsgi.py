"""
wsgi.py
--------
Production entry point for Gunicorn on Render.

Gunicorn command:  gunicorn wsgi:app

This file must be at the project root because Render runs commands from
the repo root directory.
"""
import sys
import os

# Ensure project root is always in Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.app import app   # noqa: F401  — gunicorn uses this 'app' object
