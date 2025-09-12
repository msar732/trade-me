# Trade Hub

A consolidated, production-grade marketplace web app built with Flask. All features from the previous app are preserved and rebranded to Trade Hub.

## Quickstart

1) Install Python 3.11+ and system dependencies (ensure python3-venv if using venv).
2) Optional: python3 -m venv .venv && source .venv/bin/activate
3) pip install -r requirements.txt
4) export FLASK_APP=app.py
5) python app.py (or) flask run

The app uses SQLite by default (tradehub.db). Redis and MongoDB are optional; configure via env vars if available.
