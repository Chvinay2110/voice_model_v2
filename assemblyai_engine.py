"""
assemblyai_engine.py

Provides create_temporary_token() — mints a short-lived AssemblyAI streaming
token so the browser can open a WebSocket directly to AssemblyAI's real-time
Universal-3 Pro API without the real API key ever reaching the client.

Speaker diarization is handled by AssemblyAI's built-in `speaker_labels=true`
connection parameter — no local embedding model required.

Setup:
    set ASSEMBLYAI_API_KEY=your-real-key-here      (Windows cmd)
    $env:ASSEMBLYAI_API_KEY = "your-real-key-here" (PowerShell)

Never hardcode the API key in this file or commit it to source control.
"""

import logging
import os

import requests

log = logging.getLogger("assemblyai_engine")

API_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "4aca9f9aa3d84960a45c30e06e2c43e2")
STREAMING_TOKEN_URL = "https://streaming.assemblyai.com/v3/token"


def _headers():
    if not API_KEY:
        raise RuntimeError(
            "ASSEMBLYAI_API_KEY is not set. Set it as an environment "
            "variable before running app.py."
        )
    return {"authorization": API_KEY}


def create_temporary_token(expires_in_seconds=60):
    """Mint a short-lived AssemblyAI streaming token server-side.

    https://www.assemblyai.com/docs/streaming/authenticate-with-a-temporary-token
    """
    resp = requests.get(
        STREAMING_TOKEN_URL,
        headers=_headers(),
        params={"expires_in_seconds": expires_in_seconds},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()  # {"token": "...", "expires_in_seconds": ...}