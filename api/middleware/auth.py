"""API key authentication middleware."""
from __future__ import annotations
import os
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_key_env: str = "PHI4_API_KEY", exclude_paths: list[str] | None = None):
        super().__init__(app)
        self._key = os.getenv(api_key_env, "phi4-key-change-me")
        self._exclude = set(exclude_paths or [])

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self._exclude:
            return await call_next(request)
        key = request.headers.get("X-API-Key", "")
        if key != self._key:
            return JSONResponse({"detail": "Invalid or missing API key"}, status_code=401)
        return await call_next(request)
