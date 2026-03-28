"""Simple in-memory rate limiting middleware."""
from __future__ import annotations
import time
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 30, exclude_paths: list[str] | None = None):
        super().__init__(app)
        self._rpm = requests_per_minute
        self._exclude = set(exclude_paths or [])
        self._counts: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self._exclude:
            return await call_next(request)
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = [t for t in self._counts[ip] if now - t < 60]
        if len(window) >= self._rpm:
            return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)
        window.append(now)
        self._counts[ip] = window
        return await call_next(request)
