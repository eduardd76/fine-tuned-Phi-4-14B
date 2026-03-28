"""
Prometheus metrics exporter for Phi-4 Network Architect API.

Exposes a /metrics endpoint compatible with Prometheus scraping.
Run standalone (default port 9100) or import and call register_metrics()
to attach to an existing FastAPI app.

Usage (standalone):
    python monitoring/prometheus_exporter.py --port 9100

Usage (integrated with FastAPI):
    from monitoring.prometheus_exporter import register_metrics
    register_metrics(app)
    # /metrics is now available on the main app
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Callable

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
        CONTENT_TYPE_LATEST,
        start_http_server,
        CollectorRegistry,
        REGISTRY,
    )
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

try:
    import fastapi
    from fastapi import FastAPI, Response
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if HAS_PROMETHEUS:
    REQUEST_COUNT = Counter(
        "phi4_api_requests_total",
        "Total number of API requests",
        ["method", "endpoint", "status_code"],
    )

    REQUEST_LATENCY = Histogram(
        "phi4_api_request_duration_seconds",
        "API request duration in seconds",
        ["endpoint"],
        buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0],
    )

    INFERENCE_LATENCY = Histogram(
        "phi4_inference_duration_seconds",
        "Model inference duration in seconds",
        ["task_type"],
        buckets=[1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0],
    )

    CONFIDENCE_SCORE = Histogram(
        "phi4_confidence_score",
        "Model confidence score distribution",
        ["task_type"],
        buckets=[0.5, 0.6, 0.7, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0],
    )

    THINK_BLOCK_RATE = Counter(
        "phi4_think_block_total",
        "Number of responses that included a <think> block",
        ["has_think"],
    )

    HUMAN_APPROVAL_REQUESTS = Counter(
        "phi4_human_approval_requests_total",
        "Number of responses that required human approval",
    )

    MODEL_INFO = Info("phi4_model", "Model metadata")

    ACTIVE_REQUESTS = Gauge(
        "phi4_api_active_requests",
        "Currently processing requests",
    )

    CACHE_HITS = Counter(
        "phi4_response_cache_hits_total",
        "Response cache hits (idempotency)",
    )

    CACHE_MISSES = Counter(
        "phi4_response_cache_misses_total",
        "Response cache misses",
    )


# ---------------------------------------------------------------------------
# Public helpers (called from inference / API code)
# ---------------------------------------------------------------------------

def record_inference(
    task_type: str,
    latency_ms: float,
    confidence: float,
    has_think: bool,
    requires_human: bool,
    cache_hit: bool = False,
) -> None:
    """Record metrics for one completed inference. Call from api/routes/*.py."""
    if not HAS_PROMETHEUS:
        return
    INFERENCE_LATENCY.labels(task_type=task_type).observe(latency_ms / 1000)
    CONFIDENCE_SCORE.labels(task_type=task_type).observe(confidence)
    THINK_BLOCK_RATE.labels(has_think=str(has_think)).inc()
    if requires_human:
        HUMAN_APPROVAL_REQUESTS.inc()
    if cache_hit:
        CACHE_HITS.inc()
    else:
        CACHE_MISSES.inc()


def set_model_info(model_path: str, backend: str, version: str = "1.0.0") -> None:
    """Call once at startup with model metadata."""
    if HAS_PROMETHEUS:
        MODEL_INFO.info({"path": model_path, "backend": backend, "version": version})


# ---------------------------------------------------------------------------
# FastAPI middleware
# ---------------------------------------------------------------------------

if HAS_FASTAPI and HAS_PROMETHEUS:
    class PrometheusMiddleware(BaseHTTPMiddleware):
        """Tracks request count, latency and active requests."""

        async def dispatch(self, request: Request, call_next: Callable):
            endpoint = request.url.path
            ACTIVE_REQUESTS.inc()
            start = time.perf_counter()
            try:
                response = await call_next(request)
                status = str(response.status_code)
            except Exception:
                status = "500"
                raise
            finally:
                duration = time.perf_counter() - start
                ACTIVE_REQUESTS.dec()
                REQUEST_COUNT.labels(
                    method=request.method,
                    endpoint=endpoint,
                    status_code=status,
                ).inc()
                REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
            return response


def register_metrics(app: "FastAPI") -> None:
    """Attach Prometheus middleware and /metrics endpoint to a FastAPI app."""
    if not HAS_PROMETHEUS:
        print("WARNING: prometheus_client not installed — metrics disabled")
        return
    if not HAS_FASTAPI:
        raise RuntimeError("fastapi is required to use register_metrics()")

    app.add_middleware(PrometheusMiddleware)

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        return Response(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST,
        )


# ---------------------------------------------------------------------------
# Standalone exporter
# ---------------------------------------------------------------------------

def run_standalone(port: int = 9100) -> None:
    """Start a standalone Prometheus HTTP server (no FastAPI required)."""
    if not HAS_PROMETHEUS:
        raise ImportError("Install prometheus_client: pip install prometheus-client")

    print(f"Prometheus exporter listening on :{port}/metrics")
    set_model_info(
        model_path="standalone-mode",
        backend="unknown",
    )
    start_http_server(port)

    # Keep alive
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("Exporter stopped.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phi-4 Prometheus metrics exporter")
    parser.add_argument("--port", type=int, default=9100, help="Port to expose /metrics on")
    args = parser.parse_args()
    run_standalone(args.port)
