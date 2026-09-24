"""Versioned HTTP JSON API for tool-radius compensation auditing."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .audit.pipeline import AuditFailure, run_audit
from .models import AuditRequest


def _clean_msg(msg: str) -> str:
    # Pydantic v2 prefixes ValueError text with "Value error, ".
    return msg.split("Value error, ", 1)[-1]


def _validation_body(exc) -> dict:
    try:
        errors = exc.errors()
    except Exception:  # pragma: no cover - defensive
        errors = [{"msg": str(exc), "loc": (), "type": "value_error"}]
    simplified = [
        {
            "location": [str(part) for part in err.get("loc", [])],
            "code": err.get("type", "value_error"),
            "message": _clean_msg(err.get("msg", "")),
        }
        for err in errors
    ]
    return {
        "status": "rejected",
        "error": {
            "code": "invalid_request",
            "message": "request failed exact-input validation",
            "issues": simplified,
        },
    }


def create_app() -> FastAPI:
    app = FastAPI(
        title="CAM Cutter Compensation Audit",
        version="1.0.0",
        description=(
            "Exact (floating-point-free) audit of closed line/arc tool paths "
            "after left/right cutter radius compensation."
        ),
        # The service is backend-only: no browser/HTML frontend is served.
        docs_url=None,
        redoc_url=None,
    )

    @app.get("/health")
    @app.get("/v1/health")
    def health():
        return {
            "status": "ok",
            "service": "cam-comp-audit",
            "api_versions": ["v1"],
        }

    @app.get("/v1")
    def index_v1():
        return {
            "api": "v1",
            "endpoint": "/v1/audit",
            "method": "POST",
            "health": "/v1/health",
        }

    @app.post("/v1/audit")
    async def audit_v1(request: Request):
        try:
            raw = await request.json()
        except Exception:
            return JSONResponse(
                {
                    "status": "rejected",
                    "error": {
                        "code": "invalid_request",
                        "message": "request body must be a JSON object",
                        "issues": [],
                    },
                },
                status_code=422,
            )
        if not isinstance(raw, dict):
            return JSONResponse(
                {
                    "status": "rejected",
                    "error": {
                        "code": "invalid_request",
                        "message": "request body must be a JSON object",
                        "issues": [],
                    },
                },
                status_code=422,
            )
        payload = AuditRequest.model_validate(raw)
        result = run_audit(payload)
        return JSONResponse(result, status_code=200)

    @app.exception_handler(AuditFailure)
    def audit_failure_handler(request: Request, exc: AuditFailure):
        return JSONResponse(exc.payload, status_code=exc.http)

    @app.exception_handler(RequestValidationError)
    def request_validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(_validation_body(exc), status_code=422)

    @app.exception_handler(ValidationError)
    def validation_handler(request: Request, exc: ValidationError):
        return JSONResponse(_validation_body(exc), status_code=422)

    @app.exception_handler(Exception)
    def unhandled_handler(request: Request, exc: Exception):
        return JSONResponse(
            {
                "status": "error",
                "error": {
                    "code": "internal_error",
                    "message": "unexpected internal error while auditing",
                },
            },
            status_code=500,
        )

    return app


app = create_app()
