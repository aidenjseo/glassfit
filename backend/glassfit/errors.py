"""Domain exceptions and their HTTP mapping.

Every non-2xx response uses the envelope: {"error": {"code", "message", "details"}}.
"""

from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class GlassFitError(Exception):
    code = "GLASSFIT_ERROR"
    status_code = 500

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InvalidImage(GlassFitError):
    code = "INVALID_IMAGE"
    status_code = 400


class NoFaceDetected(GlassFitError):
    code = "NO_FACE_DETECTED"
    status_code = 422


class MultipleFaces(GlassFitError):
    code = "MULTIPLE_FACES"
    status_code = 422


class PoorScanQuality(GlassFitError):
    code = "POOR_SCAN_QUALITY"
    status_code = 422


class NotFound(GlassFitError):
    code = "NOT_FOUND"
    status_code = 404


class MediapipeUnavailable(GlassFitError):
    code = "MEDIAPIPE_UNAVAILABLE"
    status_code = 503


def _envelope(code: str, message: str, details: dict[str, Any]) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details}}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(GlassFitError)
    async def _domain_error(_request: Request, exc: GlassFitError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(exc.code, exc.message, exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = jsonable_encoder(exc.errors(), custom_encoder={Exception: str})
        return JSONResponse(
            status_code=422,
            content=_envelope("VALIDATION_ERROR", "Request validation failed", {"errors": errors}),
        )
