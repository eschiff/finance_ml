import logging
from typing import Any

from ddtrace import helpers
from fastapi import HTTPException
from starlette import status
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger()


class ServiceHTTPException(HTTPException):
    def __init__(
            self,
            status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail: str = None,
            exception: Any = None,  # use to capture actual error/stack trace in logs if you wish
    ):
        super(ServiceHTTPException, self).__init__(
            status_code=status_code, detail=detail,
        )
        trace_id, span_id = helpers.get_correlation_ids()
        self.headers = {"trace_id": str(trace_id)}
        self.exception = exception

    def json(self):
        return JSONResponse(
            status_code=self.status_code,
            content={"message": self.detail},
            headers=self.headers,
        )

    def __str__(self):
        return "Status: {}, Message: {}".format(self.status_code, self.detail)


class InternalServerHTTPException(ServiceHTTPException):
    def __init__(
            self, detail="Internal server error", exception="",
    ):
        super(InternalServerHTTPException, self).__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            exception=exception,
        )


class ServiceUnavailableHTTPException(ServiceHTTPException):
    def __init__(
            self, detail="Service unavailable", exception="",
    ):
        super(ServiceUnavailableHTTPException, self).__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            exception=exception,
        )


def handle_service_http_exception(
        request: Request, exception: ServiceHTTPException
):
    """
    FastAPI override for service defined HTTP exceptions. By default log errors on 500s
    but log other exceptions as info. Each log will include a stack trace if provided.
    """
    if exception.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR:
        logger.error(
            "throwing service exception: {}".format(exception),
            exc_info=exception.exception,
        )
    else:
        logger.info("{}".format(exception.exception or exception))

    return exception.json()
