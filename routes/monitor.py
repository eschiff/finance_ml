from fastapi import APIRouter
import logging
from pydantic import BaseModel

monitor_router = APIRouter()
logger = logging.getLogger()


class GenericResponse(BaseModel):
    response: str = None


GENERIC_SUCCESS = GenericResponse(response="success")


@monitor_router.get("/ping")
async def ping():
    """
    Ping returns true if health conditions of the service are met.
    """
    return GENERIC_SUCCESS


@monitor_router.get("/ready")
async def ready():
    """
    Ready returns true if health conditions of the service are met.
    """
    return GENERIC_SUCCESS
