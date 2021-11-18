from fastapi import APIRouter

router = APIRouter()

"""
Ping endpoint for livenessProbe and readinessProbe in kubernetes yml files.
"""


@router.get("/ping")
async def ping():
    return {"response": "success"}
