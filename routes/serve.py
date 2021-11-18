from fastapi import APIRouter
from pydantic import BaseModel


class FinanceMLRequest(BaseModel):
    pass


class FinanceMLResponse(BaseModel):
    pass


router = APIRouter()


@router.post('/financeML', response_model=FinanceMLResponse)
async def retrieve_or_create_spec_level(request: FinanceMLRequest):
    return FinanceMLResponse()
