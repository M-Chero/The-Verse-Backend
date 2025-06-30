from pydantic import BaseModel

class ClaimAmountDensityItem(BaseModel):
    claim_amount: float
    Gold: float
    Platinum: float
    Silver: float