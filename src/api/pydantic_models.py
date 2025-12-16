from pydantic import BaseModel
from typing import Dict, Any

class CreditScoringRequest(BaseModel):
    # We accept a dictionary of features
    # Example: {"Amount_mean": 500.0, "Recency": 10, "ChannelId_ChannelId_3": 1}
    features: Dict[str, Any]

class CreditScoringResponse(BaseModel):
    is_high_risk: int
    risk_probability: float