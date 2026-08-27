"""Pydantic request/response contracts for the real-time inference API.

Field names mirror the ISO 20022 pacs.008 (FIToFICustomerCreditTransfer)
concepts most relevant to PayShap instant credit transfers, mapped onto a
flat JSON contract suited to a low-latency REST endpoint.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProxyType(str, Enum):
    MOBILE = "MOBILE"
    ID_NUMBER = "ID_NUMBER"
    EMAIL = "EMAIL"
    ACCOUNT = "ACCOUNT"


class PayShapTransactionRequest(BaseModel):
    transaction_id: str = Field(..., description="Unique PayShap transaction identifier")
    debtor_shap_id: str = Field(..., description="Proxy ID of the paying party")
    creditor_shap_id: str = Field(..., description="Proxy ID of the receiving party")
    amount_zar: float = Field(..., gt=0, description="Transfer amount in South African Rand")
    clearing_system_property: str = Field(
        default="pacs.008", description="ISO 20022 message type used for clearing"
    )
    timestamp: datetime = Field(..., description="Transaction timestamp (ISO 8601, UTC)")
    proxy_type: ProxyType = Field(..., description="Type of proxy used to resolve the creditor")
    debtor_agent_bic: str | None = Field(default=None, description="Debtor's participant BIC")
    creditor_agent_bic: str | None = Field(default=None, description="Creditor's participant BIC")
    purpose_code: str | None = Field(default=None, description="ISO 20022 purpose code")
    charge_bearer: str | None = Field(default=None, description="ISO 20022 charge bearer code")

    @field_validator("amount_zar")
    @classmethod
    def amount_within_payshap_limit(cls, v: float) -> float:
        # PayShap's real-time low-value rail enforces a per-transaction cap;
        # keep this configurable via the settings module rather than hardcoding
        # in production. Flagged here defensively as a request-shape guard.
        if v > 5_000_000:
            raise ValueError("amount_zar exceeds plausible PayShap transaction ceiling")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transaction_id": "PS-20260827-000123456",
                "debtor_shap_id": "+27821234567",
                "creditor_shap_id": "+27839876543",
                "amount_zar": 1250.00,
                "clearing_system_property": "pacs.008",
                "timestamp": "2026-08-27T09:14:32Z",
                "proxy_type": "MOBILE",
                "debtor_agent_bic": "FIRNZAJJ",
                "creditor_agent_bic": "SBZAZAJJ",
                "purpose_code": "CASH",
                "charge_bearer": "SLEV",
            }
        }
    )


class PayShapTransactionResponse(BaseModel):
    transaction_id: str
    fraud_score: float = Field(..., ge=0, le=1, description="Predicted probability of fraud")
    is_high_risk: bool
    decision_threshold: float
    model_version: str
