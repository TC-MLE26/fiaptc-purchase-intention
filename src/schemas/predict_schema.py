"""Request and response contracts for the purchase intention endpoint.

Request fields use snake_case; `FEATURE_COLUMNS` maps each one to the original
dataset column expected by the registered model. Domains follow the Online Shoppers
dataset (UCI).
"""

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

FEATURE_COLUMNS: dict[str, str] = {
    "administrative": "Administrative",
    "administrative_duration": "Administrative_Duration",
    "informational": "Informational",
    "informational_duration": "Informational_Duration",
    "product_related": "ProductRelated",
    "product_related_duration": "ProductRelated_Duration",
    "bounce_rates": "BounceRates",
    "exit_rates": "ExitRates",
    "page_values": "PageValues",
    "special_day": "SpecialDay",
    "month": "Month",
    "operating_systems": "OperatingSystems",
    "browser": "Browser",
    "region": "Region",
    "traffic_type": "TrafficType",
    "visitor_type": "VisitorType",
    "weekend": "Weekend",
}

_EXAMPLE_REQUEST = {
    "administrative": 9,
    "administrative_duration": 111.5,
    "informational": 1,
    "informational_duration": 48.5,
    "product_related": 49,
    "product_related_duration": 1868.819697,
    "bounce_rates": 0.0,
    "exit_rates": 0.020708874,
    "page_values": 1.706014966,
    "special_day": 0.0,
    "month": "Mar",
    "operating_systems": 2,
    "browser": 2,
    "region": 7,
    "traffic_type": 2,
    "visitor_type": "Returning_Visitor",
    "weekend": False,
}


class PredictRequest(BaseModel):
    """Features of one browsing session, as in the original dataset."""

    administrative: int = Field(ge=0, description="Páginas administrativas visitadas.")
    administrative_duration: float = Field(
        ge=0, description="Tempo (s) em páginas administrativas."
    )
    informational: int = Field(ge=0, description="Páginas informativas visitadas.")
    informational_duration: float = Field(
        ge=0, description="Tempo (s) em páginas informativas."
    )
    product_related: int = Field(ge=0, description="Páginas de produto visitadas.")
    product_related_duration: float = Field(
        ge=0, description="Tempo (s) em páginas de produto."
    )
    bounce_rates: float = Field(ge=0, le=1, description="Taxa média de rejeição.")
    exit_rates: float = Field(ge=0, le=1, description="Taxa média de saída.")
    page_values: float = Field(ge=0, description="Valor médio das páginas visitadas.")
    special_day: float = Field(
        ge=0, le=1, description="Proximidade de uma data especial (0 a 1)."
    )
    month: Literal[
        "Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]
    operating_systems: int = Field(ge=1, le=8)
    browser: int = Field(ge=1, le=13)
    region: int = Field(ge=1, le=9)
    traffic_type: int = Field(ge=1, le=20)
    visitor_type: Literal["New_Visitor", "Returning_Visitor", "Other"]
    weekend: bool = Field(description="Sessão realizada em fim de semana.")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": _EXAMPLE_REQUEST},
    )

    def to_frame(self) -> pd.DataFrame:
        """Build the one-row DataFrame with the original dataset columns."""
        values = self.model_dump()
        row = {column: values[field] for field, column in FEATURE_COLUMNS.items()}
        return pd.DataFrame([row])


class PredictResponse(BaseModel):
    """Response contract for a purchase intention prediction."""

    purchase_probability: float = Field(
        ge=0, le=1, description="Probabilidade prevista de a sessão gerar compra."
    )
    purchase_prediction: bool = Field(
        description="Indica se a probabilidade atingiu o threshold de decisão."
    )
    threshold: float = Field(
        ge=0, le=1, description="Threshold de decisão aplicado à probabilidade."
    )
    model_name: str = Field(description="Nome do modelo que gerou a predição.")
    model_version: str = Field(description="Versão do modelo no Model Registry.")
