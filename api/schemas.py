"""
Pydantic schemas for request validation and response formatting.
These define the exact shape of data coming in and going out of the API.
"""

from pydantic import BaseModel, Field
from typing import List, Literal


# ─────────────────────────────────────────────
# INPUT SCHEMA  (what the client sends us)
# ─────────────────────────────────────────────

class ShopperSession(BaseModel):
    """
    Represents one online shopping session.
    All 17 original dataset features are required.
    """

    # Numerical features
    Administrative: int = Field(
        default=0, ge=0,
        description="Number of administrative pages visited"
    )
    Administrative_Duration: float = Field(
        default=0.0, ge=0.0,
        description="Total time (seconds) spent on administrative pages"
    )
    Informational: int = Field(
        default=0, ge=0,
        description="Number of informational pages visited"
    )
    Informational_Duration: float = Field(
        default=0.0, ge=0.0,
        description="Total time (seconds) spent on informational pages"
    )
    ProductRelated: int = Field(
        default=0, ge=0,
        description="Number of product-related pages visited"
    )
    ProductRelated_Duration: float = Field(
        default=0.0, ge=0.0,
        description="Total time (seconds) spent on product-related pages"
    )
    BounceRates: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Average bounce rate of pages visited (0–1)"
    )
    ExitRates: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Average exit rate of pages visited (0–1)"
    )
    PageValues: float = Field(
        default=0.0, ge=0.0,
        description="Average value of pages visited before completing a transaction"
    )
    SpecialDay: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Closeness of visit to a special day e.g. Valentine's Day (0–1)"
    )

    # Categorical features
    Month: Literal[
        "Jan", "Feb", "Mar", "Apr", "May", "June",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ] = Field(default="May", description="Month of the visit (Aug is the baseline/reference category)")

    OperatingSystems: int = Field(
        default=2, ge=1,
        description="Operating system used (integer code)"
    )
    Browser: int = Field(
        default=2, ge=1,
        description="Browser used (integer code)"
    )
    Region: int = Field(
        default=1, ge=1,
        description="Geographic region of the visitor (integer code)"
    )
    TrafficType: int = Field(
        default=2, ge=1,
        description="Traffic source type (integer code)"
    )
    VisitorType: Literal["Returning_Visitor", "New_Visitor", "Other"] = Field(
        default="Returning_Visitor",
        description="Type of visitor"
    )
    Weekend: bool = Field(
        default=False,
        description="Whether the visit occurred on a weekend"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Administrative": 0,
                    "Administrative_Duration": 0.0,
                    "Informational": 0,
                    "Informational_Duration": 0.0,
                    "ProductRelated": 35,
                    "ProductRelated_Duration": 2500.0,
                    "BounceRates": 0.01,
                    "ExitRates": 0.03,
                    "PageValues": 25.4,
                    "SpecialDay": 0.0,
                    "Month": "Nov",
                    "OperatingSystems": 2,
                    "Browser": 2,
                    "Region": 1,
                    "TrafficType": 2,
                    "VisitorType": "Returning_Visitor",
                    "Weekend": False
                }
            ]
        }
    }


# ─────────────────────────────────────────────
# OUTPUT SCHEMAS  (what we send back)
# ─────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Single prediction result."""
    prediction: int = Field(description="0 = No Purchase, 1 = Purchase")
    label: str = Field(description="Human-readable label")
    purchase_probability: float = Field(description="Probability of purchase (0–1)")
    no_purchase_probability: float = Field(description="Probability of no purchase (0–1)")


class BatchPredictionResponse(BaseModel):
    """Response for a batch of predictions."""
    total: int = Field(description="Number of sessions submitted")
    predictions: List[PredictionResponse]


class ModelInfoResponse(BaseModel):
    """Information about the deployed model."""
    model_name: str
    description: str
    week2_metrics: dict
    dataset: dict
    how_to_use: str
