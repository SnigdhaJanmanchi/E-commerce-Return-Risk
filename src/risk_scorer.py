"""Score one purchase with the saved XGBoost pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.features import build_model_input

LOW_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.60

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "xgboost_return_risk_pipeline.joblib"
)


@dataclass(frozen=True)
class RiskAssessment:
    probability: float
    risk_score: float
    risk_level: str
    discount_pct: float
    customer_historical_return_rate: float
    product_historical_return_rate: float


def load_pipeline(model_path: Path | None = None) -> Pipeline:
    """Load the saved preprocessing and classifier pipeline. Does not write the file."""
    path = Path(model_path) if model_path is not None else MODEL_PATH
    return joblib.load(path)


def risk_score_from_probability(probability: float) -> float:
    """Map a 0–1 probability onto a 0–100 score. Same scale as probability * 100."""
    return float(probability) * 100


def assign_risk_level(probability: float) -> str:
    """Low below 0.30, medium below 0.60, high at 0.60 and above."""
    if probability < LOW_RISK_THRESHOLD:
        return "LOW"
    if probability < HIGH_RISK_THRESHOLD:
        return "MEDIUM"
    return "HIGH"


def predict_return_probability(pipeline: Pipeline, model_input: pd.DataFrame) -> float:
    """Positive-class probability as a native Python float."""
    return float(pipeline.predict_proba(model_input)[0][1])


def score_purchase(
    pipeline: Pipeline,
    *,
    purchase_date: date | datetime | pd.Timestamp,
    quantity: float,
    price: float,
    rrp: float | None,
    voucher_amount: float,
    product_group: float,
    device_id: float,
    payment_method: str,
    customer_previous_orders: float,
    customer_previous_items: float,
    customer_previous_returns: float,
    product_previous_orders: float,
    product_previous_items: float,
    product_previous_returns: float,
    rrp_is_missing: bool = False,
) -> RiskAssessment:
    """Build the model row from known pre-purchase inputs and score it."""
    model_input = build_model_input(
        purchase_date=purchase_date,
        quantity=quantity,
        price=price,
        rrp=rrp,
        voucher_amount=voucher_amount,
        product_group=product_group,
        device_id=device_id,
        payment_method=payment_method,
        customer_previous_orders=customer_previous_orders,
        customer_previous_items=customer_previous_items,
        customer_previous_returns=customer_previous_returns,
        product_previous_orders=product_previous_orders,
        product_previous_items=product_previous_items,
        product_previous_returns=product_previous_returns,
        rrp_is_missing=rrp_is_missing,
    )
    probability = predict_return_probability(pipeline, model_input)
    return RiskAssessment(
        probability=probability,
        risk_score=risk_score_from_probability(probability),
        risk_level=assign_risk_level(probability),
        discount_pct=float(model_input.loc[0, "discount_pct"]),
        customer_historical_return_rate=float(
            model_input.loc[0, "customer_historical_return_rate"]
        ),
        product_historical_return_rate=float(
            model_input.loc[0, "product_historical_return_rate"]
        ),
    )
