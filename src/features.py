"""Build the single-row feature frame expected by the saved XGBoost pipeline.

Historical rates use only counts supplied by the caller. Those counts must
already exclude the purchase being scored.
"""

from __future__ import annotations

import math
from datetime import date, datetime

import pandas as pd

MODEL_COLUMNS = [
    "quantity",
    "price",
    "rrp",
    "voucherAmount",
    "order_month",
    "order_dayofweek",
    "order_weekend",
    "discount_pct",
    "price_above_rrp",
    "rrp_missing",
    "customer_previous_orders",
    "customer_previous_items",
    "customer_previous_returns",
    "customer_historical_return_rate",
    "product_previous_orders",
    "product_previous_items",
    "product_previous_returns",
    "product_historical_return_rate",
    "productGroup",
    "deviceID",
    "paymentMethod",
]


def purchase_date_features(purchase_date: date | datetime | pd.Timestamp) -> dict[str, int]:
    """Calendar fields from the purchase date. Monday is day 0."""
    timestamp = pd.Timestamp(purchase_date)
    day_of_week = int(timestamp.dayofweek)
    return {
        "order_month": int(timestamp.month),
        "order_dayofweek": day_of_week,
        "order_weekend": int(day_of_week >= 5),
    }


def rrp_missing_flag(rrp: float | None) -> int:
    """1 when recommended retail price is absent. A provided price, including 0, is not missing."""
    if rrp is None:
        return 1
    try:
        if math.isnan(float(rrp)):
            return 1
    except (TypeError, ValueError):
        return 1
    return 0


def discount_percent(price: float, rrp: float | None) -> float:
    """Percent below RRP. Missing, zero, or negative discounts become 0."""
    if rrp_missing_flag(rrp) or float(rrp) <= 0:
        return 0.0
    discount = ((float(rrp) - float(price)) / float(rrp)) * 100
    return float(max(discount, 0.0))


def price_above_rrp_flag(price: float, rrp: float | None) -> int:
    """1 when the selling price is strictly above a known RRP."""
    if rrp_missing_flag(rrp):
        return 0
    return int(float(price) > float(rrp))


def historical_return_rate(previous_returns: float, previous_items: float) -> float:
    """Return rate from prior items only. No prior items yields 0."""
    if previous_items is None or float(previous_items) <= 0:
        return 0.0
    return float(previous_returns) / float(previous_items)


def build_model_input(
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
) -> pd.DataFrame:
    """One-row frame in the trained column order.

    productGroup is stored as float so values match the pipeline categories
    (1.0, 2.0, ...). deviceID stays an int, matching categories 1 through 5.
    """
    calendar = purchase_date_features(purchase_date)
    rrp_for_price = None if rrp_is_missing else rrp
    customer_rate = historical_return_rate(
        customer_previous_returns, customer_previous_items
    )
    product_rate = historical_return_rate(
        product_previous_returns, product_previous_items
    )
    row = {
        "quantity": quantity,
        "price": price,
        "rrp": 0.0 if rrp_for_price is None else rrp_for_price,
        "voucherAmount": voucher_amount,
        "order_month": calendar["order_month"],
        "order_dayofweek": calendar["order_dayofweek"],
        "order_weekend": calendar["order_weekend"],
        "discount_pct": discount_percent(price, rrp_for_price),
        "price_above_rrp": price_above_rrp_flag(price, rrp_for_price),
        "rrp_missing": rrp_missing_flag(rrp_for_price),
        "customer_previous_orders": customer_previous_orders,
        "customer_previous_items": customer_previous_items,
        "customer_previous_returns": customer_previous_returns,
        "customer_historical_return_rate": customer_rate,
        "product_previous_orders": product_previous_orders,
        "product_previous_items": product_previous_items,
        "product_previous_returns": product_previous_returns,
        "product_historical_return_rate": product_rate,
        "productGroup": float(product_group),
        "deviceID": int(device_id),
        "paymentMethod": payment_method,
    }
    return pd.DataFrame([row], columns=MODEL_COLUMNS)
