"""Export portfolio charts from the saved model and recorded metrics.

Does not retrain or rewrite the model.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "xgboost_return_risk_pipeline.joblib"
COMPARISON_PATH = ROOT / "models" / "model_comparison.csv"
ASSETS = ROOT / "assets"

# Holdout band results recorded in notebooks/03_model_training.ipynb.
RISK_BANDS = pd.DataFrame(
    {
        "Risk level": ["Low", "Medium", "High"],
        "Actual return rate": [17.31, 47.79, 69.63],
        "Average predicted risk": [15.75, 47.34, 70.21],
    }
)

NUMERIC_LABELS = {
    "quantity": "Quantity",
    "price": "Selling price",
    "rrp": "Recommended retail price",
    "voucherAmount": "Voucher amount",
    "order_month": "Order month",
    "order_dayofweek": "Order day of week",
    "order_weekend": "Weekend order",
    "discount_pct": "Discount percent",
    "price_above_rrp": "Price above RRP",
    "rrp_missing": "RRP missing",
    "customer_previous_orders": "Customer previous orders",
    "customer_previous_items": "Customer previous items",
    "customer_previous_returns": "Customer previous returns",
    "customer_historical_return_rate": "Customer historical return rate",
    "product_previous_orders": "Product previous orders",
    "product_previous_items": "Product previous items",
    "product_previous_returns": "Product previous returns",
    "product_historical_return_rate": "Product historical return rate",
}


def readable_feature(name: str) -> str:
    raw = name.removeprefix("num__").removeprefix("cat__")
    if raw in NUMERIC_LABELS:
        return NUMERIC_LABELS[raw]
    if raw.startswith("paymentMethod_"):
        return f"Payment method: {raw.split('_', 1)[1]}"
    if raw.startswith("productGroup_"):
        value = raw.split("_", 1)[1].removesuffix(".0")
        return f"Product group {value}"
    if raw.startswith("deviceID_"):
        return f"Device {raw.split('_', 1)[1]}"
    return raw


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_feature_importance() -> None:
    pipeline = joblib.load(MODEL_PATH)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    importance = pd.DataFrame(
        {
            "Feature": [readable_feature(name) for name in preprocessor.get_feature_names_out()],
            "Importance": classifier.feature_importances_,
        }
    )
    top = importance.sort_values("Importance", ascending=False).head(15)
    top = top.sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["Feature"], top["Importance"], color="#2f5d62")
    ax.set_xlabel("Share of model reliance")
    ax.set_title("Top Model Features by XGBoost Importance")
    fig.text(
        0.01,
        0.01,
        "Feature importance is model reliance, not a causal effect.",
        fontsize=9,
        color="#333333",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, ASSETS / "feature_importance.png")


def export_risk_validation() -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = range(len(RISK_BANDS))
    width = 0.36
    ax.bar(
        [p - width / 2 for p in positions],
        RISK_BANDS["Actual return rate"],
        width=width,
        label="Actual return rate",
        color="#1f4e79",
    )
    ax.bar(
        [p + width / 2 for p in positions],
        RISK_BANDS["Average predicted risk"],
        width=width,
        label="Average predicted risk",
        color="#7aa6c2",
    )
    ax.set_xticks(list(positions), RISK_BANDS["Risk level"])
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 100)
    ax.set_title("Holdout Return Rate by Risk Band")
    ax.legend(frameon=False)
    fig.text(
        0.01,
        0.01,
        "July–September 2015 time-based holdout. Predicted risk is the average return probability in each band.",
        fontsize=9,
        color="#333333",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, ASSETS / "risk_validation.png")


def export_model_comparison() -> None:
    comparison = pd.read_csv(COMPARISON_PATH)
    metrics = [column for column in comparison.columns if column != "Model"]
    models = comparison["Model"].tolist()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    positions = range(len(metrics))
    width = 0.36
    colors = ["#8d6e63", "#2f5d62"]
    for index, model_name in enumerate(models):
        values = comparison.loc[comparison["Model"] == model_name, metrics].iloc[0].astype(float)
        offset = -width / 2 if index == 0 else width / 2
        bars = ax.bar(
            [p + offset for p in positions],
            values,
            width=width,
            label=model_name,
            color=colors[index],
        )
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

    ax.set_xticks(list(positions), metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title("Holdout Metric Comparison")
    ax.legend(frameon=False)
    fig.text(
        0.01,
        0.01,
        "Metrics are the values in models/model_comparison.csv. Higher is better for every metric shown.",
        fontsize=9,
        color="#333333",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, ASSETS / "model_comparison.png")


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    export_feature_importance()
    export_risk_validation()
    export_model_comparison()
    print("Wrote feature_importance.png, risk_validation.png, model_comparison.png")


if __name__ == "__main__":
    main()
