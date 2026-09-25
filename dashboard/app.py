import sys
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.risk_scorer import load_pipeline, score_purchase

TRAINING_START = date(2014, 1, 1)
HOLDOUT_START = date(2015, 7, 1)
DATA_END = date(2015, 9, 30)

# Published July–September 2015 holdout results. Not recomputed at runtime.
# Rates are percent points so the table shows 17.31%, not 0.17%.
HOLDOUT_VALIDATION = pd.DataFrame(
    [
        {"Risk level": "Low", "Transactions": 62437, "Actual return rate": 17.31, "Average predicted risk": 15.75},
        {"Risk level": "Medium", "Transactions": 164793, "Actual return rate": 47.79, "Average predicted risk": 47.34},
        {"Risk level": "High", "Transactions": 141957, "Actual return rate": 69.63, "Average predicted risk": 70.21},
    ]
)

MODEL_COMPARISON = pd.DataFrame(
    [
        {"Model": "Logistic regression", "Accuracy": 0.6498, "Precision": 0.6419, "Recall": 0.7096, "F1": 0.6741, "ROC-AUC": 0.7061, "PR-AUC": 0.6861},
        {"Model": "XGBoost", "Accuracy": 0.6612, "Precision": 0.6468, "Recall": 0.7402, "F1": 0.6904, "ROC-AUC": 0.7226, "PR-AUC": 0.7067},
    ]
)

# XGBoost gain importance from the saved pipeline, as a share of total importance.
MODEL_RELIANCE = pd.DataFrame(
    [
        {"Feature": "Payment method BPRG", "Share of model reliance": 0.213576},
        {"Feature": "Quantity", "Share of model reliance": 0.097552},
        {"Feature": "Customer historical return rate", "Share of model reliance": 0.090261},
        {"Feature": "Product historical return rate", "Share of model reliance": 0.084942},
        {"Feature": "Selling price", "Share of model reliance": 0.079564},
        {"Feature": "Customer previous orders", "Share of model reliance": 0.034138},
        {"Feature": "Discount percent", "Share of model reliance": 0.026629},
        {"Feature": "Customer previous items", "Share of model reliance": 0.023879},
        {"Feature": "Recommended retail price", "Share of model reliance": 0.021883},
        {"Feature": "Product group 13", "Share of model reliance": 0.021653},
    ]
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="E-Commerce Return Risk",
    page_icon=":material/package_2:",
    layout="wide"
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return load_pipeline()


model = load_model()


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title(":material/package_2: E-commerce return risk")

st.write(
    """
    Predict the probability that an e-commerce purchase will be returned
    before the transaction is completed.

    The model uses product, pricing, customer behavior, and purchase
    characteristics to generate a return-risk score.
    """
)

st.caption(
    "Orders used for training run from 1 January 2014 through 30 June 2015. "
    "The check below uses 1 July 2015 through 30 September 2015. "
    "A purchase date outside that window is an extrapolation."
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Purchase Information")
purchase_date = st.sidebar.date_input(
    "Purchase Date",
    value=date.today()
)
product_group = st.sidebar.number_input(
    "Product Group",
    min_value=1,
    value=3
)

quantity = st.sidebar.number_input(
    "Quantity",
    min_value=1,
    value=1
)

price = st.sidebar.number_input(
    "Selling Price",
    min_value=0.0,
    value=49.99,
    step=1.0
)

rrp = st.sidebar.number_input(
    "Recommended Retail Price",
    min_value=0.0,
    value=59.99,
    step=1.0
)

voucher_amount = st.sidebar.number_input(
    "Voucher Amount",
    min_value=0.0,
    value=0.0
)

device_id = st.sidebar.number_input(
    "Device ID",
    min_value=1,
    value=2
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "BPRG",
        "KKE",
        "BPPL",
        "PAYPALVC",
        "BPLS",
        "VORAUS",
        "CBA",
        "NN",
        "KGRG",
        "RG"
    ]
)

st.sidebar.subheader("Customer History")

customer_previous_orders = st.sidebar.number_input(
    "Previous Orders",
    min_value=0,
    value=5
)

customer_previous_items = st.sidebar.number_input(
    "Previous Items",
    min_value=0,
    value=7
)

customer_previous_returns = st.sidebar.number_input(
    "Previous Returns",
    min_value=0,
    value=2
)

st.sidebar.subheader("Product History")

product_previous_orders = st.sidebar.number_input(
    "Previous Product Orders",
    min_value=0,
    value=50
)

product_previous_items = st.sidebar.number_input(
    "Previous Product Items",
    min_value=0,
    value=55
)

product_previous_returns = st.sidebar.number_input(
    "Previous Product Returns",
    min_value=0,
    value=20
)

if purchase_date < TRAINING_START or purchase_date > DATA_END:
    st.warning(
        f"The selected purchase date ({purchase_date.isoformat()}) is outside "
        "1 January 2014–30 September 2015. The score still runs, but it applies "
        "a model fit on that earlier period to a different time."
    )

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.sidebar.button(
    "Calculate Return Risk",
    type="primary",
    width="stretch",
):
    assessment = score_purchase(
        model,
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
    )

    probability = assessment.probability
    risk_score = assessment.risk_score
    risk_level = assessment.risk_level

    st.subheader("Return risk assessment")

    with st.container(horizontal=True):
        st.metric("Return probability", f"{probability:.1%}", border=True)
        st.metric("Risk score", f"{risk_score:.0f}/100", border=True)
        st.metric("Risk level", risk_level, border=True)

    st.progress(int(risk_score))

    st.subheader("Purchase summary")

    summary = pd.DataFrame({
        "Metric": [
            "Selling Price",
            "RRP",
            "Discount",
            "Customer Historical Return Rate",
            "Product Historical Return Rate"
        ],
        "Value": [
            f"${price:.2f}",
            f"${rrp:.2f}",
            f"{assessment.discount_pct:.1f}%",
            f"{assessment.customer_historical_return_rate:.1%}",
            f"{assessment.product_historical_return_rate:.1%}"
        ]
    })

    st.dataframe(summary, hide_index=True, width="stretch")

    st.subheader("Risk interpretation")

    if risk_level == "HIGH":
        st.error(
            "High return risk detected. Consider additional validation "
            "or targeted intervention before fulfillment."
        )

        st.markdown("""
        **Suggested actions**
        - Review the customer's historical return behavior.
        - Check whether this product has an unusually high historical return rate.
        - Verify product information, sizing, images, and specifications.
        - Consider proactive customer confirmation for high-value orders.
        """)

    elif risk_level == "MEDIUM":
        st.warning(
            "Moderate return risk detected. The transaction may benefit "
            "from additional product or purchase guidance."
        )

        st.markdown("""
        **Suggested actions**
        - Surface detailed product information before checkout.
        - Review customer and product return history.
        - Highlight sizing, compatibility, or specification information.
        """)

    else:
        st.success(
            "Low return risk detected based on the available transaction "
            "and historical behavior."
        )

        st.markdown("""
        **Suggested actions**
        - Proceed with the standard fulfillment workflow.
        - Continue monitoring product-level return behavior.
        """)
else:
    st.info("Enter a purchase in the sidebar, then calculate return risk.")

st.divider()

st.subheader("How the model was checked")

st.caption(
    f"Time-based holdout: train on orders before {HOLDOUT_START.isoformat()}, "
    f"score orders from {HOLDOUT_START.isoformat()} through {DATA_END.isoformat()}. "
    "XGBoost was fit on a random 500,000-row sample of the training window. "
    "Logistic regression was fit on the full training window."
)

check_col, compare_col = st.columns(2)

with check_col:
    with st.container(border=True):
        st.markdown("**Holdout risk bands**")
        st.dataframe(
            HOLDOUT_VALIDATION,
            hide_index=True,
            width="stretch",
            column_config={
                "Transactions": st.column_config.NumberColumn(format="%,d"),
                "Actual return rate": st.column_config.NumberColumn(format="%.2f%%"),
                "Average predicted risk": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )
        st.caption(
            "Low is a predicted probability under 30%. "
            "Medium is 30% up to 60%. High is 60% or more."
        )

with compare_col:
    with st.container(border=True):
        st.markdown("**Model comparison**")
        st.dataframe(
            MODEL_COMPARISON,
            hide_index=True,
            width="stretch",
            column_config={
                "Accuracy": st.column_config.NumberColumn(format="%.3f"),
                "Precision": st.column_config.NumberColumn(format="%.3f"),
                "Recall": st.column_config.NumberColumn(format="%.3f"),
                "F1": st.column_config.NumberColumn(format="%.3f"),
                "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
                "PR-AUC": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        st.caption("Higher ROC-AUC means the score ranks returned orders above kept orders more often.")

st.subheader("What the model relies on")

st.caption(
    "Each bar is that input's share of XGBoost feature importance on the saved model. "
    "This is model reliance, not a causal effect of changing the input."
)

reliance_chart = (
    alt.Chart(MODEL_RELIANCE)
    .mark_bar()
    .encode(
        x=alt.X(
            "Share of model reliance:Q",
            title="Share of model reliance",
            axis=alt.Axis(format="%"),
        ),
        y=alt.Y("Feature:N", sort="-x", title=None),
        tooltip=[
            alt.Tooltip("Feature:N"),
            alt.Tooltip("Share of model reliance:Q", format=".1%"),
        ],
    )
    .properties(height=320)
)
st.altair_chart(reliance_chart, width="stretch")
