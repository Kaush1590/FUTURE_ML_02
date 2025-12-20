import pathlib
import pickle
import plotly.express as px
import streamlit as st

@st.cache_data
def load_state(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
try:
    state_path = pathlib.Path(__file__).parent.parent / "model" / "state_dump.pkl"
    state_holder = load_state(state_path)
    impact_df = state_holder["impact_df"]
    impact_summary = state_holder["impact_summary"]
except FileNotFoundError as e:
    st.error(f"Required files not found in the project. Please generate the files from notebook before proceeding.")
    st.stop()
except Exception as e:
    st.error(f"Errors encountered while starting project: {e}")
    st.stop()

high_risk_customers = impact_df[impact_df["Risk Segment"] == "High Risk"]
total_revenue_risk = impact_df["Revenue at Risk"].sum()

page_title = "Business Impact and Retention Analysis with Simulation"
st.set_page_config(
    page_title = page_title,
    page_icon = "📊",
    layout = "wide"
)
st.sidebar.header(page_title)
st.title("Business Impact and Retention Analysis")

col1, col2, col3 = st.columns(3)
col1.metric("High-Risk Customers", f"{len(high_risk_customers):,}")
col2.metric("Total Monthly Revenue at Risk", f"₹{total_revenue_risk:,.0f}")
col3.metric("Avg High-Risk Churn Probability", f"{high_risk_customers['Churn Probability'].mean():.2f}")

st.subheader("Customer Distribution by Churn Risk")

risk_counts = impact_df["Risk Segment"].value_counts().reset_index()
risk_counts.columns = ["Risk Segment", "Customer Count"]

fig = px.bar(
    data_frame = risk_counts,
    x="Risk Segment",
    y="Customer Count",
    color="Risk Segment",
    text="Customer Count",
    title="Churn Risk Segmentation"
)

st.plotly_chart(
    figure_or_data = fig,
    width = "stretch"
)

st.subheader("Retention Campaign Simulation")

col1, col2 = st.columns(2)
col1.slider(
    label = "Target top % of highest-risk customers",
    min_value=5,
    max_value=50,
    value=10,
    step=5,
    key = "target_pct"
)
col2.slider(
    label = "Retention success rate (%)",
    min_value = 10,
    max_value = 60,
    value = 30,
    step = 5,
    key = "retention_rate"
)
top_n = int((st.session_state.target_pct / 100) * len(impact_df))
target_customers = impact_df.head(top_n)
estimated_saved_revenue = (target_customers["Revenue at Risk"].sum() * (st.session_state.retention_rate/100))
st.metric("Estimated Monthly Revenue Recovered", f"₹{estimated_saved_revenue:,.0f}")

st.subheader("Risk Segment Summary")

st.dataframe(
    data = impact_summary,
    width = "stretch"
)
st.caption(f"Revenue estimates are based on MonthlyCharges as a proxy and assume a {int(st.session_state.retention_rate)}% retention (campaign success) rate among targeted customers.")