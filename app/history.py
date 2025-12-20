import pandas as pd
from session_init import init_history
import streamlit as st

page_title = "Prediction History"
st.set_page_config(
    page_title = page_title,
    page_icon = "🕰️",
    layout = "wide"
)
st.title(page_title)
st.sidebar.header(page_title)

history_path = init_history()
df = st.session_state.history_df
if df.empty:
    st.info("Prediction history is empty.")
    st.stop()
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Predictions", len(df))
c2.metric("Churn Predictions", (df["Prediction"] == "Churn").sum())
c3.metric("No Churn Predictions", (df["Prediction"] == "No Churn").sum())
c4.metric("Average Churn Probability", round(df["Prediction Probability"].mean(), 3))
st.divider()

st.subheader("Filters")

f1, f2, f3 = st.columns(3)
customer_filter = f1.text_input("Search Customer ID")
prediction_filter = f2.selectbox(
    label = "Prediction",
    options = ["All", "Churn", "No Churn"]
)

date_range = f3.date_input(
    label = "Date range",
    value = (df["Timestamp"].min().date(), df["Timestamp"].max().date())
)

filtered_df = df.copy()

if customer_filter:
    filtered_df = filtered_df[filtered_df["customerID"].str.contains(customer_filter, case=False, na=False)]

if prediction_filter != "All":
    filtered_df = filtered_df[filtered_df["Prediction"] == prediction_filter]

filtered_df = filtered_df[
    (filtered_df["Timestamp"].dt.date >= date_range[0]) &
    (filtered_df["Timestamp"].dt.date <= date_range[1])
]
st.caption(f"Showing {len(filtered_df)} of {len(df)} records")
st.divider()

st.subheader("Prediction Records")
st.dataframe(
    data = filtered_df.sort_values("Timestamp", ascending = False),
    width = "stretch",
)

st.subheader("Insights")

d1, d2 = st.columns(2)

with d1:
    st.caption("Prediction Distribution")
    st.bar_chart(filtered_df["Prediction"].value_counts())

with d2:
    st.caption("Churn Probability Over Time")
    st.line_chart(filtered_df.sort_values("Timestamp").set_index("Timestamp")["Prediction Probability"])

st.divider()

st.download_button(
    label="Download history (CSV)",
    data=filtered_df.to_csv(index = False).encode("utf-8"),
    file_name="prediction_history.csv",
    mime="text/csv"
)

st.divider()
st.subheader("Danger Zone")

st.warning("This action cannot be undone.")
if st.checkbox("I understand this will permanently delete all history"):
    if st.button("🗑 Clear History"):
        st.session_state.history_df = pd.DataFrame()
        history_path.unlink(missing_ok = True)
        st.success("History cleared successfully.")
        st.rerun()