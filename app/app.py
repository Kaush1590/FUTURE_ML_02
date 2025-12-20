import streamlit as st

mainpage = st.Page("mainpage.py", title = "Predict Chruning Customers", icon = "🔍")
history_view = st.Page("history.py", title = "Prediction History", icon = "🕰️")
data_view = st.Page("dataset.py", title = "Dataset Information", icon = "📋")
model_view = st.Page("model_info.py", title = "Model Information", icon = "💻")
impact_view = st.Page("business_impact.py", title = "Business Impact and Retention Strategy", icon = "📊")
pg = st.navigation([mainpage, history_view, data_view, model_view, impact_view])
pg.run()