import streamlit as st

mainpage = st.Page("mainpage.py", title = "Predict Chruning Customers", icon = "🔍")
history_view = st.Page("history.py", title = "Previous History", icon = "🕰️")
data_view = st.Page("dataset.py", title = "Dataset Information", icon = "📋")
model_view = st.Page("model_info.py", title = "Model Information", icon = "💻")
pg = st.navigation([mainpage, history_view, data_view, model_view])
pg.run()