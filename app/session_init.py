import pandas as pd
import pathlib
import streamlit as st

def init_history():
    history_path = pathlib.Path(__file__).parent.parent / "model" / "history.csv"
    if "history_df" not in st.session_state:
        if history_path.exists():
            st.session_state.history_df = pd.read_csv(history_path)
        else:
            st.session_state.history_df = pd.DataFrame()
    return history_path