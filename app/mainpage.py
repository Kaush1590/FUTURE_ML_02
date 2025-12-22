import joblib
import pandas as pd
import pathlib
import pickle
from session_init import init_history
import streamlit as st

# For loading application state
@st.cache_data
def load_state(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# Input field helpers
def text_field(label, placeholder, key):
    st.text_input(
        label = label,
        label_visibility = "visible",
        placeholder = placeholder,
        key = key
    )

def number_field_int (label, placeholder, key):
    st.number_input(
        label = label,
        placeholder = placeholder,
        step = 1,
        key = key
    )

def number_field (label, placeholder, key):
    st.number_input(
        label = label,
        placeholder = placeholder,
        key = key
    )

def option_field (label, options, key):
    st.radio(
        label = label,
        options = options,
        key = key,
        horizontal = True
    )

# For saving history to disk
def save_history(history_path, df):
    df.to_csv(history_path, index = False)

# Load application states
try:
    encoder_path = pathlib.Path(__file__).parent.parent / "model" / "encoder_dump.pkl"
    model_path = pathlib.Path(__file__).parent.parent / "model" / "model_dump.joblib"
    state_path = pathlib.Path(__file__).parent.parent / "model" / "state_dump.pkl"

    # Initialize prediction history
    history_path = init_history()

    # Load trained model and encoders
    encoders = load_state(encoder_path)
    state_holder = load_state(state_path)
    model = joblib.load(model_path)

    # Dataset and feature metadata
    df = state_holder["dataframe"]
    features = state_holder["feature_list"].drop("Churn", errors = "ignore")

    # Initialize prediction history in session state
    if "history_df" not in st.session_state:
        if history_path.exists():
            st.session_state.history_df = pd.read_csv(history_path)
        else:
            st.session_state.history_df = pd.DataFrame()
            
except FileNotFoundError as e:
    st.error(f"Required files not found in the project. Please generate the files by running training notebook before proceeding.")
    st.stop()
except Exception as e:
    st.error(f"Errors encountered while starting project: {e}")
    st.stop()

# Page configuration
page_title = "Predict Churning Customers"
st.set_page_config(
    page_title = page_title,
    page_icon = "🔍",
    layout = "wide"
)
st.sidebar.header(page_title)

# Initialize prediction state
if "predict" not in st.session_state:
    st.session_state.predict = "Unknown"
if "predict_proba" not in st.session_state:
    st.session_state.predict_proba = 0.00
st.title("Predict Churning Customers")

# Prediction modes
tab1, tab2 = st.tabs(["Single input", "Bulk input"])

# Single input prediction
with tab1:
    st.header("Single Input Prediction")

    # Dynamic render input fields based on feature datatypes
    for i in features:
        if df[i].dtype == "object" and i in encoders:
            option_field(i, encoders[i].classes_, i)
        elif df[i].dtype == "int64":
            number_field_int (i, "0", i)
        elif df[i].dtype == "float64":
            number_field (i, "0.0", i)
        else:
            text_field(i, i, i)

    # Decision threshold selection
    st.slider(
        label = "Threshold",
        min_value = 0.00,
        max_value = 1.00,
        value = 0.5,
        key = "threshold"
    )

    # Run prediction on user input
    if st.button("Start Prediction"):
        feature_list = []
        feature_columns = features.drop("customerID", errors = "ignore")

        # Apply encoding to categorical inputs
        for col in feature_columns:
            value = st.session_state[col]
            if col in encoders:
                value = encoders[col].transform([value])[0]
            feature_list.append(value)

        feature_vector = pd.DataFrame(
            [feature_list],
            columns = feature_columns
        )

        # Predict churn probability based on selected threshold
        predict_proba = model.predict_proba(feature_vector)
        churn_prob = predict_proba[0][1]
        if churn_prob >= st.session_state.threshold:
            st.session_state.predict = "Churn"
            st.session_state.predict_proba = churn_prob
        else:
            st.session_state.predict = "No Churn"
            st.session_state.predict_proba = 1 - churn_prob

        # Append prediction to history
        history_row = pd.DataFrame([{
            "customerID": st.session_state.customerID,
            **{col: st.session_state[col] for col in feature_columns},
            "Prediction": st.session_state.predict,
            "Prediction Probability": round(st.session_state.predict_proba, 3),
            "Threshold": st.session_state.threshold,
            "Timestamp": pd.Timestamp.now()
        }])
        st.session_state.history_df = pd.concat(
            [st.session_state.history_df, history_row],
            ignore_index = True
        )
        st.session_state.history_df.drop_duplicates(
            subset = ["customerID", "Timestamp"],
            inplace = True
        )
        save_history(history_path = history_path, df = st.session_state.history_df)
            
    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Customer prediction", st.session_state.predict)
    c2.metric("Prediction Probablility", round(st.session_state.predict_proba, 3))

# Bulk input prediction
with tab2:
    st.header("Bulk Input Prediction")

    st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        key = "bulk_input"
    )

    st.slider(
        label = "Threshold",
        min_value = 0.00,
        max_value = 1.00,
        value = 0.5,
        key = "bulk_threshold"
    )

    if st.session_state.bulk_input is not None:
        bulk_df = pd.read_csv(st.session_state.bulk_input)
        st.subheader("Preview of uploaded data")
        st.dataframe(bulk_df)
        feature_columns = features.drop("customerID", errors="ignore")

        # Validate required columns
        missing_cols = set(feature_columns) - set(bulk_df.columns)
        if missing_cols:
            st.error(f"Missing columns in CSV: {missing_cols}")
            st.stop()
        encoded = bulk_df.copy()

        # Apply encoders to categorical columns
        for col in feature_columns:
            if col in encoders:
                try:
                    encoded[col] = encoders[col].transform(encoded[col])
                except ValueError as e:
                    st.error(f"Encoding error in column '{col}'. CSV contains unseen categories.")
                    st.stop()

        # Run prediction on user input
        if st.button("Run Bulk Prediction"):
            # Predict churn probability based on selected threshold
            proba = model.predict_proba(encoded[feature_columns])
            churn_prob = proba[:, 1]
            bulk_df["Churn_Probability"] = churn_prob
            bulk_df["Prediction"] = pd.Series(
                churn_prob >= st.session_state.bulk_threshold            
            ).map({True: "Churn", False: "No Churn"})
            st.success("Prediction completed successfully")
            st.dataframe(bulk_df)
            csv = bulk_df.to_csv(index = False).encode("utf-8")

            st.download_button(
                label = "Download Predictions CSV",
                data = csv,
                file_name = "predictions.csv",
                mime = "text/csv"
            )