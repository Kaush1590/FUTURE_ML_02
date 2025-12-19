import joblib
import pandas as pd
import pathlib
import pickle
import streamlit as st

@st.cache_data
def load_state(path):
    with open(path, "rb") as f:
        return pickle.load(f)

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

try:
    encoder_path = pathlib.Path(__file__).parent.parent / "model" / "encoder_dump.pkl"
    model_path = pathlib.Path(__file__).parent.parent / "model" / "model_dump.joblib"
    state_path = pathlib.Path(__file__).parent.parent / "model" / "state_dump.pkl"
    encoders = load_state(encoder_path)
    state_holder = load_state(state_path)
    model = joblib.load(model_path)
    df = state_holder["dataframe"]
    features = state_holder["feature_list"].drop("Churn", errors = "ignore")
except FileNotFoundError as e:
    st.error(f"Required files not found in the project. Please generate the files from notebook before proceeding.")
    st.stop()
except Exception as e:
    st.error(f"Errors encountered while starting project: {e}")
    st.stop()

page_title = "Predict Churning Customers"
st.set_page_config(
    page_title = page_title,
    page_icon = "🔍",
    layout = "wide"
)
st.sidebar.header(page_title)
if "predict" not in st.session_state:
    st.session_state.predict = "Unknown"
if "predict_proba" not in st.session_state:
    st.session_state.predict_proba = 0.00
st.title("Predict Churning Customers")

tab1, tab2 = st.tabs(["Single input", "Bulk input"])

with tab1:
    st.header("Single Input Prediction")

    for i in features:
        if df[i].dtype == "object" and i in encoders:
            option_field(i, encoders[i].classes_, i)
        elif df[i].dtype == "int64":
            number_field_int (i, "0", i)
        elif df[i].dtype == "float64":
            number_field (i, "0.0", i)
        else:
            text_field(i, i, i)

    st.slider(
        label = "Threshold",
        min_value = 0.00,
        max_value = 0.98,
        value = 0.5,
        key = "threshold"
    )

    if st.button("Start Prediction"):
        feature_list = []
        feature_columns = features.drop("customerID", errors = "ignore")

        for col in feature_columns:
            value = st.session_state[col]
            if col in encoders:
                value = encoders[col].transform([value])[0]
            feature_list.append(value)

        feature_vector = pd.DataFrame(
            [feature_list],
            columns = feature_columns
        )

        predict_proba = model.predict_proba(feature_vector)
        churn_prob = predict_proba[0][1]
        if churn_prob >= st.session_state.threshold:
            st.session_state.predict = "Churn"
            st.session_state.predict_proba = churn_prob
        else:
            st.session_state.predict = "No Churn"
            st.session_state.predict_proba = 1 - churn_prob
            
    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Customer prediction", st.session_state.predict)
    c2.metric("Prediction Probablility", round(st.session_state.predict_proba, 3))

with tab2:
    st.header("Bulk Input Prediction")

    st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        key = "bulk_input"
    )

    st.slider(
        label="Threshold",
        min_value=0.01,
        max_value=0.98,
        value=0.5,
        key="bulk_threshold"
    )

    if st.session_state.bulk_input is not None:
        bulk_df = pd.read_csv(st.session_state.bulk_input)
        st.subheader("Preview of uploaded data")
        st.dataframe(bulk_df)
        feature_columns = features.drop("customerID", errors="ignore")
        missing_cols = set(feature_columns) - set(bulk_df.columns)
        if missing_cols:
            st.error(f"Missing columns in CSV: {missing_cols}")
            st.stop()
        encoded = bulk_df.copy()
        for col in feature_columns:
            if col in encoders:
                try:
                    encoded[col] = encoders[col].transform(encoded[col])
                except ValueError as e:
                    st.error(f"Encoding error in column '{col}'. CSV contains unseen categories.")
                    st.stop()

        if st.button("Run Bulk Prediction"):
            proba = model.predict_proba(encoded[feature_columns])
            churn_prob = proba[:, 1]
            bulk_df["Churn_Probability"] = churn_prob
            bulk_df["Prediction"] = pd.Series(
                churn_prob >= st.session_state.bulk_threshold            
            ).map({True: "Churn", False: "No Churn"})
            st.success("Prediction completed successfully")
            st.dataframe(bulk_df)
            csv = bulk_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Predictions CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
