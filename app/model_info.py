import io
import pandas as pd
import pathlib
import pickle
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

@st.cache_data
def load_state(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def download_button(fig, filename, key):
    buffer = io.StringIO()
    fig.write_html(buffer)
    st.download_button(
        label = "Download interactive graph (HTML)",
        data = buffer.getvalue(),
        file_name = filename,
        mime = "text/html",
        key = key
    )

try:
    state_path = pathlib.Path(__file__).parent.parent / "model" / "state_dump.pkl"
    state_holder = load_state(state_path)
    parameters = pd.DataFrame.from_dict(state_holder["model_parameters"], orient = "index").astype("str")
    classification_report = state_holder["model_classification_report"]
    confusion_matrix = state_holder["model_confusion_matrix"]
    fpr_tpr_auc = state_holder["model_fpr_tpr_auc"]
    precision_recall_prauc = state_holder["model_precision_recall_prauc"]
    feature_importance_plot = state_holder["model_feature_importance"]
    cv_scores = pd.DataFrame.from_dict(state_holder["model_cv_scores"], orient = "index")
    metrics = pd.DataFrame.from_dict(state_holder["model_metrics"], orient = "index")
    feature_df = state_holder["feature_importance_dataframe"]
except FileNotFoundError as e:
    st.error(f"Required files not found in the project. Please generate the files from notebook before proceeding.")
    st.stop()
except Exception as e:
    st.error(f"Errors encountered while starting project: {e}")
    st.stop()
    
page_title = "Overview of the model"
st.set_page_config(
    page_title = page_title,
    page_icon = "💻",
    layout = "wide"
)
st.sidebar.header(page_title)
st.title("Model Information")
st.metric("Model Name", state_holder["model_name"])

with st.expander("Show model performance summary"):
    left, right = st.columns(2)

    left.subheader("Performance metrics")
    left.dataframe(metrics)

    left.download_button(
        label = "Download preformance metrics summary",
        data = metrics.to_csv(index=False),
        file_name = "performance_scores.csv",
        mime = "text/csv"
    )

    right.subheader("Cross Validation metrics")
    right.dataframe(cv_scores)

    right.download_button(
        label = "Download cross validation metrics summary",
        data = cv_scores.to_csv(index=False),
        file_name = "cross_validation_scores.csv",
        mime = "text/csv"
    )

(parameter_tab, 
classification_report_tab, 
confusion_matrix_tab, 
fpr_tpr_auc_tab, 
precision_recall_prauc_tab,
threshold_stimulation_tab,
feature_importance_tab, 
) = st.tabs(
    ["Model Parameters",
    "Classification Report",
    "Confusion Matrix",
    "ROC-AUC",
    "Precision-Recall vs Threshold",
    "Threshold vs Metrices",
    "Feature Importance"
    ]
)

with parameter_tab:
    st.header("Model hyperparameters")
    st.caption("Hover on top-right of the table to see options.")
    st.dataframe(
        data = parameters,
        width = 'stretch'
    )

with classification_report_tab:
    st.header("Classification Report")
    st.caption("Hover on top-right of the table to see options.")
    st.dataframe(
        data = classification_report,
        width = 'stretch'
    )

with confusion_matrix_tab:
    st.header("Confusion Matrix")
    st.caption("Hover on top of the graph to see values.")
    key = "cm"

    fig = px.imshow(
        confusion_matrix,
        text_auto = True,
        color_continuous_scale="Blues",
        aspect = "auto",
        labels = dict (x = "Predicted", y = "Actual")
    )

    st.plotly_chart(
        figure_or_data = fig,
        width = 'stretch'
    )

    download_button(
        fig = fig,
        filename = "confusion_matrix.html",
        key = key
    )

    if st.session_state[key]:
        st.toast("Download Started")

with fpr_tpr_auc_tab:
    st.header("ROC-AUC Curve")
    st.caption("Hover on top of the graph to see values.")
    key = "roc_auc"

    fpr = fpr_tpr_auc[0]
    tpr = fpr_tpr_auc[1]
    roc_auc = fpr_tpr_auc[2]
    youden_j = tpr - fpr
    best_idx = youden_j.argmax()

    c1, c2, c3 = st.columns(3)
    c1.metric("False Positive Ratio", fpr[best_idx].round(3))
    c2.metric("True Positive Ratio", tpr[best_idx].round(3))
    c3.metric("Area Under Curve (AUC)", round(roc_auc, 3))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x = fpr,
            y = tpr,
            mode = "lines",
            name = "ROC Curve",
            line = dict(width = 3)
        )
    )

    fig.add_trace(
        go.Scatter(
            x = [0, 1],
            y = [0, 1],
            mode = "lines",
            name = "Random Classifier",
            line = dict(dash = "dash")
        )
    )

    fig.add_trace(
        go.Scatter(
            x = [fpr[best_idx]],
            y = [tpr[best_idx]],
            mode = "markers",
            marker = dict(size = 10),
            name = "Best Threshold (Youden J)"
        )
    )

    fig.update_layout(
        title = "ROC–AUC Curve",
        xaxis_title = "False Positive Rate",
        yaxis_title = "True Positive Rate",
        xaxis = dict(range = [0, 1]),
        yaxis = dict(range = [0, 1])
    )

    st.plotly_chart(
        figure_or_data = fig,
        width = 'stretch'
    )

    download_button(
        fig = fig,
        filename = "roc_auc.html",
        key = key
    )

with precision_recall_prauc_tab:
    st.header("Precision-Recall vs Threshold Curve")
    st.caption("Hover on top of the graph to see values.")
    key = "pr_auc"

    precision = precision_recall_prauc[0]
    recall = precision_recall_prauc[1]
    thresholds = precision_recall_prauc[2]
    pr_auc = precision_recall_prauc[3]
        
    c1, c2, c3 = st.columns(3)
    c1.metric("Precision", precision.mean().round(3))
    c2.metric("Recall", recall.mean().round(3))
    c3.metric("PR Area Under Curve (PR-AUC)", round(pr_auc, 3))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x = thresholds,
            y = precision[:-1],
            mode = "lines",
            name = "Precision Curve",
            line = dict(width = 3)
        )
    )

    fig.add_trace(
        go.Scatter(
            x = thresholds,
            y = recall[:-1],
            mode = "lines",
            name = "Recall Curve",
            line = dict(width = 3)
        )
    )

    fig.update_layout(
        title = "Precision-Recall vs Threshold",
        xaxis_title = "Threshold",
        yaxis_title = "Score"
    )

    st.plotly_chart(
        figure_or_data = fig,
        width = 'stretch'
    )

    download_button(
        fig = fig,
        filename = "pr_t.html",
        key = key
    )

    if st.session_state[key]:
        st.toast("Download Started")

with threshold_stimulation_tab:
    st.header("Decision Threshold Simulator")
    st.caption("Adjust the probability threshold to observe changes in precision, recall, and error trade-offs.")

    precision = precision_recall_prauc[0]
    recall = precision_recall_prauc[1]
    thresholds = precision_recall_prauc[2]
    fpr = fpr_tpr_auc[0]
    tpr = fpr_tpr_auc[1]

    st.slider(
        "Select decision threshold",
        min_value=float(thresholds.min()),
        max_value=float(thresholds.max()),
        value=0.5,
        step=0.01,
        key = "threshold_slider"
    )

    idx = (abs(thresholds - st.session_state.threshold_slider)).argmin()
    p = precision[idx]
    r = recall[idx] 
    f1 = 2 * (p * r) / (p + r + 1e-9)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", round(p, 3))
    c2.metric("Recall", round(r, 3))
    c3.metric("F1 Score", round(f1, 3))
    c4.metric("Threshold", round(st.session_state.threshold_slider, 2))

with feature_importance_tab:
    st.header("Feature Importance Graph")
    st.caption("Hover on top of the graph to see values.")
    key = "feature_df"

    fig = px.bar(
        data_frame = feature_df,
        x = "Importance",
        y = "Feature",
        orientation = "h",
        text = "Importance",
        color_discrete_sequence = ["#0068C9"]
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    st.plotly_chart(
        figure_or_data = fig,
        width = "stretch"
    )

    download_button(
        fig = fig,
        filename = "feature_df.html",
        key = key
    )

    if st.session_state[key]:
        st.toast("Download Started")