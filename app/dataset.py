import io
import pandas as pd
import pathlib
import pickle
import plotly.express as px
import streamlit as st

# For loading application state
@st.cache_data
def load_state(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# For computing outliers in dataset
def outlier_summary(df):
    outlier_data = []

    for col in df.select_dtypes(include = "number").columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = df[(df[col] < lower) | (df[col] > upper)]

        outlier_data.append({
            "Column": col,
            "Outlier Count": outliers.shape[0],
            "Outlier %": round((outliers.shape[0] / df.shape[0]) * 100, 2)
        })

    return pd.DataFrame(outlier_data)

# Provides download button for interactive plotly graphs
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

# Load application states
try:
    state_path = pathlib.Path(__file__).parent.parent / "model" / "state_dump.pkl"
    state_holder = load_state(state_path)

    df = state_holder["dataframe"]              # Default dataset
    numerical_info = pd.DataFrame(df.describe())
    categorical_columns = df.select_dtypes(include = "object").columns.drop("customerID", errors = "ignore")
    
except FileNotFoundError as e:
    st.error(f"Required files not found in the project. Please generate the files by running training notebook before proceeding.")
    st.stop()
except Exception as e:
    st.error(f"Errors encountered while starting project: {e}")
    st.stop()

# Dataset summary
summary = pd.DataFrame({
    "Column": df.columns,
    "Dtype": df.dtypes.astype(str),
    "Missing %": (
        df.isna().mean() * 100
    ).round(2)
})

# Page configuration
page_title = "Overview of the dataset"
st.set_page_config(
    page_title = page_title,
    page_icon = "📋",
    layout = "wide"
)
st.sidebar.header(page_title)
st.title("Dataset Overview")

# Dataset metrics
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
c3.metric("Numerical Columns", numerical_info.shape[1])
c4.metric("Categorical Columns", categorical_columns.shape[0])
c5.metric("Missing Cells", int(df.isna().sum().sum()))
c6.metric("Duplicate Rows", df.duplicated().sum())

# Column summary
with st.expander("Column Summary"):
    st.dataframe(
        data = summary,
        width = "stretch",
    )

    st.download_button(
        label = "Download Column Summary",
        data = summary.to_csv(index = False),
        file_name = "column_summary.csv",
        mime = "text/csv"
    )

# Dataset dashboard
dataset, numerical_tab, categorical_tab, eda_tab = st.tabs(tabs = ["Dataset Contents", "Numerical Overview", "Categorical Overview", "Exploratory Data Analysis (EDA)"])

# Dataset contents
with dataset:
    st.header("Dataset Contents")
    st.caption("Hover on top-right of the table or each column header to see options.")

    selected_cols = st.multiselect(
        label = "Select columns to display",
        options = df.columns,
        default = df.columns
    )

    st.dataframe(
        data = df[selected_cols],
        width = "stretch",
        height = "stretch"
    )

# Numerical overview
with numerical_tab:
    st.header("Numerical column overview")
    st.caption("Hover on top-right of the table or each column header to see options.")

    st.dataframe(
        data = numerical_info,
        width = "stretch",
        height = "stretch" 
    )

# Categorical overview
with categorical_tab:
    st.header("Category column overview")
    st.caption("Hover on top of the graph to see values or top-right of table to see options.")

    st.selectbox(
        label = "Select categorical column",
        options = categorical_columns,
        key = "selectbox"
    )

    vc = df[st.session_state.selectbox].value_counts().reset_index()
    vc.columns = ["Category", "Count"]
    st.dataframe(vc)

    bar = px.bar(
        data_frame = vc,
        x = "Category",
        y = "Count",
        color="Category",
        text = "Count",
        title = f"Count of {st.session_state.selectbox}"
    )

    key = "bar"
    st.plotly_chart(
        bar,
        width = "stretch"
    )

    download_button(
        fig = bar,
        filename = f"Count of {st.session_state.selectbox}.html",
        key = key
    )

    if st.session_state[key]:
        st.toast("Download Started")

# Exploratory data analysis (EDA)
with eda_tab:
    st.header("Exploratory Data Analysis (EDA)")
    st.caption("Hover on top of the graph to see values.")

    st.subheader("Correlation Heatmap of numeric features")
    numerical_data = df.select_dtypes(include = ["number"])
    corr = numerical_data.corr()

    cm = px.imshow(
        corr,
        text_auto = ".2f",
        color_continuous_scale = "RdBu",
        aspect = "auto"
    )

    key1 = "cm"
    st.plotly_chart(
        cm, 
        width = "stretch"
    )

    download_button(
        fig = cm,
        filename = "correlation_matrix.html",
        key = key1
    )

    if st.session_state[key1]:
        st.toast("Download Started")

    st.divider()
    st.subheader("Distribution Plot of numeric features")

    st.selectbox(
        label = "Select numerical column",
        options = numerical_data.columns,
        key = "dist_col"
    )

    hm = px.histogram(
        data_frame = df,
        x = st.session_state.dist_col,
        marginal = "box",
        title = f"Distribution of {st.session_state.dist_col}",
        color_discrete_sequence = ["#0068C9"]
    )

    key2 = "hm"
    st.plotly_chart(
        hm, 
        width = "stretch"
    )
        
    download_button(
        fig = hm,
        filename = f"Distribution of {st.session_state.dist_col}.html",
        key = key2
    )

    if st.session_state[key2]:
        st.toast("Download Started")

    st.divider()
    st.subheader("Box Plot of numeric features")

    st.selectbox(
        "Select numerical column",
        options = numerical_data.columns,
        key = "box_col"
    )

    bm = px.box(
        df,
        y = st.session_state.box_col,
        title = f"Box plot of {st.session_state.box_col}",
        color_discrete_sequence = ["#0068C9"]
    )

    key3 = "bm"
    st.plotly_chart(
        bm, 
        width = "stretch"
    )

    download_button(
        fig = bm,
        filename = f"Box plot of {st.session_state.box_col}.html",
        key = key3
    )

    if st.session_state[key3]:
        st.toast("Download Started")

    # Outlier analysis summary
    outlier_df = outlier_summary(numerical_data)
    if outlier_df["Outlier Count"].sum() == 0:
        st.success("No outliers detected in the dataset.")
    else:
        st.dataframe(
            outlier_df.sort_values("Outlier Count", ascending = False),
            width = "stretch"
        )

        fig = px.bar(
            outlier_df,
            x = "Column",
            y = "Outlier Count",
            title = "Outlier Count per Numerical Column",
            text = "Outlier Count"
        )
            
        fig.update_traces(textposition = "outside")
        st.plotly_chart(fig, width = "stretch")