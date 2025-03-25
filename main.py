import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


@st.cache_data
def load_data():
    df = pd.read_csv("strategy_returns.csv",
                     index_col=0)  # Read the pre-stored 5 strategies
    return df


# Compute correlation matrix
def compute_correlation_matrix(df):
    return df.corr()


# Function to plot heatmap (Lower Triangular with Diagonal)
def plot_heatmap(corr_matrix, title):
    # If only one row, don't use a mask
    if corr_matrix.shape[0] == 1:
        mask = None
    else:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix,
                mask=mask,
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                linewidths=0.5)
    plt.title(title)
    st.pyplot(fig)


# Load initial data and compute correlation
df = load_data()
precomputed_corr = compute_correlation_matrix(df)

# Streamlit App UI
st.title("Strategy Correlation Dashboard")

# Section 1: Display Precomputed Correlation Heatmap
st.subheader("Precomputed Correlation Heatmap of 5 Strategies")
plot_heatmap(precomputed_corr, "Correlation Heatmap of 5 Strategies")

# Section 2: Upload New Strategy
st.subheader("Upload a New Strategy CSV File")
uploaded_file = st.file_uploader(
    "Upload a CSV file (Date as first column, returns as second column)",
    type=["csv"])

if uploaded_file is not None:
    new_strategy_df = pd.read_csv(uploaded_file,
                                  index_col=0)  # Read new strategy data

    # Ensure the new strategy has only one column
    if new_strategy_df.shape[1] != 1:
        st.error(
            "Please upload a CSV file with exactly one strategy column (besides the Date column)."
        )
    else:
        new_strategy_name = new_strategy_df.columns.tolist(
        )  # Extract new strategy name
        df = df.reset_index(drop=True).iloc[:1000]
        new_strategy_df = new_strategy_df.reset_index(drop=True).iloc[:1000]
        # Combine old and new strategies
        combined_df = pd.concat(
            [df, new_strategy_df],
            axis=1).dropna()  # Merge & handle missing values

        # Compute new correlation matrix (only for new strategy vs old ones)
        new_correlation_matrix = combined_df.corr()

        # Extract correlations of the new strategy with existing ones
        new_corr_values = new_correlation_matrix.loc[new_strategy_name, :]

        # Convert to DataFrame
        new_corr_matrix = pd.DataFrame(new_corr_values).T

        # Display new strategy correlation values
        st.subheader(
            f"Correlation of '{new_strategy_name}' with Existing Strategies")
        st.dataframe(new_corr_matrix)  # Show correlation table

        # Plot heatmap for new strategy correlation
        plot_heatmap(
            new_corr_matrix,
            f"Correlation Heatmap: {new_strategy_name} vs Existing Strategies")
