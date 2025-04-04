import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# SIDEBAR: Company Logo & File Upload
# -----------------------------
st.sidebar.image("logo.png", use_container_width=True)
uploaded_file = st.sidebar.file_uploader("Upload CSV File for New Strategy(ies)", type=["csv"])

# Load backend data (existing strategies)
@st.cache_data
def load_backend_data():
    df = pd.read_csv("strategy_returns.csv", index_col=0)
    return df

backend_data = load_backend_data()
# Reset the index so that both backend and new data have a consistent index
backend_data = backend_data.reset_index(drop=True)

# Attempt to load new data from the uploaded file
new_data = None
if uploaded_file is not None:
    try:
        new_data = pd.read_csv(uploaded_file, index_col=0)
        new_data = new_data.reset_index(drop=True)
    except Exception as e:
        st.sidebar.error("Error reading uploaded CSV file.")

# Merge backend and new data if new data is available by aligning the indices
if new_data is not None:
    min_rows = min(len(backend_data), len(new_data))
    backend_trim = backend_data.iloc[:min_rows]
    new_trim = new_data.iloc[:min_rows]
    merged_data = pd.concat([backend_trim, new_trim], axis=1)
else:
    merged_data = backend_data

# -----------------------------
# MAIN DASHBOARD: Tabs for Navigation
# -----------------------------
tab_corr, tab_monte = st.tabs(["Correlation Heatmap", "Monte Carlo Simulation"])

# -----------------------------
# TAB 1: Correlation Heatmap
# -----------------------------
with tab_corr:
    st.header("Correlation Heatmap")

    # Part 1: Existing Strategies
    st.subheader("Existing Strategies Correlation Heatmap")
    existing_cols = st.multiselect("Select existing strategies", 
                                   options=backend_data.columns.tolist(), 
                                   default=backend_data.columns.tolist(),
                                   key="existing_corr")
    if len(existing_cols) < 2:
        st.error("Please select at least two existing strategies to compute correlations.")
    else:
        df_existing = backend_data[existing_cols]
        # Compute correlation using pairwise deletion (without dropping all rows)
        corr_existing = df_existing.corr()
        mask_existing = np.triu(np.ones_like(corr_existing, dtype=bool), k=1)
        fig_existing, ax_existing = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_existing,
                    mask=mask_existing,
                    annot=True,
                    cmap="coolwarm",
                    fmt=".2f",
                    linewidths=0.5,
                    ax=ax_existing)
        ax_existing.set_title("Correlation Heatmap for Existing Strategies")
        st.pyplot(fig_existing)

    # Part 2: New Strategies vs. Existing Strategies
    if new_data is not None:
        st.subheader("New Strategies vs. Existing Strategies")
        # Let user select columns from the new data and choose existing columns to compare against.
        new_cols = st.multiselect("Select new strategies", 
                                  options=new_data.columns.tolist(), 
                                  default=new_data.columns.tolist(),
                                  key="new_corr")
        existing_cols_for_new = st.multiselect("Select existing strategies to compare with", 
                                               options=backend_data.columns.tolist(), 
                                               default=backend_data.columns.tolist(),
                                               key="existing_corr_new")
        if len(new_cols) < 1 or len(existing_cols_for_new) < 1:
            st.error("Please select at least one new strategy and one existing strategy.")
        else:
            # Use the merged data as-is (with consistent indexing)
            merged_subset = merged_data[new_cols + existing_cols_for_new]
            # Compute correlation (pairwise deletion is done internally by pandas)
            corr_merged = merged_subset.corr()
            # Extract the correlations of new strategies (rows) with existing strategies (columns)
            corr_new_vs_existing = corr_merged.loc[new_cols, existing_cols_for_new]
            fig_new, ax_new = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_new_vs_existing,
                        annot=True,
                        cmap="coolwarm",
                        fmt=".2f",
                        linewidths=0.5,
                        ax=ax_new)
            ax_new.set_title("Correlation: New Strategies vs. Existing Strategies")
            st.pyplot(fig_new)
    else:
        st.info("No new strategy file uploaded. Upload a file in the sidebar to view correlations between new and existing strategies.")

# -----------------------------
# TAB 2: Monte Carlo Simulation
# -----------------------------
with tab_monte:
    st.header("Monte Carlo Simulation")
    st.subheader("Select a Strategy for Simulation")
    # Let the user select any strategy from the merged dataset (existing + new)
    strategy = st.selectbox("Select a strategy", options=merged_data.columns.tolist())
    returns = merged_data[strategy].dropna()

    # Simulation parameters
    num_simulations = st.number_input("Number of simulations", min_value=100, value=1000, step=100)
    num_days = st.number_input("Number of days", min_value=10, value=252, step=10)
    conf_level = st.selectbox("Select Confidence Level (%)", options=[88, 90, 92, 95], index=3)

    # Monte Carlo Simulation Function
    def monte_carlo_simulation(strategy_returns, num_simulations, num_days):
        mean_return = np.mean(strategy_returns)
        std_dev = np.std(strategy_returns)
        simulations = np.zeros((num_simulations, num_days))
        for i in range(num_simulations):
            daily_returns = np.random.normal(mean_return, std_dev, num_days)
            simulations[i] = np.cumprod(1 + daily_returns)  # Cumulative returns
        return simulations

    simulations = monte_carlo_simulation(returns, num_simulations, num_days)

    st.subheader(f"Simulation of {strategy} ({num_simulations} simulations)")
    fig_sim, ax_sim = plt.subplots(figsize=(10, 6))
    ax_sim.plot(simulations.T, alpha=0.1, color="blue")
    ax_sim.set_title(f"Monte Carlo Simulation of {strategy}")
    ax_sim.set_xlabel("Days")
    ax_sim.set_ylabel("Cumulative Returns")
    st.pyplot(fig_sim)

    # Function to compute maximum drawdown for one simulation path
    def compute_max_drawdown(cum_returns):
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = cum_returns / running_max - 1
        return drawdowns.min()

    max_drawdowns = np.array([compute_max_drawdown(sim) for sim in simulations])
    lower_bound = np.percentile(max_drawdowns, (100 - conf_level) / 2)
    upper_bound = np.percentile(max_drawdowns, 100 - (100 - conf_level) / 2)
    st.write(f"At a {conf_level}% confidence level, the maximum drawdown is estimated to lie between "
             f"{lower_bound:.2%} and {upper_bound:.2%}.")
