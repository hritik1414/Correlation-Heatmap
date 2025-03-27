import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Page Selection
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Strategy Correlation Dashboard", "Monte Carlo Simulation"])

# --- COMMON FUNCTIONS ---

# Load precomputed strategy data
@st.cache_data
def load_data():
    df = pd.read_csv("strategy_returns.csv", index_col=0)  # Read the pre-stored 5 strategies
    return df

# Compute correlation matrix
def compute_correlation_matrix(df):
    return df.corr()

# Function to plot heatmap (Lower Triangular with Diagonal)
def plot_heatmap(corr_matrix, title):
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # Keeps diagonal
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(title)
    st.pyplot(fig)  # Display the heatmap in Streamlit

# --- PAGE 1: STRATEGY CORRELATION DASHBOARD ---
if page == "Strategy Correlation Dashboard":
    st.title("Strategy Correlation Dashboard")

    # Load initial data and compute correlation
    df = load_data()
    precomputed_corr = compute_correlation_matrix(df)

    # Section 1: Display Precomputed Correlation Heatmap
    st.subheader("Precomputed Correlation Heatmap of 5 Strategies")
    plot_heatmap(precomputed_corr, "Correlation Heatmap of 5 Strategies")

    # Section 2: Upload New Strategy
    st.subheader("Upload a New Strategy CSV File")
    uploaded_file = st.file_uploader("Upload a CSV file (Date as first column, returns as second column)", type=["csv"])

    if uploaded_file is not None:
        new_strategy_df = pd.read_csv(uploaded_file, index_col=0)  # Read new strategy data

        # Ensure the new strategy has only one column
        if new_strategy_df.shape[1] != 1:
            st.error("Please upload a CSV file with exactly one strategy column (besides the Date column).")
        else:
            new_strategy_name = new_strategy_df.columns[0]  # Extract new strategy name

            # Combine old and new strategies
            combined_df = pd.concat([df, new_strategy_df], axis=1).dropna()  # Merge & handle missing values

            # Compute new correlation matrix (only for new strategy vs old ones)
            new_correlation = combined_df.corr().loc[new_strategy_name, df.columns]  # Extract new strategy correlations

            # Convert to DataFrame for heatmap
            new_corr_matrix = pd.DataFrame(new_correlation).T

            # Display new strategy correlation values
            st.subheader(f"Correlation of '{new_strategy_name}' with Existing Strategies")
            st.dataframe(new_corr_matrix)  # Show correlation table

            # Plot heatmap for new strategy correlation
            plot_heatmap(new_corr_matrix, f"Correlation Heatmap: {new_strategy_name} vs Existing Strategies")

# --- PAGE 2: MONTE CARLO SIMULATION ---
elif page == "Monte Carlo Simulation":
    st.title("Monte Carlo Simulation on Strategy Returns")

    # File Upload for Simulation
    uploaded_monte_file = st.file_uploader("Upload CSV file for Monte Carlo Simulation (1000 Days of Returns)", type=["csv"])

    if uploaded_monte_file is not None:
        monte_df = pd.read_csv(uploaded_monte_file, index_col=0)  # Read new strategy data

        # Ensure at least one column is present
        if monte_df.shape[1] < 1:
            st.error("Please upload a CSV file with at least one strategy column (besides the Date column).")
        else:
            # Select strategy for Monte Carlo Simulation
            strategy_name = st.selectbox("Select a Strategy for Simulation", monte_df.columns)

            # Monte Carlo Simulation Function
            def monte_carlo_simulation(strategy_returns, num_simulations=1000, num_days=252):
                mean_return = np.mean(strategy_returns)
                std_dev = np.std(strategy_returns)

                simulations = np.zeros((num_simulations, num_days))

                for i in range(num_simulations):
                    daily_returns = np.random.normal(mean_return, std_dev, num_days)
                    simulations[i] = np.cumprod(1 + daily_returns)  # Cumulative returns

                return simulations

            # Perform Simulation
            simulations = monte_carlo_simulation(monte_df[strategy_name].dropna())

            # Plot Simulation Results
            st.subheader(f"Monte Carlo Simulation for {strategy_name}")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(simulations.T, alpha=0.1, color="blue")  # Plot all simulations
            ax.set_title(f"Monte Carlo Simulation of {strategy_name} (1000 Simulations)")
            ax.set_xlabel("Days")
            ax.set_ylabel("Cumulative Returns")
            st.pyplot(fig)
