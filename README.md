# Strategy Correlation & Monte Carlo Simulation Dashboard

This Streamlit dashboard provides:
- **Correlation Analysis**: Visualizes the correlation between different trading strategies using a heatmap.
- **Monte Carlo Simulation**: Performs Monte Carlo simulations on uploaded strategy data to analyze potential future returns.
- **Navigation Tab**: Users can switch between the Correlation Dashboard and the Monte Carlo Simulator.

## Features
- Displays a precomputed correlation heatmap of five trading strategies.
- Allows users to upload a new strategy CSV file.
- Computes and visualizes the correlation between the new strategy and existing ones.
- Monte Carlo simulation for uploaded strategy data, predicting potential return distributions.
- Interactive navigation to switch between Correlation Analysis and Monte Carlo Simulator.

## Setup Instructions

### Prerequisites
Ensure you have the following installed:
- Python (>=3.8)
- `pip` (Python package manager)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd <repository-folder>
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Prepare the Data
Ensure the file `strategy_returns.csv` is placed in the project directory. This file should contain returns data for five strategies with the first column as dates.

### Step 4: Run the Streamlit App
```bash
streamlit run main.py
```

## Navigation Guide
The application consists of two main sections accessible through a navigation tab:
1. **Correlation Dashboard**:
   - View the precomputed correlation heatmap of five strategies.
   - Upload a new strategy CSV file.
   - Analyze the correlation between the new strategy and existing ones.
2. **Monte Carlo Simulation**:
   - Upload a strategy’s daily return data.
   - Perform Monte Carlo simulations to forecast possible return distributions.
   - Visualize simulation results using line plots and histograms.

## CSV File Format
### Preloaded Data (`strategy_returns.csv`)
- The first column should contain dates.
- The subsequent columns should contain returns for different strategies.

### Uploading a New Strategy
- The uploaded file should have:
  - The first column as dates.
  - One additional column containing the strategy's returns.
- Example format:
```
Date, StrategyX
2024-01-01, 0.02
2024-01-02, -0.01
...
```

## Libraries Used
- `streamlit` (for the interactive UI)
- `pandas` (for data manipulation)
- `numpy` (for numerical operations)
- `seaborn` & `matplotlib` (for visualization)
- `scipy.stats` (for Monte Carlo simulation computations)

## Usage
1. Launch the app.
2. Use the navigation tab to switch between the **Correlation Dashboard** and **Monte Carlo Simulation**.
3. View the precomputed correlation heatmap.
4. Upload a new strategy CSV file.
5. Analyze its correlation with existing strategies.
6. Perform Monte Carlo simulations on strategy data and visualize future projections.

