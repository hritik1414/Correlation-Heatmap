# Strategy Correlation Dashboard

This Streamlit dashboard visualizes the correlation between different trading strategies using a heatmap. Users can upload new strategy data to analyze its correlation with existing strategies.

## Features
- Displays a precomputed correlation heatmap of five trading strategies.
- Allows users to upload a new strategy CSV file.
- Computes and visualizes the correlation between the new strategy and existing ones.

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

## Usage
1. Launch the app.
2. View the precomputed correlation heatmap.
3. Upload a new strategy CSV file.
4. Analyze its correlation with existing strategies.


