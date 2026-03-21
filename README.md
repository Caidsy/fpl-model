# FPL Points Predictor & Squad Optimiser

A machine learning system that predicts Fantasy Premier League gameweek points and selects the optimal squad using linear programming.

## Setup

```bash
pip install -r requirements.txt
```

## Running the Pipeline

Run each step in order from the project root directory:

### 1. Data Collection (~10 minutes)
Pulls all player, team, and fixture data from the FPL API.
```bash
python -m src.data_collection
```

### 2. Feature Engineering
Computes rolling averages, form scores, fixture difficulty, and risk flags.
```bash
python -m src.feature_engineering
```

### 3. Model Training & Backtesting
Trains XGBoost + Linear Regression, evaluates on holdout, runs backtest.
```bash
python -m src.model
```

### 4. Squad Optimisation
Selects the optimal 15-man squad for the latest gameweek.
```bash
python -m src.optimiser
```

### 5. Streamlit Dashboard
```bash
streamlit run app.py
```

## Project Structure

```
fpl_model/
├── data/
│   ├── raw/              # Raw CSVs from FPL API
│   └── processed/        # Engineered features, predictions, backtest results
├── src/
│   ├── data_collection.py    # FPL API data fetching
│   ├── feature_engineering.py # Feature computation
│   ├── model.py              # XGBoost training, evaluation, backtesting
│   ├── optimiser.py          # PuLP squad optimisation
│   └── visualisation.py      # Plotting utilities
├── notebooks/
│   └── analysis.ipynb    # Exploratory analysis
├── app.py                # Streamlit dashboard
├── model.pkl             # Trained XGBoost model (generated)
├── model_meta.pkl        # Feature column metadata (generated)
└── requirements.txt
```

## Features

- **Predictions**: XGBoost model trained on rolling stats, form, fixture difficulty, and positional data
- **Squad Optimiser**: Linear programming with budget, position, and club constraints
- **Backtesting**: Simulates squad selection across GW20-38 with cumulative tracking
- **Dashboard**: Interactive Streamlit app with squad viewer, player search, and performance charts
