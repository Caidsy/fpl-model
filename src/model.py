"""
FPL Points Prediction Model

Trains XGBoost and Linear Regression models to predict gameweek points per player.
Evaluates on a GW31-38 holdout set and reports MAE/RMSE.
Includes backtesting to simulate the optimiser over GW20-38.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_XGBOOST = False

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
MODEL_DIR = os.path.dirname(os.path.dirname(__file__))


# Features used for training
FEATURE_COLS = [
    "rolling_pts_3gw", "rolling_pts_5gw",
    "rolling_goals_3gw", "rolling_goals_5gw",
    "rolling_assists_3gw", "rolling_assists_5gw",
    "rolling_mins_3gw", "rolling_mins_5gw",
    "rolling_bonus_3gw", "rolling_bonus_5gw",
    "season_avg_pts", "form_score",
    "is_home", "fdr", "upcoming_fdr",
    "position_code", "price",
    "starts_last_5",
    "rolling_xg_3gw", "rolling_xg_5gw",
    "rolling_xa_3gw", "rolling_xa_5gw",
]


def load_features():
    """Load the processed feature matrix.

    Returns:
        DataFrame with all engineered features.
    """
    path = os.path.join(PROCESSED_DIR, "features.csv")
    return pd.read_csv(path)


def prepare_data(df, train_end_gw=30, test_start_gw=31):
    """Split data into train and test sets by gameweek.

    Args:
        df: Feature DataFrame.
        train_end_gw: Last gameweek included in training (default 30).
        test_start_gw: First gameweek in test set (default 31).

    Returns:
        Tuple of (X_train, y_train, X_test, y_test, test_df).
    """
    # Use only available feature columns
    available_features = [c for c in FEATURE_COLS if c in df.columns]

    # Drop rows with NaN in features or target
    df_clean = df.dropna(subset=available_features + ["total_points"]).copy()

    train = df_clean[df_clean["round"] <= train_end_gw]
    test = df_clean[df_clean["round"] >= test_start_gw]

    X_train = train[available_features].values
    y_train = train["total_points"].values
    X_test = test[available_features].values
    y_test = test["total_points"].values

    return X_train, y_train, X_test, y_test, test, available_features


def train_xgboost(X_train, y_train):
    """Train a gradient boosting regression model (XGBoost if available, else sklearn).

    Args:
        X_train: Training feature matrix.
        y_train: Training target values.

    Returns:
        Trained gradient boosting model.
    """
    if HAS_XGBOOST:
        model = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
    model.fit(X_train, y_train)
    return model


def train_linear(X_train, y_train):
    """Train a baseline linear regression model.

    Args:
        X_train: Training feature matrix.
        y_train: Training target values.

    Returns:
        Trained LinearRegression.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate a model and print MAE and RMSE.

    Args:
        model: Trained model with a predict method.
        X_test: Test feature matrix.
        y_test: Test target values.
        model_name: Name for display purposes.

    Returns:
        Dict with predictions, MAE, and RMSE.
    """
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"\n{model_name} Performance:")
    print(f"  MAE:  {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")

    return {"predictions": preds, "mae": mae, "rmse": rmse}


def print_feature_importance(model, feature_names):
    """Print the feature importance ranking from an XGBoost model.

    Args:
        model: Trained XGBRegressor.
        feature_names: List of feature names in order.
    """
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]

    print("\nFeature Importance Ranking:")
    print("-" * 40)
    for i, idx in enumerate(sorted_idx):
        print(f"  {i+1:2d}. {feature_names[idx]:<25s} {importance[idx]:.4f}")


def predict_for_gameweek(model, df, gameweek, feature_cols):
    """Generate predictions for all players in a specific gameweek.

    Args:
        model: Trained model.
        df: Full feature DataFrame.
        gameweek: The gameweek to predict for.
        feature_cols: List of feature column names.

    Returns:
        DataFrame with player info and predicted points.
    """
    gw_data = df[df["round"] == gameweek].copy()
    if gw_data.empty:
        return pd.DataFrame()

    available = [c for c in feature_cols if c in gw_data.columns]
    gw_clean = gw_data.dropna(subset=available)

    if gw_clean.empty:
        return pd.DataFrame()

    preds = model.predict(gw_clean[available].values)
    gw_clean = gw_clean.copy()
    gw_clean["predicted_points"] = preds

    return gw_clean


def backtest(df, feature_cols, start_gw=20, end_gw=None):
    """Simulate running the optimiser each gameweek and track performance.

    Trains on all data up to GW-1 for each gameweek, predicts, runs a simple
    top-15 selection, and tracks actual points scored.

    Args:
        df: Full feature DataFrame.
        feature_cols: List of feature column names.
        start_gw: First gameweek to simulate (default 20).
        end_gw: Last gameweek to simulate (default: max available).

    Returns:
        DataFrame with backtest results per gameweek.
    """
    if end_gw is None:
        end_gw = int(df["round"].max())

    available = [c for c in feature_cols if c in df.columns]

    # Try to load average manager scores from gameweeks data
    gw_path = os.path.join(os.path.dirname(PROCESSED_DIR), "raw", "gameweeks.csv")
    avg_scores = {}
    if os.path.exists(gw_path):
        gw_df = pd.read_csv(gw_path)
        if "average_entry_score" in gw_df.columns:
            avg_scores = dict(zip(gw_df["id"], gw_df["average_entry_score"]))

    results = []
    cumulative_pred = 0
    cumulative_actual = 0
    cumulative_avg = 0

    print(f"\nBacktesting GW{start_gw} to GW{end_gw}...")
    print("-" * 75)

    for gw in range(start_gw, end_gw + 1):
        # Train on all data before this GW
        train_data = df[df["round"] < gw].dropna(subset=available + ["total_points"])
        test_data = df[df["round"] == gw].dropna(subset=available + ["total_points"])

        if train_data.empty or test_data.empty:
            continue

        X_train = train_data[available].values
        y_train = train_data["total_points"].values

        # Train a fresh model for this GW
        if HAS_XGBOOST:
            model = XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
        model.fit(X_train, y_train)

        # Predict
        test_data = test_data.copy()
        test_data["predicted_points"] = model.predict(test_data[available].values)

        # Simple squad selection: top 11 by predicted points (excluding minutes risk)
        safe_players = test_data[test_data["minutes_risk"] == False].copy()
        if len(safe_players) < 11:
            safe_players = test_data.copy()

        squad = safe_players.nlargest(11, "predicted_points")
        pred_pts = squad["predicted_points"].sum()
        actual_pts = squad["total_points"].sum()

        avg_manager = avg_scores.get(gw, 0)

        cumulative_pred += pred_pts
        cumulative_actual += actual_pts
        cumulative_avg += avg_manager

        results.append({
            "gameweek": gw,
            "predicted_points": round(pred_pts, 1),
            "actual_points": int(actual_pts),
            "avg_manager_score": int(avg_manager) if avg_manager else None,
            "cumulative_actual": int(cumulative_actual),
            "cumulative_avg_manager": int(cumulative_avg),
        })

        print(f"  GW{gw:2d}: predicted={pred_pts:6.1f}  actual={actual_pts:3d}  "
              f"avg_mgr={int(avg_manager):3d}  cum_actual={int(cumulative_actual):4d}  "
              f"cum_avg={int(cumulative_avg):4d}")

    results_df = pd.DataFrame(results)
    output_path = os.path.join(PROCESSED_DIR, "backtest_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nBacktest results saved to {output_path}")

    return results_df


def run_training():
    """Run the full training pipeline: load data, train models, evaluate, save."""
    print("Loading feature matrix...")
    df = load_features()
    print(f"  Shape: {df.shape}")

    print("\nPreparing train/test split (train GW1-30, test GW31+)...")
    X_train, y_train, X_test, y_test, test_df, feature_cols = prepare_data(df)
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test:  {len(X_test)} samples")

    if len(X_test) == 0:
        print("\nWARNING: No test data available (season may not have reached GW31 yet).")
        print("Training on all available data instead.")
        # Train on all data, no test evaluation
        X_train_all = df.dropna(subset=feature_cols + ["total_points"])
        available = [c for c in feature_cols if c in df.columns]
        X_all = X_train_all[available].values
        y_all = X_train_all["total_points"].values

        print("\nTraining XGBoost...")
        xgb_model = train_xgboost(X_all, y_all)
        print_feature_importance(xgb_model, feature_cols)

        model_path = os.path.join(MODEL_DIR, "model.pkl")
        joblib.dump(xgb_model, model_path)
        print(f"\nModel saved to {model_path}")

        # Save feature columns for later use
        meta_path = os.path.join(MODEL_DIR, "model_meta.pkl")
        joblib.dump({"feature_cols": feature_cols}, meta_path)

        # Run backtest
        print("\n" + "=" * 75)
        backtest(df, feature_cols)

        return xgb_model, None, feature_cols

    # Train XGBoost
    print("\nTraining XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    xgb_results = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    # Train Linear Regression baseline
    print("\nTraining Linear Regression baseline...")
    lr_model = train_linear(X_train, y_train)
    lr_results = evaluate_model(lr_model, X_test, y_test, "Linear Regression")

    # Side-by-side comparison
    print("\n" + "=" * 50)
    print("Model Comparison:")
    print("=" * 50)
    print(f"{'Metric':<10} {'XGBoost':>12} {'Linear Reg':>12}")
    print("-" * 34)
    print(f"{'MAE':<10} {xgb_results['mae']:>12.3f} {lr_results['mae']:>12.3f}")
    print(f"{'RMSE':<10} {xgb_results['rmse']:>12.3f} {lr_results['rmse']:>12.3f}")

    # Feature importance
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    print_feature_importance(xgb_model, available_features)

    # Save model
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    joblib.dump(xgb_model, model_path)
    print(f"\nXGBoost model saved to {model_path}")

    # Save feature columns metadata
    meta_path = os.path.join(MODEL_DIR, "model_meta.pkl")
    joblib.dump({"feature_cols": available_features}, meta_path)

    # Save test predictions for visualisation
    test_preds = test_df.copy()
    test_preds["predicted_points"] = xgb_results["predictions"]
    pred_path = os.path.join(PROCESSED_DIR, "test_predictions.csv")
    test_preds.to_csv(pred_path, index=False)
    print(f"Test predictions saved to {pred_path}")

    # Run backtest
    print("\n" + "=" * 75)
    backtest(df, available_features)

    return xgb_model, lr_model, available_features


if __name__ == "__main__":
    run_training()
