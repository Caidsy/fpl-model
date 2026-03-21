"""
FPL Visualisation Module

Provides plotting functions for model performance, player analysis,
and backtest results using matplotlib and seaborn.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")

# Position colour mapping
POS_COLORS = {"GK": "#FFD700", "DEF": "#2ECC71", "MID": "#3498DB", "FWD": "#E74C3C"}


def plot_predicted_vs_actual(test_preds_df, save_path=None):
    """Scatter plot of predicted vs actual points on the test set.

    Args:
        test_preds_df: DataFrame with 'predicted_points' and 'total_points' columns.
        save_path: Optional file path to save the plot.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        test_preds_df["total_points"],
        test_preds_df["predicted_points"],
        alpha=0.3, s=10, color="#3498DB",
    )

    # Perfect prediction line
    max_val = max(test_preds_df["total_points"].max(), test_preds_df["predicted_points"].max())
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1, label="Perfect prediction")

    ax.set_xlabel("Actual Points")
    ax.set_ylabel("Predicted Points")
    ax.set_title("Predicted vs Actual Gameweek Points")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_gw_accuracy(test_preds_df, save_path=None):
    """Line chart of mean predicted vs actual points per gameweek.

    Args:
        test_preds_df: DataFrame with 'round', 'predicted_points', 'total_points'.
        save_path: Optional file path to save the plot.

    Returns:
        matplotlib Figure.
    """
    gw_summary = test_preds_df.groupby("round").agg(
        pred_mean=("predicted_points", "mean"),
        actual_mean=("total_points", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gw_summary["round"], gw_summary["actual_mean"], "o-", label="Actual", color="#2ECC71")
    ax.plot(gw_summary["round"], gw_summary["pred_mean"], "s--", label="Predicted", color="#E74C3C")

    ax.set_xlabel("Gameweek")
    ax.set_ylabel("Mean Points per Player")
    ax.set_title("Model Accuracy by Gameweek")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_player_form(features_df, player_id, save_path=None):
    """Plot a player's gameweek points history and rolling form.

    Args:
        features_df: Full feature DataFrame.
        player_id: The player's element ID.
        save_path: Optional file path to save the plot.

    Returns:
        matplotlib Figure.
    """
    player = features_df[features_df["player_id"] == player_id].sort_values("round")
    if player.empty:
        return None

    name = player["web_name"].iloc[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(player["round"], player["total_points"], alpha=0.5, color="#3498DB", label="GW Points")
    ax.plot(player["round"], player["form_score"], "r-", linewidth=2, label="Form Score (EWM)")
    if "rolling_pts_5gw" in player.columns:
        ax.plot(player["round"], player["rolling_pts_5gw"], "g--", linewidth=1.5, label="5GW Rolling Avg")

    ax.set_xlabel("Gameweek")
    ax.set_ylabel("Points")
    ax.set_title(f"{name} — Points & Form")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_backtest_cumulative(backtest_df, save_path=None):
    """Plot cumulative points from backtest vs average manager.

    Args:
        backtest_df: DataFrame from backtest with cumulative columns.
        save_path: Optional file path to save the plot.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        backtest_df["gameweek"], backtest_df["cumulative_actual"],
        "o-", label="Model Portfolio", color="#2ECC71", linewidth=2,
    )

    if "cumulative_avg_manager" in backtest_df.columns:
        ax.plot(
            backtest_df["gameweek"], backtest_df["cumulative_avg_manager"],
            "s--", label="Average Manager", color="#95A5A6", linewidth=2,
        )

    ax.set_xlabel("Gameweek")
    ax.set_ylabel("Cumulative Points")
    ax.set_title("Backtest: Model Portfolio vs Average Manager")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_fixture_difficulty(features_df, player_id, save_path=None):
    """Plot a player's upcoming fixture difficulty ratings.

    Args:
        features_df: Full feature DataFrame.
        player_id: The player's element ID.
        save_path: Optional file path to save the plot.

    Returns:
        matplotlib Figure.
    """
    player = features_df[features_df["player_id"] == player_id].sort_values("round")
    if player.empty:
        return None

    name = player["web_name"].iloc[0]
    fdr_colors = {1: "#1e8449", 2: "#27ae60", 3: "#f39c12", 4: "#e74c3c", 5: "#922b21"}

    fig, ax = plt.subplots(figsize=(10, 3))
    colors = [fdr_colors.get(int(f), "#999") for f in player["fdr"].fillna(3)]
    ax.bar(player["round"], player["fdr"], color=colors, edgecolor="white")

    ax.set_xlabel("Gameweek")
    ax.set_ylabel("FDR")
    ax.set_title(f"{name} — Fixture Difficulty")
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
