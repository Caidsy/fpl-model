"""
FPL Prediction Dashboard — Streamlit App

Pages:
1. Squad Selector — optimised 15-man squad for a selected gameweek
2. Player Search — GW history, form chart, fixture difficulty
3. Model Performance — predicted vs actual, MAE/RMSE
4. Backtest Results — cumulative points chart
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Paths
BASE_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
META_PATH = os.path.join(BASE_DIR, "model_meta.pkl")

# Position colour mapping
POS_COLORS = {"GK": "#FFD700", "DEF": "#2ECC71", "MID": "#3498DB", "FWD": "#E74C3C"}
POS_BG = {"GK": "#FFF8DC", "DEF": "#E8F8F5", "MID": "#EBF5FB", "FWD": "#FDEDEC"}


@st.cache_data
def load_data():
    """Load all required data files."""
    features = pd.read_csv(os.path.join(PROCESSED_DIR, "features.csv"))
    players = pd.read_csv(os.path.join(RAW_DIR, "players.csv"))
    teams = pd.read_csv(os.path.join(RAW_DIR, "teams.csv"))
    team_map = dict(zip(teams["id"], teams["short_name"]))
    return features, players, teams, team_map


@st.cache_resource
def load_model():
    """Load trained model and metadata."""
    model = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)
    return model, meta["feature_cols"]


def get_predictions(features, model, feature_cols, gameweek):
    """Generate predictions for a specific gameweek."""
    gw_data = features[features["round"] == gameweek].copy()
    if gw_data.empty:
        return pd.DataFrame()
    available = [c for c in feature_cols if c in gw_data.columns]
    clean = gw_data.dropna(subset=available).copy()
    if clean.empty:
        return pd.DataFrame()
    clean["predicted_points"] = model.predict(clean[available].values)
    return clean


def player_card(player, team_map):
    """Render a player card with position-coloured styling."""
    pos = player.get("position", "MID")
    color = POS_COLORS.get(pos, "#999")
    bg = POS_BG.get(pos, "#f9f9f9")
    team = team_map.get(player.get("team", 0), "???")
    pred = player.get("predicted_points", 0)
    price = player.get("price", 0)
    own = player.get("selected_by_percent", "N/A")

    st.markdown(f"""
    <div style="background:{bg}; border-left:4px solid {color}; padding:10px; margin:4px 0; border-radius:4px;">
        <strong style="color:{color};">{player.get('web_name', 'Unknown')}</strong>
        <span style="float:right; color:#666;">{pos}</span><br>
        <span style="color:#666;">{team} · £{price:.1f}m · {own}% owned</span><br>
        <strong>Predicted: {pred:.1f} pts</strong>
    </div>
    """, unsafe_allow_html=True)


# --- Page Functions ---

def get_next_gw_predictions(features, model, feature_cols):
    """Generate predictions for the next unplayed gameweek.

    Uses the latest GW's rolling features as a carry-forward proxy and updates
    FDR/home-away flags from the fixture list if available.

    Args:
        features: Full feature DataFrame.
        model: Trained model.
        feature_cols: List of feature column names.

    Returns:
        Tuple of (predictions_df, next_gw_number) or (empty DataFrame, gw).
    """
    latest_gw = int(features["round"].max())
    next_gw = latest_gw + 1

    gw_data = features[features["round"] == latest_gw].copy()
    if gw_data.empty:
        return pd.DataFrame(), next_gw

    # Update FDR from fixture list for the upcoming GW
    fixtures_path = os.path.join(RAW_DIR, "fixtures.csv")
    if os.path.exists(fixtures_path) and "upcoming_fdr" in gw_data.columns:
        fixtures = pd.read_csv(fixtures_path)
        next_fixtures = fixtures[fixtures["event"] == next_gw]
        if not next_fixtures.empty:
            fdr_lookup = {}
            for _, fix in next_fixtures.iterrows():
                fdr_lookup[fix["team_h"]] = fix.get("team_h_difficulty", 3)
                fdr_lookup[fix["team_a"]] = fix.get("team_a_difficulty", 3)
            gw_data["upcoming_fdr"] = gw_data["team"].map(fdr_lookup).fillna(3)
            home_teams = set(next_fixtures["team_h"].unique())
            away_teams = set(next_fixtures["team_a"].unique())
            gw_data["is_home"] = gw_data["team"].apply(
                lambda t: 1 if t in home_teams else (0 if t in away_teams else 0.5)
            )

    gw_data["round"] = next_gw

    available = [c for c in feature_cols if c in gw_data.columns]
    clean = gw_data.dropna(subset=available).copy()
    if clean.empty:
        return pd.DataFrame(), next_gw
    clean["predicted_points"] = model.predict(clean[available].values)
    return clean, next_gw


def page_squad_selector(features, model, feature_cols, team_map):
    """Squad Selector page."""
    st.header("Squad Selector")

    latest_gw = int(features["round"].max())
    next_gw = latest_gw + 1

    # Toggle between completed GWs and next-GW prediction
    mode = st.radio(
        "Mode",
        [f"Predict Next GW (GW{next_gw})", "Review Completed GW"],
        horizontal=True,
    )

    budget = st.slider("Budget (£m)", 90.0, 110.0, 100.0, 0.5)

    if mode.startswith("Predict Next"):
        preds, display_gw = get_next_gw_predictions(features, model, feature_cols)
        st.info(f"Showing predicted squad for **GW{display_gw}** (upcoming). "
                f"Uses GW{latest_gw} rolling stats with GW{display_gw} fixtures.")
    else:
        available_gws = sorted(features["round"].unique())
        selected_gw = st.selectbox("Select Gameweek", available_gws, index=len(available_gws) - 1)
        preds = get_predictions(features, model, feature_cols, selected_gw)
        display_gw = selected_gw

    if preds.empty:
        st.warning(f"No data available for GW{display_gw}")
        return

    # Import optimiser functions
    from src.optimiser import optimise_squad, select_starting_11, get_captain_picks

    squad = optimise_squad(preds, budget=budget)
    if squad.empty:
        st.error("Could not find a valid squad. Try adjusting the budget.")
        return

    starting_11, bench, formation = select_starting_11(squad)
    captain, vice_captain = get_captain_picks(starting_11)

    total_cost = squad["price"].sum()
    total_pred = squad["predicted_points"].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gameweek", f"GW{display_gw}")
    col2.metric("Formation", formation)
    col3.metric("Total Cost", f"£{total_cost:.1f}m")
    col4.metric("Predicted Points", f"{total_pred:.1f}")

    st.subheader(f"Captain: {captain['web_name']} ({captain['predicted_points']:.1f} pts)")
    st.caption(f"Vice Captain: {vice_captain['web_name']} ({vice_captain['predicted_points']:.1f} pts)")

    st.subheader("Starting 11")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        pos_players = starting_11[starting_11["position"] == pos].sort_values(
            "predicted_points", ascending=False
        )
        for _, p in pos_players.iterrows():
            player_card(p.to_dict(), team_map)

    st.subheader("Bench")
    for _, p in bench.iterrows():
        player_card(p.to_dict(), team_map)


def page_player_search(features, team_map):
    """Player Search page."""
    st.header("Player Search")

    player_names = sorted(features["web_name"].dropna().unique())
    selected = st.selectbox("Search for a player", player_names)

    player_data = features[features["web_name"] == selected].sort_values("round")
    if player_data.empty:
        st.warning("No data found.")
        return

    pid = player_data["player_id"].iloc[0]
    pos = player_data["position"].iloc[0]
    team = team_map.get(player_data["team"].iloc[0], "???")
    price = player_data["price"].iloc[-1]

    st.markdown(f"**{selected}** | {team} | {pos} | £{price:.1f}m")

    # Points history
    st.subheader("Gameweek Points History")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(player_data["round"], player_data["total_points"], alpha=0.6,
           color=POS_COLORS.get(pos, "#999"), label="GW Points")
    if "form_score" in player_data.columns:
        ax.plot(player_data["round"], player_data["form_score"], "r-",
                linewidth=2, label="Form Score")
    if "rolling_pts_5gw" in player_data.columns:
        ax.plot(player_data["round"], player_data["rolling_pts_5gw"], "g--",
                linewidth=1.5, label="5GW Rolling Avg")
    ax.set_xlabel("Gameweek")
    ax.set_ylabel("Points")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

    # Fixture difficulty
    if "fdr" in player_data.columns:
        st.subheader("Fixture Difficulty")
        fdr_colors_map = {1: "#1e8449", 2: "#27ae60", 3: "#f39c12", 4: "#e74c3c", 5: "#922b21"}
        fig2, ax2 = plt.subplots(figsize=(10, 2.5))
        colors = [fdr_colors_map.get(int(f), "#999") for f in player_data["fdr"].fillna(3)]
        ax2.bar(player_data["round"], player_data["fdr"], color=colors, edgecolor="white")
        ax2.set_xlabel("Gameweek")
        ax2.set_ylabel("FDR")
        ax2.set_yticks([1, 2, 3, 4, 5])
        ax2.grid(True, alpha=0.3, axis="y")
        st.pyplot(fig2)
        plt.close()

    # Stats table
    st.subheader("Season Stats")
    stat_cols = ["round", "total_points", "goals_scored", "assists", "clean_sheets",
                 "bonus", "minutes", "form_score"]
    display_cols = [c for c in stat_cols if c in player_data.columns]
    st.dataframe(player_data[display_cols].set_index("round"), use_container_width=True)


def page_model_performance(features, model, feature_cols):
    """Model Performance page."""
    st.header("Model Performance")

    # Load test predictions if available
    pred_path = os.path.join(PROCESSED_DIR, "test_predictions.csv")
    if os.path.exists(pred_path):
        test_preds = pd.read_csv(pred_path)
    else:
        # Generate predictions for GW31+
        test_data = features[features["round"] >= 31].copy()
        if test_data.empty:
            st.info("No test data available yet (season hasn't reached GW31).")
            # Show training performance instead
            st.subheader("Training Data Overview")
            max_gw = int(features["round"].max())
            st.write(f"Data available through GW{max_gw}")
            return

        available = [c for c in feature_cols if c in test_data.columns]
        clean = test_data.dropna(subset=available + ["total_points"])
        clean = clean.copy()
        clean["predicted_points"] = model.predict(clean[available].values)
        test_preds = clean

    if test_preds.empty:
        st.warning("No prediction data available.")
        return

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(test_preds["total_points"], test_preds["predicted_points"])
    rmse = np.sqrt(mean_squared_error(test_preds["total_points"], test_preds["predicted_points"]))

    col1, col2 = st.columns(2)
    col1.metric("MAE", f"{mae:.3f}")
    col2.metric("RMSE", f"{rmse:.3f}")

    # Scatter plot
    st.subheader("Predicted vs Actual Points")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(test_preds["total_points"], test_preds["predicted_points"],
               alpha=0.3, s=10, color="#3498DB")
    max_val = max(test_preds["total_points"].max(), test_preds["predicted_points"].max())
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual Points")
    ax.set_ylabel("Predicted Points")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

    # Per-GW accuracy
    st.subheader("Accuracy by Gameweek")
    gw_summary = test_preds.groupby("round").agg(
        pred_mean=("predicted_points", "mean"),
        actual_mean=("total_points", "mean"),
    ).reset_index()

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(gw_summary["round"], gw_summary["actual_mean"], "o-",
             label="Actual", color="#2ECC71")
    ax2.plot(gw_summary["round"], gw_summary["pred_mean"], "s--",
             label="Predicted", color="#E74C3C")
    ax2.set_xlabel("Gameweek")
    ax2.set_ylabel("Mean Points per Player")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    plt.close()


def page_backtest(features):
    """Backtest Results page."""
    st.header("Backtest Results")

    bt_path = os.path.join(PROCESSED_DIR, "backtest_results.csv")
    if not os.path.exists(bt_path):
        st.warning("No backtest results found. Run model.py first.")
        return

    bt = pd.read_csv(bt_path)

    # Summary metrics
    total_actual = bt["cumulative_actual"].iloc[-1] if not bt.empty else 0
    total_avg = bt["cumulative_avg_manager"].iloc[-1] if "cumulative_avg_manager" in bt.columns else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Total Points", f"{total_actual}")
    col2.metric("Avg Manager Total", f"{total_avg}")
    if total_avg > 0:
        col3.metric("Difference", f"+{total_actual - total_avg}")

    # Cumulative chart
    st.subheader("Cumulative Points Comparison")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(bt["gameweek"], bt["cumulative_actual"], "o-",
            label="Model Portfolio", color="#2ECC71", linewidth=2)
    if "cumulative_avg_manager" in bt.columns:
        ax.plot(bt["gameweek"], bt["cumulative_avg_manager"], "s--",
                label="Average Manager", color="#95A5A6", linewidth=2)
    ax.set_xlabel("Gameweek")
    ax.set_ylabel("Cumulative Points")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

    # Per-GW table
    st.subheader("Gameweek Breakdown")
    display_cols = ["gameweek", "predicted_points", "actual_points",
                    "avg_manager_score", "cumulative_actual", "cumulative_avg_manager"]
    display_cols = [c for c in display_cols if c in bt.columns]
    st.dataframe(bt[display_cols].set_index("gameweek"), use_container_width=True)


# --- Main App ---

def main():
    """Main Streamlit app entry point."""
    st.set_page_config(page_title="FPL Predictor", page_icon="⚽", layout="wide")
    st.title("FPL Points Predictor & Squad Optimiser")

    # Check for required files
    features_path = os.path.join(PROCESSED_DIR, "features.csv")
    if not os.path.exists(features_path):
        st.error("Feature data not found. Run the pipeline first:\n"
                 "1. `python -m src.data_collection`\n"
                 "2. `python -m src.feature_engineering`\n"
                 "3. `python -m src.model`")
        return

    features, players, teams, team_map = load_data()

    model_available = os.path.exists(MODEL_PATH)
    if model_available:
        model, feature_cols = load_model()

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Squad Selector", "Player Search", "Model Performance", "Backtest Results"],
    )

    if page == "Squad Selector":
        if not model_available:
            st.error("Model not found. Run `python -m src.model` first.")
        else:
            page_squad_selector(features, model, feature_cols, team_map)

    elif page == "Player Search":
        page_player_search(features, team_map)

    elif page == "Model Performance":
        if not model_available:
            st.error("Model not found. Run `python -m src.model` first.")
        else:
            page_model_performance(features, model, feature_cols)

    elif page == "Backtest Results":
        page_backtest(features)


if __name__ == "__main__":
    main()
