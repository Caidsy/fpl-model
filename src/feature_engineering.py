"""
FPL Feature Engineering Module

Transforms raw gameweek data into a feature matrix for modelling.
Computes rolling averages, form scores, fixture difficulty, and risk flags.
Saves the processed feature matrix to data/processed/features.csv.
"""

import os
import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")


def load_raw_data():
    """Load all raw CSV files into DataFrames.

    Returns:
        Tuple of (player_gw, players, teams, fixtures, gameweeks) DataFrames.
    """
    player_gw = pd.read_csv(os.path.join(RAW_DIR, "player_gw_history.csv"))
    players = pd.read_csv(os.path.join(RAW_DIR, "players.csv"))
    teams = pd.read_csv(os.path.join(RAW_DIR, "teams.csv"))
    fixtures = pd.read_csv(os.path.join(RAW_DIR, "fixtures.csv"))
    gameweeks = pd.read_csv(os.path.join(RAW_DIR, "gameweeks.csv"))
    return player_gw, players, teams, fixtures, gameweeks


def add_player_info(player_gw, players):
    """Merge player master data onto gameweek history.

    Args:
        player_gw: Gameweek history DataFrame.
        players: Player master DataFrame.

    Returns:
        Merged DataFrame with player metadata attached.
    """
    player_cols = ["id", "web_name", "team", "element_type", "position", "price", "selected_by_percent"]
    merged = player_gw.merge(
        players[player_cols],
        left_on="player_id",
        right_on="id",
        how="left",
        suffixes=("", "_master"),
    )
    # Position encoding: GK=0, DEF=1, MID=2, FWD=3
    merged["position_code"] = merged["element_type"] - 1
    return merged


def compute_rolling_features(df, group_col="player_id", sort_col="round"):
    """Compute rolling averages for key stats per player.

    Computes 3GW and 5GW rolling averages for points, goals, assists,
    minutes, bonus, expected goals, and expected assists.

    Args:
        df: DataFrame sorted by player and gameweek.
        group_col: Column to group by (player_id).
        sort_col: Column to sort by within each group (round).

    Returns:
        DataFrame with rolling average columns added.
    """
    df = df.sort_values([group_col, sort_col]).copy()

    roll_cols = {
        "total_points": "pts",
        "goals_scored": "goals",
        "assists": "assists",
        "minutes": "mins",
        "bonus": "bonus",
    }

    # Add xG/xA if available
    if "expected_goals" in df.columns:
        roll_cols["expected_goals"] = "xg"
    if "expected_assists" in df.columns:
        roll_cols["expected_assists"] = "xa"

    for col, short in roll_cols.items():
        # Convert to numeric, coerce errors
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        for window in [3, 5]:
            new_col = f"rolling_{short}_{window}gw"
            df[new_col] = (
                df.groupby(group_col)[col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )

    return df


def compute_season_avg(df, group_col="player_id", sort_col="round"):
    """Compute expanding season average points to date (excluding current GW).

    Args:
        df: DataFrame sorted by player and gameweek.
        group_col: Column to group by.
        sort_col: Column to sort by.

    Returns:
        DataFrame with season_avg_pts column added.
    """
    df = df.sort_values([group_col, sort_col]).copy()
    df["season_avg_pts"] = (
        df.groupby(group_col)["total_points"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )
    return df


def compute_form_score(df, group_col="player_id", sort_col="round", span=5):
    """Compute exponentially weighted average of recent points as a form score.

    Uses an EWM with the given span, shifted by 1 to avoid data leakage.

    Args:
        df: DataFrame sorted by player and gameweek.
        group_col: Column to group by.
        sort_col: Column to sort by.
        span: EWM span (default 5).

    Returns:
        DataFrame with form_score column added.
    """
    df = df.sort_values([group_col, sort_col]).copy()
    df["form_score"] = (
        df.groupby(group_col)["total_points"]
        .transform(lambda x: x.shift(1).ewm(span=span, min_periods=1).mean())
    )
    return df


def add_home_away_flag(df):
    """Add a binary home/away indicator.

    Args:
        df: DataFrame with 'was_home' column.

    Returns:
        DataFrame with 'is_home' binary column (1=home, 0=away).
    """
    df["is_home"] = df["was_home"].astype(int)
    return df


def add_fixture_difficulty(df, fixtures):
    """Add the fixture difficulty rating for each player's match.

    For home players, uses team_h_difficulty; for away, team_a_difficulty.

    Args:
        df: Player gameweek DataFrame with 'fixture' and 'was_home' columns.
        fixtures: Fixtures DataFrame with difficulty ratings.

    Returns:
        DataFrame with 'fdr' column added.
    """
    fixture_cols = ["id", "team_h_difficulty", "team_a_difficulty"]
    avail_cols = [c for c in fixture_cols if c in fixtures.columns]
    if len(avail_cols) < 3:
        df["fdr"] = 3  # default medium difficulty
        return df

    merged = df.merge(
        fixtures[avail_cols],
        left_on="fixture",
        right_on="id",
        how="left",
        suffixes=("", "_fix"),
    )
    merged["fdr"] = np.where(
        merged["was_home"] == True,
        merged["team_h_difficulty"],
        merged["team_a_difficulty"],
    )
    # Drop merge artifacts
    drop_cols = [c for c in merged.columns if c.endswith("_fix")]
    if "id_fix" in merged.columns:
        drop_cols.append("id_fix")
    # Also drop the fixture id column we merged in
    for c in ["id"]:
        if c in merged.columns and f"{c}_fix" not in merged.columns:
            # Be careful not to drop the original id
            pass
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns], errors="ignore")

    return merged


def add_minutes_risk_flag(df, group_col="player_id", sort_col="round", threshold=3):
    """Flag players who started fewer than `threshold` of their last 5 gameweeks.

    Args:
        df: DataFrame with 'starts' column.
        group_col: Column to group by.
        sort_col: Column to sort by.
        threshold: Minimum starts in last 5 GW to be considered safe (default 3).

    Returns:
        DataFrame with 'minutes_risk' boolean column.
    """
    df = df.sort_values([group_col, sort_col]).copy()

    # 'starts' column: 1 if started, 0 otherwise
    if "starts" in df.columns:
        df["starts"] = pd.to_numeric(df["starts"], errors="coerce").fillna(0)
        df["starts_last_5"] = (
            df.groupby(group_col)["starts"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
        )
    else:
        # Fallback: use minutes > 0 as proxy for starting
        df["started_proxy"] = (df["minutes"] > 0).astype(int)
        df["starts_last_5"] = (
            df.groupby(group_col)["started_proxy"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
        )

    df["minutes_risk"] = df["starts_last_5"] < threshold
    return df


def add_value_metric(df):
    """Compute predicted points per £1m cost.

    Uses form_score as a proxy for predicted points at this stage.

    Args:
        df: DataFrame with 'form_score' and 'price' columns.

    Returns:
        DataFrame with 'value_metric' column.
    """
    df["value_metric"] = np.where(
        df["price"] > 0,
        df["form_score"] / df["price"],
        0,
    )
    return df


def add_upcoming_fdr(df, fixtures):
    """Add the FDR for a player's next upcoming fixture.

    For each player-gameweek row, looks up the fixture in the next gameweek
    and attaches its difficulty rating.

    Args:
        df: Player gameweek DataFrame.
        fixtures: Fixtures DataFrame.

    Returns:
        DataFrame with 'upcoming_fdr' column.
    """
    if "event" not in fixtures.columns:
        df["upcoming_fdr"] = 3
        return df

    # Build a lookup: for each team and gameweek, what's the FDR?
    home_fdr = fixtures[["event", "team_h", "team_h_difficulty"]].rename(
        columns={"team_h": "team_id", "team_h_difficulty": "next_fdr", "event": "next_gw"}
    )
    away_fdr = fixtures[["event", "team_a", "team_a_difficulty"]].rename(
        columns={"team_a": "team_id", "team_a_difficulty": "next_fdr", "event": "next_gw"}
    )
    fdr_lookup = pd.concat([home_fdr, away_fdr], ignore_index=True)
    fdr_lookup = fdr_lookup.dropna(subset=["next_gw"])
    fdr_lookup["next_gw"] = fdr_lookup["next_gw"].astype(int)

    # For each player row, the "upcoming" GW is round + 1
    df["next_gw"] = df["round"] + 1

    # Use opponent_team from the fixture data - but we need the player's own team
    df_merged = df.merge(
        fdr_lookup,
        left_on=["team", "next_gw"],
        right_on=["team_id", "next_gw"],
        how="left",
    )
    df_merged["upcoming_fdr"] = df_merged["next_fdr"].fillna(3)

    # Clean up
    drop_cols = ["team_id", "next_fdr"]
    df_merged = df_merged.drop(columns=[c for c in drop_cols if c in df_merged.columns])

    # Handle duplicates from double gameweeks (take the harder fixture)
    if df_merged.duplicated(subset=["player_id", "round"]).any():
        df_merged = df_merged.sort_values("upcoming_fdr", ascending=False)
        df_merged = df_merged.drop_duplicates(subset=["player_id", "round"], keep="first")

    return df_merged


def build_feature_matrix():
    """Run the full feature engineering pipeline.

    Loads raw data, engineers all features, and saves to data/processed/features.csv.

    Returns:
        The processed feature DataFrame.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Loading raw data...")
    player_gw, players, teams, fixtures, gameweeks = load_raw_data()
    print(f"  {len(player_gw)} gameweek records for {player_gw['player_id'].nunique()} players")

    print("Merging player info...")
    df = add_player_info(player_gw, players)

    print("Computing rolling averages (3GW, 5GW)...")
    df = compute_rolling_features(df)

    print("Computing season average points...")
    df = compute_season_avg(df)

    print("Computing form score (EWM)...")
    df = compute_form_score(df)

    print("Adding home/away flag...")
    df = add_home_away_flag(df)

    print("Adding fixture difficulty rating...")
    df = add_fixture_difficulty(df, fixtures)

    print("Adding minutes risk flag...")
    df = add_minutes_risk_flag(df)

    print("Adding value metric...")
    df = add_value_metric(df)

    print("Adding upcoming FDR...")
    df = add_upcoming_fdr(df, fixtures)

    # Drop the next_gw helper column
    if "next_gw" in df.columns:
        df = df.drop(columns=["next_gw"])

    # Select final feature columns
    feature_cols = [
        # Identifiers
        "player_id", "web_name", "team", "position", "position_code", "round",
        "price", "selected_by_percent",
        # Target
        "total_points",
        # Rolling features
        "rolling_pts_3gw", "rolling_pts_5gw",
        "rolling_goals_3gw", "rolling_goals_5gw",
        "rolling_assists_3gw", "rolling_assists_5gw",
        "rolling_mins_3gw", "rolling_mins_5gw",
        "rolling_bonus_3gw", "rolling_bonus_5gw",
        # Season average
        "season_avg_pts",
        # Form
        "form_score",
        # Context
        "is_home", "fdr", "upcoming_fdr",
        # Risk
        "minutes_risk", "starts_last_5",
        # Value
        "value_metric",
        # Raw stats for reference
        "minutes", "goals_scored", "assists", "clean_sheets",
        "goals_conceded", "bonus", "saves",
    ]

    # Add xG/xA rolling if they exist
    for col in ["rolling_xg_3gw", "rolling_xg_5gw", "rolling_xa_3gw", "rolling_xa_5gw"]:
        if col in df.columns:
            feature_cols.append(col)

    # Keep only columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    df_features = df[feature_cols].copy()

    # Sort for reproducibility
    df_features = df_features.sort_values(["player_id", "round"]).reset_index(drop=True)

    # Save
    output_path = os.path.join(PROCESSED_DIR, "features.csv")
    df_features.to_csv(output_path, index=False)
    print(f"\nSaved feature matrix: {output_path}")
    print(f"  Shape: {df_features.shape}")
    print(f"  Columns: {list(df_features.columns)}")
    print(f"  GW range: {df_features['round'].min()} - {df_features['round'].max()}")
    print(f"  Players: {df_features['player_id'].nunique()}")

    return df_features


if __name__ == "__main__":
    build_feature_matrix()
