"""
FPL Squad Optimisation Module

Uses linear programming (PuLP) to select the optimal 15-man FPL squad
given predicted points, budget constraints, and squad rules.
"""

import os
import numpy as np
import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")


# Valid formations: (DEF, MID, FWD) — GK is always 1
VALID_FORMATIONS = [
    (3, 4, 3), (3, 5, 2),
    (4, 3, 3), (4, 4, 2), (4, 5, 1),
    (5, 3, 2), (5, 4, 1),
]


def optimise_squad(predictions_df, budget=100.0, max_per_club=3):
    """Select the optimal 15-man FPL squad using linear programming.

    Hard constraints:
    - Total cost <= budget (default £100m)
    - Max 3 players per club
    - Exactly 2 GK, 5 DEF, 5 MID, 3 FWD
    - All selected players must have minutes_risk = False

    Args:
        predictions_df: DataFrame with columns: player_id, web_name, team, position,
            position_code, price, predicted_points, minutes_risk, selected_by_percent.
        budget: Total budget in millions (default 100.0).
        max_per_club: Maximum players from one club (default 3).

    Returns:
        DataFrame of the selected 15-man squad with all relevant info.
    """
    # Filter out players with minutes risk
    df = predictions_df[predictions_df["minutes_risk"] == False].copy()
    df = df.drop_duplicates(subset=["player_id"], keep="first")
    df = df.reset_index(drop=True)

    if len(df) < 15:
        print("WARNING: Not enough eligible players after filtering.")
        return pd.DataFrame()

    n = len(df)
    prob = LpProblem("FPL_Squad_Selection", LpMaximize)

    # Binary variables: 1 if player selected, 0 otherwise
    x = [LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    # Objective: maximise total predicted points
    prob += lpSum(x[i] * df.iloc[i]["predicted_points"] for i in range(n))

    # Budget constraint
    prob += lpSum(x[i] * df.iloc[i]["price"] for i in range(n)) <= budget

    # Total squad size = 15
    prob += lpSum(x[i] for i in range(n)) == 15

    # Position constraints
    position_counts = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    for pos, count in position_counts.items():
        indices = df.index[df["position"] == pos].tolist()
        prob += lpSum(x[i] for i in indices) == count

    # Max per club
    for team_id in df["team"].unique():
        indices = df.index[df["team"] == team_id].tolist()
        prob += lpSum(x[i] for i in indices) <= max_per_club

    # Solve
    prob.solve()

    if LpStatus[prob.status] != "Optimal":
        print(f"Optimisation failed: {LpStatus[prob.status]}")
        return pd.DataFrame()

    # Extract selected players
    selected_indices = [i for i in range(n) if x[i].varValue == 1]
    squad = df.iloc[selected_indices].copy()

    return squad


def select_starting_11(squad):
    """Choose the best starting 11 from the 15-man squad in a valid formation.

    Tries all valid formations and picks the one that maximises predicted points.

    Args:
        squad: DataFrame of the 15-man squad.

    Returns:
        Tuple of (starting_11_df, bench_df, formation_string).
    """
    best_score = -1
    best_starting = None
    best_formation = None

    for d, m, f in VALID_FORMATIONS:
        gks = squad[squad["position"] == "GK"].nlargest(1, "predicted_points")
        defs = squad[squad["position"] == "DEF"].nlargest(d, "predicted_points")
        mids = squad[squad["position"] == "MID"].nlargest(m, "predicted_points")
        fwds = squad[squad["position"] == "FWD"].nlargest(f, "predicted_points")

        starting = pd.concat([gks, defs, mids, fwds])
        score = starting["predicted_points"].sum()

        if score > best_score:
            best_score = score
            best_starting = starting
            best_formation = f"1-{d}-{m}-{f}"

    bench = squad[~squad.index.isin(best_starting.index)].sort_values(
        "predicted_points", ascending=False
    )

    return best_starting, bench, best_formation


def get_captain_picks(starting_11):
    """Suggest captain and vice-captain from the starting 11.

    Args:
        starting_11: DataFrame of starting 11 players.

    Returns:
        Tuple of (captain_row, vice_captain_row).
    """
    sorted_11 = starting_11.sort_values("predicted_points", ascending=False)
    captain = sorted_11.iloc[0]
    vice_captain = sorted_11.iloc[1]
    return captain, vice_captain


def print_squad(squad, starting_11, bench, formation, captain, vice_captain):
    """Pretty-print the full squad details.

    Args:
        squad: Full 15-man squad DataFrame.
        starting_11: Starting 11 DataFrame.
        bench: Bench DataFrame.
        formation: Formation string.
        captain: Captain row (Series).
        vice_captain: Vice-captain row (Series).
    """
    # Load team names
    teams_path = os.path.join(RAW_DIR, "teams.csv")
    team_names = {}
    if os.path.exists(teams_path):
        teams_df = pd.read_csv(teams_path)
        team_names = dict(zip(teams_df["id"], teams_df["short_name"]))

    total_cost = squad["price"].sum()
    total_pred = squad["predicted_points"].sum()

    print("\n" + "=" * 80)
    print(f"OPTIMAL SQUAD  |  Formation: {formation}  |  "
          f"Cost: £{total_cost:.1f}m  |  Predicted: {total_pred:.1f} pts")
    print("=" * 80)

    print(f"\n  Captain:      {captain['web_name']} ({captain['predicted_points']:.1f} pts)")
    print(f"  Vice Captain: {vice_captain['web_name']} ({vice_captain['predicted_points']:.1f} pts)")

    header = f"{'':>3} {'Name':<20} {'Team':<5} {'Pos':<4} {'Price':>6} {'Pred Pts':>9} {'Own%':>6}"
    print(f"\n  STARTING 11")
    print(f"  {header}")
    print(f"  {'-'*len(header)}")

    for pos in ["GK", "DEF", "MID", "FWD"]:
        pos_players = starting_11[starting_11["position"] == pos].sort_values(
            "predicted_points", ascending=False
        )
        for _, p in pos_players.iterrows():
            team_str = team_names.get(p["team"], str(int(p["team"])))
            marker = " (C)" if p["player_id"] == captain["player_id"] else \
                     " (V)" if p["player_id"] == vice_captain["player_id"] else ""
            print(f"  {'':>3} {p['web_name'] + marker:<20} {team_str:<5} {p['position']:<4} "
                  f"£{p['price']:>5.1f} {p['predicted_points']:>9.1f} {p.get('selected_by_percent', 'N/A'):>6}")

    print(f"\n  BENCH (ordered by predicted points)")
    print(f"  {header}")
    print(f"  {'-'*len(header)}")
    for i, (_, p) in enumerate(bench.iterrows()):
        team_str = team_names.get(p["team"], str(int(p["team"])))
        print(f"  {i+1:>3} {p['web_name']:<20} {team_str:<5} {p['position']:<4} "
              f"£{p['price']:>5.1f} {p['predicted_points']:>9.1f} {p.get('selected_by_percent', 'N/A'):>6}")

    print()


def run_optimiser(gameweek=None):
    """Run the full squad optimisation pipeline.

    Args:
        gameweek: Specific gameweek to optimise for. If None, uses the latest available.

    Returns:
        Dict with squad, starting_11, bench, captain, vice_captain.
    """
    import joblib

    # Load model and metadata
    model_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(model_dir, "model.pkl")
    meta_path = os.path.join(model_dir, "model_meta.pkl")

    if not os.path.exists(model_path):
        print("ERROR: model.pkl not found. Run model.py first.")
        return None

    model = joblib.load(model_path)
    meta = joblib.load(meta_path)
    feature_cols = meta["feature_cols"]

    # Load feature data
    features_path = os.path.join(PROCESSED_DIR, "features.csv")
    df = pd.read_csv(features_path)

    if gameweek is None:
        gameweek = int(df["round"].max())

    print(f"Optimising squad for GW{gameweek}...")

    # Get predictions for the target gameweek
    gw_data = df[df["round"] == gameweek].copy()
    if gw_data.empty:
        print(f"No data for GW{gameweek}")
        return None

    available = [c for c in feature_cols if c in gw_data.columns]
    gw_clean = gw_data.dropna(subset=available).copy()

    gw_clean["predicted_points"] = model.predict(gw_clean[available].values)

    # Run optimisation
    squad = optimise_squad(gw_clean)
    if squad.empty:
        print("Optimisation returned empty squad.")
        return None

    starting_11, bench, formation = select_starting_11(squad)
    captain, vice_captain = get_captain_picks(starting_11)

    print_squad(squad, starting_11, bench, formation, captain, vice_captain)

    return {
        "squad": squad,
        "starting_11": starting_11,
        "bench": bench,
        "formation": formation,
        "captain": captain,
        "vice_captain": vice_captain,
    }


def predict_next_gameweek():
    """Predict the optimal squad for the NEXT unplayed gameweek.

    Uses the latest completed gameweek's rolling features (form, averages, etc.)
    as a proxy for next-GW features, but updates the upcoming FDR from the
    fixture list if available. This is the function to call when you want
    a forward-looking squad recommendation.

    Returns:
        Dict with squad, starting_11, bench, captain, vice_captain, or None.
    """
    import joblib

    model_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(model_dir, "model.pkl")
    meta_path = os.path.join(model_dir, "model_meta.pkl")

    if not os.path.exists(model_path):
        print("ERROR: model.pkl not found. Run model.py first.")
        return None

    model = joblib.load(model_path)
    meta = joblib.load(meta_path)
    feature_cols = meta["feature_cols"]

    # Load feature data
    features_path = os.path.join(PROCESSED_DIR, "features.csv")
    df = pd.read_csv(features_path)

    latest_gw = int(df["round"].max())
    next_gw = latest_gw + 1

    print(f"Latest completed GW: {latest_gw}")
    print(f"Predicting squad for upcoming GW{next_gw}...\n")

    # Use latest GW data as the base (rolling stats carry forward)
    gw_data = df[df["round"] == latest_gw].copy()
    if gw_data.empty:
        print(f"No data for GW{latest_gw}")
        return None

    # Try to update upcoming FDR from fixtures
    fixtures_path = os.path.join(RAW_DIR, "fixtures.csv")
    if os.path.exists(fixtures_path) and "upcoming_fdr" in gw_data.columns:
        fixtures = pd.read_csv(fixtures_path)
        next_fixtures = fixtures[fixtures["event"] == next_gw]
        if not next_fixtures.empty:
            # Build FDR lookup: team_id -> difficulty they face
            fdr_lookup = {}
            for _, fix in next_fixtures.iterrows():
                fdr_lookup[fix["team_h"]] = fix.get("team_h_difficulty", 3)
                fdr_lookup[fix["team_a"]] = fix.get("team_a_difficulty", 3)
            # Update FDR for each player based on their team
            gw_data["upcoming_fdr"] = gw_data["team"].map(fdr_lookup).fillna(3)
            # Update home/away flag
            home_teams = set(next_fixtures["team_h"].unique())
            away_teams = set(next_fixtures["team_a"].unique())
            gw_data["is_home"] = gw_data["team"].apply(
                lambda t: 1 if t in home_teams else (0 if t in away_teams else 0.5)
            )
            print(f"  Updated FDR and home/away flags from GW{next_gw} fixture list.")

    # Update the round column for clarity
    gw_data["round"] = next_gw

    available = [c for c in feature_cols if c in gw_data.columns]
    gw_clean = gw_data.dropna(subset=available).copy()
    gw_clean["predicted_points"] = model.predict(gw_clean[available].values)

    # Run optimisation
    squad = optimise_squad(gw_clean)
    if squad.empty:
        print("Optimisation returned empty squad.")
        return None

    starting_11, bench, formation = select_starting_11(squad)
    captain, vice_captain = get_captain_picks(starting_11)

    print_squad(squad, starting_11, bench, formation, captain, vice_captain)

    return {
        "squad": squad,
        "starting_11": starting_11,
        "bench": bench,
        "formation": formation,
        "captain": captain,
        "vice_captain": vice_captain,
        "target_gw": next_gw,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--next":
        predict_next_gameweek()
    else:
        run_optimiser()
