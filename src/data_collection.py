"""
FPL Data Collection Module

Pulls data from the official Fantasy Premier League API for the 2024/25 season.
Collects player-level gameweek histories, team stats, and fixture data.
Saves all raw data as CSVs in data/raw/.
"""

import os
import time
import requests
import pandas as pd

BASE_URL = "https://fantasy.premierleague.com/api"
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")


def fetch_bootstrap_static():
    """Fetch the main bootstrap-static endpoint containing all player and team data."""
    url = f"{BASE_URL}/bootstrap-static/"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_player_history(player_id):
    """Fetch gameweek-by-gameweek history for a single player.

    Args:
        player_id: The FPL element ID for the player.

    Returns:
        List of dicts, one per gameweek the player has data for.
    """
    url = f"{BASE_URL}/element-summary/{player_id}/"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("history", [])


def fetch_fixtures():
    """Fetch the full fixture list for the season including FDR ratings."""
    url = f"{BASE_URL}/fixtures/"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def collect_player_master(bootstrap_data):
    """Extract the player master table from bootstrap-static data.

    Args:
        bootstrap_data: The full JSON response from bootstrap-static.

    Returns:
        DataFrame with one row per player and key attributes.
    """
    elements = bootstrap_data["elements"]
    df = pd.DataFrame(elements)

    # Keep the most useful columns
    cols = [
        "id", "first_name", "second_name", "web_name", "team", "element_type",
        "now_cost", "selected_by_percent", "total_points", "minutes",
        "goals_scored", "assists", "clean_sheets", "saves",
        "goals_conceded", "yellow_cards", "red_cards", "bonus",
        "starts", "expected_goals", "expected_assists",
        "expected_goal_involvements", "expected_goals_conceded",
        "form", "points_per_game", "value_form", "value_season",
        "status", "chance_of_playing_next_round",
    ]
    # Only keep columns that exist in the data
    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()

    # Map element_type to position name
    position_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    df["position"] = df["element_type"].map(position_map)

    # Convert cost from tenths of millions to millions
    df["price"] = df["now_cost"] / 10.0

    return df


def collect_team_data(bootstrap_data):
    """Extract team-level data from bootstrap-static.

    Args:
        bootstrap_data: The full JSON response from bootstrap-static.

    Returns:
        DataFrame with one row per team.
    """
    teams = bootstrap_data["teams"]
    df = pd.DataFrame(teams)
    cols = [
        "id", "name", "short_name", "strength", "strength_overall_home",
        "strength_overall_away", "strength_attack_home", "strength_attack_away",
        "strength_defence_home", "strength_defence_away",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()


def collect_gameweek_data(bootstrap_data):
    """Extract gameweek metadata (deadlines, averages, etc.).

    Args:
        bootstrap_data: The full JSON response from bootstrap-static.

    Returns:
        DataFrame with one row per gameweek.
    """
    events = bootstrap_data["events"]
    df = pd.DataFrame(events)
    cols = [
        "id", "name", "deadline_time", "average_entry_score",
        "highest_score", "most_selected", "most_transferred_in",
        "most_captained", "most_vice_captained", "finished",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()


def collect_all_player_histories(player_ids, delay=0.5):
    """Fetch gameweek history for every player, with rate limiting.

    Args:
        player_ids: List of player element IDs to fetch.
        delay: Seconds to wait between API calls (default 0.5).

    Returns:
        DataFrame with all players' gameweek histories combined.
    """
    all_rows = []
    total = len(player_ids)

    for i, pid in enumerate(player_ids):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Fetching player {i + 1}/{total} (id={pid})...")

        try:
            history = fetch_player_history(pid)
            for row in history:
                row["player_id"] = pid
            all_rows.extend(history)
        except requests.exceptions.RequestException as e:
            print(f"  WARNING: Failed to fetch player {pid}: {e}")

        time.sleep(delay)

    df = pd.DataFrame(all_rows)
    return df


def collect_fixtures_df():
    """Fetch fixtures and return as a DataFrame.

    Returns:
        DataFrame with one row per fixture including FDR ratings.
    """
    fixtures = fetch_fixtures()
    df = pd.DataFrame(fixtures)
    cols = [
        "id", "event", "team_h", "team_a", "team_h_score", "team_a_score",
        "team_h_difficulty", "team_a_difficulty", "finished",
        "kickoff_time",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()


def compute_team_stats(player_gw_df, fixtures_df, teams_df):
    """Compute team-level per-game stats from fixture results.

    Args:
        player_gw_df: Player gameweek history DataFrame (unused but available).
        fixtures_df: Fixtures DataFrame with scores.
        teams_df: Teams DataFrame with team IDs and names.

    Returns:
        DataFrame with per-team stats: goals scored/conceded per game, home/away splits.
    """
    finished = fixtures_df[fixtures_df["finished"] == True].copy()
    if finished.empty:
        print("  WARNING: No finished fixtures found for team stats.")
        return pd.DataFrame()

    rows = []
    for _, team in teams_df.iterrows():
        tid = team["id"]
        tname = team["name"]

        home = finished[finished["team_h"] == tid]
        away = finished[finished["team_a"] == tid]

        home_gf = home["team_h_score"].sum() if not home.empty else 0
        home_ga = home["team_a_score"].sum() if not home.empty else 0
        away_gf = away["team_a_score"].sum() if not away.empty else 0
        away_ga = away["team_h_score"].sum() if not away.empty else 0

        total_games = len(home) + len(away)
        if total_games == 0:
            continue

        rows.append({
            "team_id": tid,
            "team_name": tname,
            "games_played": total_games,
            "home_games": len(home),
            "away_games": len(away),
            "total_goals_scored": home_gf + away_gf,
            "total_goals_conceded": home_ga + away_ga,
            "goals_scored_per_game": round((home_gf + away_gf) / total_games, 2),
            "goals_conceded_per_game": round((home_ga + away_ga) / total_games, 2),
            "home_goals_scored_per_game": round(home_gf / len(home), 2) if len(home) > 0 else 0,
            "home_goals_conceded_per_game": round(home_ga / len(home), 2) if len(home) > 0 else 0,
            "away_goals_scored_per_game": round(away_gf / len(away), 2) if len(away) > 0 else 0,
            "away_goals_conceded_per_game": round(away_ga / len(away), 2) if len(away) > 0 else 0,
        })

    return pd.DataFrame(rows)


def run_collection():
    """Run the full data collection pipeline and save all CSVs to data/raw/."""
    os.makedirs(RAW_DIR, exist_ok=True)

    # 1. Bootstrap static
    print("Fetching bootstrap-static data...")
    bootstrap = fetch_bootstrap_static()

    # 2. Player master
    print("Processing player master table...")
    players_df = collect_player_master(bootstrap)
    players_path = os.path.join(RAW_DIR, "players.csv")
    players_df.to_csv(players_path, index=False)
    print(f"  Saved {len(players_df)} players to {players_path}")

    # 3. Teams
    print("Processing team data...")
    teams_df = collect_team_data(bootstrap)
    teams_path = os.path.join(RAW_DIR, "teams.csv")
    teams_df.to_csv(teams_path, index=False)
    print(f"  Saved {len(teams_df)} teams to {teams_path}")

    # 4. Gameweek metadata
    print("Processing gameweek metadata...")
    gw_df = collect_gameweek_data(bootstrap)
    gw_path = os.path.join(RAW_DIR, "gameweeks.csv")
    gw_df.to_csv(gw_path, index=False)
    print(f"  Saved {len(gw_df)} gameweeks to {gw_path}")

    # 5. Fixtures
    print("Fetching fixtures...")
    fixtures_df = collect_fixtures_df()
    fixtures_path = os.path.join(RAW_DIR, "fixtures.csv")
    fixtures_df.to_csv(fixtures_path, index=False)
    print(f"  Saved {len(fixtures_df)} fixtures to {fixtures_path}")

    # 6. Player gameweek histories (this takes a while)
    print("Fetching all player gameweek histories (this may take 10-15 minutes)...")
    player_ids = players_df["id"].tolist()
    gw_history_df = collect_all_player_histories(player_ids, delay=0.5)
    gw_history_path = os.path.join(RAW_DIR, "player_gw_history.csv")
    gw_history_df.to_csv(gw_history_path, index=False)
    print(f"  Saved {len(gw_history_df)} gameweek records to {gw_history_path}")

    # 7. Team stats
    print("Computing team-level stats...")
    team_stats_df = compute_team_stats(gw_history_df, fixtures_df, teams_df)
    team_stats_path = os.path.join(RAW_DIR, "team_stats.csv")
    team_stats_df.to_csv(team_stats_path, index=False)
    print(f"  Saved {len(team_stats_df)} team stat rows to {team_stats_path}")

    print("\n=== Data collection complete! ===")
    print(f"Files saved in: {RAW_DIR}")
    print(f"  - players.csv: {len(players_df)} players")
    print(f"  - teams.csv: {len(teams_df)} teams")
    print(f"  - gameweeks.csv: {len(gw_df)} gameweeks")
    print(f"  - fixtures.csv: {len(fixtures_df)} fixtures")
    print(f"  - player_gw_history.csv: {len(gw_history_df)} records")
    print(f"  - team_stats.csv: {len(team_stats_df)} teams")

    return {
        "players": players_df,
        "teams": teams_df,
        "gameweeks": gw_df,
        "fixtures": fixtures_df,
        "player_gw_history": gw_history_df,
        "team_stats": team_stats_df,
    }


if __name__ == "__main__":
    run_collection()
