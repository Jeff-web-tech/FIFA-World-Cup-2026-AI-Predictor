# app.py

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import sqlite3
from datetime import datetime
import difflib

app = Flask(__name__)


# Helper: resolve team name against dataset (aliases + fuzzy)
def resolve_team_name(name):
    aliases = {
        "USA": "United States",
        "US": "United States",
        "United States of America": "United States",
        "Turkiye": "Turkey",
        "Cabo Verde": "Cape Verde",
        "Curacao": "Curaçao",
        "Czechia": "Czech Republic",
    }

    if name in aliases:
        return aliases[name]

    unique_names = set(data["home_team"]).union(set(data["away_team"]))
    if name in unique_names:
        return name

    lower_map = {n.lower(): n for n in unique_names}
    if name.lower() in lower_map:
        return lower_map[name.lower()]

    choices = difflib.get_close_matches(name, list(unique_names), n=1, cutoff=0.8)
    if choices:
        return choices[0]

    return None

# =========================
# LOAD MODEL + SCALER
# =========================

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# LOAD DATA
# =========================

data = pd.read_csv("engineered_results.csv")

# =========================
# DATABASE
# =========================

def init_db():

    connection = sqlite3.connect(
        "predictions.db"
    )

    cursor = connection.cursor()

    cursor.execute("""

        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            timestamp TEXT,

            home_team TEXT,

            away_team TEXT,

            prediction TEXT,

            home_prob REAL,

            draw_prob REAL,

            away_prob REAL

        )

    """)

    connection.commit()

    connection.close()

# =========================
# FLAGS
# =========================

flags = {

    "Argentina": "ARG",
    "Brazil": "BRA",
    "France": "FRA",
    "England": "ENG",
    "Spain": "ESP",
    "Germany": "GER",
    "Portugal": "POR",
    "Belgium": "BEL",
    "USA": "USA",
    "Mexico": "MEX",
    "Japan": "JPN",
    "Morocco": "MAR",
    "Croatia": "CRO",
    "Uruguay": "URU",
    "Senegal": "SEN",
    "Netherlands": "NED",
    "Sweden": "SWE",
    "South Africa": "SA",
    "South Korea": "RSA",
    "Czechia": "CZE",
    "Canada": "CAN",
    "Bosnia and Herzegovina": "BIH",
    "Qatar": "QAT",
    "Switzerland": "SUI",
    "Haiti": "HAI",
    "Scotland": "SCO",
    "Paraguay": "PAR",
    "Australia": "AUS",
    "Turkiye": "TUR",
    "Curacao": "CUW",
    "Ivory Coast": "CIV",
    "Ecuador": "ECU",
    "Tunisia": "TUN",
    "Egypt": "EGY",
    "Iran": "IRN",
    "New Zealand": "NZL",
    "Cabo Verde": "CPV",
    "Saudi Arabia": "KSA",
    "Iraq": "IRQ",
    "Norway": "NOR",
    "Algeria": "ALG",
    "Austria": "AUT",
    "Jordan": "JOR",
    "DR Congo": "COD",
    "Uzbekistan": "UZB",
    "Colombia": "COL",
    "Ghana": "GHA",
    "Panama": "PAN"
}

# =========================
# GROUPS (World Cup 2026)
# =========================

groups = {
    "Group A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "Group B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "Group C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "Group D": ["USA", "Paraguay", "Australia", "Turkiye"],
    "Group E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "Group F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "Group G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "Group H": ["Spain", "Cabo Verde", "Saudi Arabia", "Uruguay"],
    "Group I": ["France", "Senegal", "Iraq", "Norway"],
    "Group J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "Group K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "Group L": ["England", "Croatia", "Ghana", "Panama"]
}

# Create reverse mapping for teams to groups
team_to_group = {}
for group_name, teams in groups.items():
    for team in teams:
        team_to_group[team] = group_name

# =========================
# AUTO-GENERATED FIXTURES (Group Stage)
# =========================

group_match_schedule = {
    "Group A": [
        ("2026-06-11", "18:00"),
        ("2026-06-11", "21:00"),
        ("2026-06-16", "18:00"),
        ("2026-06-16", "21:00"),
        ("2026-06-21", "18:00"),
        ("2026-06-21", "21:00")
    ],
    "Group B": [
        ("2026-06-12", "15:00"),
        ("2026-06-12", "18:00"),
        ("2026-06-17", "15:00"),
        ("2026-06-17", "18:00"),
        ("2026-06-22", "15:00"),
        ("2026-06-22", "18:00")
    ],
    "Group C": [
        ("2026-06-11", "15:00"),
        ("2026-06-11", "18:00"),
        ("2026-06-16", "15:00"),
        ("2026-06-16", "21:00"),
        ("2026-06-21", "15:00"),
        ("2026-06-21", "21:00")
    ],
    "Group D": [
        ("2026-06-12", "21:00"),
        ("2026-06-13", "18:00"),
        ("2026-06-17", "21:00"),
        ("2026-06-18", "18:00"),
        ("2026-06-22", "21:00"),
        ("2026-06-23", "18:00")
    ],
    "Group E": [
        ("2026-06-13", "15:00"),
        ("2026-06-13", "21:00"),
        ("2026-06-18", "15:00"),
        ("2026-06-18", "21:00"),
        ("2026-06-23", "15:00"),
        ("2026-06-23", "21:00")
    ],
    "Group F": [
        ("2026-06-14", "15:00"),
        ("2026-06-14", "18:00"),
        ("2026-06-19", "15:00"),
        ("2026-06-19", "18:00"),
        ("2026-06-24", "15:00"),
        ("2026-06-24", "18:00")
    ],
    "Group G": [
        ("2026-06-14", "21:00"),
        ("2026-06-15", "15:00"),
        ("2026-06-19", "21:00"),
        ("2026-06-20", "15:00"),
        ("2026-06-24", "21:00"),
        ("2026-06-25", "15:00")
    ],
    "Group H": [
        ("2026-06-15", "18:00"),
        ("2026-06-15", "21:00"),
        ("2026-06-20", "18:00"),
        ("2026-06-20", "21:00"),
        ("2026-06-25", "18:00"),
        ("2026-06-25", "21:00")
    ],
    "Group I": [
        ("2026-06-16", "18:00"),
        ("2026-06-16", "21:00"),
        ("2026-06-21", "18:00"),
        ("2026-06-21", "21:00"),
        ("2026-06-26", "18:00"),
        ("2026-06-26", "21:00")
    ],
    "Group J": [
        ("2026-06-17", "15:00"),
        ("2026-06-17", "18:00"),
        ("2026-06-22", "15:00"),
        ("2026-06-22", "18:00"),
        ("2026-06-27", "15:00"),
        ("2026-06-27", "18:00")
    ],
    "Group K": [
        ("2026-06-17", "21:00"),
        ("2026-06-18", "15:00"),
        ("2026-06-23", "21:00"),
        ("2026-06-24", "15:00"),
        ("2026-06-27", "21:00"),
        ("2026-06-28", "15:00")
    ],
    "Group L": [
        ("2026-06-18", "18:00"),
        ("2026-06-18", "21:00"),
        ("2026-06-24", "18:00"),
        ("2026-06-24", "21:00"),
        ("2026-06-28", "18:00"),
        ("2026-06-28", "21:00")
    ]
}

fixtures = []
for group_name, teams in groups.items():
    schedule = group_match_schedule.get(group_name, [("TBA", "TBA")] * 6)
    match_idx = 0
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            date, time = schedule[match_idx]
            fixtures.append({
                "home": teams[i],
                "away": teams[j],
                "date": date,
                "time": time,
                "group": group_name
            })
            match_idx += 1

# =========================
# FEATURE ENGINEERING
# =========================

def build_features(home_team, away_team):

    home_resolved = resolve_team_name(home_team)
    away_resolved = resolve_team_name(away_team)

    home_matches = data[ data["home_team"] == home_resolved ] if home_resolved is not None else pd.DataFrame()
    away_matches = data[ data["away_team"] == away_resolved ] if away_resolved is not None else pd.DataFrame()

    if len(home_matches) == 0 or len(away_matches) == 0:

        return None

    home_data = home_matches.iloc[-1]
    away_data = away_matches.iloc[-1]

    features = [[

        home_data["HomeForm"],
        away_data["AwayForm"],

        home_data["HomeGoalsFor"],
        away_data["AwayGoalsFor"],

        home_data["HomeGoalsAgainst"],
        away_data["AwayGoalsAgainst"],

        home_data["HomeXGFor"],
        away_data["AwayXGFor"],

        home_data["HomeXGAgainst"],
        away_data["AwayXGAgainst"],

        home_data["HomeStrength"],
        away_data["AwayStrength"],

        home_data["TournamentImportance"]

    ]]

    return features

def simulate_match(home_team, away_team):
    """Backward-compatible wrapper: returns 'H', 'A' or 'D' using the score simulator."""
    score = simulate_score(home_team, away_team)
    if score is None:
        return None
    if score["home_goals"] > score["away_goals"]:
        return "H"
    elif score["away_goals"] > score["home_goals"]:
        return "A"
    else:
        return "D"


def simulate_score(home_team, away_team, knockout=False, force_result=None):
    """Simulate a match scoreline using Poisson goals derived from recent xG stats.

    Returns dict: {home_goals, away_goals, result, decided_by}
    """
    # Resolve names and fetch latest stats
    home_resolved = resolve_team_name(home_team)
    away_resolved = resolve_team_name(away_team)

    if home_resolved is None or away_resolved is None:
        return None

    home_matches = data[ data["home_team"] == home_resolved ]
    away_matches = data[ data["away_team"] == away_resolved ]

    if len(home_matches) == 0 or len(away_matches) == 0:
        return None

    home_data = home_matches.iloc[-1]
    away_data = away_matches.iloc[-1]

    # Estimate expected goals (lambda) using recent xG metrics when available
    # Fall back to simple goals-for/against if xG missing
    try:
        lam_home = np.nanmean([
            float(home_data.get("HomeXGFor", np.nan)),
            float(away_data.get("AwayXGAgainst", np.nan))
        ])
    except Exception:
        lam_home = np.nan

    try:
        lam_away = np.nanmean([
            float(away_data.get("AwayXGFor", np.nan)),
            float(home_data.get("HomeXGAgainst", np.nan))
        ])
    except Exception:
        lam_away = np.nan

    if np.isnan(lam_home) or lam_home <= 0:
        lam_home = max(0.4, float(home_data.get("HomeGoalsFor", 1.0)) * 0.6)
    if np.isnan(lam_away) or lam_away <= 0:
        lam_away = max(0.4, float(away_data.get("AwayGoalsFor", 1.0)) * 0.6)

    # Small home/neutral adjustment: group matches are neutral, keep as is

    # Sample goals from Poisson
    home_goals = int(np.random.poisson(lam_home))
    away_goals = int(np.random.poisson(lam_away))

    decided_by = "90"

    # Knockout handling: simulate extra time and penalties if required
    if knockout and home_goals == away_goals:
        # Extra time: scale lambdas to 30 minutes (1/3 of 90)
        lam_home_et = max(0.05, lam_home / 3.0)
        lam_away_et = max(0.05, lam_away / 3.0)

        et_home = int(np.random.poisson(lam_home_et))
        et_away = int(np.random.poisson(lam_away_et))

        home_goals += et_home
        away_goals += et_away

        decided_by = "ET"

        if home_goals == away_goals:
            # Penalty shootout: decide by strength or 50/50
            hs = float(home_data.get("HomeStrength", 1500))
            as_ = float(away_data.get("AwayStrength", 1500))
            p_home = hs / (hs + as_) if (hs + as_) > 0 else 0.5
            winner = np.random.choice(["home", "away"], p=[p_home, 1 - p_home])
            if winner == "home":
                home_goals += 1  # mark winner with +1 (not a real goal)
            else:
                away_goals += 1
            decided_by = "PEN"

    if force_result == "H" and home_goals <= away_goals:
        home_goals = away_goals + 1
    elif force_result == "A" and away_goals <= home_goals:
        away_goals = home_goals + 1
    elif force_result == "D" and home_goals != away_goals:
        away_goals = home_goals

    result = "D"
    if home_goals > away_goals:
        result = "H"
    elif away_goals > home_goals:
        result = "A"

    return {
        "home_goals": int(home_goals),
        "away_goals": int(away_goals),
        "result": result,
        "decided_by": decided_by
    }

#Group Stage Simulator
def simulate_group(group_teams):

    table = {}

    for team in group_teams:

        table[team] = {
            "points": 0,
            "gf": 0,
            "ga": 0
        }

    matches = []

    for i in range(len(group_teams)):

        for j in range(i + 1, len(group_teams)):

            home = group_teams[i]
            away = group_teams[j]

            score = simulate_score(home, away, knockout=False)

            if score is None:
                # if simulation failed, skip match
                continue

            hg = score["home_goals"]
            ag = score["away_goals"]

            table[home]["gf"] += hg
            table[home]["ga"] += ag

            table[away]["gf"] += ag
            table[away]["ga"] += hg

            if score["result"] == "H":
                table[home]["points"] += 3
            elif score["result"] == "A":
                table[away]["points"] += 3
            else:
                table[home]["points"] += 1
                table[away]["points"] += 1

            matches.append({
                "home": home,
                "away": away,
                "home_goals": hg,
                "away_goals": ag,
                "result": score["result"],
                "decided_by": score["decided_by"]
            })

    # Sort by points, goal difference, goals for
    standings = sorted(
        table.items(),
        key=lambda x: (
            x[1]["points"],
            x[1]["gf"] - x[1]["ga"],
            x[1]["gf"]
        ),
        reverse=True
    )

    return {
        "standings": standings,
        "matches": matches
    }


# Helper: rank third-placed teams across all groups
def rank_third_placed(group_results):

    thirds = []

    for group_name, group_data in group_results.items():
        standings = group_data.get("standings", [])
        if len(standings) >= 3:
            team, stats = standings[2]
            # use points then goal difference then goals for
            pts = stats.get("points", 0)
            gd = stats.get("gf", 0) - stats.get("ga", 0)
            gf = stats.get("gf", 0)

            thirds.append({
                "group": group_name,
                "team": team,
                "points": pts,
                "gd": gd,
                "gf": gf
            })

    # sort by FIFA criteria: points, goal difference, goals for
    thirds_sorted = sorted(
        thirds,
        key=lambda x: (x["points"], x["gd"], x["gf"]),
        reverse=True
    )

    return thirds_sorted


# FIFA Round-of-32 combination table placeholder.
# The key is a sorted tuple of group names that finished third (e.g. ("A","B","C",...)).
# The value should map Round-of-32 slots that expect a third-placed team to the supplying group.
# Example entry (NOT real):
# ("A","B","C","D","E","F","G","H") : {
#     "slot_1": "C",
#     "slot_2": "D",
#     ...
# }
# You must populate this with the official FIFA mapping for exact behavior.
FIFA_R32_COMBINATIONS = {
    # TODO: fill with official FIFA combination table mapping
}


def build_round_of_32(group_results):

    # Top two from each group in order (A-L)
    winners = {}
    runners = {}

    for group_name, standings in group_results.items():
        winners[group_name] = standings[0][0]
        runners[group_name] = standings[1][0]

    # Rank third-placed teams and select best 8
    thirds_sorted = rank_third_placed(group_results)
    best_thirds = thirds_sorted[:8]
    third_groups = [t["group"] for t in best_thirds]

    # Try to use FIFA official mapping if available
    key = tuple(sorted(third_groups))

    r32 = []

    if key in FIFA_R32_COMBINATIONS:
        mapping = FIFA_R32_COMBINATIONS[key]
        # mapping should specify which group supplies each "3X" slot
        # We'll build pairs according to FIFA official bracket using mapping
        # (Implementation depends on exact mapping format)
        # For now assume mapping maps strings like "3a"->"GroupC" etc.
        for slot, group_source in mapping.items():
            # slot should indicate which first/second it faces; skip detailed parsing here
            pass
        # NOTE: detailed implementation requires the official table format
    else:
        # Fallback deterministic fill: place best thirds into predefined slots.
        # This is NOT the official FIFA mapping; it's a sensible deterministic fallback.
        slot_order = [
            ("1A", "3"), ("1B", "3"), ("1C", "3"), ("1D", "3"),
            ("1E", "3"), ("1F", "3"), ("1G", "3"), ("1H", "3"),
            ("1I", "3"), ("1J", "3"), ("1K", "3"), ("1L", "3")
        ]

        # combine winners and best_thirds into R32 pairs: winner vs best_third in order
        # if fewer thirds than slots, match winners vs runners as fallback
        for i, group_name in enumerate(sorted(groups.keys())):
            # match winner of group_name vs either best_thirds[i] or runner-up of next group
            home = winners[group_name]
            if i < len(best_thirds):
                away = best_thirds[i]["team"]
            else:
                # fallback: face runner-up of next group in list
                next_groups = list(groups.keys())
                away_group = next_groups[(i + 1) % len(next_groups)]
                away = runners[away_group]

            r32.append({"home": home, "away": away})

    return r32

#Full Tournament Simulator
def simulate_tournament():

    qualified = []

    group_results = {}

    # GROUP STAGE

    for group_name, teams in groups.items():

        group_data = simulate_group(teams)

        group_results[group_name] = group_data

        qualified.append(
            group_data["standings"][0][0]
        )

        qualified.append(
            group_data["standings"][1][0]
        )

    # QUARTER FINALS

    quarter_pairs = [

        (qualified[0], qualified[3]),
        (qualified[1], qualified[2]),

        (qualified[4], qualified[7]),
        (qualified[5], qualified[6])
    ]

    semi_finalists = []

    quarter_results = []

    for home, away in quarter_pairs:

        score = simulate_score(home, away, knockout=True)
        if score is None:
            continue

        if score["result"] == "H":
            winner = home

        elif score["result"] == "A":
            winner = away

        else:
            winner = np.random.choice(
                [home, away]
            )

        semi_finalists.append(
            winner
        )

        quarter_results.append({

            "home": home,
            "away": away,
            "home_goals": score["home_goals"],
            "away_goals": score["away_goals"],
            "decided_by": score["decided_by"],
            "winner": winner
        })

    # SEMI FINALS

    semi_pairs = [

        (
            semi_finalists[0],
            semi_finalists[1]
        ),

        (
            semi_finalists[2],
            semi_finalists[3]
        )
    ]

    finalists = []

    semi_results = []

    for home, away in semi_pairs:

        score = simulate_score(home, away, knockout=True)
        if score is None:
            continue

        if score["result"] == "H":
            winner = home

        elif score["result"] == "A":
            winner = away

        else:
            winner = np.random.choice(
                [home, away]
            )

        finalists.append(winner)

        semi_results.append({

            "home": home,
            "away": away,
            "home_goals": score["home_goals"],
            "away_goals": score["away_goals"],
            "decided_by": score["decided_by"],
            "winner": winner
        })

    # FINAL

    final_home = finalists[0]
    final_away = finalists[1]

    score = simulate_score(final_home, final_away, knockout=True)
    if score is None:
        champion = final_home
        score = {"home_goals": 0, "away_goals": 0, "decided_by": "90"}
    else:
        if score["result"] == "H":
            champion = final_home

        elif score["result"] == "A":
            champion = final_away

        else:
            champion = np.random.choice(
                [final_home, final_away]
            )

    final_data = {

        "home": final_home,

        "away": final_away,

        "home_goals": score["home_goals"],

        "away_goals": score["away_goals"],

        "decided_by": score["decided_by"],

        "winner": champion
    }

    return {

        "groups": group_results,

        "qualified": qualified,

        "quarters": quarter_results,

        "semis": semi_results,

        "final": final_data,

        "champion": champion
    }

#Monte Carlo Simulation
def calculate_champion_odds(simulations=100):

    champions = {}

    for i in range(simulations):

        tournament = simulate_tournament()

        winner = tournament["champion"]

        if winner not in champions:

            champions[winner] = 0

        champions[winner] += 1

    odds = []

    for team, wins in champions.items():

        percentage = round(

            (wins / simulations) * 100,

            2
        )

        odds.append({

            "team": team,

            "percentage": percentage
        })

    odds = sorted(

        odds,

        key=lambda x: x["percentage"],

        reverse=True
    )

    return odds


# HOME PAGE

@app.route("/")
def home():

    connection = sqlite3.connect(
        "predictions.db"
    )

    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""

        SELECT *

        FROM predictions

        ORDER BY id DESC

        LIMIT 10

    """)

    history = cursor.fetchall()

    connection.close()

    return render_template(

        "index.html",

        fixtures=fixtures,

        groups=groups,

        team_to_group=team_to_group,

        history=history,

        flags=flags
    )


# API PREDICTION ROUTE

@app.route("/api/groups")
def api_groups():

    return jsonify({
        "groups": groups,
        "team_to_group": team_to_group
    })


@app.route("/api/featured-matches")
def api_featured_matches():

    # Get the most important matches (first matches of each group)
    featured = [f for f in fixtures if f["date"] in ["2026-06-11", "2026-06-12", "2026-06-13", "2026-06-14"]]

    return jsonify({
        "featured": featured,
        "total_matches": len(fixtures)
    })

# API PREDICTION ROUTE

@app.route("/api/predict", methods=["POST"])
def api_predict():

    print("API HIT")

    data_json = request.get_json()

    print(data_json)

    home_team = data_json["home_team"]
    away_team = data_json["away_team"]

    features = build_features(
        home_team,
        away_team
    )

    if features is None:

        return jsonify({

            "error": "Team data not found"
        })
    
    print(features)

    scaled_features = scaler.transform(
        features
    )

    prediction = model.predict(
        scaled_features
    )[0]

    probabilities = model.predict_proba(
        scaled_features
    )[0]

    prob_map = {}

    for i, cls in enumerate(model.classes_):

        prob_map[cls] = round(
            probabilities[i] * 100,
            2
        )

    away_prob = prob_map.get("A", 0)

    draw_prob = prob_map.get("D", 0)

    home_prob = prob_map.get("H", 0)

    # SAVE TO DATABASE

    connection = sqlite3.connect(
        "predictions.db"
    )

    cursor = connection.cursor()

    cursor.execute("""

        INSERT INTO predictions (

            timestamp,
            home_team,
            away_team,
            prediction,
            home_prob,
            draw_prob,
            away_prob

        )

        VALUES (?, ?, ?, ?, ?, ?, ?)

    """, (

        str(datetime.now()),

        home_team,
        away_team,

        prediction,

        home_prob,
        draw_prob,
        away_prob
    ))

    connection.commit()

    connection.close()

    score = simulate_score(home_team, away_team, force_result=prediction)
    home_goals = score["home_goals"] if score is not None else None
    away_goals = score["away_goals"] if score is not None else None
    decided_by = score["decided_by"] if score is not None else None

    return jsonify({

        "prediction": prediction,

        "home_prob": home_prob,

        "draw_prob": draw_prob,

        "away_prob": away_prob,

        "home_goals": home_goals,

        "away_goals": away_goals,

        "decided_by": decided_by
    })

# SIMULATOR ROUTE
@app.route("/simulate-world-cup")
def simulate_world_cup():

    results = simulate_tournament()

    champion_odds = calculate_champion_odds(
        simulations=200
    )

    return render_template(

        "simulation.html",

        results=results,

        champion_odds=champion_odds
    )

# =========================
# START APP
# =========================

init_db()

if __name__ == "__main__":

    app.run(debug=True)