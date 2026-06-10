import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
import numpy as np
import joblib

df = pd.read_csv("results.csv")

# print(df.columns)

important_columns = [

    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "neutral"

]

df = df[important_columns]

# print(df.head())

def get_result(row):

    if row["home_score"] > row["away_score"]:
        return "H"

    elif row["home_score"] < row["away_score"]:
        return "A"

    else:
        return "D"

df["result"] = df.apply(get_result, axis=1)

# print(df[[
#     "home_team",
#     "away_team",
#     "home_score",
#     "away_score",
#     "result"
# ]].head())

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
# print(df.head())

major_tournaments = [

    "FIFA World Cup",
    "UEFA Euro",
    "Copa America",
    "African Cup of Nations",
    "AFC Asian Cup"

]

df = df[
    df["tournament"].isin(major_tournaments)
]

# print(df["tournament"].value_counts())

tournament_strength = {

    "FIFA World Cup": 5,

    "UEFA Euro": 4,

    "Copa America": 4,

    "African Cup of Nations": 3,

    "AFC Asian Cup": 3,

    "Friendly": 1
}

rankings = pd.read_csv(
    "fifa_rankings.csv"
)

rank_dict = dict(
    zip(
        rankings["Country"],
        rankings["Rank"]
    )
)

df["TournamentImportance"] = 1

#Points Earned in the last 5 matches
df["HomeForm"] = 0.0
df["AwayForm"] = 0.0

#Average goals scored in the last 5 matches
df["HomeGoalsFor"] = 0.0
df["AwayGoalsFor"] = 0.0

#Average goals conceded in the last 5 matches
df["HomeGoalsAgainst"] = 0.0
df["AwayGoalsAgainst"] = 0.0

#Expected Goals
df["HomeXG"] = (
    0.1 * df["home_score"]
)

df["AwayXG"] = (
    0.1 * df["away_score"]
)

df["HomeXGFor"] = 0.0
df["AwayXGFor"] = 0.0

df["HomeXGAgainst"] = 0.0
df["AwayXGAgainst"] = 0.0

df["HomeStrength"] = 0.0
df["AwayStrength"] = 0.0

df["HomeRank"] = 200
df["AwayRank"] = 200

df["RankDifference"] = 0

team_points = {}
team_goals_for = {}
team_goals_against = {}
team_xg_for = {}
team_xg_against = {}
team_strength = {}

for index, row in df.iterrows():

    home_team = row["home_team"]
    away_team = row["away_team"]

    if home_team not in team_points:
        team_points[home_team] = []
        team_goals_for[home_team] = []
        team_goals_against[home_team] = []

    if away_team not in team_points:
        team_points[away_team] = []
        team_goals_for[away_team] = []
        team_goals_against[away_team] = []

    if home_team not in team_xg_for:
        team_xg_for[home_team] = []
        team_xg_against[home_team] = []

    if away_team not in team_xg_for:
        team_xg_for[away_team] = []
        team_xg_against[away_team] = []

    if home_team not in team_strength:
        team_strength[home_team] = 1500

    if away_team not in team_strength:
        team_strength[away_team] = 1500

    home_form = sum(team_points[home_team][-5:])
    away_form = sum(team_points[away_team][-5:])
    df.at[index, "HomeForm"] = home_form
    df.at[index, "AwayForm"] = away_form

    home_goals_for = sum(
        team_goals_for[home_team][-5:]
    ) / max(len(team_goals_for[home_team][-5:]), 1)

    away_goals_for = sum(
        team_goals_for[away_team][-5:]
    ) / max(len(team_goals_for[away_team][-5:]), 1)

    home_goals_against = sum(
        team_goals_against[home_team][-5:]
    ) / max(len(team_goals_against[home_team][-5:]), 1)

    away_goals_against = sum(
        team_goals_against[away_team][-5:]
    ) / max(len(team_goals_against[away_team][-5:]), 1)

    df.at[index, "HomeGoalsFor"] = home_goals_for
    df.at[index, "AwayGoalsFor"] = away_goals_for

    df.at[index, "HomeGoalsAgainst"] = home_goals_against
    df.at[index, "AwayGoalsAgainst"] = away_goals_against

    home_xg_for = sum(
        team_xg_for[home_team][-5:]
    ) / max(len(team_xg_for[home_team][-5:]), 1)

    away_xg_for = sum(
        team_xg_for[away_team][-5:]
    ) / max(len(team_xg_for[away_team][-5:]), 1)

    home_xg_against = sum(
        team_xg_against[home_team][-5:]
    ) / max(len(team_xg_against[home_team][-5:]), 1)

    away_xg_against = sum(
        team_xg_against[away_team][-5:]
    ) / max(len(team_xg_against[away_team][-5:]), 1)

    df.at[index, "HomeXGFor"] = home_xg_for
    df.at[index, "AwayXGFor"] = away_xg_for

    df.at[index, "HomeXGAgainst"] = home_xg_against
    df.at[index, "AwayXGAgainst"] = away_xg_against

    df.at[index, "HomeStrength"] = (
        team_strength[home_team]
    )

    df.at[index, "AwayStrength"] = (
        team_strength[away_team]
    )

    if row["result"] == "H":
        home_points = 3
        away_points = 0

    elif row["result"] == "A":
        home_points = 0
        away_points = 3

    else:
        home_points = 1
        away_points = 1

    team_points[home_team].append(home_points)
    team_points[away_team].append(away_points)

    team_goals_for[home_team].append(row["home_score"])
    team_goals_against[home_team].append(row["away_score"])

    team_goals_for[away_team].append(row["away_score"])
    team_goals_against[away_team].append(row["home_score"])

    team_xg_for[home_team].append(
        row["HomeXG"]
    )

    team_xg_against[home_team].append(
        row["AwayXG"]
    )

    team_xg_for[away_team].append(
        row["AwayXG"]
    )

    team_xg_against[away_team].append(
        row["HomeXG"]
    )

    home_rating = team_strength[home_team]
    away_rating = team_strength[away_team]

    home_advantage = 100
    if row["neutral"]:
        home_advantage = 0

    home_rank = rank_dict.get(
        home_team,
        200
    )

    away_rank = rank_dict.get(
        away_team,
        200
    )

    df.at[index, "HomeRank"] = home_rank
    df.at[index, "AwayRank"] = away_rank

    df.at[index, "RankDifference"] = (
        away_rank - home_rank
    )

    expected_home = 1 / (
        1 + 10 ** (
            (
                away_rating
                -
                (home_rating + home_advantage)
            ) / 400
        )
    )

    expected_away = 1 / (
        1 + 10 ** ((home_rating - away_rating) / 400)
    )

    if row["result"] == "H":
        actual_home = 1
        actual_away = 0

    elif row["result"] == "A":
        actual_home = 0
        actual_away = 1

    else:
        actual_home = 0.5
        actual_away = 0.5

    importance = tournament_strength.get(
        row["tournament"],
        1
    )

    goal_difference = abs(
        row["home_score"]
        - row["away_score"]
    )

    if goal_difference <= 1:
        goal_multiplier = 1

    elif goal_difference == 2:
        goal_multiplier = 1.5

    elif goal_difference == 3:
        goal_multiplier = 1.75

    else:
        goal_multiplier = 2

    df.at[index, "TournamentImportance"] = (
        importance
    )

    k = (
        20 * importance * goal_multiplier
    )

    team_strength[home_team] = (
        home_rating
        + k * (actual_home - expected_home)
    )

    team_strength[away_team] = (
        away_rating
        + k * (actual_away - expected_away)
    )

# print(df[[

#     "home_team",
#     "away_team",

#     "HomeForm",
#     "AwayForm",

#     "HomeGoalsFor",
#     "AwayGoalsFor",

#     "HomeGoalsAgainst",
#     "AwayGoalsAgainst",

#     "result"

# ]].head(20))

features = [

    "HomeForm",
    "AwayForm",

    "HomeGoalsFor",
    "AwayGoalsFor",

    "HomeGoalsAgainst",
    "AwayGoalsAgainst",

    "HomeXGFor",
    "AwayXGFor",

    "HomeXGAgainst",
    "AwayXGAgainst",

    "HomeStrength",
    "AwayStrength",

    "TournamentImportance",

    "HomeRank",
    "AwayRank",
    "RankDifference"

]



imputer = SimpleImputer(strategy="mean")

x = imputer.fit_transform(df[features])

y = df["result"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, shuffle=False
)
# print(x_train.shape)
# print(x_test.shape)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)

model.fit(x_train_scaled, y_train)

predictions = model.predict(x_test_scaled)
probabilities = model.predict_proba(x_test_scaled)
confidence_scores = probabilities.max(
    axis=1
)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, predictions)
print(cm)

print(np.unique(predictions, return_counts=True))

# for i in range(10):

#     print("\nMatch", i + 1)

#     print(
#         "Predicted:",
#         predictions[i]
#     )

#     print(
#         "Actual:",
#         y_test.iloc[i]
#     )

#     print(
#         "Away Win Probability:",
#         round(probabilities[i][0] * 100, 2),
#         "%"
#     )

#     print(
#         "Draw Probability:",
#         round(probabilities[i][1] * 100, 2),
#         "%"
#     )

#     print(
#         "Home Win Probability:",
#         round(probabilities[i][2] * 100, 2),
#         "%"
#     )

results = pd.DataFrame({

    "Actual": y_test.values,

    "Predicted": predictions,

    "Confidence": confidence_scores
})

results = results.sort_values(

    by="Confidence",

    ascending=False
)

strong_predictions = results[

    results["Confidence"] >= 0.70
]

# print(strong_predictions)
# print(df["result"].value_counts())

df.to_csv(
    "engineered_results.csv",
    index=False
)

world_cup_teams = [

    "Argentina",
    "Brazil",
    "France",
    "England",

    "Spain",
    "Germany",
    "Portugal",
    "Netherlands",

    "Belgium",
    "Uruguay",
    "Croatia",
    "Mexico",

    "USA",
    "Japan",
    "Morocco",
    "Senegal"
]

def build_features(home_team, away_team):

    home_matches = df[
        df["home_team"] == home_team
    ]

    away_matches = df[
        df["away_team"] == away_team
    ]

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
    features = build_features(home_team, away_team)
    if features is None:
        return None
    
    scaled = scaler.transform(features)
    probs = model.predict_proba(scaled)[0]

    classes = model.classes_

    result = np.random.choice(
        classes,
        p=probs
    )
    return result

# print(

#     simulate_match(
#         "Argentina",
#         "France"
#     )
# )

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")