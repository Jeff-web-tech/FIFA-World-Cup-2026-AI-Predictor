import requests

url = "https://api.football-data.org/v4/competitions/PL/matches"
headers = {
    "X-Auth-Token": "a62be72c16e24bce90543e81b10bcf7c"
}
response = requests.get(url, headers=headers)
data = response.json()

matches = data["matches"]
for match in matches:
    home = match["homeTeam"]["name"]
    away = match["awayTeam"]["name"]

    print(home, "vs", away)