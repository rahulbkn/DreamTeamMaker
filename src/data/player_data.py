import requests
from typing import List, Dict
import sqlite3
from datetime import datetime, timedelta
import json
import numpy as np

CRICAPI_KEY = "4a1c0cf5-f217-44e4-aa8f-381ac1b529ea"
CRICAPI_URL = "https://api.cricapi.com/v1/"
SPORTS_RADAR_KEY = "5u6P8br5tuEVwHclQenkU3yA51HIKmMGb1hQh2ol"
SPORTS_RADAR_URL = "https://api.sportradar.us/cricket-t2/en/"

DB_FILE = "player_stats.db"
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS players (
    name TEXT PRIMARY KEY, role TEXT, team TEXT, credits REAL, avg_points REAL, form_factor REAL,
    ownership REAL, available INTEGER, recent_points TEXT, venue_avg REAL, matches_played_last_30_days INTEGER,
    last_updated DATETIME, injury_status TEXT, injury_history TEXT, fitness_score REAL, fitness_history TEXT,
    last_match_date TEXT
)''')
conn.commit()

def fetch_cricapi_data(match_id: str) -> Dict:
    response = requests.get(f"{CRICAPI_URL}currentMatches?apikey={CRICAPI_KEY}&match_id={match_id}")
    return response.json() if response.status_code == 200 else {}

def fetch_sportsradar_data(match_id: str) -> Dict:
    response = requests.get(f"{SPORTS_RADAR_URL}matches/{match_id}/summary.json?api_key={SPORTS_RADAR_KEY}")
    return response.json() if response.status_code == 200 else {}

def fetch_fitness_data(player_name: str) -> Dict:
    response = requests.get(f"{SPORTS_RADAR_URL}players/{player_name}/profile.json?api_key={SPORTS_RADAR_KEY}")
    if response.status_code == 200:
        data = response.json()
        return {
            "fitness_score": data.get("fitness_score", 80.0),
            "workload": data.get("workload", 5),
            "last_match_date": data.get("last_match_date", str(datetime.now() - timedelta(days=2))),
            "injury_status": data.get("injury_status", "fit")
        }
    return {"fitness_score": 80.0, "workload": 5, "last_match_date": str(datetime.now() - timedelta(days=2)), "injury_status": "fit"}

def calculate_fitness_score(player: Dict) -> float:
    days_since_last_match = (datetime.now() - datetime.strptime(player["last_match_date"], "%Y-%m-%d %H:%M:%S.%f")).days
    workload_penalty = min(20, player["matches_played_last_30_days"]) / 20
    rest_bonus = min(10, days_since_last_match) / 10
    injury_factor = 0.5 if player["injury_status"] != "fit" else 1.0
    return 100 * (0.4 * (1 - workload_penalty) + 0.4 * rest_bonus + 0.2 * injury_factor)

def sync_player_data(players: List[Dict], pitch: str, weather: str, match_id: str = None, city: str = "Unknown") -> List[Dict]:
    live_data = {}
    if match_id:
        cricapi_data = fetch_cricapi_data(match_id)
        sportsradar_data = fetch_sportsradar_data(match_id)
        live_data = cricapi_data.get("data", {}) or sportsradar_data.get("match", {})
    
    for player in players:
        live_player = next((p for p in live_data.get("players", []) if p["name"] == player["name"]), None)
        fitness_data = fetch_fitness_data(player["name"])
        
        if live_player:
            player["recent_points"] = live_player.get("recent_points_list", player.get("recent_points", [player["avg_points"]] * 5))
            player["form_factor"] = min(1.5, max(0.5, np.mean(player["recent_points"][-5:]) / player["avg_points"]))
            player["venue_avg"] = live_player.get("venue_avg", player.get("venue_avg", player["avg_points"]))
            player["matches_played_last_30_days"] = live_player.get("matches_played", player.get("matches_played_last_30_days", 0))
        
        player["injury_status"] = fitness_data.get("injury_status", player.get("injury_status", "fit"))
        player["injury_history"] = player.get("injury_history", "[]")
        player["fitness_score"] = calculate_fitness_score(player) if "fitness_score" not in fitness_data else fitness_data["fitness_score"]
        player["last_match_date"] = fitness_data["last_match_date"]
        player["fitness_history"] = player.get("fitness_history", "[]")
        fitness_history = json.loads(player["fitness_history"])
        fitness_history.append({"score": player["fitness_score"], "date": str(datetime.now())})
        player["fitness_history"] = json.dumps(fitness_history[-10:])
        player["available"] = player["injury_status"] == "fit" and player["fitness_score"] > 50
        
        recent_points_str = ','.join(map(str, player["recent_points"]))
        c.execute('''INSERT OR REPLACE INTO players (name, role, team, credits, avg_points, form_factor, ownership, available,
                     recent_points, venue_avg, matches_played_last_30_days, last_updated, injury_status, injury_history,
                     fitness_score, fitness_history, last_match_date)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (player["name"], player["role"], player["team"], player["credits"], player["avg_points"], player["form_factor"],
                   player["ownership"], int(player["available"]), recent_points_str, player["venue_avg"],
                   player["matches_played_last_30_days"], datetime.now(), player["injury_status"], player["injury_history"],
                   player["fitness_score"], player["fitness_history"], player["last_match_date"]))
    conn.commit()
    return players

def load_players() -> List[Dict]:
    c.execute("SELECT * FROM players")
    rows = c.fetchall()
    if rows:
        return [{
            "name": row[0], "role": row[1], "team": row[2], "credits": row[3], "avg_points": row[4],
            "form_factor": row[5], "ownership": row[6], "available": bool(row[7]),
            "recent_points": list(map(float, row[8].split(','))), "venue_avg": row[9],
            "matches_played_last_30_days": row[10], "injury_status": row[12], "injury_history": row[13],
            "fitness_score": row[14], "fitness_history": row[15], "last_match_date": row[16]
        } for row in rows]
    players = [
        {"name": "Player A", "role": "BAT", "team": "Team1", "credits": 9.0, "avg_points": 50, "form_factor": 1.0, "ownership": 20, "available": True,
         "recent_points": [50, 55, 45, 60, 40], "venue_avg": 52, "matches_played_last_30_days": 5, "injury_status": "fit", "injury_history": "[]",
         "fitness_score": 80.0, "fitness_history": "[]", "last_match_date": str(datetime.now() - timedelta(days=2))},
        {"name": "Player B", "role": "BOW", "team": "Team2", "credits": 8.5, "avg_points": 45, "form_factor": 1.0, "ownership": 15, "available": True,
         "recent_points": [40, 50, 35, 55, 30], "venue_avg": 47, "matches_played_last_30_days": 3, "injury_status": "fit", "injury_history": "[]",
         "fitness_score": 85.0, "fitness_history": "[]", "last_match_date": str(datetime.now() - timedelta(days=3))},
    ]
    sync_player_data(players, "batting", "sunny")
    return players