from typing import List, Dict
from src.models.predictor import predict_points, predict_injury_risk, predict_fitness_score

def generate_team(players: List[Dict], pitch: str, weather: str, must_include: List[str], risk_level: str) -> List[Dict]:
    match_context = {"opponent_strength": 1.0, "toss_winner": ""}
    available_players = [
        p for p in players 
        if p["available"] and predict_injury_risk(p, pitch, weather, match_context) < 0.5 and predict_fitness_score(p, pitch, weather, match_context) > 60
    ]
    must_include_players = [p for p in available_players if p["name"] in must_include]
    remaining_slots = 11 - len(must_include_players)
    remaining_players = sorted(
        [p for p in available_players if p["name"] not in must_include],
        key=lambda x: predict_points(x, pitch, weather, match_context),
        reverse=True
    )
    team = must_include_players + remaining_players[:remaining_slots]
    return team[:11]  # Simplified; add role/credit constraints in production

def select_leaders(team: List[Dict], pitch: str, weather: str) -> tuple:
    match_context = {"opponent_strength": 1.0, "toss_winner": ""}
    sorted_team = sorted(team, key=lambda x: predict_points(x, pitch, weather, match_context), reverse=True)
    return sorted_team[0], sorted_team[1]  # Captain, Vice-Captain

def get_differential_pick(team: List[Dict], all_players: List[Dict], pitch: str, weather: str) -> Dict:
    match_context = {"opponent_strength": 1.0, "toss_winner": ""}
    differentials = [p for p in all_players if p not in team and p["ownership"] < 10]
    return max(differentials, key=lambda x: predict_points(x, pitch, weather, match_context), default=None)