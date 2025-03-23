from typing import List, Dict
from config.settings import MAX_CREDITS, ROLE_LIMITS, TEAM_SIZE

def is_valid_team(team: List[Dict]) -> bool:
    if len(team) != TEAM_SIZE:
        return False
    total_credits = sum(p["credits"] for p in team)
    if total_credits > MAX_CREDITS:
        return False
    roles = {"BAT": 0, "BOW": 0, "ALL": 0, "WK": 0}
    for player in team:
        roles[player["role"]] += 1
    for role, (min_count, max_count) in ROLE_LIMITS.items():
        if not (min_count <= roles[role] <= max_count):
            return False
    return True