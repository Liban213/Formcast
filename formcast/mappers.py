"""Pure functions: raw FPL JSON in, row dicts (keyed by model column) out.

No I/O here, so the API-to-schema mapping can be tested against saved responses.
"""

from datetime import datetime
from decimal import Decimal


def _parse_time(value: str | None) -> datetime | None:
    # FPL uses "2026-08-21T19:00:00Z"; fromisoformat handles "Z" from 3.11.
    return datetime.fromisoformat(value) if value else None


def _tenths(value: int) -> Decimal:
    # FPL stores prices as integers in tenths of a million: 61 -> £6.1m.
    return Decimal(value) / 10


def map_teams(bootstrap: dict) -> list[dict]:
    return [
        {"id": t["id"], "name": t["name"], "short_name": t["short_name"]}
        for t in bootstrap["teams"]
    ]


def map_gameweeks(bootstrap: dict) -> list[dict]:
    return [
        {
            "id": e["id"],
            "deadline_time": _parse_time(e["deadline_time"]),
            "finished": e["finished"],
            "is_current": e["is_current"],
            "is_next": e["is_next"],
            "ranked_count": e["ranked_count"] or None,  # FPL sends 0 for future weeks
        }
        for e in bootstrap["events"]
    ]


def map_players(bootstrap: dict) -> list[dict]:
    positions = {t["id"]: t["singular_name_short"] for t in bootstrap["element_types"]}
    return [
        {
            "id": p["id"],
            "first_name": p["first_name"],
            "second_name": p["second_name"],
            "web_name": p["web_name"],
            "team_id": p["team"],
            "position": positions[p["element_type"]],
            "price": _tenths(p["now_cost"]),
            "ownership_pct": Decimal(p["selected_by_percent"]),
            "status": p["status"],
            "chance_of_playing": p["chance_of_playing_next_round"],
            "news": p["news"],
        }
        for p in bootstrap["elements"]
    ]


def map_fixtures(fixtures: list[dict]) -> list[dict]:
    return [
        {
            "id": f["id"],
            "gameweek": f["event"],
            "kickoff_time": _parse_time(f["kickoff_time"]),
            "home_team_id": f["team_h"],
            "away_team_id": f["team_a"],
            "home_score": f["team_h_score"],
            "away_score": f["team_a_score"],
            "finished": f["finished"],
            "home_difficulty": f["team_h_difficulty"],
            "away_difficulty": f["team_a_difficulty"],
        }
        for f in fixtures
    ]


def map_player_history(element_summary: dict) -> list[dict]:
    return [
        {
            "player_id": h["element"],
            "fixture_id": h["fixture"],
            "gameweek": h["round"],
            "opponent_team_id": h["opponent_team"],
            "was_home": h["was_home"],
            "kickoff_time": _parse_time(h["kickoff_time"]),
            "minutes": h["minutes"],
            "total_points": h["total_points"],
            "goals_scored": h["goals_scored"],
            "assists": h["assists"],
            "clean_sheets": h["clean_sheets"],
            "goals_conceded": h["goals_conceded"],
            "bonus": h["bonus"],
            "expected_goals": Decimal(h["expected_goals"]),
            "expected_assists": Decimal(h["expected_assists"]),
            "price": _tenths(h["value"]),
            "selected": h["selected"],
        }
        for h in element_summary["history"]
    ]
