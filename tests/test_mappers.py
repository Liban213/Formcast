"""Raw FPL responses map into the schema's shape, checked against saved real responses."""

from datetime import datetime, timezone
from decimal import Decimal

from formcast import mappers
from formcast.models import Fixture, Gameweek, Player, PlayerGameweekStats, Team
from tests.conftest import load


def columns(model) -> set[str]:
    return {c.name for c in model.__table__.columns}


def test_every_mapper_produces_exactly_the_model_columns():
    bootstrap = load("bootstrap_static.json")
    cases = [
        (mappers.map_teams(bootstrap), Team),
        (mappers.map_gameweeks(bootstrap), Gameweek),
        (mappers.map_players(bootstrap), Player),
        (mappers.map_fixtures(load("fixtures.json")), Fixture),
        (mappers.map_player_history(load("element_summary_124.json")), PlayerGameweekStats),
    ]
    for rows, model in cases:
        assert rows, model.__name__
        for row in rows:
            assert set(row) == columns(model), model.__name__


def test_map_teams():
    teams = mappers.map_teams(load("bootstrap_static.json"))
    assert len(teams) == 20
    assert teams[0] == {"id": 1, "name": "Arsenal", "short_name": "ARS"}


def test_map_gameweeks_flags_current_and_next():
    gameweeks = {g["id"]: g for g in mappers.map_gameweeks(load("bootstrap_static.json"))}
    assert len(gameweeks) == 38
    assert gameweeks[5]["is_current"] and gameweeks[5]["finished"]
    assert gameweeks[6]["is_next"] and not gameweeks[6]["finished"]
    assert gameweeks[6]["deadline_time"] == datetime(2026, 10, 10, 10, 0, tzinfo=timezone.utc)
    assert gameweeks[5]["ranked_count"] == 10833606
    assert gameweeks[6]["ranked_count"] is None  # sent as 0 before the week starts


def test_map_players_converts_price_ownership_and_position():
    players = {p["id"]: p for p in mappers.map_players(load("bootstrap_static.json"))}
    assert players[1] == {
        "id": 1,
        "first_name": "David",
        "second_name": "Raya Martín",
        "web_name": "Raya",
        "team_id": 1,
        "position": "GKP",
        "price": Decimal("6.1"),  # now_cost 61 is tenths of a million
        "ownership_pct": Decimal("42.3"),  # arrives as the string "42.3"
        "status": "a",
        "chance_of_playing": None,  # no injury news
        "news": "",
    }
    assert players[124]["position"] == "MID"
    assert players[124]["second_name"] == "Groß"


def test_map_fixtures_finished_and_upcoming():
    fixtures = {f["id"]: f for f in mappers.map_fixtures(load("fixtures.json"))}
    assert len(fixtures) == 380

    finished = fixtures[1]
    assert finished["home_team_id"] == 1 and finished["away_team_id"] == 7
    assert (finished["home_score"], finished["away_score"]) == (3, 0)
    assert finished["finished"] is True
    assert finished["kickoff_time"] == datetime(2026, 8, 21, 19, 0, tzinfo=timezone.utc)

    upcoming = fixtures[51]
    assert upcoming["gameweek"] == 6
    assert upcoming["home_score"] is None and upcoming["finished"] is False


def test_map_fixtures_handles_postponed_match():
    # A postponed, unrescheduled match has no gameweek and no kickoff time.
    raw = dict(load("fixtures.json")[0], event=None, kickoff_time=None, finished=False,
               team_h_score=None, team_a_score=None)
    [row] = mappers.map_fixtures([raw])
    assert row["gameweek"] is None
    assert row["kickoff_time"] is None


def test_map_player_history():
    history = mappers.map_player_history(load("element_summary_124.json"))
    assert [h["gameweek"] for h in history] == [1, 2, 3, 4, 5]
    gw4 = history[3]
    assert gw4 == {
        "player_id": 124,
        "fixture_id": 38,
        "gameweek": 4,
        "opponent_team_id": 7,
        "was_home": False,
        "kickoff_time": datetime(2026, 9, 13, 13, 0, tzinfo=timezone.utc),
        "minutes": 90,
        "total_points": 17,
        "goals_scored": 1,
        "assists": 2,
        "clean_sheets": gw4["clean_sheets"],
        "goals_conceded": gw4["goals_conceded"],
        "bonus": 3,
        "expected_goals": Decimal("0.82"),
        "expected_assists": Decimal("0.19"),
        "price": Decimal("5.6"),
        "selected": 1745184,
    }
    # Price and ownership are captured per gameweek, not just today's values.
    assert history[0]["price"] == Decimal("5.5")
    assert history[0]["selected"] == 1308033


def test_map_player_history_double_gameweek_keeps_both_fixtures():
    summary = load("element_summary_124.json")
    second = dict(summary["history"][4], fixture=999)  # same round, another fixture
    summary["history"].append(second)
    history = mappers.map_player_history(summary)
    gw5 = [h for h in history if h["gameweek"] == 5]
    assert {h["fixture_id"] for h in gw5} == {42, 999}
