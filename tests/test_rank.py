"""Edge cases of the captain, differential and avoid lists."""

import pandas as pd

from formcast.predict.rank import RankParams, rank_players


def make(rows):
    """rows: dicts with player_id plus any overrides of the defaults below."""
    defaults = {"window_gameweeks": 3, "form": 6.0, "avg_minutes": 90.0, "total_minutes": 270,
                "fixtures_played": 3, "n_fixtures": 1, "fixture_multiplier": 1.0,
                "ownership_pct": 20.0, "status": "a", "chance_of_playing": None}
    full = pd.DataFrame([{**defaults, "predicted_points": r.get("form", 6.0), **r} for r in rows])
    full["gameweek"] = 6
    player_cols = ["player_id", "ownership_pct", "status", "chance_of_playing"]
    players = full[player_cols]
    predictions = full.drop(columns=player_cols[1:])
    return predictions, players


def ids(df) -> list[int]:
    return list(df["player_id"])


def test_one_lucky_game_does_not_top_the_lists():
    # Player 1: a single 90-minute, 20-point cameo in a 3-gameweek window.
    predictions, players = make([
        {"player_id": 1, "predicted_points": 20, "total_minutes": 90, "fixtures_played": 1},
        {"player_id": 2, "predicted_points": 6},
        {"player_id": 3, "predicted_points": 20, "total_minutes": 90, "ownership_pct": 2},
        {"player_id": 4, "predicted_points": 5, "ownership_pct": 2},
    ])
    r = rank_players(predictions, players)
    assert ids(r.captains) == [2]
    assert ids(r.differentials) == [4]


def test_minimum_minutes_scale_with_the_form_window():
    # Before GW2 there is one gameweek of form: 60 minutes is enough.
    predictions, players = make([
        {"player_id": 1, "window_gameweeks": 1, "total_minutes": 60},
        {"player_id": 2, "window_gameweeks": 1, "total_minutes": 59},
    ])
    assert ids(rank_players(predictions, players).captains) == [1]


def test_ownership_threshold_splits_captains_from_differentials():
    predictions, players = make([
        {"player_id": 1, "ownership_pct": 10.0},
        {"player_id": 2, "ownership_pct": 9.9},
    ])
    r = rank_players(predictions, players)
    assert ids(r.captains) == [1]
    assert ids(r.differentials) == [2]


def test_lists_are_sorted_by_predicted_points_and_capped():
    predictions, players = make(
        [{"player_id": i, "predicted_points": i} for i in range(1, 9)]
        + [{"player_id": 100 + i, "predicted_points": i, "ownership_pct": 1} for i in range(1, 15)]
    )
    r = rank_players(predictions, players, RankParams(n_captains=3, n_differentials=4))
    assert ids(r.captains) == [8, 7, 6]
    assert ids(r.differentials) == [114, 113, 112, 111]


def test_unavailable_and_blank_players_are_not_picked_but_are_flagged():
    predictions, players = make([
        {"player_id": 1, "predicted_points": 12, "status": "i", "ownership_pct": 40},
        {"player_id": 2, "predicted_points": 11, "n_fixtures": 0, "ownership_pct": 30},
        {"player_id": 3, "predicted_points": 10, "status": "d", "ownership_pct": 2},
        {"player_id": 4, "predicted_points": 5},
    ])
    r = rank_players(predictions, players)
    assert ids(r.captains) == [4]
    assert r.differentials.empty
    reasons = dict(zip(r.avoid["player_id"], r.avoid["reasons"]))
    assert reasons == {1: "injured", 2: "no fixture"}  # player 3 is barely owned


def test_doubtful_at_75_percent_is_still_picked_but_50_percent_is_flagged():
    predictions, players = make([
        {"player_id": 1, "predicted_points": 9, "status": "d", "chance_of_playing": 75},
        {"player_id": 2, "predicted_points": 8, "status": "d", "chance_of_playing": 50},
        {"player_id": 3, "predicted_points": 7, "status": "d"},  # doubtful, no % given
    ])
    r = rank_players(predictions, players)
    assert ids(r.captains) == [1]
    reasons = dict(zip(r.avoid["player_id"], r.avoid["reasons"]))
    assert reasons == {2: "doubtful (50%)", 3: "doubtful"}


def test_avoid_flags_form_fixture_and_minutes():
    predictions, players = make([
        {"player_id": 1, "form": 1.5},
        {"player_id": 2, "fixture_multiplier": 0.85},
        {"player_id": 3, "total_minutes": 100, "form": 1.0},  # limited, not "poor form" too
        {"player_id": 4, "fixture_multiplier": 0.86},  # just above the line
        {"player_id": 5, "form": 1.0, "ownership_pct": 4.9},  # not owned enough to matter
        {"player_id": 6, "form": 1.0, "fixture_multiplier": 0.7, "status": "s"},
    ])
    avoid = rank_players(predictions, players).avoid
    reasons = dict(zip(avoid["player_id"], avoid["reasons"]))
    assert reasons == {
        1: "poor form",
        2: "tough fixture",
        3: "limited minutes",
        6: "suspended, tough fixture, poor form",
    }


def test_avoid_list_puts_the_most_owned_first():
    predictions, players = make([
        {"player_id": 1, "form": 1.0, "ownership_pct": 8},
        {"player_id": 2, "form": 1.0, "ownership_pct": 55},
        {"player_id": 3, "form": 1.0, "ownership_pct": 21},
    ])
    assert ids(rank_players(predictions, players).avoid) == [2, 3, 1]
