"""Predicted points per player for one gameweek: recent form × fixture difficulty."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from formcast.predict.form import weighted_form
from formcast.predict.strength import LeagueAverages, expected_goals, team_strengths

# Attackers earn points from goals, so their fixture depends on the opponent's
# defense. Goalkeepers and defenders earn them from clean sheets, so theirs
# depends on the opponent's attack.
ATTACKING_POSITIONS = {"MID", "FWD"}


@dataclass(frozen=True)
class ModelParams:
    n_gameweeks: int = 5  # form window
    half_life: float = 2.0  # gameweeks until a game's weight halves
    prior_matches: float = 5.0  # shrinkage of team strengths toward average
    # How strongly the fixture moves the prediction: 0 ignores fixtures entirely,
    # 1 scales points one-for-one with the expected-goals ratio. Partial by
    # default because form already reflects a player's own quality.
    fixture_weight: float = 0.5


def fixture_multipliers(
    team_fixtures: pd.DataFrame, league: LeagueAverages, fixture_weight: float
) -> pd.DataFrame:
    """Add attack_mult and defend_mult columns: 1.0 for an average fixture.

    attack_mult > 1 means the team should score more than an average team does
    (weak opponent defense, home advantage). defend_mult > 1 means the team
    should concede less than average (blunt opponent attack).
    """
    avg = league.per_team
    return team_fixtures.assign(
        attack_mult=(team_fixtures["xg_for"] / avg) ** fixture_weight,
        defend_mult=(avg / team_fixtures["xg_against"]) ** fixture_weight,
    )


def predict_points(
    players: pd.DataFrame,
    history: pd.DataFrame,
    fixtures: pd.DataFrame,
    target_gw: int,
    params: ModelParams = ModelParams(),
) -> pd.DataFrame:
    """Predicted points for every player in `target_gw`.

    Uses only data from before `target_gw`, so it can be run for past gameweeks
    in a backtest. Players in a double gameweek get the sum over both fixtures;
    players whose team has a blank gameweek get 0.

    players:  player_id, team_id, position
    history:  player_id, gameweek, minutes, total_points
    fixtures: id, gameweek, home_team_id, away_team_id, home_score, away_score, finished

    Returns one row per player, highest predicted_points first: player_id,
    gameweek, form, avg_minutes, total_minutes, fixtures_played, n_fixtures,
    fixture_multiplier, predicted_points.
    """
    form = weighted_form(history, target_gw, params.n_gameweeks, params.half_life)
    strengths, league = team_strengths(fixtures, target_gw, params.prior_matches)
    upcoming = fixtures[fixtures["gameweek"] == target_gw]
    team_fixtures = fixture_multipliers(
        expected_goals(upcoming, strengths, league), league, params.fixture_weight
    )

    # Players with no history in the window have no form signal: predict 0.
    base = (
        players[["player_id", "team_id", "position"]]
        .merge(form, on="player_id", how="left")
        .rename(columns={"fixtures": "fixtures_played"})
        .fillna({"form": 0.0, "avg_minutes": 0.0, "total_minutes": 0, "fixtures_played": 0})
    )

    # One row per player per upcoming fixture (two in a double gameweek, a
    # single all-NaN fixture row for a blank).
    per_fixture = base.merge(team_fixtures, on="team_id", how="left")
    per_fixture["multiplier"] = np.where(
        per_fixture["position"].isin(ATTACKING_POSITIONS),
        per_fixture["attack_mult"],
        per_fixture["defend_mult"],
    )
    per_fixture["points"] = per_fixture["form"] * per_fixture["multiplier"]

    result = per_fixture.groupby("player_id", sort=False).agg(
        form=("form", "first"),
        avg_minutes=("avg_minutes", "first"),
        total_minutes=("total_minutes", "first"),
        fixtures_played=("fixtures_played", "first"),
        n_fixtures=("fixture_id", "count"),
        fixture_multiplier=("multiplier", "mean"),
        predicted_points=("points", "sum"),  # NaN for a blank sums to 0
    )
    result.insert(0, "gameweek", target_gw)
    return (
        result.reset_index()
        .astype({"total_minutes": int, "fixtures_played": int})
        .sort_values("predicted_points", ascending=False, ignore_index=True)
    )
