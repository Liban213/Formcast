"""Build next gameweek's three lists from the database, with display fields attached.

Shared by the refresh job (which stores them) and the preview script (which
prints them), so both always show the same thing.
"""

from dataclasses import dataclass

import pandas as pd
from sqlalchemy.engine import Engine

from formcast import loaders
from formcast.predict import ModelParams, predict_points
from formcast.predict.rank import RankParams, Rankings, rank_players


@dataclass(frozen=True)
class GameweekLists:
    gameweek: int
    rankings: Rankings


def opponent_labels(fixtures: pd.DataFrame, teams: pd.DataFrame, gw: int) -> pd.Series:
    """team_id -> opponent(s) in `gw`, e.g. "ARS (H)" or "SUN (A), CHE (H)"."""
    upcoming = fixtures[fixtures["gameweek"] == gw]
    names = teams.set_index("team_id")["short_name"]
    return pd.concat([
        pd.DataFrame({"team_id": upcoming["home_team_id"],
                      "opp": upcoming["away_team_id"].map(names) + " (H)"}),
        pd.DataFrame({"team_id": upcoming["away_team_id"],
                      "opp": upcoming["home_team_id"].map(names) + " (A)"}),
    ]).groupby("team_id")["opp"].agg(", ".join)


def build_lists(
    engine: Engine,
    model_params: ModelParams = ModelParams(),
    rank_params: RankParams = RankParams(),
) -> GameweekLists | None:
    """The lists for the next gameweek, or None when there isn't one (season over)."""
    gw = loaders.next_gameweek(engine)
    if gw is None:
        return None

    players = loaders.load_players(engine)
    teams = loaders.load_teams(engine)
    fixtures = loaders.load_fixtures(engine)
    predictions = predict_points(players, loaders.load_history(engine), fixtures, gw, model_params)

    display = players.merge(teams, on="team_id")
    display["opponent"] = display["team_id"].map(opponent_labels(fixtures, teams, gw)).fillna("blank")
    return GameweekLists(gameweek=gw, rankings=rank_players(predictions, display, rank_params))
