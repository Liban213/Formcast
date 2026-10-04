"""Read Postgres tables into the DataFrame shapes the prediction engine expects."""

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


def load_players(engine: Engine) -> pd.DataFrame:
    df = pd.read_sql(
        "SELECT id AS player_id, web_name, team_id, position, price, ownership_pct, status"
        " FROM players",
        engine,
    )
    # NUMERIC columns arrive as Decimal objects; pandas maths wants floats.
    return df.astype({"price": float, "ownership_pct": float})


def load_history(engine: Engine) -> pd.DataFrame:
    return pd.read_sql(
        "SELECT player_id, fixture_id, gameweek, opponent_team_id, was_home,"
        " minutes, total_points, selected FROM player_gameweek_stats",
        engine,
    )


def load_fixtures(engine: Engine) -> pd.DataFrame:
    return pd.read_sql(
        "SELECT id, gameweek, kickoff_time, home_team_id, away_team_id,"
        " home_score, away_score, finished FROM fixtures",
        engine,
    )


def load_teams(engine: Engine) -> pd.DataFrame:
    return pd.read_sql("SELECT id AS team_id, short_name FROM teams", engine)


def next_gameweek(engine: Engine) -> int | None:
    with engine.connect() as conn:
        return conn.scalar(text("SELECT id FROM gameweeks WHERE is_next"))
