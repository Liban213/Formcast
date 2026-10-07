"""Turn predicted points into the three lists the site shows.

- Captain picks: highest predicted points among already-popular players.
- Differentials: high predicted points, low ownership.
- Avoid: popular players with a red flag (injury, no fixture, poor form,
  limited minutes, brutal fixture), most-owned first.
"""

from dataclasses import dataclass

import pandas as pd

STATUS_REASONS = {
    "i": "injured",
    "s": "suspended",
    "u": "unavailable",
    "n": "unavailable",
}


@dataclass(frozen=True)
class RankParams:
    # A player needs this many minutes per gameweek of the form window (on
    # average) to be picked, so one lucky cameo can't top the lists.
    min_minutes_per_gw: float = 60
    # FPL marks a player doubtful even at 75%, which usually means a minor knock
    # and he plays. Below this he isn't picked and is flagged.
    min_chance_of_playing: float = 75
    captain_min_ownership: float = 10.0  # %
    differential_max_ownership: float = 10.0  # %
    avoid_min_ownership: float = 5.0  # % - nobody needs warning off unowned players
    poor_form: float = 2.0  # points per fixture: barely more than turning up
    tough_fixture: float = 0.85  # fixture multiplier at or below this is flagged
    n_captains: int = 5
    n_differentials: int = 10
    n_avoid: int = 10


@dataclass(frozen=True)
class Rankings:
    captains: pd.DataFrame
    differentials: pd.DataFrame
    avoid: pd.DataFrame


def flag_players(
    predictions: pd.DataFrame, players: pd.DataFrame, params: RankParams = RankParams()
) -> pd.DataFrame:
    """Join predictions to players and mark who is eligible to be picked."""
    df = predictions.merge(players, on="player_id", how="inner")
    min_minutes = params.min_minutes_per_gw * df["window_gameweeks"]
    df["enough_minutes"] = df["total_minutes"] >= min_minutes
    df["available"] = (df["status"] == "a") | (
        (df["status"] == "d") & (df["chance_of_playing"] >= params.min_chance_of_playing)
    )
    df["pickable"] = df["enough_minutes"] & df["available"] & (df["n_fixtures"] > 0)
    return df


def _avoid_reasons(row, params: RankParams) -> str:
    reasons = []
    if row.status in STATUS_REASONS:
        reasons.append(STATUS_REASONS[row.status])
    elif row.status == "d" and not row.available:
        chance = row.chance_of_playing
        reasons.append("doubtful" if pd.isna(chance) else f"doubtful ({chance:.0f}%)")
    if row.n_fixtures == 0:
        reasons.append("no fixture")
    elif row.fixture_multiplier <= params.tough_fixture:
        reasons.append("tough fixture")
    if not row.enough_minutes:
        reasons.append("limited minutes")
    elif row.form < params.poor_form:
        reasons.append("poor form")
    return ", ".join(reasons)


def rank_players(
    predictions: pd.DataFrame, players: pd.DataFrame, params: RankParams = RankParams()
) -> Rankings:
    """Build the captain, differential and avoid lists for one gameweek.

    predictions: output of predict_points.
    players: player_id, ownership_pct, status, chance_of_playing (+ any display
             columns, which are carried through to the lists).
    """
    df = flag_players(predictions, players, params)
    by_points = df.sort_values("predicted_points", ascending=False, kind="stable")
    pickable = by_points[by_points["pickable"]]

    captains = pickable[pickable["ownership_pct"] >= params.captain_min_ownership]
    differentials = pickable[pickable["ownership_pct"] < params.differential_max_ownership]

    owned = df[df["ownership_pct"] >= params.avoid_min_ownership].copy()
    owned["reasons"] = [_avoid_reasons(r, params) for r in owned.itertuples()]
    avoid = owned[owned["reasons"] != ""].sort_values(
        ["ownership_pct", "predicted_points"], ascending=[False, True], kind="stable"
    )

    return Rankings(
        captains=captains.head(params.n_captains).reset_index(drop=True),
        differentials=differentials.head(params.n_differentials).reset_index(drop=True),
        avoid=avoid.head(params.n_avoid).reset_index(drop=True),
    )
