"""Replay finished gameweeks as if they hadn't happened yet, then score the picks.

For each gameweek G the model sees only what was knowable at G's deadline:
form from gameweeks before G, team strengths from results before G, each
player's club and ownership *as they were in G* (not today's), and no injury
news (FPL doesn't keep a history of player status, so nobody is filtered as
injured). The picks are then compared with the points actually scored in G.

Headline metric, fixed before any results were looked at: the hit rate - the
share of captain and differential picks that outscored the average pickable
player in their own ownership group - against the base rate, the same share for
every player in those groups (what picking at random would score).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from formcast.predict.model import ModelParams, predict_points
from formcast.predict.rank import RankParams, flag_players, rank_players


@dataclass(frozen=True)
class BacktestResult:
    per_gameweek: pd.DataFrame  # one row per replayed gameweek
    summary: dict


def players_as_of(
    players: pd.DataFrame, history: pd.DataFrame, fixtures: pd.DataFrame,
    gameweeks: pd.DataFrame, gw: int,
) -> pd.DataFrame:
    """Each player's club and ownership % as they were in `gw`.

    Built from that week's history rows: a row's fixture and was_home give the
    club (so mid-season transfers are handled), and `selected` divided by the
    managers playing that week gives ownership at the deadline. Players with no
    row in `gw` weren't at a Premier League club that week, or their club had no
    fixture, and are left out.
    """
    rows = (
        history[history["gameweek"] == gw]
        .sort_values("fixture_id")
        .drop_duplicates("player_id")  # a double gameweek has two rows per player
    )
    fx = fixtures.set_index("id")
    team = np.where(
        rows["was_home"],
        rows["fixture_id"].map(fx["home_team_id"]),
        rows["fixture_id"].map(fx["away_team_id"]),
    )
    managers = gameweeks.set_index("id").loc[gw, "ranked_count"]
    return pd.DataFrame(
        {
            "player_id": rows["player_id"].to_numpy(),
            "team_id": team,
            "position": rows["player_id"].map(players.set_index("player_id")["position"]).to_numpy(),
            "ownership_pct": (100 * rows["selected"] / managers).to_numpy(),
            "status": "a",
            "chance_of_playing": np.nan,
        }
    )


def _rank_corr(a: pd.Series, b: pd.Series) -> float:
    """Spearman correlation: Pearson on ranks (avoids a scipy dependency).

    Undefined (NaN) when either side has no variation, e.g. everyone scored 0.
    """
    if a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(a.rank().corr(b.rank()))


def backtest_gameweek(
    players: pd.DataFrame, history: pd.DataFrame, fixtures: pd.DataFrame,
    gameweeks: pd.DataFrame, gw: int,
    model_params: ModelParams = ModelParams(), rank_params: RankParams = RankParams(),
) -> dict:
    as_of = players_as_of(players, history, fixtures, gameweeks, gw)
    predictions = predict_points(as_of, history, fixtures, gw, model_params)
    rankings = rank_players(predictions, as_of, rank_params)
    actual = history[history["gameweek"] == gw].groupby("player_id")["total_points"].sum()

    def points(df: pd.DataFrame) -> pd.Series:
        return df["player_id"].map(actual).fillna(0)

    flagged = flag_players(predictions, as_of, rank_params)
    pool = flagged[flagged["pickable"]].assign(actual=points)
    captain_pool = pool[pool["ownership_pct"] >= rank_params.captain_min_ownership]
    differential_pool = pool[pool["ownership_pct"] < rank_params.differential_max_ownership]
    owned = flagged[flagged["ownership_pct"] >= rank_params.avoid_min_ownership]
    template = as_of.sort_values("ownership_pct", ascending=False).head(1)

    captains, differentials = points(rankings.captains), points(rankings.differentials)
    captain_bar, differential_bar = captain_pool["actual"].mean(), differential_pool["actual"].mean()
    hits = (captains > captain_bar).sum() + (differentials > differential_bar).sum()
    base_hits = (captain_pool["actual"] > captain_bar).sum() + (
        differential_pool["actual"] > differential_bar
    ).sum()

    return {
        "gameweek": gw,
        "captain_pick": rankings.captains["player_id"].iloc[0] if len(captains) else None,
        "captain_pick_points": captains.iloc[0] if len(captains) else np.nan,
        "template_captain": template["player_id"].iloc[0],
        "template_captain_points": points(template).iloc[0],
        "captains_avg": captains.mean(),
        "captain_pool_avg": captain_bar,
        "differentials_avg": differentials.mean(),
        "differential_pool_avg": differential_bar,
        "avoid_avg": points(rankings.avoid).mean(),
        "owned_avg": points(owned).mean(),
        "picks": len(captains) + len(differentials),
        "hits": int(hits),
        "hit_rate": hits / (len(captains) + len(differentials)),
        "base_hits": int(base_hits),
        "base_n": len(captain_pool) + len(differential_pool),
        "base_rate": base_hits / (len(captain_pool) + len(differential_pool)),
        "rank_corr": _rank_corr(pool["predicted_points"], pool["actual"]),
        "pool_size": len(pool),
    }


def run_backtest(
    players: pd.DataFrame, history: pd.DataFrame, fixtures: pd.DataFrame,
    gameweeks: pd.DataFrame, gameweek_ids: list[int],
    model_params: ModelParams = ModelParams(), rank_params: RankParams = RankParams(),
) -> BacktestResult:
    per_gw = pd.DataFrame([
        backtest_gameweek(players, history, fixtures, gameweeks, gw, model_params, rank_params)
        for gw in gameweek_ids
    ])
    picks, hits = per_gw["picks"].sum(), per_gw["hits"].sum()
    beat = per_gw["captain_pick_points"] - per_gw["template_captain_points"]
    summary = {
        "gameweeks": list(gameweek_ids),
        "picks": int(picks),
        "hit_rate": hits / picks,  # pooled over every pick, not averaged per week
        "base_rate": per_gw["base_hits"].sum() / per_gw["base_n"].sum(),
        "captains_avg": per_gw["captains_avg"].mean(),
        "captain_pool_avg": per_gw["captain_pool_avg"].mean(),
        "differentials_avg": per_gw["differentials_avg"].mean(),
        "differential_pool_avg": per_gw["differential_pool_avg"].mean(),
        "avoid_avg": per_gw["avoid_avg"].mean(),
        "owned_avg": per_gw["owned_avg"].mean(),
        "captain_vs_template": f"{(beat > 0).sum()}W {(beat == 0).sum()}D {(beat < 0).sum()}L",
        "rank_corr": per_gw["rank_corr"].mean(),
    }
    return BacktestResult(per_gameweek=per_gw, summary=summary)
