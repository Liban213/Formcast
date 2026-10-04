"""Recent form: a player's points over the last few gameweeks, recent games weighted more."""

import numpy as np
import pandas as pd


def decay_weights(ages, half_life: float) -> np.ndarray:
    """Weight for a game `age` gameweeks before the most recent one.

    Age 0 (the latest gameweek) weighs 1.0; a game `half_life` gameweeks older
    weighs 0.5, twice that weighs 0.25, and so on.
    """
    return 0.5 ** (np.asarray(ages, dtype=float) / half_life)


def weighted_form(
    history: pd.DataFrame, target_gw: int, n_gameweeks: int, half_life: float
) -> pd.DataFrame:
    """Decay-weighted points per fixture over the `n_gameweeks` before `target_gw`.

    `history` has one row per player per fixture (player_id, gameweek, minutes,
    total_points). Only gameweeks strictly before `target_gw` are used, so the
    same function is safe to call for past gameweeks in a backtest.

    Fixtures where the player sat on the bench (0 minutes, 0 points) stay in the
    average on purpose: a player who is regularly benched should have lower
    expected points than one who always starts.

    Returns one row per player: form, avg_minutes (decay-weighted, per fixture),
    total_minutes and fixtures (unweighted counts in the window).
    """
    window = history[
        (history["gameweek"] < target_gw) & (history["gameweek"] >= target_gw - n_gameweeks)
    ]
    weights = decay_weights(target_gw - 1 - window["gameweek"], half_life)
    weighted = window.assign(
        w=weights,
        w_points=weights * window["total_points"],
        w_minutes=weights * window["minutes"],
    )
    totals = weighted.groupby("player_id").agg(
        w=("w", "sum"),
        w_points=("w_points", "sum"),
        w_minutes=("w_minutes", "sum"),
        total_minutes=("minutes", "sum"),
        fixtures=("minutes", "size"),
    )
    return pd.DataFrame(
        {
            "form": totals["w_points"] / totals["w"],
            "avg_minutes": totals["w_minutes"] / totals["w"],
            "total_minutes": totals["total_minutes"],
            "fixtures": totals["fixtures"],
        }
    ).reset_index()
