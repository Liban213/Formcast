"""Run the backtest over every finished gameweek and print the results.

Run with: python -m formcast.backtest           (default parameters)
          python -m formcast.backtest --sweep   (also compare a few model settings)
"""

import argparse
from dataclasses import replace

import pandas as pd

from formcast import loaders
from formcast.db import get_engine
from formcast.predict import ModelParams
from formcast.predict.backtest import run_backtest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="compare a grid of model settings")
    args = parser.parse_args()

    engine = get_engine()
    players = loaders.load_players(engine)
    history = loaders.load_history(engine)
    fixtures = loaders.load_fixtures(engine)
    gameweeks = loaders.load_gameweeks(engine)
    # GW1 can't be predicted: there is no earlier data.
    finished = [int(g) for g in gameweeks.loc[gameweeks["finished"], "id"] if g >= 2]

    result = run_backtest(players, history, fixtures, gameweeks, finished)
    names = players.set_index("player_id")["web_name"]
    per_gw = result.per_gameweek.assign(
        captain_pick=lambda d: d["captain_pick"].map(names),
        template_captain=lambda d: d["template_captain"].map(names),
    )
    pd.set_option("display.width", 160)
    print(f"Backtest over GW{finished[0]}-{finished[-1]}\n")
    print(per_gw.drop(columns=["base_hits", "base_n"]).round(2).to_string(index=False))

    s = result.summary
    print(f"""
Headline: {s['hit_rate']:.0%} of {s['picks']} picks beat their group's average \
(random pick: {s['base_rate']:.0%})
  Captain picks      {s['captains_avg']:.2f} pts  vs  {s['captain_pool_avg']:.2f} avg popular player
  Differentials      {s['differentials_avg']:.2f} pts  vs  {s['differential_pool_avg']:.2f} avg low-owned player
  Avoid list         {s['avoid_avg']:.2f} pts  vs  {s['owned_avg']:.2f} avg owned player (lower is better)
  #1 captain vs most-owned player: {s['captain_vs_template']}
  Rank correlation (predicted vs actual): {s['rank_corr']:.2f}""")

    if args.sweep:
        print("\nSweep (same gameweeks; too few to tune on - read as a sanity check):")
        rows = []
        for fixture_weight in (0.0, 0.5, 1.0):
            for half_life in (1.0, 2.0, 4.0):
                params = replace(ModelParams(), fixture_weight=fixture_weight, half_life=half_life)
                r = run_backtest(players, history, fixtures, gameweeks, finished, params).summary
                rows.append({"fixture_weight": fixture_weight, "half_life": half_life,
                             "hit_rate": r["hit_rate"], "rank_corr": r["rank_corr"],
                             "captains_avg": r["captains_avg"],
                             "differentials_avg": r["differentials_avg"]})
        print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
