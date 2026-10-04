"""Print the model's predictions for the next gameweek, for eyeballing.

Run with: python -m formcast.preview
"""

import pandas as pd

from formcast import loaders
from formcast.db import get_engine
from formcast.predict import predict_points


def main() -> None:
    engine = get_engine()
    gw = loaders.next_gameweek(engine)
    if gw is None:
        raise SystemExit("No upcoming gameweek: the season may be over.")

    players = loaders.load_players(engine)
    teams = loaders.load_teams(engine)
    fixtures = loaders.load_fixtures(engine)
    preds = predict_points(players, loaders.load_history(engine), fixtures, gw)

    # Name each player's opponent(s), e.g. "ARS (H)".
    upcoming = fixtures[fixtures["gameweek"] == gw]
    names = teams.set_index("team_id")["short_name"]
    opponents = pd.concat([
        pd.DataFrame({"team_id": upcoming["home_team_id"],
                      "opp": upcoming["away_team_id"].map(names) + " (H)"}),
        pd.DataFrame({"team_id": upcoming["away_team_id"],
                      "opp": upcoming["home_team_id"].map(names) + " (A)"}),
    ]).groupby("team_id")["opp"].agg(", ".join)

    table = preds.merge(players, on="player_id").merge(teams, on="team_id")
    table["opponent"] = table["team_id"].map(opponents).fillna("blank")
    cols = ["web_name", "short_name", "position", "price", "ownership_pct", "opponent",
            "form", "fixture_multiplier", "predicted_points"]

    pd.set_option("display.width", 140)
    print(f"Gameweek {gw}: top 20 by predicted points\n")
    print(table[cols].head(20).round(2).to_string(index=False))
    for position in ["GKP", "DEF", "MID", "FWD"]:
        top = table[table["position"] == position].head(5)
        print(f"\nTop {position}: " + ", ".join(
            f"{r.web_name} {r.predicted_points:.1f}" for r in top.itertuples()))


if __name__ == "__main__":
    main()
