"""Print next gameweek's captain picks, differentials and avoid list, for eyeballing.

Run with: python -m formcast.preview
"""

import pandas as pd

from formcast import loaders
from formcast.db import get_engine
from formcast.predict import predict_points
from formcast.predict.rank import rank_players


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

    display = players.merge(teams, on="team_id")
    display["opponent"] = display["team_id"].map(opponents).fillna("blank")
    rankings = rank_players(preds, display)

    cols = ["web_name", "short_name", "position", "price", "ownership_pct", "opponent",
            "form", "fixture_multiplier", "predicted_points"]
    pd.set_option("display.width", 160)
    for title, df, extra in [
        ("Captain picks", rankings.captains, []),
        ("Differentials", rankings.differentials, []),
        ("Avoid", rankings.avoid, ["reasons"]),
    ]:
        print(f"\nGW{gw} {title}\n")
        print(df[cols + extra].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
