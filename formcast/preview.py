"""Print next gameweek's captain picks, differentials and avoid list, for eyeballing.

Computes fresh from the database without storing anything; the refresh job is
what the site serves. Run with: python -m formcast.preview
"""

import pandas as pd

from formcast.db import get_engine
from formcast.lists import build_lists


def main() -> None:
    lists = build_lists(get_engine())
    if lists is None:
        raise SystemExit("No upcoming gameweek: the season may be over.")

    cols = ["web_name", "short_name", "position", "price", "ownership_pct", "opponent",
            "form", "fixture_multiplier", "predicted_points"]
    pd.set_option("display.width", 160)
    r = lists.rankings
    for title, df, extra in [
        ("Captain picks", r.captains, []),
        ("Differentials", r.differentials, []),
        ("Avoid", r.avoid, ["reasons"]),
    ]:
        print(f"\nGW{lists.gameweek} {title}\n")
        print(df[cols + extra].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
