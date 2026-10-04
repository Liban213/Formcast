"""Team attack/defense strength and expected goals, the same concept as Pitchside's Poisson model.

A team's attack strength is its goals scored per match relative to the league
average; defense strength is goals conceded per match relative to the average.
Both are 1.0 for an average team. Defense above 1.0 means a *leaky* defense.

Expected goals for a fixture then follow the classic Poisson-model form:
    home λ = league avg home goals × home attack × away defense
    away λ = league avg away goals × away attack × home defense
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class LeagueAverages:
    home_goals: float  # mean goals per match by the home side
    away_goals: float

    @property
    def per_team(self) -> float:
        """Mean goals one team scores in a match, home or away."""
        return (self.home_goals + self.away_goals) / 2


def team_strengths(
    fixtures: pd.DataFrame, before_gw: int, prior_matches: float
) -> tuple[pd.DataFrame, LeagueAverages]:
    """Attack and defense strength per team, from finished fixtures before `before_gw`.

    Early in the season a team's ratios swing wildly on a single 5-0. To damp
    that, each team starts with `prior_matches` imaginary matches at the league
    average, so strengths begin near 1.0 and move toward the team's real record
    as games are played (Bayesian shrinkage). prior_matches=0 disables it.
    """
    played = fixtures[fixtures["finished"] & (fixtures["gameweek"] < before_gw)]
    if played.empty:
        raise ValueError(f"No finished fixtures before gameweek {before_gw}")

    league = LeagueAverages(
        home_goals=float(played["home_score"].mean()),
        away_goals=float(played["away_score"].mean()),
    )

    # One row per team per match, from that team's point of view.
    as_home = played.rename(
        columns={"home_team_id": "team_id", "home_score": "scored", "away_score": "conceded"}
    )
    as_away = played.rename(
        columns={"away_team_id": "team_id", "away_score": "scored", "home_score": "conceded"}
    )
    cols = ["team_id", "scored", "conceded"]
    per_team = pd.concat([as_home[cols], as_away[cols]]).groupby("team_id").agg(
        matches=("scored", "size"), scored=("scored", "sum"), conceded=("conceded", "sum")
    )

    avg, k = league.per_team, prior_matches
    strengths = pd.DataFrame(
        {
            "attack": (per_team["scored"] + k * avg) / (per_team["matches"] + k) / avg,
            "defense": (per_team["conceded"] + k * avg) / (per_team["matches"] + k) / avg,
        }
    ).reset_index()
    return strengths, league


def expected_goals(
    fixtures: pd.DataFrame, strengths: pd.DataFrame, league: LeagueAverages
) -> pd.DataFrame:
    """Expected goals for and against, for both teams in each fixture.

    Returns two rows per fixture (one per team): fixture_id, team_id,
    opponent_team_id, was_home, xg_for, xg_against. A team with no matches yet
    is treated as average (strength 1.0).
    """
    s = strengths.set_index("team_id")

    def attack(ids):
        return ids.map(s["attack"]).fillna(1.0)

    def defense(ids):
        return ids.map(s["defense"]).fillna(1.0)

    home, away = fixtures["home_team_id"], fixtures["away_team_id"]
    home_xg = league.home_goals * attack(home) * defense(away)
    away_xg = league.away_goals * attack(away) * defense(home)

    return pd.concat(
        [
            pd.DataFrame({"fixture_id": fixtures["id"], "team_id": home,
                          "opponent_team_id": away, "was_home": True,
                          "xg_for": home_xg, "xg_against": away_xg}),
            pd.DataFrame({"fixture_id": fixtures["id"], "team_id": away,
                          "opponent_team_id": home, "was_home": False,
                          "xg_for": away_xg, "xg_against": home_xg}),
        ],
        ignore_index=True,
    )
