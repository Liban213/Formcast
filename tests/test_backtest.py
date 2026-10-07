"""The backtest replays gameweeks without peeking, and scores picks correctly."""

import pandas as pd
import pytest

from formcast.predict.backtest import backtest_gameweek, players_as_of, run_backtest
from formcast.predict.rank import RankParams

MANAGERS = 1000
# Three gameweeks, every match 1-1: all teams exactly average, so predicted
# points equal form and fixtures don't distort anything.
MATCHES = {1: [(1, 2), (3, 4)], 2: [(1, 3), (2, 4)], 3: [(1, 4), (2, 3)]}
SMALL = RankParams(n_captains=5, n_differentials=5)


def build_season(points=lambda pid, gw: pid if pid <= 10 else pid - 10):
    """20 midfielders on 4 teams who score the same every week.

    Players 1-10 are popular (21-30% owned) and score 1..10 points; players
    11-20 are differentials (0.5-5% owned) and also score 1..10.
    """
    fixtures, fid = [], 0
    for gw, matches in MATCHES.items():
        for home, away in matches:
            fid += 1
            fixtures.append({"id": fid, "gameweek": gw, "home_team_id": home,
                             "away_team_id": away, "home_score": 1, "away_score": 1,
                             "finished": True})
    fixtures = pd.DataFrame(fixtures)

    players = pd.DataFrame({"player_id": range(1, 21), "position": "MID",
                            "team_id": [(i % 4) + 1 for i in range(1, 21)]})
    ownership = {i: 20 + i if i <= 10 else (i - 10) * 0.5 for i in range(1, 21)}
    history = []
    for p in players.itertuples():
        for f in fixtures.itertuples():
            if p.team_id in (f.home_team_id, f.away_team_id):
                history.append({
                    "player_id": p.player_id, "fixture_id": f.id, "gameweek": f.gameweek,
                    "was_home": p.team_id == f.home_team_id, "minutes": 90,
                    "total_points": points(p.player_id, f.gameweek),
                    "selected": int(ownership[p.player_id] * MANAGERS / 100),
                })
    gameweeks = pd.DataFrame({"id": [1, 2, 3], "ranked_count": MANAGERS})
    return players, pd.DataFrame(history), fixtures, gameweeks


def test_players_as_of_uses_that_weeks_club_and_ownership():
    players, history, fixtures, gameweeks = build_season()
    # Player 1 (team 2) is shown as transferred to team 3 in GW3, and widely bought.
    gw3 = (history["player_id"] == 1) & (history["gameweek"] == 3)
    history.loc[gw3, ["fixture_id", "was_home", "selected"]] = [6, False, 600]

    as_of = players_as_of(players, history, fixtures, gameweeks, 3).set_index("player_id")
    assert as_of.loc[1, "team_id"] == 3  # fixture 6 is 2 v 3, away side
    assert as_of.loc[1, "ownership_pct"] == pytest.approx(60.0)
    gw2 = players_as_of(players, history, fixtures, gameweeks, 2).set_index("player_id")
    assert gw2.loc[1, "team_id"] == 2
    assert gw2.loc[1, "ownership_pct"] == pytest.approx(21.0)


def test_players_as_of_counts_a_double_gameweek_player_once():
    players, history, fixtures, gameweeks = build_season()
    extra = history[(history["player_id"] == 1) & (history["gameweek"] == 3)].assign(fixture_id=99)
    as_of = players_as_of(players, pd.concat([history, extra]), fixtures, gameweeks, 3)
    assert (as_of["player_id"] == 1).sum() == 1


def test_perfectly_predictable_season_scores_perfectly():
    result = backtest_gameweek(*build_season(), gw=3, rank_params=SMALL)
    # Top 5 of each group scored 6..10 against a group average of 5.5.
    assert result["captains_avg"] == pytest.approx(8.0)
    assert result["captain_pool_avg"] == pytest.approx(5.5)
    assert result["differentials_avg"] == pytest.approx(8.0)
    assert (result["hits"], result["picks"], result["hit_rate"]) == (10, 10, 1.0)
    assert result["base_rate"] == pytest.approx(0.5)  # half of each group beats its mean
    assert result["rank_corr"] == pytest.approx(1.0)
    # Player 10 is both the most-owned and the model's top captain.
    assert result["captain_pick"] == result["template_captain"] == 10
    assert result["template_captain_points"] == 10


def test_the_target_gameweek_is_not_peeked_at():
    # Same form history, but everyone scores 0 in GW3 itself.
    zero_gw3 = build_season(
        points=lambda pid, gw: 0 if gw == 3 else (pid if pid <= 10 else pid - 10)
    )
    result = backtest_gameweek(*zero_gw3, gw=3, rank_params=SMALL)
    assert result["captain_pick"] == 10  # picks unchanged: they come from GW1-2 only
    assert result["captains_avg"] == 0  # but scored against what really happened
    assert result["hit_rate"] == 0
    assert pd.isna(result["rank_corr"])  # nobody scored: no ranking to correlate with


def test_later_gameweeks_do_not_change_an_earlier_replay():
    players, history, fixtures, gameweeks = build_season()
    before = backtest_gameweek(players, history, fixtures, gameweeks, 2, rank_params=SMALL)
    # Rewrite GW3 completely: different results and points.
    history.loc[history["gameweek"] == 3, "total_points"] = 50
    fixtures.loc[fixtures["gameweek"] == 3, ["home_score", "away_score"]] = [7, 0]
    after = backtest_gameweek(players, history, fixtures, gameweeks, 2, rank_params=SMALL)
    assert before == after


def test_run_backtest_pools_picks_across_gameweeks():
    result = run_backtest(*build_season(), gameweek_ids=[2, 3], rank_params=SMALL)
    assert list(result.per_gameweek["gameweek"]) == [2, 3]
    assert result.summary["picks"] == 20
    assert result.summary["hit_rate"] == 1.0
    assert result.summary["captain_vs_template"] == "0W 2D 0L"
