"""Known-input/known-output checks on the prediction engine. No database needed."""

import pandas as pd
import pytest

from formcast.predict import ModelParams, predict_points
from formcast.predict.form import decay_weights, weighted_form
from formcast.predict.strength import LeagueAverages, expected_goals, team_strengths


def history_rows(player_id, points_by_gw, minutes=90):
    return [
        {"player_id": player_id, "gameweek": gw, "minutes": minutes, "total_points": pts}
        for gw, pts in points_by_gw.items()
    ]


def fixture(id, gw, home, away, home_score=None, away_score=None):
    return {"id": id, "gameweek": gw, "home_team_id": home, "away_team_id": away,
            "home_score": home_score, "away_score": away_score,
            "finished": home_score is not None}


# --- form ---------------------------------------------------------------------


def test_decay_weights_halve_every_half_life():
    assert decay_weights([0, 1, 2, 3], half_life=2) == pytest.approx(
        [1.0, 0.70711, 0.5, 0.35355], abs=1e-5
    )
    assert decay_weights([0, 1, 2], half_life=1) == pytest.approx([1.0, 0.5, 0.25])


def test_weighted_form_by_hand():
    # GW1..4 = 2, 4, 6, 8 points; predicting GW5 with half-life 1.
    # Weights (newest first) 1, 0.5, 0.25, 0.125:
    #   (8·1 + 6·0.5 + 4·0.25 + 2·0.125) / 1.875 = 12.25 / 1.875
    history = pd.DataFrame(history_rows(7, {1: 2, 2: 4, 3: 6, 4: 8}))
    [row] = weighted_form(history, target_gw=5, n_gameweeks=6, half_life=1).to_dict("records")
    assert row["form"] == pytest.approx(12.25 / 1.875)
    assert row["form"] > (2 + 4 + 6 + 8) / 4  # rising form beats the plain mean
    assert row["total_minutes"] == 360
    assert row["fixtures"] == 4


def test_weighted_form_ignores_target_gameweek_and_later():
    # A 20-point GW5 haul must not leak into the prediction for GW5.
    clean = pd.DataFrame(history_rows(7, {3: 4, 4: 6}))
    leaky = pd.DataFrame(history_rows(7, {3: 4, 4: 6, 5: 20, 6: 20}))
    a = weighted_form(clean, 5, n_gameweeks=6, half_life=2)
    b = weighted_form(leaky, 5, n_gameweeks=6, half_life=2)
    assert a["form"].item() == pytest.approx(b["form"].item())


def test_weighted_form_window_drops_old_gameweeks():
    # Window of 2 before GW5 is GW3-4 only; the 15 in GW1 is ignored.
    history = pd.DataFrame(history_rows(7, {1: 15, 3: 2, 4: 2}))
    form = weighted_form(history, target_gw=5, n_gameweeks=2, half_life=2)
    assert form["form"].item() == pytest.approx(2.0)
    assert form["fixtures"].item() == 2


def test_weighted_form_benched_fixture_counts_as_zero():
    history = pd.DataFrame(
        history_rows(7, {3: 6}) + history_rows(7, {4: 0}, minutes=0)
    )
    form = weighted_form(history, target_gw=5, n_gameweeks=6, half_life=1)
    # (0·1 + 6·0.5) / 1.5 = 2: the benching drags form down.
    assert form["form"].item() == pytest.approx(2.0)
    assert form["total_minutes"].item() == 90


def test_weighted_form_double_gameweek_fixtures_share_a_weight():
    history = pd.DataFrame(history_rows(7, {4: 2}) + history_rows(7, {4: 8}))
    form = weighted_form(history, target_gw=5, n_gameweeks=6, half_life=1)
    assert form["form"].item() == pytest.approx(5.0)
    assert form["fixtures"].item() == 2


# --- team strength ------------------------------------------------------------

# Team 1 beat team 2 3-1 at home; teams 3 and 4 drew 0-0.
# 4 goals across 4 team-matches: league average 1 goal per team per match.
TWO_GAMES = pd.DataFrame([fixture(1, 1, 1, 2, 3, 1), fixture(2, 1, 3, 4, 0, 0)])


def test_league_averages():
    _, league = team_strengths(TWO_GAMES, before_gw=2, prior_matches=0)
    assert league == LeagueAverages(home_goals=1.5, away_goals=0.5)
    assert league.per_team == 1.0


def test_team_strengths_without_shrinkage_are_raw_ratios():
    strengths, _ = team_strengths(TWO_GAMES, before_gw=2, prior_matches=0)
    s = strengths.set_index("team_id")
    assert s.loc[1, "attack"] == pytest.approx(3.0)  # 3 goals vs average 1
    assert s.loc[1, "defense"] == pytest.approx(1.0)  # conceded 1: exactly average
    assert s.loc[2, "attack"] == pytest.approx(1.0)
    assert s.loc[2, "defense"] == pytest.approx(3.0)  # conceded 3: leaky
    assert s.loc[3, "attack"] == pytest.approx(0.0)


def test_team_strengths_shrink_toward_average():
    # One prior match at the league average (1 goal):
    # team 1 attack = (3 + 1) / (1 + 1) / 1 = 2.0, team 3 attack = (0 + 1) / 2 = 0.5
    strengths, _ = team_strengths(TWO_GAMES, before_gw=2, prior_matches=1)
    s = strengths.set_index("team_id")
    assert s.loc[1, "attack"] == pytest.approx(2.0)
    assert s.loc[3, "attack"] == pytest.approx(0.5)
    assert s.loc[2, "defense"] == pytest.approx(2.0)


def test_team_strengths_use_only_finished_fixtures_before_gameweek():
    fixtures = pd.concat([
        TWO_GAMES,
        pd.DataFrame([
            fixture(3, 2, 1, 3, 9, 9),  # GW2 result must not count when predicting GW2
            fixture(4, 1, 2, 4),  # unplayed GW1 fixture (postponed)
        ]),
    ])
    with_extra, _ = team_strengths(fixtures, before_gw=2, prior_matches=0)
    baseline, _ = team_strengths(TWO_GAMES, before_gw=2, prior_matches=0)
    pd.testing.assert_frame_equal(with_extra, baseline)


def test_team_strengths_need_some_results():
    with pytest.raises(ValueError, match="before gameweek 1"):
        team_strengths(TWO_GAMES, before_gw=1, prior_matches=0)


def test_expected_goals_poisson_formula():
    strengths = pd.DataFrame({"team_id": [1, 2], "attack": [1.2, 0.8], "defense": [0.9, 1.5]})
    league = LeagueAverages(home_goals=1.5, away_goals=1.2)
    xg = expected_goals(pd.DataFrame([fixture(10, 2, 1, 2)]), strengths, league)
    home = xg[xg["was_home"]].iloc[0]
    away = xg[~xg["was_home"]].iloc[0]
    assert home["xg_for"] == pytest.approx(1.5 * 1.2 * 1.5)  # home avg × att 1 × def 2
    assert away["xg_for"] == pytest.approx(1.2 * 0.8 * 0.9)  # away avg × att 2 × def 1
    assert home["xg_against"] == away["xg_for"]
    assert (home["team_id"], home["opponent_team_id"]) == (1, 2)


def test_expected_goals_unknown_team_is_average():
    strengths = pd.DataFrame({"team_id": [1], "attack": [2.0], "defense": [1.0]})
    league = LeagueAverages(home_goals=1.0, away_goals=1.0)
    xg = expected_goals(pd.DataFrame([fixture(10, 2, 1, 99)]), strengths, league)
    assert xg[xg["was_home"]]["xg_for"].item() == pytest.approx(2.0)  # 99 defends at 1.0
    assert xg[~xg["was_home"]]["xg_for"].item() == pytest.approx(1.0)


# --- predict_points -----------------------------------------------------------

# GW1: four teams, all games 1-1, so every team is exactly average (1.0/1.0)
# and the league averages 1 goal home and away.
LEVEL_GW1 = [fixture(1, 1, 1, 2, 1, 1), fixture(2, 1, 3, 4, 1, 1)]
NO_SHRINK = dict(prior_matches=0, n_gameweeks=5, half_life=2)


def predict(players, history, fixtures, gw=2, **params):
    return predict_points(
        pd.DataFrame(players), pd.DataFrame(history), pd.DataFrame(fixtures), gw,
        ModelParams(**{**NO_SHRINK, **params}),
    ).set_index("player_id")


def test_average_fixture_leaves_form_unchanged():
    preds = predict(
        [{"player_id": 7, "team_id": 1, "position": "MID"}],
        history_rows(7, {1: 6}),
        LEVEL_GW1 + [fixture(3, 2, 1, 3)],
    )
    assert preds.loc[7, "fixture_multiplier"] == pytest.approx(1.0)
    assert preds.loc[7, "predicted_points"] == pytest.approx(6.0)


def test_attacker_boosted_by_leaky_defense_defender_by_blunt_attack():
    # GW1: team 2 lost 0-4 to team 1 (attack 0, defense 2 before shrinkage);
    # teams 3 and 4 drew 2-2. League average 2 goals per team per match.
    gw1 = [fixture(1, 1, 1, 2, 4, 0), fixture(2, 1, 3, 4, 2, 2)]
    players = [{"player_id": 7, "team_id": 3, "position": "FWD"},
               {"player_id": 8, "team_id": 3, "position": "DEF"},
               {"player_id": 9, "team_id": 4, "position": "FWD"}]
    history = history_rows(7, {1: 5}) + history_rows(8, {1: 5}) + history_rows(9, {1: 5})
    fixtures = gw1 + [fixture(3, 2, 3, 2), fixture(4, 2, 1, 4)]

    preds = predict(players, history, fixtures, prior_matches=1, fixture_weight=1)
    # With 1 prior match: team 2 attack (0 + 2)/2/2 = 0.5, defense (4 + 2)/2/2 = 1.5.
    # Team 3 (attack 1, defense 1) hosts team 2; home avg 3, away avg 1:
    #   xg_for = 3 × 1 × 1.5 = 4.5 → attack mult 4.5 / 2 = 2.25
    #   xg_against = 1 × 0.5 × 1 = 0.5 → defend mult 2 / 0.5 = 4.0
    assert preds.loc[7, "predicted_points"] == pytest.approx(5 * 2.25)
    assert preds.loc[8, "predicted_points"] == pytest.approx(5 * 4.0)
    # Team 4 away at team 1 (attack 1.5, defense 0.5): xg_for = 1 × 1 × 0.5 = 0.5
    assert preds.loc[9, "fixture_multiplier"] == pytest.approx(0.25)


def test_fixture_weight_zero_ignores_fixtures():
    gw1 = [fixture(1, 1, 1, 2, 4, 0), fixture(2, 1, 3, 4, 2, 2)]
    preds = predict(
        [{"player_id": 7, "team_id": 3, "position": "FWD"}],
        history_rows(7, {1: 5}),
        gw1 + [fixture(3, 2, 3, 2)],
        fixture_weight=0,
    )
    assert preds.loc[7, "predicted_points"] == pytest.approx(5.0)


def test_fixture_weight_dampens_the_multiplier():
    gw1 = [fixture(1, 1, 1, 2, 4, 0), fixture(2, 1, 3, 4, 2, 2)]
    preds = predict(
        [{"player_id": 7, "team_id": 3, "position": "FWD"}],
        history_rows(7, {1: 5}),
        gw1 + [fixture(3, 2, 3, 2)],
        prior_matches=1, fixture_weight=0.5,
    )
    assert preds.loc[7, "fixture_multiplier"] == pytest.approx(2.25 ** 0.5)


def test_double_gameweek_sums_both_fixtures_and_blank_is_zero():
    players = [{"player_id": 7, "team_id": 1, "position": "MID"},
               {"player_id": 8, "team_id": 4, "position": "MID"}]
    history = history_rows(7, {1: 6}) + history_rows(8, {1: 6})
    # GW2: team 1 plays twice, team 4 doesn't play at all.
    fixtures = LEVEL_GW1 + [fixture(3, 2, 1, 2), fixture(4, 2, 3, 1)]
    preds = predict(players, history, fixtures)
    assert preds.loc[7, "n_fixtures"] == 2
    assert preds.loc[7, "predicted_points"] == pytest.approx(12.0)
    assert preds.loc[8, "n_fixtures"] == 0
    assert preds.loc[8, "predicted_points"] == 0


def test_player_without_history_predicts_zero():
    preds = predict(
        [{"player_id": 7, "team_id": 1, "position": "MID"},
         {"player_id": 8, "team_id": 1, "position": "MID"}],
        history_rows(7, {1: 6}),
        LEVEL_GW1 + [fixture(3, 2, 1, 3)],
    )
    assert preds.loc[8, "predicted_points"] == 0
    assert preds.loc[8, "total_minutes"] == 0


def test_output_is_sorted_and_stamped_with_gameweek():
    players = [{"player_id": i, "team_id": 1, "position": "MID"} for i in (1, 2, 3)]
    history = history_rows(1, {1: 2}) + history_rows(2, {1: 9}) + history_rows(3, {1: 5})
    preds = predict_points(
        pd.DataFrame(players), pd.DataFrame(history),
        pd.DataFrame(LEVEL_GW1 + [fixture(3, 2, 1, 3)]), 2, ModelParams(**NO_SHRINK),
    )
    assert list(preds["player_id"]) == [2, 3, 1]
    assert (preds["gameweek"] == 2).all()
    assert (preds["window_gameweeks"] == 1).all()  # only GW1 exists before GW2
