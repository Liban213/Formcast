"""Ingestion runs end-to-end into a real Postgres test database."""

import pytest
from sqlalchemy import func, select, text

from formcast.fpl_client import FPLClient
from formcast.ingest import run_ingestion
from formcast.models import Fixture, Gameweek, Player, PlayerGameweekStats, Team


def count(engine, model) -> int:
    with engine.connect() as conn:
        return conn.scalar(select(func.count()).select_from(model))


def test_ingestion_runs_end_to_end(engine, fake_client):
    result = run_ingestion(fake_client, engine)

    assert count(engine, Team) == result.teams == 20
    assert count(engine, Gameweek) == result.gameweeks == 38
    assert count(engine, Player) == result.players == 3
    assert count(engine, Fixture) == result.fixtures == 380
    assert count(engine, PlayerGameweekStats) == result.player_gameweek_stats == 10

    with engine.connect() as conn:
        gross = conn.execute(
            text("SELECT price, ownership_pct, position FROM players WHERE id = 124")
        ).one()
        assert (str(gross.price), str(gross.ownership_pct), gross.position) == (
            "5.8", "29.10", "MID"
        )


def test_players_with_no_minutes_are_not_fetched(engine, fake_client):
    run_ingestion(fake_client, engine)
    # Player 2 has played 0 minutes: stored in players, but no history request.
    assert sorted(fake_client.summary_calls) == [1, 124]


def test_rerunning_ingestion_updates_instead_of_duplicating(engine, fake_client):
    run_ingestion(fake_client, engine)

    original = fake_client.bootstrap_static
    def bootstrap_with_price_rise():
        data = original()
        data["elements"][0]["now_cost"] = 62
        return data
    fake_client.bootstrap_static = bootstrap_with_price_rise

    run_ingestion(fake_client, engine)

    assert count(engine, Player) == 3
    assert count(engine, PlayerGameweekStats) == 10
    with engine.connect() as conn:
        assert str(conn.scalar(text("SELECT price FROM players WHERE id = 1"))) == "6.2"


def test_failed_fetch_leaves_database_untouched(engine, fake_client):
    def broken_summary(player_id):
        raise ConnectionError("FPL is down")
    fake_client.element_summary = broken_summary

    with pytest.raises(ConnectionError):
        run_ingestion(fake_client, engine)

    assert count(engine, Team) == 0
    assert count(engine, Player) == 0


@pytest.mark.live
def test_live_fpl_api_matches_expected_shape():
    """Hits the real API. Run with: pytest -m live"""
    client = FPLClient(request_delay=0)
    bootstrap = client.bootstrap_static()
    assert {"teams", "events", "elements", "element_types"} <= bootstrap.keys()
    assert len(bootstrap["teams"]) == 20

    player = next(p for p in bootstrap["elements"] if p["minutes"] > 0)
    summary = client.element_summary(player["id"])
    assert summary["history"], "an active player should have match history"
