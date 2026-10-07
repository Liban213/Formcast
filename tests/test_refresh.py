"""The refresh job stores complete, timestamped runs, and never a partial one."""

import pandas as pd
import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from formcast.lists import GameweekLists
from formcast.models import Prediction, PredictionRun
from formcast.predict.rank import Rankings
from formcast.refresh import latest_run_created_at, run_refresh, store_run


def count(engine, model) -> int:
    with engine.connect() as conn:
        return conn.scalar(select(func.count()).select_from(model))


def test_refresh_ingests_predicts_and_stores_a_run(engine, fake_client):
    result = run_refresh(fake_client, engine)
    assert (result.run_id, result.gameweek) == (1, 6)
    assert result.ingest.players == 3

    with engine.connect() as conn:
        run = conn.execute(select(PredictionRun)).one()
        rows = conn.execute(
            select(Prediction).order_by(Prediction.list_name, Prediction.rank)
        ).all()
    assert run.gameweek == 6
    assert run.params["model"]["half_life"] == 2.0  # settings recorded with the run
    assert run.params["rank"]["captain_min_ownership"] == 10.0

    # Of the 3 sample players, Raya and Groß are popular and have minutes;
    # Arrizabalaga hasn't played, so he's on no list.
    assert [(r.list_name, r.rank, r.web_name) for r in rows] == [
        ("captains", 1, "Groß"),
        ("captains", 2, "Raya"),
    ]
    gross = rows[0]
    assert (gross.team, gross.position, gross.opponent) == ("BHA", "MID", "SUN (A)")
    assert gross.predicted_points == pytest.approx(15.19, abs=0.01)
    assert gross.reasons is None


def test_each_refresh_adds_a_run_and_keeps_old_ones(engine, fake_client):
    run_refresh(fake_client, engine)
    second = run_refresh(fake_client, engine)
    assert second.run_id == 2
    assert count(engine, PredictionRun) == 2
    assert count(engine, Prediction) == 4
    assert latest_run_created_at(engine) is not None


def test_failed_ingestion_stores_no_run(engine, fake_client):
    def broken(player_id):
        raise ConnectionError("FPL is down")
    fake_client.element_summary = broken
    with pytest.raises(ConnectionError):
        run_refresh(fake_client, engine)
    assert count(engine, PredictionRun) == 0


def test_a_run_is_all_or_nothing(engine, fake_client):
    run_refresh(fake_client, engine)
    with engine.connect() as conn:
        good = pd.read_sql(text("SELECT * FROM predictions"), conn)
    good = good.rename(columns={"team": "short_name"})
    bad = good.assign(player_id=99999)  # no such player: fails the foreign key
    lists = GameweekLists(6, Rankings(captains=good, differentials=bad, avoid=good.iloc[:0]))

    with pytest.raises(IntegrityError):
        store_run(engine, lists, params={})
    # The run row and its valid captains were rolled back with the bad rows.
    assert count(engine, PredictionRun) == 1
    assert count(engine, Prediction) == 2


def test_no_upcoming_gameweek_stores_nothing(engine, fake_client):
    original = fake_client.bootstrap_static
    def season_over():
        data = original()
        for event in data["events"]:
            event["is_next"] = False
        return data
    fake_client.bootstrap_static = season_over

    result = run_refresh(fake_client, engine)
    assert result.run_id is None
    assert count(engine, PredictionRun) == 0
