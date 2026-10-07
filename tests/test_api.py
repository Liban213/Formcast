"""The endpoints return the latest stored run, and never compute anything themselves."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import insert, update

from formcast import api, lists, refresh
from formcast.models import Prediction, PredictionRun
from formcast.predict import model as predict_model


@pytest.fixture
def client(engine):
    api.app.dependency_overrides[api.get_engine] = lambda: engine
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


@pytest.fixture
def stored(engine, fake_client):
    """Ingest the sample data and store one real run (GW6: Groß, Raya captains)."""
    return refresh.run_refresh(fake_client, engine)


ENDPOINTS = ["/api/captain-picks", "/api/differentials", "/api/avoid"]


@pytest.mark.parametrize("path", ENDPOINTS)
def test_no_run_yet_is_503_not_a_crash(client, path):
    response = client.get(path)
    assert response.status_code == 503
    assert "first refresh" in response.json()["detail"]


def test_captain_picks_returns_the_stored_list(client, stored):
    response = client.get("/api/captain-picks")
    assert response.status_code == 200
    body = response.json()
    assert body["gameweek"] == 6
    assert body["generated_at"]
    assert [p["name"] for p in body["players"]] == ["Groß", "Raya"]
    gross = body["players"][0]
    assert gross == {
        "rank": 1, "player_id": 124, "name": "Groß", "team": "BHA", "position": "MID",
        "price": 5.8, "ownership_pct": 29.1, "opponent": "SUN (A)", "form": 11.25,
        "fixture_multiplier": 1.35, "predicted_points": 15.19, "news": "",
    }


def test_empty_list_is_an_empty_array(client, stored):
    body = client.get("/api/differentials").json()
    assert body["gameweek"] == 6
    assert body["players"] == []


def test_avoid_includes_reasons(client, engine, stored):
    with engine.begin() as conn:
        conn.execute(insert(Prediction).values(
            run_id=stored.run_id, list_name="avoid", rank=1, player_id=1, web_name="Raya",
            team="ARS", position="GKP", price=6.1, ownership_pct=42.3, opponent="blank",
            form=5.7, fixture_multiplier=None, predicted_points=0.0,
            reasons="injured, no fixture", news="Knee injury - Expected back 20 Oct",
        ))
    [player] = client.get("/api/avoid").json()["players"]
    assert player["reasons"] == "injured, no fixture"
    assert player["fixture_multiplier"] is None
    assert player["news"].startswith("Knee injury")


def test_serves_only_the_newest_run(client, engine, fake_client, stored):
    second = refresh.run_refresh(fake_client, engine)
    # Make the newer run distinguishable: Raya first.
    with engine.begin() as conn:
        conn.execute(update(PredictionRun).where(PredictionRun.id == second.run_id)
                     .values(gameweek=7))
        conn.execute(update(Prediction).where(Prediction.run_id == second.run_id)
                     .values(web_name=Prediction.web_name + " (new)"))
    body = client.get("/api/captain-picks").json()
    assert body["gameweek"] == 7
    assert [p["name"] for p in body["players"]] == ["Groß (new)", "Raya (new)"]


def test_requests_never_run_the_model(client, stored, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("an API request triggered a model run")
    monkeypatch.setattr(predict_model, "predict_points", forbidden)
    monkeypatch.setattr(lists, "predict_points", forbidden)
    monkeypatch.setattr(lists, "build_lists", forbidden)
    monkeypatch.setattr(refresh, "run_refresh", forbidden)
    for path in ENDPOINTS:
        assert client.get(path).status_code == 200


@pytest.mark.parametrize("path", ENDPOINTS)
def test_responses_are_cacheable(client, stored, path):
    assert client.get(path).headers["cache-control"] == "public, max-age=300"


def test_health_reports_freshness(client, stored):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["latest_gameweek"] == 6
    assert body["latest_run_at"].endswith("Z") or body["latest_run_at"].endswith("+00:00")


def test_health_before_any_run(client):
    assert client.get("/api/health").json() == {
        "status": "ok", "latest_run_at": None, "latest_gameweek": None,
    }
